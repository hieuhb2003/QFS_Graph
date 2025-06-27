"""
Cluster Summary Generator for GraphRAG System
Tạo summary cho clusters bằng LLM với map-reduce
"""

import asyncio
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

from ..utils.logger_config import get_logger
from ..clients.llm_client import BaseLLMClient


class ClusterSummaryGenerator:
    """Tạo summary cho clusters bằng LLM với map-reduce"""
    
    def __init__(self, 
                 llm_client: BaseLLMClient,
                 max_workers: int = 4,
                 context_length: int = 4096,
                 summary_prompt_template: Optional[str] = None,
                 map_reduce_prompt_template: Optional[str] = None):
        """
        Khởi tạo ClusterSummaryGenerator
        
        Args:
            llm_client: LLM client để tạo summary
            max_workers: Số worker tối đa cho parallel processing
            context_length: Độ dài context tối đa
            summary_prompt_template: Template cho summary prompt
            map_reduce_prompt_template: Template cho map-reduce prompt
        """
        self.llm_client = llm_client
        self.max_workers = max_workers
        self.context_length = context_length
        self.logger = get_logger(__name__)
        
        # Default prompt templates
        self.summary_prompt_template = summary_prompt_template or """
You are an expert document analyst. Please create a concise and accurate summary for the following documents:

Documents:
{documents}

Requirements:
1. Summarize the main content of the documents
2. Identify key topics and important concepts
3. Create a well-structured summary
4. Keep the summary under 300 words

Summary:
"""
        
        self.map_reduce_prompt_template = map_reduce_prompt_template or """
You are an expert document analyst. Please create a comprehensive summary from the following sub-summaries:

Sub-summaries:
{summaries}

Requirements:
1. Synthesize information from all sub-summaries
2. Remove duplicate information
3. Create a final well-structured summary
4. Keep the summary under 500 words
5. Ensure consistency and logical flow

Comprehensive Summary:
"""
    
    async def generate_cluster_summaries(self, 
                                       cluster_documents: Dict[int, List[str]], 
                                       doc_content_map: Dict[str, str]) -> Dict[int, str]:
        """
        Tạo summary cho tất cả clusters
        
        Args:
            cluster_documents: {cluster_id: [doc_hash_ids]}
            doc_content_map: {doc_hash_id: content}
            
        Returns:
            {cluster_id: summary}
        """
        self.logger.info(f"Bắt đầu tạo summary cho {len(cluster_documents)} clusters...")
        
        # Tạo semaphore để giới hạn số worker
        semaphore = asyncio.Semaphore(self.max_workers)
        
        # Tạo tasks cho từng cluster
        tasks = []
        for cluster_id, doc_hash_ids in cluster_documents.items():
            if cluster_id == -1:  # Bỏ qua outlier cluster
                continue
            task = self._generate_single_cluster_summary(
                cluster_id, doc_hash_ids, doc_content_map, semaphore
            )
            tasks.append(task)
        
        # Chạy parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Xử lý kết quả
        summaries = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                cluster_id = list(cluster_documents.keys())[i]
                self.logger.error(f"Lỗi khi tạo summary cho cluster {cluster_id}: {result}")
                summaries[cluster_id] = f"Lỗi tạo summary: {str(result)}"
            else:
                cluster_id, summary = result
                summaries[cluster_id] = summary
        
        self.logger.info(f"Hoàn thành tạo summary cho {len(summaries)} clusters")
        return summaries
    
    async def _generate_single_cluster_summary(self, 
                                             cluster_id: int, 
                                             doc_hash_ids: List[str], 
                                             doc_content_map: Dict[str, str],
                                             semaphore: asyncio.Semaphore) -> Tuple[int, str]:
        """Tạo summary cho một cluster cụ thể"""
        async with semaphore:
            try:
                self.logger.info(f"Đang tạo summary cho cluster {cluster_id} với {len(doc_hash_ids)} documents...")
                
                # Lấy nội dung documents
                documents = []
                for doc_hash_id in doc_hash_ids:
                    if doc_hash_id in doc_content_map:
                        documents.append(doc_content_map[doc_hash_id])
                
                if not documents:
                    return cluster_id, "Không có documents để tạo summary"
                
                # Tạo summary bằng map-reduce
                summary = await self._map_reduce_summary(documents)
                
                self.logger.info(f"Hoàn thành summary cho cluster {cluster_id}")
                return cluster_id, summary
                
            except Exception as e:
                self.logger.error(f"Lỗi khi tạo summary cho cluster {cluster_id}: {e}")
                raise
    
    async def _map_reduce_summary(self, documents: List[str]) -> str:
        """
        Tạo summary bằng phương pháp map-reduce
        
        Args:
            documents: Danh sách nội dung documents
            
        Returns:
            Summary cuối cùng
        """
        # Map phase: Tạo summary cho từng chunk documents
        chunk_summaries = await self._map_phase(documents)
        
        # Reduce phase: Tổng hợp các summary
        final_summary = await self._reduce_phase(chunk_summaries)
        
        return final_summary
    
    async def _map_phase(self, documents: List[str]) -> List[str]:
        """Map phase: Tạo summary cho từng chunk documents"""
        summaries = []
        
        # Chia documents thành chunks dựa trên context length
        chunks = self._split_documents_into_chunks(documents)
        
        # Tạo summary cho từng chunk
        for chunk in chunks:
            try:
                summary = await self._generate_chunk_summary(chunk)
                summaries.append(summary)
            except Exception as e:
                self.logger.error(f"Lỗi khi tạo summary cho chunk: {e}")
                summaries.append(f"Lỗi tạo summary: {str(e)}")
        
        return summaries
    
    async def _reduce_phase(self, summaries: List[str]) -> str:
        """Reduce phase: Tổng hợp các summary"""
        if not summaries:
            return "Không có summary để tổng hợp"
        
        if len(summaries) == 1:
            return summaries[0]
        
        try:
            # Tạo prompt cho reduce phase
            summaries_text = "\n\n".join([f"Summary {i+1}: {summary}" for i, summary in enumerate(summaries)])
            prompt = self.map_reduce_prompt_template.format(summaries=summaries_text)
            
            # Gọi LLM để tạo summary tổng hợp
            response = await self.llm_client.generate_text(prompt)
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Lỗi trong reduce phase: {e}")
            # Fallback: nối các summary lại
            return "\n\n".join(summaries)
    
    def _split_documents_into_chunks(self, documents: List[str]) -> List[List[str]]:
        """
        Chia documents thành chunks dựa trên context length
        
        Args:
            documents: Danh sách documents
            
        Returns:
            List các chunks
        """
        chunks = []
        current_chunk = []
        current_length = 0
        
        for doc in documents:
            doc_length = len(doc)
            
            # Nếu thêm document này vượt quá context length
            if current_length + doc_length > self.context_length and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [doc]
                current_length = doc_length
            else:
                current_chunk.append(doc)
                current_length += doc_length
        
        # Thêm chunk cuối cùng
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    async def _generate_chunk_summary(self, documents: List[str]) -> str:
        """
        Tạo summary cho một chunk documents
        
        Args:
            documents: Danh sách documents trong chunk
            
        Returns:
            Summary cho chunk
        """
        try:
            # Tạo prompt
            documents_text = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(documents)])
            prompt = self.summary_prompt_template.format(documents=documents_text)
            
            # Gọi LLM để tạo summary
            response = await self.llm_client.generate_text(prompt)
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo chunk summary: {e}")
            # Fallback: tạo summary đơn giản
            return f"Summary cho {len(documents)} documents: " + " ".join([doc[:100] + "..." for doc in documents[:3]])
    
    async def generate_single_cluster_summary(self, 
                                            cluster_id: int, 
                                            doc_hash_ids: List[str], 
                                            doc_content_map: Dict[str, str]) -> str:
        """
        Tạo summary cho một cluster cụ thể (public method)
        
        Args:
            cluster_id: ID của cluster
            doc_hash_ids: Danh sách hash IDs của documents
            doc_content_map: Map từ doc_hash_id sang content
            
        Returns:
            Summary của cluster
        """
        try:
            # Lấy nội dung documents
            documents = []
            for doc_hash_id in doc_hash_ids:
                if doc_hash_id in doc_content_map:
                    documents.append(doc_content_map[doc_hash_id])
            
            if not documents:
                return "Không có documents để tạo summary"
            
            # Tạo summary
            summary = await self._map_reduce_summary(documents)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo summary cho cluster {cluster_id}: {e}")
            return f"Lỗi tạo summary: {str(e)}"
    
    def update_prompt_templates(self, 
                              summary_prompt_template: Optional[str] = None,
                              map_reduce_prompt_template: Optional[str] = None):
        """
        Cập nhật prompt templates
        
        Args:
            summary_prompt_template: Template mới cho summary
            map_reduce_prompt_template: Template mới cho map-reduce
        """
        if summary_prompt_template:
            self.summary_prompt_template = summary_prompt_template
        
        if map_reduce_prompt_template:
            self.map_reduce_prompt_template = map_reduce_prompt_template
        
        self.logger.info("Đã cập nhật prompt templates") 