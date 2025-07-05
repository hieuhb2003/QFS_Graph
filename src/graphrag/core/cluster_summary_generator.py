"""
Cluster Summary Generator for GraphRAG System
Tạo summary cho clusters bằng LLM với map-reduce
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
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
                 map_reduce_prompt_template: Optional[str] = None,
                 query_generation_prompt_template: Optional[str] = None):
        """
        Khởi tạo ClusterSummaryGenerator
        
        Args:
            llm_client: LLM client để tạo summary
            max_workers: Số worker tối đa cho parallel processing
            context_length: Độ dài context tối đa
            summary_prompt_template: Template cho summary prompt
            map_reduce_prompt_template: Template cho map-reduce prompt
            query_generation_prompt_template: Template cho query generation prompt
        """
        self.llm_client = llm_client
        self.max_workers = max_workers
        self.context_length = context_length
        self.logger = get_logger()
        
        # Default prompt templates
        self.summary_prompt_template = summary_prompt_template or """
You are a highly skilled data analyst AI. Your task is to extract the core information from a set of documents. This summary will be used later to synthesize a larger picture.

Your goal is to create a dense, factual summary. Focus on answering "Who? What? When? Where? Why?".

Documents:
{documents}

Instructions:
1.  Identify the Central Theme: What is the single most important topic, event, or entity discussed in these documents?
2.  Extract Key Entities: List the key people, organizations, locations, and specific terms mentioned.
3.  Synthesize: Based on the above, write a concise, neutral, and factual paragraph summarizing the main points. Do NOT add opinions or interpretations. Focus only on the information present in the text.

Dense Summary:
"""
        
        self.map_reduce_prompt_template = map_reduce_prompt_template or """
You are an expert information synthesizer. You have been given several sub-summaries from a single, coherent cluster of documents. Your task is to intelligently merge them into a final, comprehensive, and non-redundant summary.

Sub-summaries:
{summaries}

Instructions:
1.  Synthesize, Don't Stack: Read all sub-summaries to understand the complete picture. Do NOT simply list the points from each one. Instead, weave them into a single, coherent narrative.
2.  De-duplicate: Identify and merge overlapping information. If multiple summaries mention the same event or fact, present it only once in the final summary.
3.  Provide a Holistic View: The final summary must represent the entire cluster as a whole, capturing its main theme, key actors, and primary outcomes. It should be a self-contained, easy-to-read paragraph.

Final Comprehensive Summary:
"""
        
        self.query_generation_prompt_template = query_generation_prompt_template or """
You are an expert assistant. Based on the following cluster summaries, please provide a comprehensive answer to the user's question.

User Question: {query}

Relevant Cluster Summaries:
{summaries}

Requirements:
1. Synthesize information from all sub-summaries
2. Remove duplicate information
3. Create a final well-structured summary
4. Keep the summary under 500 words
5. Ensure consistency and logical flow

Comprehensive Summary:
"""
        
        self.query_generation_prompt_template = query_generation_prompt_template or """
You are an expert assistant. Based on the following cluster summaries, please provide a comprehensive answer to the user's question.

User Question: {query}

Relevant Cluster Summaries:
{summaries}

Requirements:
1. Answer the question based on the information in the cluster summaries
2. Provide specific details and examples from the summaries
3. If the information is not available in the summaries, clearly state that
4. Keep the answer well-structured and informative
5. Cite which clusters the information comes from when relevant

Answer:
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
            response = await self.llm_client.generate(prompt)
            
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
            response = await self.llm_client.generate(prompt)
            
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
                              map_reduce_prompt_template: Optional[str] = None,
                              query_generation_prompt_template: Optional[str] = None):
        """
        Cập nhật prompt templates
        
        Args:
            summary_prompt_template: Template mới cho summary
            map_reduce_prompt_template: Template mới cho map-reduce
            query_generation_prompt_template: Template mới cho query generation
        """
        if summary_prompt_template:
            self.summary_prompt_template = summary_prompt_template
        
        if map_reduce_prompt_template:
            self.map_reduce_prompt_template = map_reduce_prompt_template
        
        if query_generation_prompt_template:
            self.query_generation_prompt_template = query_generation_prompt_template
        
        self.logger.info("Đã cập nhật prompt templates")
    
    async def query_cluster_summaries(self, 
                                    query: str, 
                                    cluster_summaries: List[Dict[str, Any]], 
                                    mode: str = "retrieval") -> Dict[str, Any]:
        """
        Query cluster summaries với 2 mode: retrieval và generation
        
        Args:
            query: Query string
            cluster_summaries: List các cluster summaries từ vector DB (đã được top_k)
            mode: "retrieval" hoặc "generation"
            
        Returns:
            Dict chứa kết quả query
        """
        try:
            self.logger.info(f"Querying cluster summaries with mode: {mode}")
            
            if mode == "retrieval":
                return await self._retrieval_mode(query, cluster_summaries)
            elif mode == "generation":
                return await self._generation_mode(query, cluster_summaries)
            else:
                raise ValueError(f"Mode không hợp lệ: {mode}. Chỉ hỗ trợ 'retrieval' hoặc 'generation'")
                
        except Exception as e:
            self.logger.error(f"Lỗi khi query cluster summaries: {e}")
            return {
                "error": str(e),
                "mode": mode,
                "query": query
            }
    
    async def _retrieval_mode(self, 
                            query: str, 
                            cluster_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Mode retrieval: chỉ trả về summaries đã được top_k từ vector DB"""
        try:
            # Format kết quả từ vector DB
            results = []
            for summary in cluster_summaries:
                results.append({
                    "cluster_id": summary.get("cluster_id"),
                    "summary": summary.get("summary_text", summary.get("summary", "")),
                    "doc_hash_ids": summary.get("doc_hash_ids", []),
                    "score": summary.get("distance", 0.0)  # Vector DB trả về distance
                })
            
            return {
                "mode": "retrieval",
                "query": query,
                "results": results,
                "total_found": len(results)
            }
            
        except Exception as e:
            self.logger.error(f"Lỗi trong retrieval mode: {e}")
            return {
                "mode": "retrieval",
                "query": query,
                "error": str(e),
                "results": []
            }
    
    async def _generation_mode(self, 
                             query: str, 
                             cluster_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Mode generation: gen câu trả lời từ summaries đã được top_k"""
        try:
            if not self.llm_client:
                self.logger.warning("Không có LLM client, fallback về retrieval mode")
                return await self._retrieval_mode(query, cluster_summaries)
            
            if not cluster_summaries:
                return {
                    "mode": "generation",
                    "query": query,
                    "answer": "Không tìm thấy thông tin liên quan để trả lời câu hỏi.",
                    "used_summaries": [],
                    "total_found": 0
                }
            
            # Tạo prompt cho generation
            summaries_text = "\n\n".join([
                f"Cluster {summary.get('cluster_id')}: {summary.get('summary_text', summary.get('summary', ''))}"
                for summary in cluster_summaries
            ])
            
            prompt = self.query_generation_prompt_template.format(
                query=query,
                summaries=summaries_text
            )
            
            # Gọi LLM để tạo câu trả lời
            self.logger.info("Đang tạo câu trả lời bằng LLM...")
            answer = await self.llm_client.generate_text(prompt)
            
            # Format kết quả
            used_summaries = []
            for summary in cluster_summaries:
                used_summaries.append({
                    "cluster_id": summary.get("cluster_id"),
                    "summary": summary.get("summary_text", summary.get("summary", "")),
                    "doc_hash_ids": summary.get("doc_hash_ids", []),
                    "score": summary.get("distance", 0.0)
                })
            
            return {
                "mode": "generation",
                "query": query,
                "answer": answer.strip(),
                "used_summaries": used_summaries,
                "total_found": len(used_summaries)
            }
            
        except Exception as e:
            self.logger.error(f"Lỗi trong generation mode: {e}")
            return {
                "mode": "generation",
                "query": query,
                "error": str(e),
                "answer": f"Lỗi khi tạo câu trả lời: {str(e)}",
                "used_summaries": []
            } 