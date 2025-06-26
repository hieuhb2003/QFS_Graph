"""
Cluster Summary Generator for GraphRAG System
Uses map-reduce approach to generate cluster summaries with LLM
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import math

from ..utils.logger_config import get_logger
from ..clients.llm_client import BaseLLMClient


@dataclass
class ClusterSummary:
    """Summary information for a cluster"""
    cluster_id: int
    summary: str
    keywords: List[str]
    doc_count: int
    processing_time: float
    token_count: int


class ClusterSummaryGenerator:
    """
    Generates summaries for clusters using map-reduce approach
    """
    
    def __init__(self, 
                 llm_client: BaseLLMClient,
                 max_tokens_per_batch: int = 4000,
                 max_concurrent_batches: int = 3):
        """
        Initialize cluster summary generator
        
        Args:
            llm_client: LLM client for generating summaries
            max_tokens_per_batch: Maximum tokens per batch for map-reduce
            max_concurrent_batches: Maximum concurrent batches for parallel processing
        """
        self.llm_client = llm_client
        self.max_tokens_per_batch = max_tokens_per_batch
        self.max_concurrent_batches = max_concurrent_batches
        self.logger = get_logger()
        
        # Statistics
        self.summary_stats = {
            'total_clusters_processed': 0,
            'total_summaries_generated': 0,
            'total_processing_time': 0.0,
            'total_tokens_used': 0
        }
    
    async def generate_cluster_summaries(self, 
                                       cluster_documents: Dict[int, List[str]],
                                       cluster_keywords: Dict[int, List[str]]) -> Dict[int, ClusterSummary]:
        """
        Generate summaries for multiple clusters using map-reduce approach
        
        Args:
            cluster_documents: Dict[cluster_id, List[document_contents]]
            cluster_keywords: Dict[cluster_id, List[keywords]]
            
        Returns:
            Dict[cluster_id, ClusterSummary]: Generated summaries
        """
        self.logger.info(f"Starting cluster summary generation for {len(cluster_documents)} clusters")
        
        # Semaphore for controlling concurrent batches
        semaphore = asyncio.Semaphore(self.max_concurrent_batches)
        
        async def process_single_cluster(cluster_id: int) -> Tuple[int, ClusterSummary]:
            async with semaphore:
                return await self._generate_single_cluster_summary(
                    cluster_id, 
                    cluster_documents[cluster_id], 
                    cluster_keywords[cluster_id]
                )
        
        # Process all clusters concurrently
        tasks = [
            process_single_cluster(cluster_id) 
            for cluster_id in cluster_documents.keys()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        summaries = {}
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Error in cluster summary generation: {result}")
                continue
            
            cluster_id, summary = result
            summaries[cluster_id] = summary
            
            # Update statistics
            self.summary_stats['total_clusters_processed'] += 1
            self.summary_stats['total_summaries_generated'] += 1
            self.summary_stats['total_processing_time'] += summary.processing_time
            self.summary_stats['total_tokens_used'] += summary.token_count
        
        self.logger.info(f"Completed summary generation: {len(summaries)} summaries created")
        return summaries
    
    async def _generate_single_cluster_summary(self, 
                                             cluster_id: int, 
                                             documents: List[str], 
                                             keywords: List[str]) -> ClusterSummary:
        """
        Generate summary for a single cluster using map-reduce approach
        
        Args:
            cluster_id: Cluster ID
            documents: List of document contents in the cluster
            keywords: List of keywords for the cluster
            
        Returns:
            ClusterSummary: Generated summary
        """
        start_time = asyncio.get_event_loop().time()
        
        self.logger.info(f"Generating summary for cluster {cluster_id} with {len(documents)} documents")
        
        try:
            # Step 1: Map - Split documents into batches
            batches = self._create_document_batches(documents)
            self.logger.info(f"Created {len(batches)} batches for cluster {cluster_id}")
            
            # Step 2: Reduce - Process batches and combine results
            batch_summaries = []
            for i, batch in enumerate(batches):
                batch_summary = await self._process_document_batch(batch, i, len(batches))
                batch_summaries.append(batch_summary)
            
            # Step 3: Final reduction - Combine all batch summaries
            final_summary = await self._combine_batch_summaries(
                batch_summaries, 
                keywords, 
                cluster_id
            )
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return ClusterSummary(
                cluster_id=cluster_id,
                summary=final_summary,
                keywords=keywords,
                doc_count=len(documents),
                processing_time=processing_time,
                token_count=self._estimate_token_count(final_summary)
            )
            
        except Exception as e:
            self.logger.error(f"Error generating summary for cluster {cluster_id}: {e}")
            return ClusterSummary(
                cluster_id=cluster_id,
                summary=f"Error generating summary: {str(e)}",
                keywords=keywords,
                doc_count=len(documents),
                processing_time=asyncio.get_event_loop().time() - start_time,
                token_count=0
            )
    
    def _create_document_batches(self, documents: List[str]) -> List[List[str]]:
        """
        Create batches of documents that fit within token limits
        
        Args:
            documents: List of document contents
            
        Returns:
            List[List[str]]: Batches of documents
        """
        batches = []
        current_batch = []
        current_tokens = 0
        
        for doc in documents:
            # Estimate tokens for this document (rough estimation: 1 token ≈ 4 characters)
            doc_tokens = len(doc) // 4
            
            # If adding this document would exceed limit, start new batch
            if current_tokens + doc_tokens > self.max_tokens_per_batch and current_batch:
                batches.append(current_batch)
                current_batch = [doc]
                current_tokens = doc_tokens
            else:
                current_batch.append(doc)
                current_tokens += doc_tokens
        
        # Add final batch if not empty
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    async def _process_document_batch(self, 
                                    batch: List[str], 
                                    batch_index: int, 
                                    total_batches: int) -> str:
        """
        Process a single batch of documents to generate intermediate summary
        
        Args:
            batch: List of document contents
            batch_index: Index of this batch
            total_batches: Total number of batches
            
        Returns:
            str: Intermediate summary for this batch
        """
        # Create prompt for batch processing
        documents_text = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(batch)])
        
        prompt = f"""You are analyzing a batch of documents (batch {batch_index + 1} of {total_batches}). 
Please provide a concise summary of the key topics and themes in these documents.

Documents:
{documents_text}

Please provide a comprehensive summary that captures:
1. Main topics and themes
2. Key entities mentioned
3. Important relationships or concepts
4. Overall context and significance

Summary:"""

        try:
            response = await self.llm_client.generate(prompt, max_tokens=1000)
            return response.strip()
        except Exception as e:
            self.logger.error(f"Error processing batch {batch_index}: {e}")
            return f"Error processing batch {batch_index}: {str(e)}"
    
    async def _combine_batch_summaries(self, 
                                     batch_summaries: List[str], 
                                     keywords: List[str], 
                                     cluster_id: int) -> str:
        """
        Combine all batch summaries into a final cluster summary
        
        Args:
            batch_summaries: List of batch summaries
            keywords: Cluster keywords
            cluster_id: Cluster ID
            
        Returns:
            str: Final cluster summary
        """
        if not batch_summaries:
            return "No documents available for summary generation."
        
        # Create prompt for final combination
        summaries_text = "\n\n".join([f"Batch {i+1}: {summary}" for i, summary in enumerate(batch_summaries)])
        keywords_text = ", ".join(keywords[:10])  # Use top 10 keywords
        
        prompt = f"""You are creating a final summary for Cluster {cluster_id}. 
This cluster contains the following batch summaries and is characterized by these keywords: {keywords_text}

Batch Summaries:
{summaries_text}

Please create a comprehensive, coherent summary that:
1. Synthesizes all the batch summaries into a unified narrative
2. Highlights the main themes and topics of this cluster
3. Identifies key entities and their relationships
4. Provides context about the cluster's significance
5. Uses clear, professional language

Final Cluster Summary:"""

        try:
            response = await self.llm_client.generate(prompt, max_tokens=1500)
            return response.strip()
        except Exception as e:
            self.logger.error(f"Error combining batch summaries for cluster {cluster_id}: {e}")
            return f"Error generating final summary: {str(e)}"
    
    def _estimate_token_count(self, text: str) -> int:
        """
        Estimate token count for text (rough estimation)
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            int: Estimated token count
        """
        # Rough estimation: 1 token ≈ 4 characters
        return len(text) // 4
    
    async def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary generation statistics
        
        Returns:
            Dict: Statistics about summary generation
        """
        return {
            **self.summary_stats,
            'avg_processing_time': (
                self.summary_stats['total_processing_time'] / 
                max(self.summary_stats['total_clusters_processed'], 1)
            ),
            'avg_tokens_per_summary': (
                self.summary_stats['total_tokens_used'] / 
                max(self.summary_stats['total_summaries_generated'], 1)
            )
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        self.summary_stats = {
            'total_clusters_processed': 0,
            'total_summaries_generated': 0,
            'total_processing_time': 0.0,
            'total_tokens_used': 0
        } 