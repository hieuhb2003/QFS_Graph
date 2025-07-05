import asyncio
import os
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

from ..utils.logger_config import get_logger
from ..db.json_doc_status_impl import JsonDocStatusStorage
from ..db.json_kv_impl import JsonKVStorage
from ..db.nano_vector_db_impl import NanoVectorDBStorage
from ..db.networkx_impl import NetworkXStorage
from .llm_extractor import LLMExtractor
from .operators import DocumentOperator, QueryOperator
from .clustering_manager import ClusteringManager
from .cluster_summary_generator import ClusterSummaryGenerator
from ..utils.utils import (
    compute_hash_with_prefix, 
    create_chunks, 
    normalize_entity_name,
    Entity,
    Relation,
    Document,
    ClusterInfo
)
from ..clients.llm_client import BaseLLMClient
from ..clients.embedding_client import BaseEmbeddingClient


class GraphRAGSystem:
    """Hệ thống GraphRAG tích hợp với NetworkX và Vector DB"""
    
    def __init__(self, working_dir: str, embedding_client: BaseEmbeddingClient, global_config: Dict[str, Any], llm_client: Optional[BaseLLMClient] = None):
        """
        Khởi tạo hệ thống GraphRAG
        
        Args:
            working_dir: Thư mục làm việc
            embedding_client: Embedding client
            global_config: Cấu hình toàn cục
            llm_client: LLM client (optional)
        """
        self.working_dir = working_dir
        self.embedding_client = embedding_client
        self.global_config = global_config
        self.llm_client = llm_client
        self.logger = get_logger()
        
        # Tạo thư mục làm việc nếu chưa tồn tại
        os.makedirs(working_dir, exist_ok=True)
        
        # Khởi tạo các database
        self._doc_status_db = JsonDocStatusStorage(
            namespace="doc_status",
            global_config=global_config
        )
        
        self._chunk_db = JsonKVStorage(
            namespace="chunks",
            global_config=global_config,
            embedding_func=self._embedding_wrapper
        )
        
        # Thêm chunk VDB để lưu chunks vào vector database
        self._chunk_vdb = NanoVectorDBStorage(
            namespace="chunks_vdb",
            embedding_func=self._embedding_wrapper,
            global_config=global_config,
            meta_fields=["doc_id", "chunk_index"]
        )
        
        self._entity_db = NanoVectorDBStorage(
            namespace="entities",
            embedding_func=self._embedding_wrapper,
            global_config=global_config,
            meta_fields=["entity_name", "description", "chunk_id", "doc_id"]
        )
        
        self._relation_db = NanoVectorDBStorage(
            namespace="relations", 
            embedding_func=self._embedding_wrapper,
            global_config=global_config,
            meta_fields=["source_entity", "relation_description", "target_entity", "chunk_id", "doc_id"]
        )
        
        # Thêm cluster summary VDB riêng biệt
        self._cluster_summary_db = NanoVectorDBStorage(
            namespace="cluster_summaries",
            embedding_func=self._embedding_wrapper,
            global_config=global_config,
            meta_fields=["cluster_id", "doc_hash_ids", "summary_text", "created_at"]
        )
        
        self._graph_db = NetworkXStorage(
            namespace="knowledge_graph",
            global_config=global_config
        )
        
        # Khởi tạo LLM extractor
        self.llm_extractor = LLMExtractor(llm_client=llm_client, max_workers=4)
        
        # Khởi tạo clustering manager
        clustering_config = global_config.get('clustering', {})
        self.clustering_manager = ClusteringManager(
            embedding_client=embedding_client,
            outlier_threshold=clustering_config.get('outlier_threshold', 10),
            max_tokens=clustering_config.get('max_tokens', 8192),
            batch_size=clustering_config.get('batch_size', 16),
            model_save_path=os.path.join(working_dir, "clustering_models")
        )
        
        # Khởi tạo cluster summary generator
        summary_config = global_config.get('summary', {})
        self.cluster_summary_generator = ClusterSummaryGenerator(
            llm_client=llm_client,
            max_workers=summary_config.get('max_workers', 4),
            context_length=summary_config.get('context_length', 4096),
            summary_prompt_template=summary_config.get('summary_prompt_template'),
            map_reduce_prompt_template=summary_config.get('map_reduce_prompt_template'),
            query_generation_prompt_template=summary_config.get('query_generation_prompt_template')
        )
        
        # Khởi tạo operators
        self.doc_operator = DocumentOperator(
            self._doc_status_db,
            self._chunk_db,
            self._entity_db,
            self._relation_db,
            self._graph_db,
            self.llm_extractor
        )
        
        self.query_operator = QueryOperator(
            self._entity_db,
            self._relation_db,
            self._graph_db,
            self._chunk_db,
            self._doc_status_db
        )
        
        self.logger.info(f"GraphRAG System initialized successfully in {working_dir}")
        self.logger.info(f"Embedding dimension: {self.embedding_client.embedding_dimension}")
        if llm_client:
            self.logger.info(f"LLM client: {llm_client.model_name}")
    
    async def _embedding_wrapper(self, texts: List[str]) -> List:
        """
        Wrapper cho embedding function để tương thích với NanoVectorDBStorage
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List: List of embeddings
        """
        try:
            embeddings = await self.embedding_client.embed(texts)
            return embeddings
        except Exception as e:
            self.logger.error(f"Error in embedding wrapper: {e}")
            # Return zero vectors as fallback
            dim = self.embedding_client.embedding_dimension
            return [np.zeros(dim, dtype=np.float32) for _ in texts]
    
    async def insert_document(self, doc_id: str, content: str, chunk_size: int = 1000) -> bool:
        """
        Insert document vào hệ thống
        
        Args:
            doc_id: ID của document
            content: Nội dung document
            chunk_size: Kích thước chunk
            
        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        return await self.doc_operator.insert_document(doc_id, content, chunk_size)
    
    async def insert_document_with_llm(self, doc_id: str, content: str, chunk_size: int = 1000) -> bool:
        """
        Insert document với LLM extraction (one-shot)
        
        Args:
            doc_id: ID của document
            content: Nội dung document
            chunk_size: Kích thước chunk
            
        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        try:
            # Check document status
            doc_status = await self._doc_status_db.get_by_id(doc_id)
            
            if doc_status and doc_status.get("status") == "success":
                self.logger.info(f"Document {doc_id} already processed successfully, skipping")
                return True
            
            # Update status to pending
            await self._doc_status_db.upsert({
                doc_id: {
                    "status": "pending",
                    "content": content,
                    "timestamp": asyncio.get_event_loop().time()
                }
            })
            
            # Extract entities and relations using LLM (one-shot)
            entities, relations = await self.llm_extractor.extract_from_text(content)
            
            # Set doc_id and chunk_id đúng chuẩn
            chunk_hash = compute_hash_with_prefix(content, "chunk-")
            
            for entity in entities:
                entity.doc_id = doc_id
                entity.chunk_id = chunk_hash  # Sử dụng chunk- prefix cho chunk_id
            
            for relation in relations:
                relation.doc_id = doc_id
                relation.chunk_id = chunk_hash  # Sử dụng chunk- prefix cho chunk_id
            
            # Save document as single chunk using _save_chunks
            await self._save_chunks([content], doc_id)
            
            # Save entities to vector DB
            if entities:
                await self.doc_operator._save_entities(entities)
            
            # Save relations to vector DB
            if relations:
                await self.doc_operator._save_relations(relations)
            
            # Update graph
            await self._update_graph(entities, relations)
            
            # Save all data immediately
            await self._save_all_data()
            
            # Update status to success
            await self._doc_status_db.upsert({
                doc_id: {
                    "status": "success",
                    "content": content,
                    "timestamp": asyncio.get_event_loop().time(),
                    "entities_count": len(entities),
                    "relations_count": len(relations),
                    "chunks_count": 1,
                    "processing_type": "one_shot"
                }
            })
            
            self.logger.info(f"Successfully processed document {doc_id} with one-shot LLM extraction: {len(entities)} entities and {len(relations)} relations")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing document {doc_id} with LLM: {e}")
            
            # Update status to failed
            await self._doc_status_db.upsert({
                doc_id: {
                    "status": "failed",
                    "content": content,
                    "timestamp": asyncio.get_event_loop().time(),
                    "error": str(e),
                    "processing_type": "one_shot"
                }
            })
            
            return False
    
    async def insert_documents_batch(self, documents: List[str], chunk_size: int = 1000, max_concurrent_docs: int = 5) -> List[bool]:
        """
        Insert nhiều documents song song
        
        Args:
            documents: List of document contents
            chunk_size: Kích thước chunk
            max_concurrent_docs: Số documents tối đa chạy song song
            
        Returns:
            List[bool]: Kết quả xử lý từng document
        """
        self.logger.info(f"Starting batch processing of {len(documents)} documents with max {max_concurrent_docs} concurrent")
        
        # Semaphore để giới hạn số documents chạy song song
        semaphore = asyncio.Semaphore(max_concurrent_docs)
        
        async def process_single_document(content: str) -> bool:
            async with semaphore:
                # Tính doc_id từ content
                doc_id = compute_hash_with_prefix(content, "doc-")
                
                try:
                    # Check document status
                    doc_status = await self._doc_status_db.get_by_id(doc_id)
                    
                    if doc_status and doc_status.get("status") == "success":
                        self.logger.info(f"Document {doc_id} already processed successfully, skipping")
                        return True
                    
                    # Update status to pending
                    await self._doc_status_db.upsert({
                        doc_id: {
                            "status": "pending",
                            "content": content,
                            "timestamp": asyncio.get_event_loop().time()
                        }
                    })
                    
                    # Chunk document
                    chunks = create_chunks(content, chunk_size)
                    self.logger.info(f"Created {len(chunks)} chunks for document {doc_id}")
                    
                    # Extract entities and relations from chunks (song song)
                    all_entities, all_relations = await self.llm_extractor.extract_from_chunks(chunks, doc_id)
                    
                    # Save chunks
                    await self._save_chunks(chunks, doc_id)
                    
                    # Save entities to vector DB
                    if all_entities:
                        await self._save_entities(all_entities)
                    
                    # Save relations to vector DB
                    if all_relations:
                        await self._save_relations(all_relations)
                    
                    # Update graph
                    await self._update_graph(all_entities, all_relations)
                    
                    # Save all data immediately
                    await self._save_all_data()
                    
                    # Update status to success
                    await self._doc_status_db.upsert({
                        doc_id: {
                            "status": "success",
                            "content": content,
                            "timestamp": asyncio.get_event_loop().time(),
                            "entities_count": len(all_entities),
                            "relations_count": len(all_relations),
                            "chunks_count": len(chunks),
                            "processing_type": "batch_chunked"
                        }
                    })
                    
                    self.logger.info(f"Successfully processed document {doc_id} with {len(all_entities)} entities and {len(all_relations)} relations")
                    return True
                    
                except Exception as e:
                    self.logger.error(f"Error processing document {doc_id}: {e}")
                    
                    # Update status to failed
                    await self._doc_status_db.upsert({
                        doc_id: {
                            "status": "failed",
                            "content": content,
                            "timestamp": asyncio.get_event_loop().time(),
                            "error": str(e),
                            "processing_type": "batch_chunked"
                        }
                    })
                    
                    return False
        
        # Tạo tasks cho tất cả documents
        tasks = [process_single_document(content) for content in documents]
        
        # Chạy song song tất cả documents
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Xử lý kết quả
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                doc_id = compute_hash_with_prefix(documents[i], "doc-")
                self.logger.error(f"Exception in document {doc_id}: {result}")
                processed_results.append(False)
            else:
                processed_results.append(result)
        
        success_count = sum(processed_results)
        self.logger.info(f"Batch processing completed: {success_count}/{len(documents)} documents successful")
        
        return processed_results

    async def insert_documents_batch_with_llm(self, documents: List[str], max_concurrent_docs: int = 5) -> List[bool]:
        """
        Insert nhiều documents song song với LLM extraction (one-shot)
        
        Args:
            documents: List of document contents
            max_concurrent_docs: Số documents tối đa chạy song song
            
        Returns:
            List[bool]: Kết quả xử lý từng document
        """
        self.logger.info(f"Starting batch LLM processing of {len(documents)} documents with max {max_concurrent_docs} concurrent")
        
        # Semaphore để giới hạn số documents chạy song song
        semaphore = asyncio.Semaphore(max_concurrent_docs)
        
        async def process_single_document_llm(content: str) -> bool:
            async with semaphore:
                # Tính doc_id từ content
                doc_id = compute_hash_with_prefix(content, "doc-")
                
                try:
                    # Check document status
                    doc_status = await self._doc_status_db.get_by_id(doc_id)
                    
                    if doc_status and doc_status.get("status") == "success":
                        self.logger.info(f"Document {doc_id} already processed successfully, skipping")
                        return True
                    
                    # Update status to pending
                    await self._doc_status_db.upsert({
                        doc_id: {
                            "status": "pending",
                            "content": content,
                            "timestamp": asyncio.get_event_loop().time()
                        }
                    })
                    
                    # Extract entities and relations using LLM (one-shot)
                    entities, relations = await self.llm_extractor.extract_from_text(content)
                    
                    # Set doc_id and chunk_id đúng chuẩn
                    chunk_hash = compute_hash_with_prefix(content, "chunk-")
                    
                    for entity in entities:
                        entity.doc_id = doc_id
                        entity.chunk_id = chunk_hash  # Sử dụng chunk- prefix cho chunk_id
                    
                    for relation in relations:
                        relation.doc_id = doc_id
                        relation.chunk_id = chunk_hash  # Sử dụng chunk- prefix cho chunk_id
                    
                    # Save document as single chunk using _save_chunks
                    await self._save_chunks([content], doc_id)
                    
                    # Save entities to vector DB
                    if entities:
                        await self._save_entities(entities)
                    
                    # Save relations to vector DB
                    if relations:
                        await self._save_relations(relations)
                    
                    # Update graph
                    await self._update_graph(entities, relations)
                    
                    # Save all data immediately
                    await self._save_all_data()
                    
                    # Update status to success
                    await self._doc_status_db.upsert({
                        doc_id: {
                            "status": "success",
                            "content": content,
                            "timestamp": asyncio.get_event_loop().time(),
                            "entities_count": len(entities),
                            "relations_count": len(relations),
                            "chunks_count": 1,
                            "processing_type": "one_shot"
                        }
                    })
                    
                    self.logger.info(f"Successfully processed document {doc_id} with one-shot LLM extraction: {len(entities)} entities and {len(relations)} relations")
                    return True
                    
                except Exception as e:
                    self.logger.error(f"Error processing document {doc_id} with LLM: {e}")
                    
                    # Update status to failed
                    await self._doc_status_db.upsert({
                        doc_id: {
                            "status": "failed",
                            "content": content,
                            "timestamp": asyncio.get_event_loop().time(),
                            "error": str(e),
                            "processing_type": "one_shot"
                        }
                    })
                    
                    return False
        
        # Tạo tasks cho tất cả documents
        tasks = [process_single_document_llm(content) for content in documents]
        
        # Chạy song song tất cả documents
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Xử lý kết quả
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                doc_id = compute_hash_with_prefix(documents[i], "doc-")
                self.logger.error(f"Exception in document {doc_id}: {result}")
                processed_results.append(False)
            else:
                processed_results.append(result)
        
        success_count = sum(processed_results)
        self.logger.info(f"Batch LLM processing completed: {success_count}/{len(documents)} documents successful")
        
        return processed_results

    async def _save_all_data(self):
        """Save tất cả data ngay lập tức"""
        self.logger.info("Saving all data immediately...")
        
        # Save all databases
        await self._doc_status_db.index_done_callback()
        await self._chunk_db.index_done_callback()
        await self._chunk_vdb.index_done_callback()
        await self._entity_db.index_done_callback()
        await self._relation_db.index_done_callback()
        await self._cluster_summary_db.index_done_callback()
        await self._graph_db.index_done_callback()
        
        self.logger.info("All data saved successfully")

    async def _save_chunks(self, chunks: List[str], doc_id: str):
        """Lưu chunks vào database và VDB"""
        chunk_data = {}
        chunk_vdb_data = {}
        
        for i, chunk_content in enumerate(chunks):
            chunk_id = compute_hash_with_prefix(chunk_content, "chunk-")
            
            # Lưu vào JSON database
            chunk_data[chunk_id] = {
                "content": chunk_content,
                "doc_id": doc_id,
                "chunk_index": i
            }
            
            # Lưu vào VDB
            chunk_vdb_data[chunk_id] = {
                "content": chunk_content,
                "doc_id": doc_id,
                "chunk_index": i
            }
        
        if chunk_data:
            await self._chunk_db.upsert(chunk_data)
            self.logger.info(f"Saved {len(chunk_data)} chunks to JSON database for document {doc_id}")
        
        if chunk_vdb_data:
            await self._chunk_vdb.upsert(chunk_vdb_data)
            self.logger.info(f"Saved {len(chunk_vdb_data)} chunks to VDB for document {doc_id}")

    async def _save_entities(self, entities: List[Entity]):
        """Lưu entities vào vector DB"""
        entity_data = {}
        for entity in entities:
            entity_key = compute_hash_with_prefix(f"{entity.entity_name} {entity.description}", "ent-")
            entity_data[entity_key] = {
                "content": f"{entity.entity_name} {entity.description}",
                "entity_name": entity.entity_name,
                "description": entity.description,
                "chunk_id": entity.chunk_id,
                "doc_id": entity.doc_id
            }
        
        if entity_data:
            await self._entity_db.upsert(entity_data)
            self.logger.info(f"Saved {len(entity_data)} entities to vector DB")

    async def _save_relations(self, relations: List[Relation]):
        """Lưu relations vào vector DB"""
        relation_data = {}
        for relation in relations:
            relation_key = compute_hash_with_prefix(
                f"{relation.source_entity} {relation.relation_description} {relation.target_entity}",
                "rel-"
            )
            relation_data[relation_key] = {
                "content": f"{relation.source_entity} {relation.relation_description} {relation.target_entity}",
                "source_entity": relation.source_entity,
                "relation_description": relation.relation_description,
                "target_entity": relation.target_entity,
                "chunk_id": relation.chunk_id,
                "doc_id": relation.doc_id
            }
        
        if relation_data:
            await self._relation_db.upsert(relation_data)
            self.logger.info(f"Saved {len(relation_data)} relations to vector DB")

    async def _update_graph(self, entities: List[Entity], relations: List[Relation]):
        """Cập nhật knowledge graph"""
        # Group entities by normalized name
        entity_groups = {}
        
        for entity in entities:
            normalized_name = normalize_entity_name(entity.entity_name)
            if normalized_name not in entity_groups:
                entity_groups[normalized_name] = []
            entity_groups[normalized_name].append(entity)
        
        # Merge entities with same normalized name
        for normalized_name, entity_list in entity_groups.items():
            # Combine descriptions
            descriptions = [entity.description for entity in entity_list]
            combined_description = " <|> ".join(descriptions)
            
            # Lưu chunk IDs theo format cũ với ký tự đặc biệt
            chunk_ids = list(set([entity.chunk_id for entity in entity_list]))  # Remove duplicates
            combined_chunk_ids = " <|> ".join(chunk_ids)
            
            # Create or update node
            node_data = {
                "entity_name": normalized_name,
                "description": combined_description,
                "source": combined_chunk_ids,  # Lưu theo format cũ
                "topic_id": None
            }
            
            await self._graph_db.upsert_node(normalized_name, node_data)
        
        # Add relations as edges
        for relation in relations:
            source_normalized = normalize_entity_name(relation.source_entity)
            target_normalized = normalize_entity_name(relation.target_entity)
            
            edge_data = {
                "relation_description": relation.relation_description,
                "chunk_id": relation.chunk_id,  # Lưu chunk ID cụ thể của relation
                "doc_id": relation.doc_id
            }
            
            await self._graph_db.upsert_edge(source_normalized, target_normalized, edge_data)
        
        # Thêm document nodes và belong_to relations
        await self._add_document_nodes_and_belong_relations(entities, relations)
        
        self.logger.info(f"Updated graph with {len(entity_groups)} entities and {len(relations)} relations")

    async def _add_document_nodes_and_belong_relations(self, entities: List[Entity], relations: List[Relation]):
        """Thêm document nodes và belong_to relations"""
        try:
            # Lấy unique doc_ids từ entities và relations
            doc_ids = set()
            for entity in entities:
                doc_ids.add(entity.doc_id)
            for relation in relations:
                doc_ids.add(relation.doc_id)
            
            # Thêm document nodes
            for doc_id in doc_ids:
                # Lấy document content từ doc_status_db
                doc_status = await self._doc_status_db.get_by_id(doc_id)
                if doc_status:
                    content = doc_status.get("content", "")
                    # Thêm document node (không lưu embedding)
                    doc_node_data = {
                        "doc_id": doc_id,
                        "content": content[:200] + "..." if len(content) > 200 else content,  # Truncate
                        "type": "document",
                        "cluster_id": None  # Sẽ được cập nhật sau khi clustering
                    }
                    await self._graph_db.upsert_node(doc_id, doc_node_data)
            
            # Thêm belong_to relations từ entities đến documents
            for entity in entities:
                entity_normalized = normalize_entity_name(entity.entity_name)
                doc_id = entity.doc_id
                
                # Thêm belong_to relation (không lưu vào vector DB)
                edge_data = {
                    "relation_type": "belong_to",
                    "entity_name": entity.entity_name,
                    "doc_id": doc_id,
                    "chunk_id": entity.chunk_id
                }
                await self._graph_db.upsert_edge(entity_normalized, doc_id, edge_data)
            
            self.logger.info(f"Added {len(doc_ids)} document nodes and belong_to relations")
            
        except Exception as e:
            self.logger.error(f"Lỗi khi thêm document nodes và belong_to relations: {e}")

    # Query methods
    async def query_entities(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Query entities từ vector DB"""
        return await self.query_operator.query_entities(query, top_k)
    
    async def query_relations(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Query relations từ vector DB"""
        return await self.query_operator.query_relations(query, top_k)
    
    async def get_entity_neighbors(self, entity_name: str) -> List[Tuple[str, Dict, Dict]]:
        """Lấy neighbors của một entity trong graph"""
        return await self.query_operator.get_entity_neighbors(entity_name)
    
    async def get_document_status(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Lấy trạng thái của document"""
        return await self.query_operator.get_document_status(doc_id)
    
    async def get_chunk_content(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Lấy nội dung của chunk"""
        return await self.query_operator.get_chunk_content(chunk_id)
    
    async def get_status_counts(self) -> Dict[str, int]:
        """Lấy số lượng documents theo trạng thái"""
        return await self.query_operator.get_status_counts()
    
    async def search_entities_by_name(self, entity_name: str) -> List[Dict[str, Any]]:
        """Tìm entities theo tên"""
        return await self.query_operator.search_entities_by_name(entity_name)
    
    async def get_entity_graph_context(self, entity_name: str, max_depth: int = 2) -> Dict[str, Any]:
        """Lấy context graph của một entity"""
        return await self.query_operator.get_entity_graph_context(entity_name, max_depth)
    
    # Database access methods
    @property
    def doc_status_db(self):
        return self._doc_status_db
    
    @property
    def chunk_db(self):
        return self._chunk_db
    
    @property
    def chunk_vdb(self):
        return self._chunk_vdb
    
    @property
    def entity_db(self):
        return self._entity_db
    
    @property
    def relation_db(self):
        return self._relation_db
    
    @property
    def graph_db(self):
        return self._graph_db
    
    # Cleanup
    async def cleanup(self):
        """Cleanup resources và save tất cả databases"""
        self.logger.info("Starting cleanup and saving all databases...")
        
        # Cleanup LLM extractor
        self.llm_extractor.cleanup()
        
        # Save all databases
        await self._doc_status_db.index_done_callback()
        await self._chunk_db.index_done_callback()
        await self._chunk_vdb.index_done_callback()
        await self._entity_db.index_done_callback()
        await self._relation_db.index_done_callback()
        await self._cluster_summary_db.index_done_callback()
        await self._graph_db.index_done_callback()
        
        self.logger.info("Cleanup completed successfully")
    
    # Statistics methods
    async def get_system_stats(self) -> Dict[str, Any]:
        """Lấy thống kê tổng quan của hệ thống"""
        try:
            # Get document status counts
            status_counts = await self.get_status_counts()
            
            # Get graph stats
            graph = self._graph_db._graph
            graph_stats = {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges()
            }
            
            # Get vector DB stats
            entity_count = len(self._entity_db.client_storage["data"])
            relation_count = len(self._relation_db.client_storage["data"])
            
            # Get chunk count
            chunk_count = len(self._chunk_db._data)
            
            return {
                "document_status": status_counts,
                "graph": graph_stats,
                "vector_dbs": {
                    "entities": entity_count,
                    "relations": relation_count
                },
                "chunks": chunk_count,
                "working_dir": self.working_dir,
                "embedding_dimension": self.embedding_client.embedding_dimension,
                "llm_model": self.llm_client.model_name if self.llm_client else None
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system stats: {e}")
            return {"error": str(e)}
    
    # Clustering methods
    async def cluster_documents(self, outlier_threshold: int = 10) -> Dict[str, Any]:
        """
        Thực hiện clustering cho tất cả documents đã được xử lý
        
        Args:
            outlier_threshold: Ngưỡng số lượng outlier để tạo cluster mới
            
        Returns:
            Dict chứa thông tin clustering
        """
        try:
            self.logger.info("Bắt đầu clustering documents...")
            
            # Lấy tất cả documents đã xử lý thành công
            all_docs = await self._doc_status_db.get_all()
            successful_docs = []
            doc_hash_ids = []
            
            for doc_id, doc_info in all_docs.items():
                if doc_info.get("status") == "success":
                    content = doc_info.get("content", "")
                    if content:
                        successful_docs.append(content)
                        doc_hash_ids.append(doc_id)
            
            if not successful_docs:
                self.logger.warning("Không có documents nào để cluster")
                return {"error": "Không có documents nào để cluster"}
            
            self.logger.info(f"Tìm thấy {len(successful_docs)} documents để cluster")
            
            # Thực hiện clustering
            clustering_result = await self.clustering_manager.cluster_documents(successful_docs, doc_hash_ids)
            
            # Tạo cluster summaries
            await self._generate_cluster_summaries(clustering_result)
            
            # Cập nhật graph với cluster information
            await self._update_graph_with_clusters(clustering_result)
            
            self.logger.info("Hoàn thành clustering và tạo summaries")
            return clustering_result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi thực hiện clustering: {e}")
            return {"error": str(e)}
    
    async def update_clusters_with_new_data(self, new_documents: List[str]) -> Dict[str, Any]:
        """
        Cập nhật clusters với documents mới
        
        Args:
            new_documents: Danh sách documents mới
            
        Returns:
            Dict chứa thông tin cập nhật
        """
        try:
            self.logger.info(f"Cập nhật clusters với {len(new_documents)} documents mới...")
            
            # Tính doc_hash_ids cho documents mới
            new_doc_hash_ids = []
            for content in new_documents:
                doc_hash_id = compute_hash_with_prefix(content, "doc-")
                new_doc_hash_ids.append(doc_hash_id)
            
            # Cập nhật clusters
            clustering_result = await self.clustering_manager.update_clusters_with_new_data(
                new_documents, new_doc_hash_ids
            )
            
            # Tạo cluster summaries mới
            await self._generate_cluster_summaries(clustering_result)
            
            # Cập nhật graph với cluster information mới
            await self._update_graph_with_clusters(clustering_result)
            
            self.logger.info("Hoàn thành cập nhật clusters")
            return clustering_result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi cập nhật clusters: {e}")
            return {"error": str(e)}
    
    async def get_cluster_info(self) -> Dict[str, Any]:
        """Lấy thông tin về clusters hiện tại"""
        return await self.clustering_manager.get_cluster_info()
    
    async def get_documents_by_cluster(self, cluster_id: int) -> List[str]:
        """Lấy hash doc IDs thuộc cluster cụ thể"""
        return await self.clustering_manager.get_documents_by_cluster(cluster_id)
    
    async def get_cluster_doc_ids(self, cluster_id: int) -> List[str]:
        """Lấy hash doc IDs của cluster"""
        return await self.clustering_manager.get_cluster_doc_ids(cluster_id)
    
    async def generate_cluster_summaries(self, max_workers: int = 4) -> Dict[int, str]:
        """
        Tạo summaries cho tất cả clusters
        
        Args:
            max_workers: Số worker tối đa cho parallel processing
            
        Returns:
            Dict {cluster_id: summary}
        """
        try:
            self.logger.info("Bắt đầu tạo cluster summaries...")
            
            # Lấy cluster documents
            cluster_info = await self.clustering_manager.get_cluster_info()
            cluster_documents = cluster_info.get("cluster_documents", {})
            
            # Lấy document contents
            doc_content_map = await self._get_document_content_map()
            
            # Tạo summaries
            summaries = await self.cluster_summary_generator.generate_cluster_summaries(
                cluster_documents, doc_content_map
            )
            
            # Lưu summaries vào cluster summary VDB
            await self._save_cluster_summaries_to_vdb(summaries)
            
            # Lưu summaries vào clustering manager
            for cluster_id, summary in summaries.items():
                doc_hash_ids = cluster_documents.get(cluster_id, [])
                await self.clustering_manager.save_cluster_summary(cluster_id, summary, doc_hash_ids)
            
            self.logger.info(f"Hoàn thành tạo summaries cho {len(summaries)} clusters")
            return summaries
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo cluster summaries: {e}")
            return {"error": str(e)}
    
    async def query_cluster_summaries(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query cluster summaries từ vector DB
        
        Args:
            query: Query string
            top_k: Số kết quả tối đa
            
        Returns:
            List các cluster summaries phù hợp
        """
        try:
            # Query từ cluster summary VDB
            results = await self._cluster_summary_db.query(query, top_k)
            
            # Format kết quả
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "cluster_id": result.get("cluster_id"),
                    "summary": result.get("summary_text"),
                    "doc_hash_ids": result.get("doc_hash_ids", []),
                    "score": result.get("score", 0.0)
                })
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Lỗi khi query cluster summaries: {e}")
            return []
    
    async def query_cluster_summaries_with_mode(self, 
                                              query: str, 
                                              mode: str = "retrieval", 
                                              top_k: int = 5) -> Dict[str, Any]:
        """
        Query cluster summaries với 2 mode: retrieval và generation
        
        Args:
            query: Query string
            mode: "retrieval" hoặc "generation"
            top_k: Số kết quả tối đa từ vector DB
            
        Returns:
            Dict chứa kết quả query theo mode
        """
        try:
            self.logger.info(f"Querying cluster summaries with mode: {mode}")
            
            # Query từ cluster summary VDB để lấy top_k results
            cluster_summaries = await self._cluster_summary_db.query(query, top_k)
            
            # Sử dụng cluster summary generator để xử lý theo mode
            result = await self.cluster_summary_generator.query_cluster_summaries(
                query=query,
                cluster_summaries=cluster_summaries,
                mode=mode
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi query cluster summaries với mode {mode}: {e}")
            return {
                "error": str(e),
                "mode": mode,
                "query": query,
                "results": [] if mode == "retrieval" else None,
                "answer": None if mode == "generation" else None
            }
    
    async def get_documents_with_same_cluster(self, doc_id: str) -> List[str]:
        """
        Lấy danh sách documents cùng cluster với document cho trước
        
        Args:
            doc_id: Hash ID của document
            
        Returns:
            List hash IDs của documents cùng cluster
        """
        try:
            # Lấy cluster_id của document
            cluster_assignments = await self.clustering_manager.get_cluster_info()
            cluster_id = cluster_assignments.get("cluster_assignments", {}).get(doc_id)
            
            if cluster_id is None:
                return []
            
            # Lấy tất cả documents cùng cluster
            cluster_docs = await self.clustering_manager.get_documents_by_cluster(cluster_id)
            
            # Loại bỏ document hiện tại
            cluster_docs = [doc for doc in cluster_docs if doc != doc_id]
            
            return cluster_docs
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy documents cùng cluster: {e}")
            return []
    
    # Helper methods for clustering
    async def _generate_cluster_summaries(self, clustering_result: Dict[str, Any]) -> None:
        """Tạo summaries cho clusters"""
        try:
            cluster_documents = clustering_result.get("cluster_documents", {})
            doc_content_map = await self._get_document_content_map()
            
            summaries = await self.cluster_summary_generator.generate_cluster_summaries(
                cluster_documents, doc_content_map
            )
            
            # Lưu summaries
            await self._save_cluster_summaries_to_vdb(summaries)
            
            # Lưu vào clustering manager
            for cluster_id, summary in summaries.items():
                doc_hash_ids = cluster_documents.get(cluster_id, [])
                await self.clustering_manager.save_cluster_summary(cluster_id, summary, doc_hash_ids)
                
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo cluster summaries: {e}")
    
    async def _get_document_content_map(self) -> Dict[str, str]:
        """Lấy map từ doc_hash_id sang content"""
        try:
            all_docs = await self._doc_status_db.get_all()
            doc_content_map = {}
            
            for doc_id, doc_info in all_docs.items():
                if doc_info.get("status") == "success":
                    content = doc_info.get("content", "")
                    if content:
                        doc_content_map[doc_id] = content
            
            return doc_content_map
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy document content map: {e}")
            return {}
    
    async def _save_cluster_summaries_to_vdb(self, summaries: Dict[int, str]) -> None:
        """Lưu cluster summaries vào vector DB"""
        try:
            summary_data = {}
            
            for cluster_id, summary in summaries.items():
                summary_key = f"cluster_summary_{cluster_id}"
                
                # Lấy doc_hash_ids của cluster
                doc_hash_ids = await self.clustering_manager.get_cluster_doc_ids(cluster_id)
                
                summary_data[summary_key] = {
                    "content": summary,
                    "cluster_id": cluster_id,
                    "doc_hash_ids": doc_hash_ids,
                    "summary_text": summary,
                    "created_at": asyncio.get_event_loop().time()
                }
            
            if summary_data:
                await self._cluster_summary_db.upsert(summary_data)
                self.logger.info(f"Đã lưu {len(summary_data)} cluster summaries vào VDB")
                
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu cluster summaries vào VDB: {e}")
    
    async def _update_graph_with_clusters(self, clustering_result: Dict[str, Any]) -> None:
        """Cập nhật graph với thông tin clusters"""
        try:
            cluster_assignments = clustering_result.get("cluster_assignments", {})
            cluster_documents = clustering_result.get("cluster_documents", {})
            
            # Thêm document nodes vào graph nếu chưa có
            for doc_hash_id in cluster_assignments.keys():
                doc_status = await self._doc_status_db.get_by_id(doc_hash_id)
                if doc_status:
                    # Thêm document node
                    doc_node_data = {
                        "doc_id": doc_hash_id,
                        "content": doc_status.get("content", "")[:200] + "...",  # Truncate
                        "cluster_id": cluster_assignments.get(doc_hash_id, -1),
                        "type": "document"
                    }
                    await self._graph_db.upsert_node(doc_hash_id, doc_node_data)
            
            # Thêm "has_same_cluster" relations giữa documents cùng cluster
            for cluster_id, doc_hash_ids in cluster_documents.items():
                if cluster_id == -1:  # Bỏ qua outlier cluster
                    continue
                
                # Tạo relations giữa tất cả documents trong cluster
                for i, doc1 in enumerate(doc_hash_ids):
                    for j, doc2 in enumerate(doc_hash_ids):
                        if i < j:  # Tránh duplicate và self-relation
                            edge_data = {
                                "relation_type": "has_same_cluster",
                                "cluster_id": cluster_id,
                                "doc1": doc1,
                                "doc2": doc2
                            }
                            await self._graph_db.upsert_edge(doc1, doc2, edge_data)
            
            self.logger.info("Đã cập nhật graph với thông tin clusters")
            
        except Exception as e:
            self.logger.error(f"Lỗi khi cập nhật graph với clusters: {e}")
    
    # Database access methods for clustering
    @property
    def cluster_summary_db(self):
        return self._cluster_summary_db
    
    @property
    def clustering_manager(self):
        return self.clustering_manager 