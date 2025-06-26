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
from .clustering_manager import DynamicClusteringManager, DocumentClusterResult
from .cluster_summary_generator import ClusterSummaryGenerator, ClusterSummary
from ..utils.utils import (
    compute_hash_with_prefix, 
    create_chunks, 
    normalize_entity_name,
    Entity,
    Relation
)
from ..clients.llm_client import BaseLLMClient
from ..clients.embedding_client import BaseEmbeddingClient


class GraphRAGSystem:
    """Hệ thống GraphRAG tích hợp với NetworkX, Vector DB và Dynamic Clustering"""
    
    def __init__(self, working_dir: str, embedding_client: BaseEmbeddingClient, global_config: Dict[str, Any], 
                 llm_client: Optional[BaseLLMClient] = None, enable_clustering: bool = True,
                 clustering_config: Optional[Dict[str, Any]] = None):
        """
        Khởi tạo hệ thống GraphRAG
        
        Args:
            working_dir: Thư mục làm việc
            embedding_client: Embedding client
            global_config: Cấu hình toàn cục
            llm_client: LLM client (optional)
            enable_clustering: Bật/tắt clustering
            clustering_config: Cấu hình clustering
        """
        self.working_dir = working_dir
        self.embedding_client = embedding_client
        self.global_config = global_config
        self.llm_client = llm_client
        self.enable_clustering = enable_clustering
        self.clustering_config = clustering_config or {}
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
        
        self._graph_db = NetworkXStorage(
            namespace="knowledge_graph",
            global_config=global_config
        )
        
        # Khởi tạo clustering manager nếu được bật
        self.clustering_manager = None
        if self.enable_clustering:
            self.clustering_manager = DynamicClusteringManager(
                embedding_client=embedding_client,
                working_dir=working_dir,
                update_threshold=self.clustering_config.get('update_threshold', 10),
                min_cluster_size=self.clustering_config.get('min_cluster_size', 5),
                min_samples=self.clustering_config.get('min_samples', 2),
                model_name=self.clustering_config.get('model_name', 'dynamic_bertopic_model')
            )
            self.logger.info("Clustering manager initialized")
        
        # Khởi tạo cluster summary generator nếu có LLM client
        self.summary_generator = None
        if self.enable_clustering and llm_client:
            self.summary_generator = ClusterSummaryGenerator(
                llm_client=llm_client,
                max_tokens_per_batch=self.clustering_config.get('max_tokens_per_batch', 4000),
                max_concurrent_batches=self.clustering_config.get('max_concurrent_batches', 3)
            )
            self.logger.info("Cluster summary generator initialized")
        
        # Khởi tạo LLM extractor
        self.llm_extractor = LLMExtractor(llm_client=llm_client, max_workers=4)
        
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
        if self.enable_clustering:
            self.logger.info("Dynamic clustering enabled")
    
    async def initialize_clustering(self, initial_docs: List[str]) -> bool:
        """
        Initialize clustering model with initial documents
        
        Args:
            initial_docs: List of initial documents for training
            
        Returns:
            bool: True if successful
        """
        if not self.enable_clustering or not self.clustering_manager:
            self.logger.warning("Clustering is disabled")
            return False
        
        return await self.clustering_manager.initialize_model(initial_docs)
    
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
    
    async def insert_document_with_llm(self, doc_id: str, content: str, chunk_size: int = 1000) -> Dict[str, Any]:
        """
        Insert document với LLM extraction và clustering
        
        Args:
            doc_id: ID của document
            content: Nội dung document
            chunk_size: Kích thước chunk
            
        Returns:
            Dict: Kết quả xử lý bao gồm cả clustering
        """
        try:
            # Check document status
            doc_status = await self._doc_status_db.get_by_id(doc_id)
            
            if doc_status and doc_status.get("status") == "success":
                self.logger.info(f"Document {doc_id} already processed successfully, skipping")
                return {"status": "skipped", "reason": "already_processed"}
            
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
            
            # Update graph with Document node and belong-to relations
            await self._update_graph_with_document(doc_id, content, entities, relations)
            
            # Update document status
            status_update = {
                "status": "success",
                "entities_count": len(entities),
                "relations_count": len(relations),
                "timestamp": asyncio.get_event_loop().time()
            }
            
            await self._doc_status_db.upsert({doc_id: status_update})
            
            return {
                "status": "success",
                "doc_id": doc_id,
                "entities_count": len(entities),
                "relations_count": len(relations)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing document {doc_id}: {e}")
            await self._doc_status_db.upsert({
                doc_id: {
                    "status": "error",
                    "error": str(e),
                    "timestamp": asyncio.get_event_loop().time()
                }
            })
            return {"status": "error", "error": str(e)}
    
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
        
        self.logger.info(f"Updated graph with {len(entity_groups)} entities and {len(relations)} relations")

    async def _update_graph_with_document(self, doc_id: str, content: str, entities: List[Entity], relations: List[Relation]):
        """Cập nhật knowledge graph với Document node và belong-to relations"""
        
        # 1. Add Document node to graph
        doc_node_data = {
            "type": "Doc",
            "doc_id": doc_id,
            "content_preview": content[:200] + "..." if len(content) > 200 else content,
            "entities_count": len(entities),
            "relations_count": len(relations),
            "timestamp": asyncio.get_event_loop().time()
        }
        
        await self._graph_db.upsert_node(doc_id, doc_node_data)
        self.logger.info(f"Added Document node: {doc_id}")
        
        # 2. Group entities by normalized name
        entity_groups = {}
        
        for entity in entities:
            normalized_name = normalize_entity_name(entity.entity_name)
            if normalized_name not in entity_groups:
                entity_groups[normalized_name] = []
            entity_groups[normalized_name].append(entity)
        
        # 3. Merge entities with same normalized name and add belong-to relations
        for normalized_name, entity_list in entity_groups.items():
            # Combine descriptions
            descriptions = [entity.description for entity in entity_list]
            combined_description = " <|> ".join(descriptions)
            
            # Lưu chunk IDs theo format cũ với ký tự đặc biệt
            chunk_ids = list(set([entity.chunk_id for entity in entity_list]))  # Remove duplicates
            combined_chunk_ids = " <|> ".join(chunk_ids)
            
            # Create or update entity node
            node_data = {
                "type": "Entity",
                "entity_name": normalized_name,
                "description": combined_description,
                "source": combined_chunk_ids,  # Lưu theo format cũ
                "topic_id": None
            }
            
            await self._graph_db.upsert_node(normalized_name, node_data)
            
            # 4. Add belong-to relation from entity to document
            belong_to_edge_data = {
                "relation_type": "belong_to",
                "doc_id": doc_id,
                "entity_name": normalized_name,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            await self._graph_db.upsert_edge(normalized_name, doc_id, belong_to_edge_data)
        
        # 5. Add relations as edges between entities
        for relation in relations:
            source_normalized = normalize_entity_name(relation.source_entity)
            target_normalized = normalize_entity_name(relation.target_entity)
            
            edge_data = {
                "relation_type": "entity_relation",
                "relation_description": relation.relation_description,
                "chunk_id": relation.chunk_id,  # Lưu chunk ID cụ thể của relation
                "doc_id": relation.doc_id
            }
            
            await self._graph_db.upsert_edge(source_normalized, target_normalized, edge_data)
        
        self.logger.info(f"Updated graph with Document node {doc_id}, {len(entity_groups)} entities, {len(relations)} relations, and {len(entity_groups)} belong-to edges")

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
    
    async def process_clustering(self, documents: List[str]) -> Dict[str, Any]:
        """
        Process clustering for documents (lazy clustering - only when called)
        
        Args:
            documents: List of document contents to cluster
            
        Returns:
            Dict: Clustering results
        """
        if not self.enable_clustering or not self.clustering_manager:
            return {"error": "Clustering is disabled"}
        
        try:
            self.logger.info(f"Starting clustering for {len(documents)} documents")
            
            # Process each document through clustering
            cluster_results = []
            for i, doc in enumerate(documents):
                doc_id = f"doc_{i}"
                result = await self.clustering_manager.process_document(doc_id, doc)
                cluster_results.append(result)
                
                # Update document status with clustering info
                await self._doc_status_db.upsert({
                    doc_id: {
                        "cluster_id": result.cluster_id,
                        "cluster_confidence": result.confidence,
                        "is_outlier": result.is_outlier,
                        "cluster_keywords": result.keywords,
                        "clustering_time": result.processing_time
                    }
                })
            
            # Get clustering statistics
            cluster_stats = await self.clustering_manager.get_cluster_statistics()
            
            return {
                "status": "success",
                "documents_processed": len(documents),
                "cluster_results": cluster_results,
                "cluster_statistics": cluster_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error in clustering: {e}")
            return {"error": str(e)}
    
    async def generate_cluster_summaries(self) -> Dict[str, Any]:
        """
        Generate summaries for all clusters using map-reduce approach
        
        Returns:
            Dict: Summary generation results
        """
        if not self.enable_clustering or not self.summary_generator:
            return {"error": "Summary generation is disabled or no LLM client available"}
        
        try:
            self.logger.info("Starting cluster summary generation")
            
            # Get all document statuses
            all_statuses = await self._doc_status_db.get_all()
            
            # Group documents by cluster
            cluster_documents = {}
            cluster_keywords = {}
            
            for doc_id, status in all_statuses.items():
                if status.get("status") == "success":
                    cluster_id = status.get("cluster_id")
                    if cluster_id is not None and cluster_id != -1:  # Skip outliers
                        if cluster_id not in cluster_documents:
                            cluster_documents[cluster_id] = []
                            cluster_keywords[cluster_id] = status.get("cluster_keywords", [])
                        
                        # Get document content
                        content = status.get("content", "")
                        if content:
                            cluster_documents[cluster_id].append(content)
            
            if not cluster_documents:
                return {"error": "No documents found in clusters"}
            
            # Generate summaries
            summaries = await self.summary_generator.generate_cluster_summaries(
                cluster_documents, 
                cluster_keywords
            )
            
            # Get summary statistics
            summary_stats = await self.summary_generator.get_summary_statistics()
            
            return {
                "status": "success",
                "clusters_processed": len(cluster_documents),
                "summaries_generated": len(summaries),
                "summaries": summaries,
                "summary_statistics": summary_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error generating cluster summaries: {e}")
            return {"error": str(e)}
    
    async def get_documents_by_cluster(self, cluster_id: int) -> List[Dict[str, Any]]:
        """
        Get all documents belonging to a specific cluster
        
        Args:
            cluster_id: Cluster ID
            
        Returns:
            List[Dict]: Documents in the cluster
        """
        if not self.enable_clustering:
            return []
        
        try:
            # Get all document statuses
            all_statuses = await self._doc_status_db.get_all()
            cluster_docs = []
            
            for doc_id, status in all_statuses.items():
                if status.get("cluster_id") == cluster_id:
                    cluster_docs.append({
                        "doc_id": doc_id,
                        "status": status.get("status"),
                        "cluster_confidence": status.get("cluster_confidence"),
                        "entities_count": status.get("entities_count", 0),
                        "relations_count": status.get("relations_count", 0),
                        "timestamp": status.get("timestamp"),
                        "content": status.get("content", "")
                    })
            
            return cluster_docs
            
        except Exception as e:
            self.logger.error(f"Error getting documents by cluster: {e}")
            return []
    
    async def get_clustering_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive clustering statistics
        
        Returns:
            Dict: Clustering statistics
        """
        if not self.enable_clustering or not self.clustering_manager:
            return {"clustering_enabled": False}
        
        try:
            # Get basic clustering stats
            cluster_stats = await self.clustering_manager.get_cluster_statistics()
            
            # Get document distribution across clusters
            all_statuses = await self._doc_status_db.get_all()
            cluster_distribution = {}
            outlier_count = 0
            
            for doc_id, status in all_statuses.items():
                if status.get("status") == "success":
                    cluster_id = status.get("cluster_id", -1)
                    if cluster_id == -1:
                        outlier_count += 1
                    else:
                        cluster_distribution[cluster_id] = cluster_distribution.get(cluster_id, 0) + 1
            
            return {
                "clustering_enabled": True,
                **cluster_stats,
                "cluster_distribution": cluster_distribution,
                "outlier_count": outlier_count,
                "total_processed_docs": len([s for s in all_statuses.values() if s.get("status") == "success"])
            }
            
        except Exception as e:
            self.logger.error(f"Error getting clustering statistics: {e}")
            return {"clustering_enabled": True, "error": str(e)}
    
    async def force_cluster_update(self) -> bool:
        """
        Force clustering model update
        
        Returns:
            bool: True if successful
        """
        if not self.enable_clustering or not self.clustering_manager:
            return False
        
        return await self.clustering_manager.force_update_model()
    
    async def query_with_clustering(self, query: str, top_k: int = 10, use_clusters: bool = True) -> Dict[str, Any]:
        """
        Query với clustering awareness
        
        Args:
            query: Query string
            top_k: Number of top results
            use_clusters: Whether to use clustering for enhanced search
            
        Returns:
            Dict: Query results with clustering context
        """
        try:
            # Get basic query results
            entities = await self.query_entities(query, top_k)
            relations = await self.query_relations(query, top_k)
            
            result = {
                "query": query,
                "entities": entities,
                "relations": relations,
                "clustering_info": None
            }
            
            # Add clustering context if enabled
            if use_clusters and self.enable_clustering and self.clustering_manager:
                try:
                    # Get all clusters
                    clusters = await self.clustering_manager.get_all_clusters()
                    
                    # Simple similarity search (could be enhanced)
                    cluster_details = []
                    for cluster in clusters[:3]:  # Top 3 clusters
                        cluster_details.append({
                            "cluster_id": cluster.cluster_id,
                            "keywords": cluster.keywords,
                            "doc_count": cluster.doc_count
                        })
                    
                    result["clustering_info"] = {
                        "available_clusters": cluster_details,
                        "total_clusters": len(clusters)
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error in clustering query: {e}")
                    result["clustering_info"] = {"error": str(e)}
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in clustering query: {e}")
            return {"error": str(e)} 