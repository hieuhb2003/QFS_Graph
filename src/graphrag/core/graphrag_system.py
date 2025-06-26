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
from ..utils.utils import compute_hash_with_prefix, create_chunks, normalize_entity_name
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
        self.doc_status_db = JsonDocStatusStorage(
            namespace="doc_status",
            global_config=global_config
        )
        
        self.chunk_db = JsonKVStorage(
            namespace="chunks",
            global_config=global_config
        )
        
        self.entity_db = NanoVectorDBStorage(
            namespace="entities",
            embedding_func=self._embedding_wrapper,
            global_config=global_config,
            meta_fields=["entity_name", "description", "chunk_id", "doc_id"]
        )
        
        self.relation_db = NanoVectorDBStorage(
            namespace="relations", 
            embedding_func=self._embedding_wrapper,
            global_config=global_config,
            meta_fields=["source_entity", "relation_description", "target_entity", "chunk_id", "doc_id"]
        )
        
        self.graph_db = NetworkXStorage(
            namespace="knowledge_graph",
            global_config=global_config
        )
        
        # Khởi tạo LLM extractor
        self.llm_extractor = LLMExtractor(llm_client=llm_client, max_workers=4)
        
        # Khởi tạo operators
        self.doc_operator = DocumentOperator(
            self.doc_status_db,
            self.chunk_db,
            self.entity_db,
            self.relation_db,
            self.graph_db,
            self.llm_extractor
        )
        
        self.query_operator = QueryOperator(
            self.entity_db,
            self.relation_db,
            self.graph_db,
            self.chunk_db,
            self.doc_status_db
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
            doc_status = await self.doc_status_db.get_by_id(doc_id)
            
            if doc_status and doc_status.get("status") == "success":
                self.logger.info(f"Document {doc_id} already processed successfully, skipping")
                return True
            
            # Update status to pending
            await self.doc_status_db.upsert({
                doc_id: {
                    "status": "pending",
                    "content": content,
                    "timestamp": asyncio.get_event_loop().time()
                }
            })
            
            # Extract entities and relations using LLM (one-shot)
            entities, relations = await self.llm_extractor.extract_from_text(content)
            
            # Set doc_id for all entities and relations
            doc_hash = compute_hash_with_prefix(content, "doc-")
            for entity in entities:
                entity.doc_id = doc_hash
                entity.chunk_id = doc_hash  # Use doc as chunk for one-shot
            
            for relation in relations:
                relation.doc_id = doc_hash
                relation.chunk_id = doc_hash  # Use doc as chunk for one-shot
            
            # Save document as single chunk
            await self.chunk_db.upsert({
                doc_hash: {
                    "content": content,
                    "doc_id": doc_hash,
                    "chunk_index": 0
                }
            })
            
            # Save entities to vector DB
            if entities:
                await self.doc_operator._save_entities(entities)
            
            # Save relations to vector DB
            if relations:
                await self.doc_operator._save_relations(relations)
            
            # Update graph
            await self.doc_operator._update_graph(entities, relations)
            
            # Update status to success
            await self.doc_status_db.upsert({
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
            await self.doc_status_db.upsert({
                doc_id: {
                    "status": "failed",
                    "content": content,
                    "timestamp": asyncio.get_event_loop().time(),
                    "error": str(e),
                    "processing_type": "one_shot"
                }
            })
            
            return False
    
    async def insert_documents_batch(self, documents: List[Dict[str, str]], chunk_size: int = 1000, max_concurrent_docs: int = 5) -> List[bool]:
        """
        Insert nhiều documents song song
        
        Args:
            documents: List of documents [{"doc_id": "doc1", "content": "content1"}, ...]
            chunk_size: Kích thước chunk
            max_concurrent_docs: Số documents tối đa chạy song song
            
        Returns:
            List[bool]: Kết quả xử lý từng document
        """
        self.logger.info(f"Starting batch processing of {len(documents)} documents with max {max_concurrent_docs} concurrent")
        
        # Semaphore để giới hạn số documents chạy song song
        semaphore = asyncio.Semaphore(max_concurrent_docs)
        
        async def process_single_document(doc_data: Dict[str, str]) -> bool:
            async with semaphore:
                doc_id = doc_data["doc_id"]
                content = doc_data["content"]
                
                try:
                    # Check document status
                    doc_status = await self.doc_status_db.get_by_id(doc_id)
                    
                    if doc_status and doc_status.get("status") == "success":
                        self.logger.info(f"Document {doc_id} already processed successfully, skipping")
                        return True
                    
                    # Update status to pending
                    await self.doc_status_db.upsert({
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
                    
                    # Update status to success
                    await self.doc_status_db.upsert({
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
                    await self.doc_status_db.upsert({
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
        tasks = [process_single_document(doc) for doc in documents]
        
        # Chạy song song tất cả documents
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Xử lý kết quả
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Exception in document {documents[i]['doc_id']}: {result}")
                processed_results.append(False)
            else:
                processed_results.append(result)
        
        success_count = sum(processed_results)
        self.logger.info(f"Batch processing completed: {success_count}/{len(documents)} documents successful")
        
        return processed_results

    async def insert_documents_batch_with_llm(self, documents: List[Dict[str, str]], max_concurrent_docs: int = 5) -> List[bool]:
        """
        Insert nhiều documents song song với LLM extraction (one-shot)
        
        Args:
            documents: List of documents [{"doc_id": "doc1", "content": "content1"}, ...]
            max_concurrent_docs: Số documents tối đa chạy song song
            
        Returns:
            List[bool]: Kết quả xử lý từng document
        """
        self.logger.info(f"Starting batch LLM processing of {len(documents)} documents with max {max_concurrent_docs} concurrent")
        
        # Semaphore để giới hạn số documents chạy song song
        semaphore = asyncio.Semaphore(max_concurrent_docs)
        
        async def process_single_document_llm(doc_data: Dict[str, str]) -> bool:
            async with semaphore:
                doc_id = doc_data["doc_id"]
                content = doc_data["content"]
                
                try:
                    # Check document status
                    doc_status = await self.doc_status_db.get_by_id(doc_id)
                    
                    if doc_status and doc_status.get("status") == "success":
                        self.logger.info(f"Document {doc_id} already processed successfully, skipping")
                        return True
                    
                    # Update status to pending
                    await self.doc_status_db.upsert({
                        doc_id: {
                            "status": "pending",
                            "content": content,
                            "timestamp": asyncio.get_event_loop().time()
                        }
                    })
                    
                    # Extract entities and relations using LLM (one-shot)
                    entities, relations = await self.llm_extractor.extract_from_text(content)
                    
                    # Set doc_id for all entities and relations
                    doc_hash = compute_hash_with_prefix(content, "doc-")
                    for entity in entities:
                        entity.doc_id = doc_hash
                        entity.chunk_id = doc_hash  # Use doc as chunk for one-shot
                    
                    for relation in relations:
                        relation.doc_id = doc_hash
                        relation.chunk_id = doc_hash  # Use doc as chunk for one-shot
                    
                    # Save document as single chunk
                    await self.chunk_db.upsert({
                        doc_hash: {
                            "content": content,
                            "doc_id": doc_hash,
                            "chunk_index": 0
                        }
                    })
                    
                    # Save entities to vector DB
                    if entities:
                        await self._save_entities(entities)
                    
                    # Save relations to vector DB
                    if relations:
                        await self._save_relations(relations)
                    
                    # Update graph
                    await self._update_graph(entities, relations)
                    
                    # Update status to success
                    await self.doc_status_db.upsert({
                        doc_id: {
                            "status": "success",
                            "content": content,
                            "timestamp": asyncio.get_event_loop().time(),
                            "entities_count": len(entities),
                            "relations_count": len(relations),
                            "chunks_count": 1,
                            "processing_type": "batch_one_shot"
                        }
                    })
                    
                    self.logger.info(f"Successfully processed document {doc_id} with one-shot LLM extraction: {len(entities)} entities and {len(relations)} relations")
                    return True
                    
                except Exception as e:
                    self.logger.error(f"Error processing document {doc_id} with LLM: {e}")
                    
                    # Update status to failed
                    await self.doc_status_db.upsert({
                        doc_id: {
                            "status": "failed",
                            "content": content,
                            "timestamp": asyncio.get_event_loop().time(),
                            "error": str(e),
                            "processing_type": "batch_one_shot"
                        }
                    })
                    
                    return False
        
        # Tạo tasks cho tất cả documents
        tasks = [process_single_document_llm(doc) for doc in documents]
        
        # Chạy song song tất cả documents
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Xử lý kết quả
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Exception in document {documents[i]['doc_id']}: {result}")
                processed_results.append(False)
            else:
                processed_results.append(result)
        
        success_count = sum(processed_results)
        self.logger.info(f"Batch LLM processing completed: {success_count}/{len(documents)} documents successful")
        
        return processed_results

    async def _save_chunks(self, chunks: List[str], doc_id: str):
        """Lưu chunks vào database"""
        chunk_data = {}
        for i, chunk_content in enumerate(chunks):
            chunk_id = compute_hash_with_prefix(chunk_content, "chunk-")
            chunk_data[chunk_id] = {
                "content": chunk_content,
                "doc_id": doc_id,
                "chunk_index": i
            }
        
        if chunk_data:
            await self.chunk_db.upsert(chunk_data)
            self.logger.info(f"Saved {len(chunk_data)} chunks for document {doc_id}")

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
            await self.entity_db.upsert(entity_data)
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
            await self.relation_db.upsert(relation_data)
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
            
            # Combine chunk IDs
            chunk_ids = [entity.chunk_id for entity in entity_list]
            combined_chunk_ids = " <|> ".join(chunk_ids)
            
            # Create or update node
            node_data = {
                "entity_name": normalized_name,
                "description": combined_description,
                "source": combined_chunk_ids,
                "topic_id": None
            }
            
            await self.graph_db.upsert_node(normalized_name, node_data)
        
        # Add relations as edges
        for relation in relations:
            source_normalized = normalize_entity_name(relation.source_entity)
            target_normalized = normalize_entity_name(relation.target_entity)
            
            edge_data = {
                "relation_description": relation.relation_description,
                "chunk_id": relation.chunk_id,
                "doc_id": relation.doc_id
            }
            
            await self.graph_db.upsert_edge(source_normalized, target_normalized, edge_data)
        
        self.logger.info(f"Updated graph with {len(entity_groups)} entities and {len(relations)} relations")

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
        await self.doc_status_db.index_done_callback()
        await self.chunk_db.index_done_callback()
        await self.entity_db.index_done_callback()
        await self.relation_db.index_done_callback()
        await self.graph_db.index_done_callback()
        
        self.logger.info("Cleanup completed successfully")
    
    # Statistics methods
    async def get_system_stats(self) -> Dict[str, Any]:
        """Lấy thống kê tổng quan của hệ thống"""
        try:
            # Get document status counts
            status_counts = await self.get_status_counts()
            
            # Get graph stats
            graph = self.graph_db._graph
            graph_stats = {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges()
            }
            
            # Get vector DB stats
            entity_count = len(self.entity_db.client_storage["data"])
            relation_count = len(self.relation_db.client_storage["data"])
            
            # Get chunk count
            chunk_count = len(self.chunk_db._data)
            
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