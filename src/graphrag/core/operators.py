import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from ..utils.logger_config import get_logger
from ..utils.utils import (
    Entity, Relation, Chunk,
    compute_hash_with_prefix, normalize_entity_name, create_chunks
)
from .llm_extractor import LLMExtractor


class DocumentOperator:
    """Operator để xử lý documents"""
    
    def __init__(self, doc_status_db, chunk_db, entity_db, relation_db, graph_db, llm_extractor):
        self.doc_status_db = doc_status_db
        self.chunk_db = chunk_db
        self.entity_db = entity_db
        self.relation_db = relation_db
        self.graph_db = graph_db
        self.llm_extractor = llm_extractor
        self.logger = get_logger()
    
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
            
            # Extract entities and relations from chunks
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
                    "chunks_count": len(chunks)
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
                    "error": str(e)
                }
            })
            
            return False
    
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


class QueryOperator:
    """Operator để query dữ liệu"""
    
    def __init__(self, entity_db, relation_db, graph_db, chunk_db, doc_status_db):
        self.entity_db = entity_db
        self.relation_db = relation_db
        self.graph_db = graph_db
        self.chunk_db = chunk_db
        self.doc_status_db = doc_status_db
        self.logger = get_logger()
    
    async def query_entities(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Query entities từ vector DB"""
        self.logger.debug(f"Querying entities with: '{query}', top_k: {top_k}")
        results = await self.entity_db.query(query, top_k)
        self.logger.debug(f"Found {len(results)} entities")
        return results
    
    async def query_relations(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Query relations từ vector DB"""
        self.logger.debug(f"Querying relations with: '{query}', top_k: {top_k}")
        results = await self.relation_db.query(query, top_k)
        self.logger.debug(f"Found {len(results)} relations")
        return results
    
    async def get_entity_neighbors(self, entity_name: str) -> List[Tuple[str, Dict, Dict]]:
        """Lấy neighbors của một entity trong graph"""
        normalized_name = normalize_entity_name(entity_name)
        self.logger.debug(f"Getting neighbors for entity: {normalized_name}")
        neighbors = await self.graph_db.get_connected_nodes_with_edges(normalized_name)
        self.logger.debug(f"Found {len(neighbors)} neighbors")
        return neighbors
    
    async def get_document_status(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Lấy trạng thái của document"""
        self.logger.debug(f"Getting status for document: {doc_id}")
        return await self.doc_status_db.get_by_id(doc_id)
    
    async def get_chunk_content(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Lấy nội dung của chunk"""
        self.logger.debug(f"Getting content for chunk: {chunk_id}")
        return await self.chunk_db.get_by_id(chunk_id)
    
    async def get_status_counts(self) -> Dict[str, int]:
        """Lấy số lượng documents theo trạng thái"""
        self.logger.debug("Getting document status counts")
        return await self.doc_status_db.get_status_counts()
    
    async def search_entities_by_name(self, entity_name: str) -> List[Dict[str, Any]]:
        """Tìm entities theo tên"""
        self.logger.debug(f"Searching entities by name: {entity_name}")
        
        # Query với tên entity
        results = await self.entity_db.query(entity_name, top_k=20)
        
        # Filter kết quả có tên entity chứa query
        filtered_results = []
        query_lower = entity_name.lower()
        
        for result in results:
            result_entity_name = result.get('entity_name', '').lower()
            if query_lower in result_entity_name:
                filtered_results.append(result)
        
        self.logger.debug(f"Found {len(filtered_results)} entities matching '{entity_name}'")
        return filtered_results
    
    async def get_entity_graph_context(self, entity_name: str, max_depth: int = 2) -> Dict[str, Any]:
        """Lấy context graph của một entity"""
        normalized_name = normalize_entity_name(entity_name)
        self.logger.debug(f"Getting graph context for entity: {normalized_name}")
        
        # Get entity node data
        node_data = await self.graph_db.get_node_data(normalized_name)
        if not node_data:
            self.logger.warning(f"Entity not found: {normalized_name}")
            return {"error": "Entity not found"}
        
        # Get neighbors
        neighbors = await self.graph_db.get_connected_nodes_with_edges(normalized_name)
        
        # Build context
        context = {
            "entity": {
                "name": normalized_name,
                "data": node_data
            },
            "neighbors": [
                {
                    "name": neighbor_id,
                    "data": neighbor_data,
                    "relation": edge_data.get("relation_description", "")
                }
                for neighbor_id, neighbor_data, edge_data in neighbors
            ],
            "total_neighbors": len(neighbors)
        }
        
        self.logger.debug(f"Built context with {len(neighbors)} neighbors")
        return context 