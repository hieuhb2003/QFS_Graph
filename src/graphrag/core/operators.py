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
        """Lưu entities vào vector DB với deduplication"""
        entity_data = {}
        existing_entities = {}
        
        # Get existing entities to check for duplicates
        try:
            # Query existing entities to check for duplicates
            existing_results = await self.entity_db.query("", top_k=1000)  # Get all entities
            for result in existing_results:
                existing_name = result.get('entity_name', '')
                if existing_name:
                    existing_entities[existing_name.lower()] = result
        except Exception as e:
            self.logger.warning(f"Could not fetch existing entities: {e}")
        
        for entity in entities:
            # Use normalized entity name as key for deduplication
            normalized_name = normalize_entity_name(entity.entity_name)
            entity_key = compute_hash_with_prefix(normalized_name, "ent-")
            
            # Check if entity already exists
            existing_entity = existing_entities.get(entity.entity_name.lower())
            
            if existing_entity:
                # Merge with existing entity
                existing_description = existing_entity.get('description', '')
                existing_chunk_ids = existing_entity.get('chunk_id', '')
                existing_doc_ids = existing_entity.get('doc_id', '')
                
                # Combine descriptions
                if entity.description not in existing_description:
                    combined_description = f"{existing_description} <|> {entity.description}"
                else:
                    combined_description = existing_description
                
                # Combine chunk IDs
                if entity.chunk_id not in existing_chunk_ids:
                    combined_chunk_ids = f"{existing_chunk_ids} <|> {entity.chunk_id}"
                else:
                    combined_chunk_ids = existing_chunk_ids
                
                # Combine doc IDs
                if entity.doc_id not in existing_doc_ids:
                    combined_doc_ids = f"{existing_doc_ids} <|> {entity.doc_id}"
                else:
                    combined_doc_ids = existing_doc_ids
                
                entity_data[entity_key] = {
                    "content": f"{normalized_name} {combined_description}",
                    "entity_name": normalized_name,
                    "description": combined_description,
                    "chunk_id": combined_chunk_ids,
                    "doc_id": combined_doc_ids
                }
                
                self.logger.debug(f"Merged entity: {normalized_name}")
            else:
                # New entity
                entity_data[entity_key] = {
                    "content": f"{normalized_name} {entity.description}",
                    "entity_name": normalized_name,
                    "description": entity.description,
                    "chunk_id": entity.chunk_id,
                    "doc_id": entity.doc_id
                }
        
        if entity_data:
            await self.entity_db.upsert(entity_data)
            self.logger.info(f"Saved {len(entity_data)} entities to vector DB (with deduplication)")
    
    async def _save_relations(self, relations: List[Relation]):
        """Lưu relations vào vector DB với deduplication"""
        relation_data = {}
        existing_relations = {}
        
        # Get existing relations to check for duplicates
        try:
            # Query existing relations to check for duplicates
            existing_results = await self.relation_db.query("", top_k=1000)  # Get all relations
            for result in existing_results:
                source = result.get('source_entity', '')
                target = result.get('target_entity', '')
                rel_desc = result.get('relation_description', '')
                if source and target and rel_desc:
                    key = f"{source.lower()}_{rel_desc.lower()}_{target.lower()}"
                    existing_relations[key] = result
        except Exception as e:
            self.logger.warning(f"Could not fetch existing relations: {e}")
        
        for relation in relations:
            # Normalize entity names
            source_normalized = normalize_entity_name(relation.source_entity)
            target_normalized = normalize_entity_name(relation.target_entity)
            
            # Create unique key for deduplication
            relation_key = compute_hash_with_prefix(
                f"{source_normalized} {relation.relation_description} {target_normalized}",
                "rel-"
            )
            
            # Check if relation already exists
            existing_key = f"{source_normalized.lower()}_{relation.relation_description.lower()}_{target_normalized.lower()}"
            existing_relation = existing_relations.get(existing_key)
            
            if existing_relation:
                # Merge with existing relation
                existing_chunk_ids = existing_relation.get('chunk_id', '')
                existing_doc_ids = existing_relation.get('doc_id', '')
                
                # Combine chunk IDs
                if relation.chunk_id not in existing_chunk_ids:
                    combined_chunk_ids = f"{existing_chunk_ids} <|> {relation.chunk_id}"
                else:
                    combined_chunk_ids = existing_chunk_ids
                
                # Combine doc IDs
                if relation.doc_id not in existing_doc_ids:
                    combined_doc_ids = f"{existing_doc_ids} <|> {relation.doc_id}"
                else:
                    combined_doc_ids = existing_doc_ids
                
                relation_data[relation_key] = {
                    "content": f"{source_normalized} {relation.relation_description} {target_normalized}",
                    "source_entity": source_normalized,
                    "relation_description": relation.relation_description,
                    "target_entity": target_normalized,
                    "chunk_id": combined_chunk_ids,
                    "doc_id": combined_doc_ids
                }
                
                self.logger.debug(f"Merged relation: {source_normalized} -> {relation.relation_description} -> {target_normalized}")
            else:
                # New relation
                relation_data[relation_key] = {
                    "content": f"{source_normalized} {relation.relation_description} {target_normalized}",
                    "source_entity": source_normalized,
                    "relation_description": relation.relation_description,
                    "target_entity": target_normalized,
                    "chunk_id": relation.chunk_id,
                    "doc_id": relation.doc_id
                }
        
        if relation_data:
            await self.relation_db.upsert(relation_data)
            self.logger.info(f"Saved {len(relation_data)} relations to vector DB (with deduplication)")
    
    async def _update_graph(self, entities: List[Entity], relations: List[Relation]):
        """Cập nhật knowledge graph với deduplication"""
        # Group entities by normalized name
        entity_groups = {}
        
        for entity in entities:
            normalized_name = normalize_entity_name(entity.entity_name)
            if normalized_name not in entity_groups:
                entity_groups[normalized_name] = []
            entity_groups[normalized_name].append(entity)
        
        # Merge entities with same normalized name
        for normalized_name, entity_list in entity_groups.items():
            # Check if node already exists
            existing_node = await self.graph_db.get_node(normalized_name)
            
            if existing_node:
                # Merge with existing node
                existing_description = existing_node.get('description', '')
                existing_source = existing_node.get('source', '')
                
                # Combine descriptions
                new_descriptions = [entity.description for entity in entity_list]
                for desc in new_descriptions:
                    if desc not in existing_description:
                        if existing_description:
                            existing_description = f"{existing_description} <|> {desc}"
                        else:
                            existing_description = desc
                
                # Combine chunk IDs
                new_chunk_ids = [entity.chunk_id for entity in entity_list]
                for chunk_id in new_chunk_ids:
                    if chunk_id not in existing_source:
                        if existing_source:
                            existing_source = f"{existing_source} <|> {chunk_id}"
                        else:
                            existing_source = chunk_id
                
                node_data = {
                    "entity_name": normalized_name,
                    "description": existing_description,
                    "source": existing_source,
                    "topic_id": None
                }
                
                self.logger.debug(f"Merged graph node: {normalized_name}")
            else:
                # New node
                descriptions = [entity.description for entity in entity_list]
                combined_description = " <|> ".join(descriptions)
                
                chunk_ids = [entity.chunk_id for entity in entity_list]
                combined_chunk_ids = " <|> ".join(chunk_ids)
                
                node_data = {
                    "entity_name": normalized_name,
                    "description": combined_description,
                    "source": combined_chunk_ids,
                    "topic_id": None
                }
            
            await self.graph_db.upsert_node(normalized_name, node_data)
        
        # Add relations as edges with deduplication
        for relation in relations:
            source_normalized = normalize_entity_name(relation.source_entity)
            target_normalized = normalize_entity_name(relation.target_entity)
            
            # Check if edge already exists
            existing_edge = await self.graph_db.get_edge(source_normalized, target_normalized)
            
            if existing_edge:
                # Merge with existing edge
                existing_chunk_id = existing_edge.get('chunk_id', '')
                existing_doc_id = existing_edge.get('doc_id', '')
                
                # Combine chunk IDs
                if relation.chunk_id not in existing_chunk_id:
                    if existing_chunk_id:
                        combined_chunk_id = f"{existing_chunk_id} <|> {relation.chunk_id}"
                    else:
                        combined_chunk_id = relation.chunk_id
                else:
                    combined_chunk_id = existing_chunk_id
                
                # Combine doc IDs
                if relation.doc_id not in existing_doc_id:
                    if existing_doc_id:
                        combined_doc_id = f"{existing_doc_id} <|> {relation.doc_id}"
                    else:
                        combined_doc_id = relation.doc_id
                else:
                    combined_doc_id = existing_doc_id
                
                edge_data = {
                    "relation_description": relation.relation_description,
                    "chunk_id": combined_chunk_id,
                    "doc_id": combined_doc_id
                }
                
                self.logger.debug(f"Merged graph edge: {source_normalized} -> {target_normalized}")
            else:
                # New edge
                edge_data = {
                    "relation_description": relation.relation_description,
                    "chunk_id": relation.chunk_id,
                    "doc_id": relation.doc_id
                }
            
            await self.graph_db.upsert_edge(source_normalized, target_normalized, edge_data)
        
        self.logger.info(f"Updated graph with {len(entity_groups)} entities and {len(relations)} relations (with deduplication)")


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
        
        # Try to get entity node data with normalized name
        node_data = await self.graph_db.get_node_data(normalized_name)
        
        # If not found, try to find similar node names
        if not node_data:
            # Get all nodes and find the best match
            all_nodes = list(self.graph_db._graph.nodes())
            best_match = None
            
            # Look for exact match first
            for node in all_nodes:
                if entity_name.lower() in node.lower() or node.lower() in entity_name.lower():
                    best_match = node
                    break
            
            if best_match:
                node_data = await self.graph_db.get_node_data(best_match)
                normalized_name = best_match
                self.logger.debug(f"Found similar entity: {best_match}")
            else:
                self.logger.warning(f"Entity not found: {entity_name}")
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