import asyncio
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import threading

from ..utils.logger_config import get_logger
from ..utils.utils import Entity, Relation, extract_entities_and_relations_from_text, validate_entity_relation_format, compute_hash_with_prefix
from ..clients.llm_client import BaseLLMClient


class LLMExtractor:
    """LLM Extractor để extract entities và relations từ text"""
    
    def __init__(self, llm_client: Optional[BaseLLMClient] = None, max_workers: int = 4):
        """
        Khởi tạo LLM Extractor
        
        Args:
            llm_client: LLM client (có thể là VLLMClient, OpenAIClient, etc.)
            max_workers: Số worker threads cho parallel processing
        """
        self.llm_client = llm_client
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.lock = threading.Lock()
        self.logger = get_logger()
        
        # Prompt template cho one-shot extraction
        self.extraction_prompt = """
You are an expert at extracting entities and relations from text. Please analyze the following text and extract entities and relations in the exact format specified.

Text to analyze:
{text}

Please extract entities and relations in this exact format:
<entity>Entity_Name<SEP>Entity description
<relation>Source_Entity<SEP>Relation description<SEP>Target_Entity

Rules:
1. Extract all important entities (people, organizations, places, concepts, etc.)
2. Extract all meaningful relations between entities
3. Use clear, concise descriptions
4. Follow the exact format with <entity> and <relation> tags
5. Use <SEP> as separator between fields

Output:
"""
    
    async def extract_from_text(self, text: str) -> Tuple[List[Entity], List[Relation]]:
        """
        Extract entities và relations từ text sử dụng LLM
        
        Args:
            text: Text cần extract
            
        Returns:
            Tuple[List[Entity], List[Relation]]: Entities và relations được extract
        """
        try:
            if self.llm_client:
                # Sử dụng LLM để extract
                extracted_text = await self._call_llm(text)
                entities, relations = extract_entities_and_relations_from_text(extracted_text)
                self.logger.info(f"LLM extracted {len(entities)} entities and {len(relations)} relations")
            else:
                # Fallback: sử dụng regex trực tiếp nếu text đã có format
                if validate_entity_relation_format(text):
                    entities, relations = extract_entities_and_relations_from_text(text)
                    self.logger.info(f"Regex extracted {len(entities)} entities and {len(relations)} relations")
                else:
                    self.logger.warning("No LLM client provided and text doesn't have entity/relation format")
                    entities, relations = [], []
            
            return entities, relations
            
        except Exception as e:
            self.logger.error(f"Error extracting entities and relations: {e}")
            return [], []
    
    async def _call_llm(self, text: str) -> str:
        """
        Gọi LLM để extract entities và relations
        
        Args:
            text: Text cần extract
            
        Returns:
            str: Text được extract theo format mong muốn
        """
        if not self.llm_client:
            raise ValueError("LLM client not provided")
        
        # Format prompt
        prompt = self.extraction_prompt.format(text=text)
        
        try:
            self.logger.debug(f"Calling LLM for text extraction (length: {len(text)})")
            
            # Gọi LLM client
            response = await self.llm_client.generate(
                prompt=prompt,
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=2048
            )
            
            self.logger.debug(f"LLM response received (length: {len(response)})")
            return response
            
        except Exception as e:
            self.logger.error(f"Error calling LLM: {e}")
            return ""
    
    async def extract_from_chunks(self, chunks: List[str], doc_id: str) -> Tuple[List[Entity], List[Relation]]:
        """
        Extract entities và relations từ nhiều chunks song song, chunk_id là hash md5 với prefix chunk-
        """
        all_entities = []
        all_relations = []
        
        self.logger.info(f"Starting extraction from {len(chunks)} chunks for document {doc_id}")
        
        async def process_chunk(chunk_content: str, chunk_index: int):
            chunk_id = compute_hash_with_prefix(chunk_content, "chunk-")
            try:
                entities, relations = await self.extract_from_text(chunk_content)
                # Gán chunk_id và doc_id đúng chuẩn cho entity/relation
                for entity in entities:
                    entity.chunk_id = chunk_id
                    entity.doc_id = doc_id
                for relation in relations:
                    relation.chunk_id = chunk_id
                    relation.doc_id = doc_id
                return entities, relations
            except Exception as e:
                self.logger.error(f"Error processing chunk {chunk_index}: {e}")
                return [], []
        
        # Tạo task cho từng chunk
        tasks = [process_chunk(chunk, idx) for idx, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Error in chunk {i}: {result}")
                continue
            entities, relations = result
            all_entities.extend(entities)
            all_relations.extend(relations)
        
        self.logger.info(f"Total extracted: {len(all_entities)} entities, {len(all_relations)} relations")
        return all_entities, all_relations
    
    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        self.logger.info("LLM Extractor cleanup completed") 