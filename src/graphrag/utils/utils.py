import hashlib
import re
from typing import List, Tuple
from dataclasses import dataclass
import os
import json

@dataclass
class Entity:
    """Entity data structure"""
    entity_name: str
    description: str
    chunk_id: str
    doc_id: str


@dataclass
class Relation:
    """Relation data structure"""
    source_entity: str
    relation_description: str
    target_entity: str
    chunk_id: str
    doc_id: str


@dataclass
class Chunk:
    """Chunk data structure"""
    content: str
    doc_id: str


def compute_hash_with_prefix(content: str, prefix: str) -> str:
    """Tính hash cho content với prefix"""
    return f"{prefix}{hashlib.md5(content.encode('utf-8')).hexdigest()}"


def normalize_entity_name(entity_name: str) -> str:
    """Chuẩn hóa tên entity (strip và viết hoa)"""
    return entity_name.strip().upper()


def create_chunks(content: str, chunk_size: int) -> List[str]:
    """Tạo chunks từ content"""
    words = content.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks


def extract_entities_and_relations_from_text(text: str) -> Tuple[List[Entity], List[Relation]]:
    """
    Extract entities và relations từ text sử dụng regex
    Format mong đợi:
    <entity>Name_entity<SEP>A description of entity in doc
    <relation>Entity_source<SEP> relation between entity source and target <SEP> entity_target
    """
    entities = []
    relations = []
    
    # Extract entities
    entity_pattern = r'<entity>(.*?)<SEP>(.*?)(?=<entity>|<relation>|$)'
    entity_matches = re.findall(entity_pattern, text, re.DOTALL)
    
    for entity_name, description in entity_matches:
        entities.append(Entity(
            entity_name=entity_name.strip(),
            description=description.strip(),
            chunk_id="",  # Sẽ được set sau
            doc_id=""     # Sẽ được set sau
        ))
    
    # Extract relations
    relation_pattern = r'<relation>(.*?)<SEP>(.*?)<SEP>(.*?)(?=<entity>|<relation>|$)'
    relation_matches = re.findall(relation_pattern, text, re.DOTALL)
    
    for source, relation_desc, target in relation_matches:
        relations.append(Relation(
            source_entity=source.strip(),
            relation_description=relation_desc.strip(),
            target_entity=target.strip(),
            chunk_id="",  # Sẽ được set sau
            doc_id=""     # Sẽ được set sau
        ))
    
    return entities, relations


def validate_entity_relation_format(text: str) -> bool:
    """Validate format của text có đúng pattern không"""
    # Check if text contains at least one entity or relation
    entity_pattern = r'<entity>.*?<SEP>.*?(?=<entity>|<relation>|$)'
    relation_pattern = r'<relation>.*?<SEP>.*?<SEP>.*?(?=<entity>|<relation>|$)'
    
    has_entity = bool(re.search(entity_pattern, text, re.DOTALL))
    has_relation = bool(re.search(relation_pattern, text, re.DOTALL))
    
    return has_entity or has_relation 

def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


def write_json(json_obj, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)
