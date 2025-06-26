"""
GraphRAG System - A comprehensive system for building knowledge graphs from documents
using LLM extraction, vector databases, and NetworkX graph processing.
"""

__version__ = "0.1.0"
__author__ = "GraphRAG Team"
__email__ = "contact@graphrag.com"

# Core imports
from .core.graphrag_system import GraphRAGSystem
from .core.llm_extractor import LLMExtractor
from .core.operators import DocumentOperator, QueryOperator

# Client imports
from .clients.llm_client import create_llm_client, BaseLLMClient, VLLMClient, OpenAIClient
from .clients.embedding_client import create_embedding_client, BaseEmbeddingClient, SentenceTransformersEmbeddingClient, VLLMEmbeddingClient, OpenAIEmbeddingClient

# Utils imports
from .utils.logger_config import setup_logger, get_logger, GraphRAGLogger

# Database imports
from .db.json_doc_status_impl import JsonDocStatusStorage
from .db.json_kv_impl import JsonKVStorage
from .db.nano_vector_db_impl import NanoVectorDBStorage
from .db.networkx_impl import NetworkXStorage

# Data structures
from .utils.utils import Entity, Relation, Chunk

__all__ = [
    # Core
    "GraphRAGSystem",
    "LLMExtractor", 
    "DocumentOperator",
    "QueryOperator",
    
    # Clients
    "create_llm_client",
    "BaseLLMClient",
    "VLLMClient",
    "OpenAIClient",
    "create_embedding_client",
    "BaseEmbeddingClient",
    "SentenceTransformersEmbeddingClient",
    "VLLMEmbeddingClient",
    "OpenAIEmbeddingClient",
    
    # Utils
    "setup_logger",
    "get_logger",
    "GraphRAGLogger",
    
    # Database
    "JsonDocStatusStorage",
    "JsonKVStorage", 
    "NanoVectorDBStorage",
    "NetworkXStorage",
    
    # Data structures
    "Entity",
    "Relation",
    "Chunk",
] 