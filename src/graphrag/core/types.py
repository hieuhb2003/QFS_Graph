"""
Type definitions for the GraphRAG system.
"""

from typing import TypedDict, List, Dict, Any

class KnowledgeGraph(TypedDict):
    """Knowledge graph structure"""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]] 