"""
Utility types and functions for the GraphRAG core module.
"""

from typing import Callable, List
import numpy as np

EmbeddingFunc = Callable[[List[str]], np.ndarray] 