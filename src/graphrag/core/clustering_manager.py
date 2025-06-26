"""
Dynamic Clustering Manager for GraphRAG System
Based on cluster_new.py implementation
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np

from ..utils.logger_config import get_logger
from ..clients.embedding_client import BaseEmbeddingClient


@dataclass
class ClusterInfo:
    """Information about a cluster/topic"""
    cluster_id: int
    keywords: List[str]
    doc_count: int
    documents: List[str]
    centroid_embedding: Optional[np.ndarray] = None
    confidence_scores: Optional[List[float]] = None


@dataclass
class DocumentClusterResult:
    """Result of document clustering"""
    doc_id: str
    cluster_id: int
    confidence: float
    is_outlier: bool
    keywords: List[str]
    processing_time: float


class DynamicClusteringManager:
    """
    Dynamic BERTopic Pipeline Manager for GraphRAG System
    Based on cluster_new.py implementation
    """
    
    def __init__(self, 
                 embedding_client: BaseEmbeddingClient,
                 working_dir: str,
                 update_threshold: int = 10,
                 min_cluster_size: int = 5,
                 min_samples: int = 2,
                 model_name: str = "dynamic_bertopic_model"):
        """
        Initialize clustering manager
        
        Args:
            embedding_client: Client for generating embeddings
            working_dir: Working directory for model storage
            update_threshold: Number of outliers before model update
            min_cluster_size: Minimum cluster size for new clusters
            min_samples: Minimum samples for HDBSCAN
            model_name: Name of the BERTopic model
        """
        self.embedding_client = embedding_client
        self.working_dir = working_dir
        self.update_threshold = update_threshold
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.model_name = model_name
        self.model_path = os.path.join(working_dir, model_name)
        
        self.logger = get_logger()
        
        # Clustering state (following cluster_new.py structure)
        self.outlier_buffer = []
        self.topic_model = None
        self.embedding_model = None
        self._is_initialized = False
        
        # Statistics tracking
        self.cluster_stats = {
            'total_docs_processed': 0,
            'outliers_collected': 0,
            'model_updates': 0,
            'new_clusters_created': 0
        }
        
        self.logger.info(f"DynamicClusteringManager initialized with threshold={update_threshold}")
    
    async def initialize_model(self, initial_docs: List[str]) -> bool:
        """
        Initialize BERTopic model with initial documents
        Following cluster_new.py train_initial_model logic
        """
        try:
            if os.path.exists(self.model_path):
                self.logger.info(f"Loading existing model from {self.model_path}")
                await self._load_model()
                return True
            
            self.logger.info("Training new BERTopic model...")
            
            # Import here to avoid dependency issues
            from bertopic import BERTopic
            from umap.umap_ import UMAP
            from hdbscan import HDBSCAN
            
            # Generate embeddings
            embeddings = await self.embedding_client.embed(initial_docs)
            
            # Configure models (following cluster_new.py BEST_PARAMS)
            umap_model = UMAP(
                n_neighbors=5, 
                n_components=5, 
                min_dist=0.0, 
                metric='cosine', 
                random_state=42
            )
            
            hdbscan_model = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True
            )
            
            # Train model
            topic_model = BERTopic(
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                verbose=True
            )
            
            topic_model.fit_transform(initial_docs, embeddings)
            
            # Save model
            os.makedirs(self.model_path, exist_ok=True)
            topic_model.save(self.model_path, serialization="safetensors", save_embedding_model=self.embedding_client.model_name)
            
            self.topic_model = topic_model
            self._is_initialized = True
            
            self.logger.info(f"Model trained and saved to {self.model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            return False
    
    async def _load_model(self):
        """Load BERTopic model from disk"""
        try:
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
            
            self.topic_model = BERTopic.load(self.model_path)
            self.embedding_model = SentenceTransformer(self.embedding_client.model_name)
            self._is_initialized = True
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    async def process_document(self, doc_id: str, content: str) -> DocumentClusterResult:
        """
        Process a single document for clustering
        Following cluster_new.py process_new_document logic
        """
        start_time = asyncio.get_event_loop().time()
        
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")
        
        self.cluster_stats['total_docs_processed'] += 1
        
        self.logger.info(f"Processing document: \"{content[:70]}...\"")
        
        try:
            # Generate embedding
            embedding = await self.embedding_client.embed([content])
            
            # Predict topic (following cluster_new.py logic)
            predicted_topics, _ = self.topic_model.transform([content], embedding)
            topic_id = predicted_topics[0]
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            if topic_id != -1:
                # Document assigned to existing cluster
                topic_info = self.topic_model.get_topic(topic_id)
                keywords = [word for word, _ in topic_info]
                
                self.logger.info(f"✅ Result: Belongs to existing topic ID={topic_id} ({keywords})")
                
                return DocumentClusterResult(
                    doc_id=doc_id,
                    cluster_id=topic_id,
                    confidence=1.0,  # High confidence for assigned topics
                    is_outlier=False,
                    keywords=keywords,
                    processing_time=processing_time
                )
            else:
                # Document is outlier (following cluster_new.py logic)
                self.logger.info("⚠️ Result: Is an Outlier. Adding to buffer.")
                self.outlier_buffer.append(content)
                self.cluster_stats['outliers_collected'] += 1
                
                self.logger.info(f"   [Buffer Status]: {len(self.outlier_buffer)} / {self.update_threshold} outliers.")
                
                # Check threshold and update (following cluster_new.py logic)
                if len(self.outlier_buffer) >= self.update_threshold:
                    await self._update_model_with_outliers()
                
                return DocumentClusterResult(
                    doc_id=doc_id,
                    cluster_id=-1,
                    confidence=0.0,
                    is_outlier=True,
                    keywords=[],
                    processing_time=processing_time
                )
                
        except Exception as e:
            self.logger.error(f"Error processing document {doc_id}: {e}")
            return DocumentClusterResult(
                doc_id=doc_id,
                cluster_id=-1,
                confidence=0.0,
                is_outlier=True,
                keywords=[],
                processing_time=asyncio.get_event_loop().time() - start_time
            )
    
    async def _update_model_with_outliers(self):
        """
        Update model with accumulated outliers
        Following cluster_new.py _update_model_with_outliers logic
        """
        self.logger.info("="*50)
        self.logger.info(f"🔥🔥🔥 BUFFER IS FULL! ACTIVATING MODEL UPDATE 🔥🔥🔥")
        self.logger.info(f"   Using {len(self.outlier_buffer)} outlier documents to update.")
        
        try:
            # Extract data from buffer
            docs_to_update = self.outlier_buffer.copy()
            embeddings_to_update = await self.embedding_client.embed(docs_to_update)
            
            # Count topics before update
            topics_before = len(self.topic_model.get_topics())
            
            # Update model (following cluster_new.py logic)
            self.topic_model.partial_fit(docs_to_update, embeddings_to_update)
            
            # Count new topics
            topics_after = len(self.topic_model.get_topics())
            new_topics = topics_after - topics_before
            
            # Update statistics
            self.cluster_stats['model_updates'] += 1
            self.cluster_stats['new_clusters_created'] += new_topics
            
            self.logger.info("   [SUCCESS] Model has been updated!")
            
            # Save updated model
            self.save_model()
            
            # Clear buffer (following cluster_new.py logic)
            self.outlier_buffer = []
            self.logger.info("   [INFO] Buffer has been cleared.")
            
        except Exception as e:
            self.logger.error(f"Error updating model: {e}")
            # Don't clear buffer on error
        finally:
            self.logger.info("="*50)
    
    def save_model(self):
        """Save current model state"""
        self.logger.info(f"   [INFO] Saving latest model state to: {self.model_path}")
        try:
            self.topic_model.save(self.model_path, serialization="safetensors", save_embedding_model=self.embedding_client.model_name)
            self.logger.info("   [SUCCESS] Model saved successfully!")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
    
    async def get_cluster_info(self, cluster_id: int) -> Optional[ClusterInfo]:
        """
        Get information about a specific cluster
        """
        if not self._is_initialized:
            return None
        
        try:
            topic_info = self.topic_model.get_topic(cluster_id)
            if topic_info is None:
                return None
            
            keywords = [word for word, _ in topic_info]
            
            return ClusterInfo(
                cluster_id=cluster_id,
                keywords=keywords,
                doc_count=0,  # Would need to track this separately
                documents=[],
                confidence_scores=[]
            )
            
        except Exception as e:
            self.logger.error(f"Error getting cluster info: {e}")
            return None
    
    async def get_all_clusters(self) -> List[ClusterInfo]:
        """
        Get information about all clusters
        """
        if not self._is_initialized:
            return []
        
        try:
            topic_info = self.topic_model.get_topic_info()
            clusters = []
            
            for _, row in topic_info.iterrows():
                cluster_id = row['Topic']
                if cluster_id != -1:  # Skip outlier topic
                    cluster = await self.get_cluster_info(cluster_id)
                    if cluster:
                        clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error getting all clusters: {e}")
            return []
    
    async def get_cluster_statistics(self) -> Dict[str, Any]:
        """
        Get clustering statistics
        """
        if not self._is_initialized:
            return {}
        
        try:
            topic_info = self.topic_model.get_topic_info()
            
            return {
                **self.cluster_stats,
                'total_clusters': len(topic_info) - 1,  # Exclude outlier topic
                'current_buffer_size': len(self.outlier_buffer),
                'outlier_rate': self.cluster_stats['outliers_collected'] / max(self.cluster_stats['total_docs_processed'], 1),
                'model_path': self.model_path,
                'is_initialized': self._is_initialized
            }
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}
    
    async def force_update_model(self) -> bool:
        """
        Force model update even if buffer is not full
        """
        if len(self.outlier_buffer) == 0:
            self.logger.info("No outliers in buffer to update")
            return True
        
        await self._update_model_with_outliers()
        return True
    
    async def cleanup(self):
        """Cleanup resources"""
        self.outlier_buffer = []
        self.topic_model = None
        self.embedding_model = None
        self._is_initialized = False 