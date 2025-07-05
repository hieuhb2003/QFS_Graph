"""
Clustering Manager for GraphRAG System
Quản lý clustering cho hệ thống GraphRAG
"""

import asyncio
import json
import os
import pickle
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import torch
from umap.umap_ import UMAP
from hdbscan import HDBSCAN

from ..utils.logger_config import get_logger
from ..utils.utils import ClusterInfo, compute_hash_with_prefix
from ..clients.embedding_client import BaseEmbeddingClient


class ClusteringManager:
    """Quản lý clustering cho GraphRAG system"""
    
    def __init__(self, 
                 embedding_client: BaseEmbeddingClient,
                 outlier_threshold: int = 10,
                 max_tokens: int = 8192,
                 batch_size: int = 16,
                 model_save_path: str = "clustering_models"):
        """
        Khởi tạo ClusteringManager
        
        Args:
            embedding_client: Embedding client để tạo embeddings
            outlier_threshold: Ngưỡng số lượng outlier để tạo cluster mới
            max_tokens: Số token tối đa cho mỗi document
            batch_size: Batch size cho việc tạo embedding
            model_save_path: Đường dẫn lưu model
        """
        self.embedding_client = embedding_client
        self.outlier_threshold = outlier_threshold
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.model_save_path = model_save_path
        self.logger = get_logger()
        
        # Best parameters từ grid search
        self.best_params = {
            'HDBSCAN_MIN_CLUSTER_SIZE': 5,
            'HDBSCAN_MIN_SAMPLES': 2,
            'UMAP_N_COMPONENTS': 5,
            'UMAP_N_NEIGHBORS': 5
        }
        
        # Khởi tạo các thành phần
        self.topic_model = None
        self.umap_model = None
        self.hdbscan_model = None
        
        # Lưu trữ dữ liệu
        self.all_docs = []
        self.all_embeddings = None
        self.cluster_history = []
        self.outlier_docs = []
        self.outlier_embeddings = None
        
        # Thông tin chi tiết về clusters
        self.cluster_assignments = {}  # {doc_hash_id: cluster_id}
        self.cluster_docs_map = {}    # {cluster_id: [doc_hash_ids]}
        self.document_hash_ids = []    # [doc_hash_id1, doc_hash_id2, ...]
        
        # Cluster info storage
        self.cluster_info_storage = {}  # {cluster_id: ClusterInfo}
        
        # Tạo thư mục lưu model
        os.makedirs(model_save_path, exist_ok=True)
        
    def _initialize_models(self):
        """Khởi tạo các mô hình clustering"""
        self.logger.info("Đang khởi tạo clustering models...")
        
        # Khởi tạo UMAP và HDBSCAN với best parameters
        self.umap_model = UMAP(
            n_neighbors=self.best_params['UMAP_N_NEIGHBORS'],
            n_components=self.best_params['UMAP_N_COMPONENTS'],
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        
        self.hdbscan_model = HDBSCAN(
            min_cluster_size=self.best_params['HDBSCAN_MIN_CLUSTER_SIZE'],
            min_samples=self.best_params['HDBSCAN_MIN_SAMPLES'],
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        # Khởi tạo BERTopic
        self.topic_model = BERTopic(
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            verbose=False
        )
        
        self.logger.info("Đã khởi tạo clustering models thành công")
    
    def _preprocess_documents(self, documents: List[str]) -> List[str]:
        """Tiền xử lý documents"""
        processed_docs = []
        for doc in documents:
            if isinstance(doc, str):
                # Tách documents nếu có separator
                sub_docs = doc.split("|||||")
                for sub_doc in sub_docs:
                    if not sub_doc.strip():
                        continue
                    # Giới hạn độ dài token
                    if len(sub_doc) > self.max_tokens * 4:  # Ước tính 4 chars per token
                        sub_doc = sub_doc[:self.max_tokens * 4]
                    processed_docs.append(sub_doc.strip())
        return processed_docs
    
    async def _create_embeddings(self, documents: List[str]) -> np.ndarray:
        """Tạo embeddings cho documents"""
        self.logger.info(f"Đang tạo embeddings cho {len(documents)} documents...")
        
        embeddings = []
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            batch_embeddings = await self.embedding_client.embed(batch)
            embeddings.extend(batch_embeddings)
        
        embeddings = np.array(embeddings)
        self.logger.info(f"Đã tạo embeddings với shape: {embeddings.shape}")
        return embeddings
    
    def _update_cluster_assignments(self, documents: List[str], topics: List[int], 
                                   doc_hash_ids: List[str], start_idx: int = 0):
        """Cập nhật thông tin chi tiết về cluster assignments"""
        for i, (doc, topic, doc_hash_id) in enumerate(zip(documents, topics, doc_hash_ids)):
            idx = start_idx + i
            
            # Lưu assignment
            self.cluster_assignments[doc_hash_id] = topic
            
            # Lưu vào cluster documents
            if topic not in self.cluster_docs_map:
                self.cluster_docs_map[topic] = []
            self.cluster_docs_map[topic].append(doc_hash_id)
    
    async def cluster_documents(self, documents: List[str], doc_hash_ids: List[str]) -> Dict[str, Any]:
        """
        Thực hiện clustering cho documents
        
        Args:
            documents: Danh sách nội dung documents
            doc_hash_ids: Danh sách hash IDs tương ứng
            
        Returns:
            Dict chứa thông tin clustering
        """
        self.logger.info(f"Bắt đầu clustering cho {len(documents)} documents...")
        
        # Khởi tạo models nếu chưa có
        if self.topic_model is None:
            self._initialize_models()
        
        # Tiền xử lý documents
        processed_docs = self._preprocess_documents(documents)
        
        # Tạo embeddings
        embeddings = await self._create_embeddings(processed_docs)
        
        # Fit BERTopic
        self.logger.info("Đang fit BERTopic model...")
        topics, probs = self.topic_model.fit_transform(processed_docs, embeddings)
        
        # Cập nhật cluster assignments
        self.document_hash_ids = doc_hash_ids
        self._update_cluster_assignments(processed_docs, topics, doc_hash_ids)
        
        # Phân loại documents thành clusters và outliers
        cluster_docs = []
        outlier_docs = []
        cluster_doc_hash_ids = []
        outlier_doc_hash_ids = []
        
        for i, topic in enumerate(topics):
            if topic == -1:  # Outlier
                outlier_docs.append(processed_docs[i])
                outlier_doc_hash_ids.append(doc_hash_ids[i])
            else:
                cluster_docs.append(processed_docs[i])
                cluster_doc_hash_ids.append(doc_hash_ids[i])
        
        # Lưu outliers
        if outlier_docs:
            self.outlier_docs = outlier_docs
            outlier_indices = [i for i, topic in enumerate(topics) if topic == -1]
            self.outlier_embeddings = embeddings[outlier_indices]
            self.logger.info(f"Phát hiện {len(outlier_docs)} outliers")
        
        # Cập nhật all_docs và all_embeddings
        self.all_docs = processed_docs
        self.all_embeddings = embeddings
        
        # Tạo cluster info
        await self._create_cluster_info()
        
        # Lưu lịch sử clustering
        self.cluster_history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'cluster_documents',
            'total_docs': len(processed_docs),
            'clustered_docs': len(cluster_docs),
            'outlier_docs': len(outlier_docs),
            'n_clusters': len(set(topics)) - (1 if -1 in topics else 0),
            'cluster_assignments': self.cluster_assignments.copy(),
            'cluster_docs_map': {k: v.copy() for k, v in self.cluster_docs_map.items()}
        })
        
        # Lưu model
        await self.save_clustering_model()
        
        result = {
            'total_documents': len(processed_docs),
            'clustered_documents': len(cluster_docs),
            'outlier_documents': len(outlier_docs),
            'n_clusters': len(set(topics)) - (1 if -1 in topics else 0),
            'cluster_assignments': self.cluster_assignments,
            'cluster_documents': self.cluster_docs_map,
            'cluster_info': self.cluster_info_storage
        }
        
        self.logger.info(f"Hoàn thành clustering: {result['n_clusters']} clusters, {result['outlier_documents']} outliers")
        return result
    
    async def update_clusters_with_new_data(self, new_documents: List[str], 
                                           new_doc_hash_ids: List[str]) -> Dict[str, Any]:
        """
        Cập nhật clusters với dữ liệu mới
        
        Args:
            new_documents: Danh sách documents mới
            new_doc_hash_ids: Danh sách hash IDs mới
            
        Returns:
            Dict chứa thông tin cập nhật
        """
        self.logger.info(f"Cập nhật clusters với {len(new_documents)} documents mới...")
        
        if self.topic_model is None:
            raise ValueError("Model chưa được fit. Hãy gọi cluster_documents() trước.")
        
        # Tiền xử lý documents mới
        processed_new_docs = self._preprocess_documents(new_documents)
        
        # Tạo embeddings cho documents mới
        new_embeddings = await self._create_embeddings(processed_new_docs)
        
        # Dự đoán topics
        predicted_topics, _ = self.topic_model.transform(processed_new_docs, new_embeddings)
        
        # Phân loại kết quả
        clustered_new_docs = []
        outlier_new_docs = []
        clustered_new_doc_hash_ids = []
        outlier_new_doc_hash_ids = []
        
        for i, topic in enumerate(predicted_topics):
            if topic == -1:  # Outlier
                outlier_new_docs.append(processed_new_docs[i])
                outlier_new_doc_hash_ids.append(new_doc_hash_ids[i])
            else:
                clustered_new_docs.append(processed_new_docs[i])
                clustered_new_doc_hash_ids.append(new_doc_hash_ids[i])
        
        # Thêm outliers mới vào tập outlier hiện tại
        if outlier_new_docs:
            self.outlier_docs.extend(outlier_new_docs)
            
            # Cập nhật outlier embeddings
            if self.outlier_embeddings is not None:
                self.outlier_embeddings = np.vstack([self.outlier_embeddings, new_embeddings])
            else:
                self.outlier_embeddings = new_embeddings
        
        # Kiểm tra xem có cần tạo cluster mới cho outliers không
        if len(self.outlier_docs) >= self.outlier_threshold:
            self.logger.info(f"Số lượng outliers ({len(self.outlier_docs)}) đã đạt ngưỡng ({self.outlier_threshold})")
            await self._create_new_clusters_from_outliers()
        
        # Cập nhật cluster assignments cho documents mới
        start_idx = len(self.all_docs)
        self.all_docs.extend(processed_new_docs)
        self.document_hash_ids.extend(new_doc_hash_ids)
        
        if self.all_embeddings is not None:
            self.all_embeddings = np.vstack([self.all_embeddings, new_embeddings])
        else:
            self.all_embeddings = new_embeddings
        
        self._update_cluster_assignments(processed_new_docs, predicted_topics, new_doc_hash_ids, start_idx)
        
        # Cập nhật cluster info
        await self._create_cluster_info()
        
        # Cập nhật lịch sử
        self.cluster_history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'update_clusters_with_new_data',
            'new_docs': len(new_documents),
            'clustered_new_docs': len(clustered_new_docs),
            'new_outliers': len(outlier_new_docs),
            'total_outliers': len(self.outlier_docs),
            'cluster_assignments': self.cluster_assignments.copy(),
            'cluster_docs_map': {k: v.copy() for k, v in self.cluster_docs_map.items()}
        })
        
        # Lưu model
        await self.save_clustering_model()
        
        result = {
            'new_documents': len(new_documents),
            'clustered_new_docs': len(clustered_new_docs),
            'new_outliers': len(outlier_new_docs),
            'total_outliers': len(self.outlier_docs),
            'cluster_assignments': self.cluster_assignments,
            'cluster_documents': self.cluster_docs_map,
            'cluster_info': self.cluster_info_storage
        }
        
        self.logger.info(f"Hoàn thành cập nhật clusters")
        return result
    
    async def _create_new_clusters_from_outliers(self):
        """Tạo cluster mới từ tập outliers"""
        self.logger.info("Tạo cluster mới từ outliers...")
        
        if len(self.outlier_docs) < self.outlier_threshold:
            self.logger.info(f"Chưa đủ outliers để tạo cluster mới ({len(self.outlier_docs)} < {self.outlier_threshold})")
            return
        
        # Tạo mô hình clustering mới cho outliers
        outlier_umap = UMAP(
            n_neighbors=self.best_params['UMAP_N_NEIGHBORS'],
            n_components=self.best_params['UMAP_N_COMPONENTS'],
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        
        outlier_hdbscan = HDBSCAN(
            min_cluster_size=self.best_params['HDBSCAN_MIN_CLUSTER_SIZE'],
            min_samples=self.best_params['HDBSCAN_MIN_SAMPLES'],
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        outlier_topic_model = BERTopic(
            umap_model=outlier_umap,
            hdbscan_model=outlier_hdbscan,
            verbose=False
        )
        
        # Fit mô hình mới trên outliers
        outlier_topics, _ = outlier_topic_model.fit_transform(self.outlier_docs, self.outlier_embeddings)
        
        # Phân loại outliers thành clusters mới và outliers thực sự
        new_clustered_outliers = []
        remaining_outliers = []
        new_cluster_embeddings = []
        remaining_outlier_embeddings = []
        new_clustered_doc_hash_ids = []
        remaining_outlier_doc_hash_ids = []
        
        # Lấy doc hash IDs của outliers (cần mapping lại)
        outlier_doc_hash_ids = []
        for doc in self.outlier_docs:
            # Tìm doc_hash_id tương ứng
            for i, stored_doc in enumerate(self.all_docs):
                if stored_doc == doc:
                    outlier_doc_hash_ids.append(self.document_hash_ids[i])
                    break
        
        for i, topic in enumerate(outlier_topics):
            if topic == -1:  # Vẫn là outlier
                remaining_outliers.append(self.outlier_docs[i])
                remaining_outlier_embeddings.append(self.outlier_embeddings[i])
                if i < len(outlier_doc_hash_ids):
                    remaining_outlier_doc_hash_ids.append(outlier_doc_hash_ids[i])
            else:  # Được cluster
                new_clustered_outliers.append(self.outlier_docs[i])
                new_cluster_embeddings.append(self.outlier_embeddings[i])
                if i < len(outlier_doc_hash_ids):
                    new_clustered_doc_hash_ids.append(outlier_doc_hash_ids[i])
        
        # Cập nhật dữ liệu
        self.outlier_docs = remaining_outliers
        self.outlier_embeddings = np.array(remaining_outlier_embeddings) if remaining_outlier_embeddings else None
        
        # Thêm documents được cluster vào tập chính
        if new_clustered_outliers:
            start_idx = len(self.all_docs)
            self.all_docs.extend(new_clustered_outliers)
            self.document_hash_ids.extend(new_clustered_doc_hash_ids)
            new_cluster_embeddings = np.array(new_cluster_embeddings)
            
            if self.all_embeddings is not None:
                self.all_embeddings = np.vstack([self.all_embeddings, new_cluster_embeddings])
            else:
                self.all_embeddings = new_cluster_embeddings
            
            # Cập nhật cluster assignments cho documents mới
            self._update_cluster_assignments(new_clustered_outliers, outlier_topics, new_clustered_doc_hash_ids, start_idx)
        
        self.logger.info(f"Tạo cluster mới từ outliers: {len(new_clustered_outliers)} được cluster, {len(remaining_outliers)} còn lại")
    
    async def _create_cluster_info(self):
        """Tạo cluster info từ cluster assignments"""
        self.cluster_info_storage = {}
        
        for cluster_id, doc_hash_ids in self.cluster_docs_map.items():
            if cluster_id == -1:  # Outlier cluster
                continue
                
            cluster_info = ClusterInfo(
                cluster_id=cluster_id,
                doc_hash_ids=doc_hash_ids,
                outlier_doc_hash_ids=[],
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
            self.cluster_info_storage[cluster_id] = cluster_info
        
        # Thêm outlier cluster info
        outlier_doc_hash_ids = []
        for doc_hash_id, cluster_id in self.cluster_assignments.items():
            if cluster_id == -1:
                outlier_doc_hash_ids.append(doc_hash_id)
        
        if outlier_doc_hash_ids:
            outlier_cluster_info = ClusterInfo(
                cluster_id=-1,
                doc_hash_ids=[],
                outlier_doc_hash_ids=outlier_doc_hash_ids,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
            self.cluster_info_storage[-1] = outlier_cluster_info
    
    async def save_clustering_model(self):
        """Lưu clustering model và dữ liệu"""
        self.logger.info(f"Đang lưu clustering model tại: {self.model_save_path}")
        
        # Lưu BERTopic model
        if self.topic_model:
            self.topic_model.save(os.path.join(self.model_save_path, "bertopic_model"))
        
        # Lưu dữ liệu khác
        data_to_save = {
            'all_docs': self.all_docs,
            'all_embeddings': self.all_embeddings,
            'outlier_docs': self.outlier_docs,
            'outlier_embeddings': self.outlier_embeddings,
            'cluster_history': self.cluster_history,
            'best_params': self.best_params,
            'cluster_assignments': self.cluster_assignments,
            'cluster_docs_map': self.cluster_docs_map,
            'document_hash_ids': self.document_hash_ids,
            'cluster_info_storage': self.cluster_info_storage
        }
        
        with open(os.path.join(self.model_save_path, "clustering_data.pkl"), 'wb') as f:
            pickle.dump(data_to_save, f)
        
        # Lưu lịch sử dưới dạng JSON để dễ đọc
        with open(os.path.join(self.model_save_path, "cluster_history.json"), 'w', encoding='utf-8') as f:
            json.dump(self.cluster_history, f, ensure_ascii=False, indent=2)
        
        # Lưu cluster info dưới dạng JSON
        cluster_info_dict = {}
        for cluster_id, cluster_info in self.cluster_info_storage.items():
            cluster_info_dict[cluster_id] = {
                'cluster_id': cluster_info.cluster_id,
                'doc_hash_ids': cluster_info.doc_hash_ids,
                'outlier_doc_hash_ids': cluster_info.outlier_doc_hash_ids,
                'summary': cluster_info.summary,
                'created_at': cluster_info.created_at,
                'updated_at': cluster_info.updated_at
            }
        
        with open(os.path.join(self.model_save_path, "cluster_info.json"), 'w', encoding='utf-8') as f:
            json.dump(cluster_info_dict, f, ensure_ascii=False, indent=2)
        
        self.logger.info("Đã lưu clustering model thành công")
    
    async def load_clustering_model(self):
        """Tải clustering model và dữ liệu đã lưu"""
        self.logger.info(f"Đang tải clustering model từ: {self.model_save_path}")
        
        # Tải BERTopic model
        bertopic_path = os.path.join(self.model_save_path, "bertopic_model")
        if os.path.exists(bertopic_path):
            self.topic_model = BERTopic.load(bertopic_path)
        else:
            raise FileNotFoundError(f"Không tìm thấy BERTopic model tại {bertopic_path}")
        
        # Tải dữ liệu khác
        data_path = os.path.join(self.model_save_path, "clustering_data.pkl")
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            
            self.all_docs = data['all_docs']
            self.all_embeddings = data['all_embeddings']
            self.outlier_docs = data['outlier_docs']
            self.outlier_embeddings = data['outlier_embeddings']
            self.cluster_history = data['cluster_history']
            self.best_params = data['best_params']
            self.cluster_assignments = data.get('cluster_assignments', {})
            self.cluster_docs_map = data.get('cluster_docs_map', data.get('cluster_documents', {}))
            self.document_hash_ids = data.get('document_hash_ids', [])
            self.cluster_info_storage = data.get('cluster_info_storage', {})
        else:
            raise FileNotFoundError(f"Không tìm thấy clustering data tại {data_path}")
        
        self.logger.info("Đã tải clustering model thành công")
    
    async def get_cluster_info(self) -> Dict[str, Any]:
        """Lấy thông tin về clusters hiện tại"""
        if self.topic_model is None:
            return {"error": "Model chưa được fit"}
        
        return {
            "total_documents": len(self.all_docs),
            "total_clusters": len([k for k in self.cluster_docs_map.keys() if k != -1]),
            "outlier_documents": len(self.outlier_docs),
            "cluster_assignments": self.cluster_assignments,
            "cluster_documents": self.cluster_docs_map,
            "cluster_info": self.cluster_info_storage,
            "cluster_history": self.cluster_history
        }
    
    async def get_documents_by_cluster(self, cluster_id: int) -> List[str]:
        """Lấy hash doc IDs thuộc cluster cụ thể"""
        if cluster_id not in self.cluster_docs_map:
            return []
        
        return self.cluster_docs_map[cluster_id]
    
    async def get_cluster_doc_ids(self, cluster_id: int) -> List[str]:
        """Lấy hash doc IDs của cluster"""
        if cluster_id == -1:
            # Lấy outlier doc hash IDs
            outlier_doc_hash_ids = []
            for doc_hash_id, assigned_cluster_id in self.cluster_assignments.items():
                if assigned_cluster_id == -1:
                    outlier_doc_hash_ids.append(doc_hash_id)
            return outlier_doc_hash_ids
        else:
            return self.cluster_docs_map.get(cluster_id, [])
    
    async def save_cluster_summary(self, cluster_id: int, summary: str, 
                                  doc_hash_ids: List[str]) -> None:
        """Lưu summary cho cluster (sẽ được gọi từ GraphRAGSystem)"""
        if cluster_id in self.cluster_info_storage:
            self.cluster_info_storage[cluster_id].summary = summary
            self.cluster_info_storage[cluster_id].updated_at = datetime.now().isoformat()
        else:
            cluster_info = ClusterInfo(
                cluster_id=cluster_id,
                doc_hash_ids=doc_hash_ids,
                outlier_doc_hash_ids=[],
                summary=summary,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
            self.cluster_info_storage[cluster_id] = cluster_info
        
        # Lưu ngay
        await self.save_clustering_model() 