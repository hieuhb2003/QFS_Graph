# -*- coding: utf-8 -*-
"""
Dynamic Clustering System with Outlier Management
Hệ thống phân cụm động với quản lý outlier
"""

import json
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import torch
from umap.umap_ import UMAP
from hdbscan import HDBSCAN
from sklearn import metrics

class DynamicClusteringSystem:
    def __init__(self, 
                 embedding_model_name: str = "BAAI/bge-m3",
                 max_tokens: int = 8192,
                 batch_size: int = 16,
                 outlier_threshold: int = 10,
                 model_save_path: str = "clustering_model"):
        """
        Khởi tạo hệ thống clustering động
        
        Args:
            embedding_model_name: Tên mô hình embedding
            max_tokens: Số token tối đa cho mỗi document
            batch_size: Batch size cho việc tạo embedding
            outlier_threshold: Ngưỡng số lượng outlier để tạo cluster mới
            model_save_path: Đường dẫn lưu model
        """
        self.embedding_model_name = embedding_model_name
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.outlier_threshold = outlier_threshold
        self.model_save_path = model_save_path
        
        # Best parameters từ grid search
        self.best_params = {
            'HDBSCAN_MIN_CLUSTER_SIZE': 5,
            'HDBSCAN_MIN_SAMPLES': 2,
            'UMAP_N_COMPONENTS': 5,
            'UMAP_N_NEIGHBORS': 5
        }
        
        # Khởi tạo các thành phần
        self.embedding_model = None
        self.tokenizer = None
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
        self.cluster_assignments = {}  # {doc_id: cluster_id}
        self.cluster_documents = {}    # {cluster_id: [doc_ids]}
        self.document_ids = []         # [doc_id1, doc_id2, ...]
        
        # Tạo thư mục lưu model
        os.makedirs(model_save_path, exist_ok=True)
        
    def _initialize_models(self):
        """Khởi tạo các mô hình embedding và tokenizer"""
        print(f"[INFO] Đang tải mô hình embedding: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name, device='cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
        
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
                    tokens = self.tokenizer.encode(sub_doc.strip(), add_special_tokens=False)
                    if len(tokens) > self.max_tokens:
                        tokens = tokens[:self.max_tokens]
                    processed_docs.append(self.tokenizer.decode(tokens))
        return processed_docs
    
    def _create_embeddings(self, documents: List[str]) -> np.ndarray:
        """Tạo embeddings cho documents"""
        print(f"[INFO] Đang tạo embeddings cho {len(documents)} documents...")
        
        gpu_count = torch.cuda.device_count()
        if gpu_count > 0:
            print(f"[INFO] Sử dụng {gpu_count} GPU để tạo embeddings...")
            pool = self.embedding_model.start_multi_process_pool()
            embeddings = self.embedding_model.encode_multi_process(
                documents, pool=pool, batch_size=self.batch_size
            )
            self.embedding_model.stop_multi_process_pool(pool)
        else:
            print("[INFO] Sử dụng CPU để tạo embeddings...")
            embeddings = self.embedding_model.encode(documents, batch_size=self.batch_size)
        
        print(f"[SUCCESS] Đã tạo embeddings với shape: {embeddings.shape}")
        return embeddings
    
    def _update_cluster_assignments(self, documents: List[str], topics: List[int], start_idx: int = 0):
        """Cập nhật thông tin chi tiết về cluster assignments"""
        for i, (doc, topic) in enumerate(zip(documents, topics)):
            doc_id = start_idx + i
            
            # Lưu assignment
            self.cluster_assignments[doc_id] = topic
            
            # Lưu vào cluster documents
            if topic not in self.cluster_documents:
                self.cluster_documents[topic] = []
            self.cluster_documents[topic].append(doc_id)
    
    def initial_fit(self, documents: List[str]):
        """
        Fit mô hình ban đầu với dữ liệu đầu tiên
        
        Args:
            documents: Danh sách documents để fit ban đầu
        """
        print("\n=== BẮT ĐẦU FIT MÔ HÌNH BAN ĐẦU ===")
        
        # Khởi tạo models
        self._initialize_models()
        
        # Tiền xử lý documents
        processed_docs = self._preprocess_documents(documents)
        self.all_docs = processed_docs
        
        # Tạo embeddings
        embeddings = self._create_embeddings(processed_docs)
        self.all_embeddings = embeddings
        
        # Fit BERTopic
        print("[INFO] Đang fit BERTopic model...")
        topics, probs = self.topic_model.fit_transform(processed_docs, embeddings)
        
        # Cập nhật cluster assignments
        self.document_ids = list(range(len(processed_docs)))
        self._update_cluster_assignments(processed_docs, topics)
        
        # Phân loại documents thành clusters và outliers
        cluster_docs = []
        outlier_docs = []
        
        for i, topic in enumerate(topics):
            if topic == -1:  # Outlier
                outlier_docs.append(processed_docs[i])
            else:
                cluster_docs.append(processed_docs[i])
        
        # Lưu outliers
        if outlier_docs:
            self.outlier_docs = outlier_docs
            outlier_indices = [i for i, topic in enumerate(topics) if topic == -1]
            self.outlier_embeddings = embeddings[outlier_indices]
            print(f"[INFO] Phát hiện {len(outlier_docs)} outliers")
        
        # Lưu lịch sử clustering
        self.cluster_history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'initial_fit',
            'total_docs': len(processed_docs),
            'clustered_docs': len(cluster_docs),
            'outlier_docs': len(outlier_docs),
            'n_clusters': len(set(topics)) - (1 if -1 in topics else 0),
            'cluster_assignments': self.cluster_assignments.copy(),
            'cluster_documents': {k: v.copy() for k, v in self.cluster_documents.items()}
        })
        
        # Lưu model
        self.save_model()
        
        print(f"[SUCCESS] Hoàn thành fit ban đầu:")
        print(f"  - Tổng documents: {len(processed_docs)}")
        print(f"  - Documents được cluster: {len(cluster_docs)}")
        print(f"  - Outliers: {len(outlier_docs)}")
        print(f"  - Số clusters: {len(set(topics)) - (1 if -1 in topics else 0)}")
        
        return topics, probs
    
    def predict_new_documents(self, new_documents: List[str]) -> Tuple[List[int], List[str]]:
        """
        Dự đoán cluster cho documents mới
        
        Args:
            new_documents: Danh sách documents mới
            
        Returns:
            Tuple (predicted_topics, outlier_docs)
        """
        print(f"\n=== DỰ ĐOÁN CHO {len(new_documents)} DOCUMENTS MỚI ===")
        
        if self.topic_model is None:
            raise ValueError("Model chưa được fit. Hãy gọi initial_fit() trước.")
        
        # Tiền xử lý documents mới
        processed_new_docs = self._preprocess_documents(new_documents)
        
        # Tạo embeddings cho documents mới
        new_embeddings = self._create_embeddings(processed_new_docs)
        
        # Dự đoán topics
        predicted_topics, _ = self.topic_model.transform(processed_new_docs, new_embeddings)
        
        # Phân loại kết quả
        clustered_new_docs = []
        outlier_new_docs = []
        
        for i, topic in enumerate(predicted_topics):
            if topic == -1:  # Outlier
                outlier_new_docs.append(processed_new_docs[i])
            else:
                clustered_new_docs.append(processed_new_docs[i])
        
        print(f"[INFO] Kết quả dự đoán:")
        print(f"  - Documents được cluster: {len(clustered_new_docs)}")
        print(f"  - Outliers mới: {len(outlier_new_docs)}")
        
        return predicted_topics, outlier_new_docs
    
    def update_with_new_data(self, new_documents: List[str]):
        """
        Cập nhật mô hình với dữ liệu mới
        
        Args:
            new_documents: Danh sách documents mới
        """
        print(f"\n=== CẬP NHẬT MÔ HÌNH VỚI {len(new_documents)} DOCUMENTS MỚI ===")
        
        # Dự đoán documents mới
        predicted_topics, outlier_new_docs = self.predict_new_documents(new_documents)
        
        # Thêm outliers mới vào tập outlier hiện tại
        if outlier_new_docs:
            self.outlier_docs.extend(outlier_new_docs)
            
            # Tạo embeddings cho outliers mới
            outlier_new_embeddings = self._create_embeddings(outlier_new_docs)
            
            # Cập nhật outlier embeddings
            if self.outlier_embeddings is not None:
                self.outlier_embeddings = np.vstack([self.outlier_embeddings, outlier_new_embeddings])
            else:
                self.outlier_embeddings = outlier_new_embeddings
        
        # Kiểm tra xem có cần tạo cluster mới cho outliers không
        if len(self.outlier_docs) >= self.outlier_threshold:
            print(f"[INFO] Số lượng outliers ({len(self.outlier_docs)}) đã đạt ngưỡng ({self.outlier_threshold})")
            self._create_new_clusters_from_outliers()
        
        # Cập nhật lịch sử
        self.cluster_history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'update_with_new_data',
            'new_docs': len(new_documents),
            'clustered_new_docs': len(new_documents) - len(outlier_new_docs),
            'new_outliers': len(outlier_new_docs),
            'total_outliers': len(self.outlier_docs),
            'cluster_assignments': self.cluster_assignments.copy(),
            'cluster_documents': {k: v.copy() for k, v in self.cluster_documents.items()}
        })
        
        # Lưu model
        self.save_model()
    
    def _create_new_clusters_from_outliers(self):
        """Tạo cluster mới từ tập outliers"""
        print("\n=== TẠO CLUSTER MỚI TỪ OUTLIERS ===")
        
        if len(self.outlier_docs) < self.outlier_threshold:
            print(f"[INFO] Chưa đủ outliers để tạo cluster mới ({len(self.outlier_docs)} < {self.outlier_threshold})")
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
        
        for i, topic in enumerate(outlier_topics):
            if topic == -1:  # Vẫn là outlier
                remaining_outliers.append(self.outlier_docs[i])
                remaining_outlier_embeddings.append(self.outlier_embeddings[i])
            else:  # Được cluster
                new_clustered_outliers.append(self.outlier_docs[i])
                new_cluster_embeddings.append(self.outlier_embeddings[i])
        
        # Cập nhật dữ liệu
        self.outlier_docs = remaining_outliers
        self.outlier_embeddings = np.array(remaining_outlier_embeddings) if remaining_outlier_embeddings else None
        
        # Thêm documents được cluster vào tập chính
        if new_clustered_outliers:
            start_idx = len(self.all_docs)
            self.all_docs.extend(new_clustered_outliers)
            new_cluster_embeddings = np.array(new_cluster_embeddings)
            
            if self.all_embeddings is not None:
                self.all_embeddings = np.vstack([self.all_embeddings, new_cluster_embeddings])
            else:
                self.all_embeddings = new_cluster_embeddings
            
            # Cập nhật cluster assignments cho documents mới
            new_doc_ids = list(range(start_idx, start_idx + len(new_clustered_outliers)))
            self.document_ids.extend(new_doc_ids)
            self._update_cluster_assignments(new_clustered_outliers, outlier_topics, start_idx)
        
        print(f"[SUCCESS] Tạo cluster mới từ outliers:")
        print(f"  - Outliers được cluster: {len(new_clustered_outliers)}")
        print(f"  - Outliers còn lại: {len(remaining_outliers)}")
    
    def save_model(self):
        """Lưu model và dữ liệu"""
        print(f"[INFO] Đang lưu model tại: {self.model_save_path}")
        
        # Lưu BERTopic model
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
            'cluster_documents': self.cluster_documents,
            'document_ids': self.document_ids
        }
        
        with open(os.path.join(self.model_save_path, "clustering_data.pkl"), 'wb') as f:
            pickle.dump(data_to_save, f)
        
        # Lưu lịch sử dưới dạng JSON để dễ đọc
        with open(os.path.join(self.model_save_path, "cluster_history.json"), 'w', encoding='utf-8') as f:
            json.dump(self.cluster_history, f, ensure_ascii=False, indent=2)
        
        # Lưu thông tin chi tiết về clusters
        self._save_detailed_cluster_info()
        
        print("[SUCCESS] Đã lưu model thành công")
    
    def _save_detailed_cluster_info(self):
        """Lưu thông tin chi tiết về clusters"""
        cluster_details = {}
        
        for cluster_id, doc_ids in self.cluster_documents.items():
            cluster_details[cluster_id] = {
                'documents': [self.all_docs[doc_id] for doc_id in doc_ids],
                'document_ids': doc_ids,
                'count': len(doc_ids)
            }
        
        # Lưu outliers
        cluster_details[-1] = {
            'documents': self.outlier_docs,
            'document_ids': [],
            'count': len(self.outlier_docs)
        }
        
        with open(os.path.join(self.model_save_path, "detailed_cluster_info.json"), 'w', encoding='utf-8') as f:
            json.dump(cluster_details, f, ensure_ascii=False, indent=2)
    
    def load_model(self):
        """Tải model và dữ liệu đã lưu"""
        print(f"[INFO] Đang tải model từ: {self.model_save_path}")
        
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
            self.cluster_documents = data.get('cluster_documents', {})
            self.document_ids = data.get('document_ids', [])
        else:
            raise FileNotFoundError(f"Không tìm thấy clustering data tại {data_path}")
        
        # Khởi tạo lại embedding model và tokenizer
        self._initialize_models()
        
        print("[SUCCESS] Đã tải model thành công")
    
    def get_cluster_info(self) -> Dict:
        """Lấy thông tin về clusters hiện tại"""
        if self.topic_model is None:
            return {"error": "Model chưa được fit"}
        
        topics_info = self.topic_model.get_topic_info()
        
        return {
            "total_documents": len(self.all_docs),
            "total_clusters": len(topics_info) - 1,  # Trừ outlier cluster
            "outlier_documents": len(self.outlier_docs),
            "cluster_details": topics_info.to_dict('records'),
            "cluster_history": self.cluster_history,
            "cluster_assignments": self.cluster_assignments,
            "cluster_documents": self.cluster_documents
        }
    
    def get_documents_by_cluster(self, cluster_id: int) -> List[str]:
        """Lấy documents thuộc cluster cụ thể"""
        if cluster_id not in self.cluster_documents:
            return []
        
        doc_ids = self.cluster_documents[cluster_id]
        return [self.all_docs[doc_id] for doc_id in doc_ids]
    
    def get_detailed_cluster_report(self) -> Dict:
        """Lấy báo cáo chi tiết về tất cả clusters"""
        report = {
            'summary': {
                'total_documents': len(self.all_docs),
                'total_clusters': len([k for k in self.cluster_documents.keys() if k != -1]),
                'total_outliers': len(self.outlier_docs)
            },
            'clusters': {},
            'outliers': self.outlier_docs
        }
        
        for cluster_id, doc_ids in self.cluster_documents.items():
            if cluster_id == -1:  # Bỏ qua outlier cluster
                continue
            
            cluster_docs = [self.all_docs[doc_id] for doc_id in doc_ids]
            report['clusters'][cluster_id] = {
                'count': len(doc_ids),
                'documents': cluster_docs,
                'document_ids': doc_ids
            }
        
        return report
    
    def print_cluster_summary(self):
        """In tóm tắt về clusters"""
        print("\n=== TÓM TẮT CLUSTERS ===")
        print(f"Tổng số documents: {len(self.all_docs)}")
        print(f"Số clusters: {len([k for k in self.cluster_documents.keys() if k != -1])}")
        print(f"Số outliers: {len(self.outlier_docs)}")
        
        print("\nChi tiết từng cluster:")
        for cluster_id, doc_ids in sorted(self.cluster_documents.items()):
            if cluster_id == -1:
                print(f"  Outliers: {len(doc_ids)} documents")
            else:
                print(f"  Cluster {cluster_id}: {len(doc_ids)} documents")
                # In 2 documents đầu tiên làm ví dụ
                sample_docs = [self.all_docs[doc_id] for doc_id in doc_ids[:2]]
                for i, doc in enumerate(sample_docs):
                    print(f"    - Doc {i+1}: {doc[:100]}...")


def example_usage():
    """Ví dụ sử dụng hệ thống"""
    
    # Khởi tạo hệ thống
    clustering_system = DynamicClusteringSystem(
        embedding_model_name="BAAI/bge-m3",
        outlier_threshold=10
    )
    
    # Dữ liệu ban đầu
    initial_docs = [
        "Document về machine learning và AI",
        "Document về deep learning",
        "Document về natural language processing",
        "Document về computer vision",
        "Document về reinforcement learning",
        "Document về data science",
        "Document về big data",
        "Document về database systems",
        "Document về web development",
        "Document về mobile development",
        # Thêm một số outliers
        "Document về cooking recipes",
        "Document về travel destinations",
        "Document về fashion trends"
    ]
    
    # Fit ban đầu
    topics, probs = clustering_system.initial_fit(initial_docs)
    
    # In tóm tắt
    clustering_system.print_cluster_summary()
    
    # Dữ liệu mới sau một thời gian
    new_docs = [
        "Document về neural networks",
        "Document về convolutional networks",
        "Document về transformers",
        "Document về BERT model",
        "Document về GPT models",
        # Thêm outliers mới
        "Document về gardening tips",
        "Document về fitness exercises"
    ]
    
    # Cập nhật với dữ liệu mới
    clustering_system.update_with_new_data(new_docs)
    
    # In tóm tắt sau khi cập nhật
    clustering_system.print_cluster_summary()
    
    # Lấy báo cáo chi tiết
    detailed_report = clustering_system.get_detailed_cluster_report()
    print("\n=== BÁO CÁO CHI TIẾT ===")
    print(json.dumps(detailed_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    example_usage() 