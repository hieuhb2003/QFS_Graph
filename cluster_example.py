# -*- coding: utf-8 -*-
"""
File: grid_search_bertopic.py
Mô tả: Tự động hóa việc tìm kiếm tham số tốt nhất cho BERTopic bằng Grid Search.
       Script sẽ chạy nhiều tổ hợp tham số UMAP/HDBSCAN, đánh giá từng cái,
       và báo cáo kết quả tổng hợp cùng với bộ tham số tốt nhất.
"""

import json
import os
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import torch
# from umap import UMAP
from umap.umap_ import UMAP

from hdbscan import HDBSCAN
from sklearn import metrics
from sklearn.model_selection import ParameterGrid # <<< Thêm import này

def load_and_preprocess_data(data_path, model_name, max_tokens):
    """Tải và xử lý dữ liệu từ file."""
    print(f"\n[INFO] Đang tải và xử lý dữ liệu từ: {data_path}...")
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
    except Exception as e:
        print(f"[LỖI] Đọc file thất bại: {e}")
        return None
    
    docs = []
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for item in all_data:
        if isinstance(item, str):
            sub_docs = item.split("|||||")
            for sub_doc in sub_docs:
                if not sub_doc.strip(): continue
                tokens = tokenizer.encode(sub_doc.strip(), add_special_tokens=False)
                if len(tokens) > max_tokens: tokens = tokens[:max_tokens]
                docs.append(tokenizer.decode(tokens))
    print(f"[INFO] Tổng số văn bản sau khi xử lý: {len(docs)}")
    return docs

def create_embeddings(docs, model_name, batch_size):
    """Tạo embedding song song trên nhiều GPU."""
    print(f"\n[INFO] Đang tải mô hình embedding: {model_name}...")
    embedding_model = SentenceTransformer(model_name, device='cpu')
    
    gpu_count = torch.cuda.device_count()
    print(f"\n[INFO] Bắt đầu tạo embeddings song song trên {gpu_count} GPU...")
    pool = embedding_model.start_multi_process_pool()
    embeddings = embedding_model.encode_multi_process(docs, pool=pool, batch_size=batch_size)
    embedding_model.stop_multi_process_pool(pool)
    print(f"[SUCCESS] Đã tạo xong embeddings với shape: {embeddings.shape}")
    return embeddings

def evaluate_run(true_labels, predicted_labels):
    """Hàm đánh giá và TRẢ VỀ một dictionary chứa các điểm số."""
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    non_outlier_mask = predicted_labels != -1
    
    # Nếu tất cả là outlier, trả về điểm số tệ nhất
    if np.sum(non_outlier_mask) == 0:
        return {"ARI": -1, "AMI": -1, "Homogeneity": -1, "Completeness": -1, "V-measure": -1, "Outlier %": 100.0}

    filtered_true = true_labels[non_outlier_mask]
    filtered_pred = predicted_labels[non_outlier_mask]
    
    scores = {
        "ARI": metrics.adjusted_rand_score(filtered_true, filtered_pred),
        "AMI": metrics.adjusted_mutual_info_score(filtered_true, filtered_pred),
        "Homogeneity": metrics.homogeneity_score(filtered_true, filtered_pred),
        "Completeness": metrics.completeness_score(filtered_true, filtered_pred),
        "V-measure": metrics.v_measure_score(filtered_true, filtered_pred),
        "Outlier %": 100 * (1 - (np.sum(non_outlier_mask) / len(predicted_labels)))
    }
    return scores

def main():
    # =============================================================================
    # BƯỚC 1: CÁC THAM SỐ CẤU HÌNH VÀ LƯỚI TÌM KIẾM
    # =============================================================================
    # --- Cấu hình tĩnh ---
    DATA_FILE_PATH = "/home/hungpv/projects/next_work/raw_data.json"
    GROUND_TRUTH_PATH = "/home/hungpv/projects/next_work/grouth_truth_cluster.json"
    EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
    MAX_TOKENS_PER_DOC = 8192
    BATCH_SIZE = 16
    OUTPUT_DIR = "results_grid_search"
    PRIMARY_METRIC = "AMI" # Chỉ số dùng để chọn model tốt nhất

    # --- ĐỊNH NGHĨA LƯỚI THAM SỐ ĐỂ TÌM KIẾM ---
# --- ĐỊNH NGHĨA LƯỚI THAM SỐ ĐÃ ĐƯỢC TINH CHỈNH ---
    param_grid = {
        # Ưu tiên các giá trị thấp để tập trung vào cấu trúc local
        'UMAP_N_NEIGHBORS': [5, 10, 15], 
        
        # Giữ nguyên để khám phá
        'UMAP_N_COMPONENTS': [5, 10], 
        
        # Neo quanh con số bạn biết (5-10 docs)
        'HDBSCAN_MIN_CLUSTER_SIZE': [5, 8], 
        
        # Chỉ dùng các giá trị rất thấp để tối đa độ nhạy
        'HDBSCAN_MIN_SAMPLES': [1, 2, 3] 
    }
    
    print("--- BẮT ĐẦU QUY TRÌNH GRID SEARCH CHO BERTOPIC ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =============================================================================
    # BƯỚC 2: CHUẨN BỊ DỮ LIỆU (CHẠY 1 LẦN)
    # =============================================================================
    docs = load_and_preprocess_data(DATA_FILE_PATH, EMBEDDING_MODEL_NAME, MAX_TOKENS_PER_DOC)
    embeddings = create_embeddings(docs, EMBEDDING_MODEL_NAME, BATCH_SIZE)
    
    # Tải và xử lý ground truth để đánh giá
    with open(GROUND_TRUTH_PATH, 'r', encoding='utf-8') as f: true_clusters_list = json.load(f)
    doc_to_true_label = {}
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    for i, cluster_docs in enumerate(true_clusters_list):
        for doc in cluster_docs:
            tokens = tokenizer.encode(doc.strip(), add_special_tokens=False)
            if len(tokens) > MAX_TOKENS_PER_DOC: tokens = tokens[:MAX_TOKENS_PER_DOC]
            doc_to_true_label[tokenizer.decode(tokens).strip()] = f"True_Cluster_{i}"
    true_labels = [doc_to_true_label.get(doc.strip(), "N/A_Unlabeled") for doc in docs]

    # =============================================================================
    # BƯỚC 3: VÒNG LẶP GRID SEARCH
    # =============================================================================
    grid = ParameterGrid(param_grid)
    results = []
    best_score = -1
    best_params = None

    print(f"\n[INFO] Bắt đầu tìm kiếm trên {len(grid)} tổ hợp tham số...")

    for i, params in enumerate(grid):
        print(f"\n--- Thử nghiệm {i+1}/{len(grid)}: {params} ---")
        
        try:
            # Khởi tạo các thành phần với tham số hiện tại
            umap_model = UMAP(n_neighbors=params['UMAP_N_NEIGHBORS'], n_components=params['UMAP_N_COMPONENTS'], min_dist=0.0, metric='cosine', random_state=42)
            hdbscan_model = HDBSCAN(min_cluster_size=params['HDBSCAN_MIN_CLUSTER_SIZE'], min_samples=params['HDBSCAN_MIN_SAMPLES'], metric='euclidean', cluster_selection_method='eom', prediction_data=True)
            
            # Khởi tạo và huấn luyện BERTopic
            topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, verbose=False)
            topics, _ = topic_model.fit_transform(docs, embeddings)

            # Đánh giá kết quả
            scores = evaluate_run(true_labels, topics)
            
            # Lưu kết quả
            current_run = {**params, **scores}
            results.append(current_run)
            
            # Cập nhật kết quả tốt nhất
            if scores[PRIMARY_METRIC] > best_score:
                best_score = scores[PRIMARY_METRIC]
                best_params = params
                print(f"🔥🔥🔥 New best score found! {PRIMARY_METRIC}: {best_score:.4f}")

        except Exception as e:
            print(f"[LỖI] Gặp lỗi với tham số {params}. Lỗi: {e}")
            results.append({**params, "ARI": "Error", "AMI": "Error"})

    # =============================================================================
    # BƯỚC 4: BÁO CÁO KẾT QUẢ
    # =============================================================================
    print("\n\n--- HOÀN TẤT GRID SEARCH ---")
    
    # Chuyển kết quả sang DataFrame để dễ xem
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=PRIMARY_METRIC, ascending=False)
    
    # Lưu bảng tổng hợp kết quả
    results_df.to_csv(os.path.join(OUTPUT_DIR, "grid_search_summary.csv"), index=False, encoding='utf-8-sig')
    
    print(f"\nBảng tổng hợp kết quả đã được lưu tại: {os.path.join(OUTPUT_DIR, 'grid_search_summary.csv')}")
    
    print("\n--- Top 5 cấu hình tốt nhất ---")
    print(results_df.head(5))
    
    print("\n--- Cấu hình tốt nhất được tìm thấy ---")
    print(f"Điểm {PRIMARY_METRIC} cao nhất: {best_score:.4f}")
    print("Với các tham số:")
    print(best_params)

if __name__ == "__main__":
    main()