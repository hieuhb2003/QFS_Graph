#!/usr/bin/env python3
"""
Demo GraphRAG với clustering integration
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphrag.clients.embedding_client import SentenceTransformersEmbeddingClient
from graphrag.clients.llm_client import OpenAIClient
from graphrag.core.graphrag_system import GraphRAGSystem


async def main():
    """Main demo function"""
    
    # Configuration
    working_dir = "graphrag_data"
    global_config = {
        "embedding_dimension": 768,
        "chunk_size": 1000,
        "max_workers": 4
    }
    
    # Initialize clients
    embedding_client = SentenceTransformersEmbeddingClient(
        model_name="BAAI/bge-m3",
        global_config=global_config
    )
    
    llm_client = OpenAIClient(
        model_name="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY"),
        global_config=global_config
    )
    
    # Clustering configuration
    clustering_config = {
        "update_threshold": 5,  # Update model when 5 outliers collected
        "min_cluster_size": 3,
        "min_samples": 2,
        "similarity_threshold": 0.6,
        "model_name": "dynamic_bertopic_model"
    }
    
    # Initialize GraphRAG system with clustering
    graphrag = GraphRAGSystem(
        working_dir=working_dir,
        embedding_client=embedding_client,
        global_config=global_config,
        llm_client=llm_client,
        enable_clustering=True,
        clustering_config=clustering_config
    )
    
    print("🚀 GraphRAG System with Clustering initialized!")
    
    # Sample documents for initial clustering training
    initial_docs = [
        "Python là một ngôn ngữ lập trình rất phổ biến hiện nay.",
        "Thư viện Pandas giúp xử lý dữ liệu dạng bảng hiệu quả.",
        "TensorFlow và PyTorch là hai framework học sâu hàng đầu.",
        "Lỗi tràn bộ nhớ thường xảy ra khi xử lý các tập dữ liệu lớn.",
        "Cấu trúc dữ liệu và giải thuật là nền tảng của khoa học máy tính.",
        "Trận đấu bóng đá giữa Việt Nam và Thái Lan diễn ra kịch tính.",
        "Nhiều cầu thủ đã thể hiện phong độ xuất sắc trong giải đấu này.",
        "Huấn luyện viên trưởng đã có những chỉ đạo chiến thuật hợp lý.",
        "Kết quả của môn điền kinh tại SEA Games thật bất ngờ.",
        "Lịch thi đấu của đội tuyển quốc gia đã được công bố."
    ]
    
    # Initialize clustering model
    print("\n📊 Initializing clustering model...")
    clustering_success = await graphrag.initialize_clustering(initial_docs)
    if clustering_success:
        print("✅ Clustering model initialized successfully!")
    else:
        print("❌ Failed to initialize clustering model")
    
    # Sample documents for processing (including outliers for clustering)
    documents = [
        # Technology documents (should cluster with existing tech docs)
        "Một lập trình viên giỏi cần nắm vững các thuật toán cốt lõi.",
        "Lỗi 'NoneType' object has no attribute '...' là lỗi phổ biến trong Python.",
        "Machine learning algorithms require careful hyperparameter tuning.",
        
        # Sports documents (should cluster with existing sports docs)
        "Bàn thắng ở phút cuối đã định đoạt trận đấu.",
        "Cầu thủ xuất sắc nhất trận đấu đã được bình chọn.",
        "Đội tuyển quốc gia chuẩn bị cho trận đấu quan trọng.",
        
        # Food documents (should be outliers, create new cluster)
        "Công thức làm món phở bò Hà Nội chuẩn vị.",
        "Cách chọn một trái sầu riêng ngon không phải ai cũng biết.",
        "Để làm bánh mì, bạn cần chuẩn bị bột và men nở.",
        "Review nhà hàng buffet lẩu nướng mới mở tại quận 1.",
        "Cách ướp sườn nướng BBQ ngon như ngoài hàng.",
        "Món bún chả là đặc sản không thể bỏ qua khi đến Hà Nội.",
        
        # More technology documents
        "Deep learning models require large amounts of training data.",
        "Version control with Git is essential for collaborative development.",
        
        # More sports documents
        "Championship finals will be held next month.",
        "Team strategy focuses on defensive play.",
        
        # More food documents (should trigger cluster update)
        "Traditional Vietnamese cuisine includes many rice-based dishes.",
        "Street food culture is vibrant in Southeast Asian cities."
    ]
    
    print(f"\n📄 Processing {len(documents)} documents with clustering...")
    
    # Process documents in batch with clustering
    results = await graphrag.insert_documents_batch_with_llm(
        documents=documents,
        max_concurrent_docs=3
    )
    
    # Analyze results
    success_count = sum(1 for r in results if r.get("status") == "success")
    print(f"\n✅ Successfully processed {success_count}/{len(documents)} documents")
    
    # Show clustering results
    print("\n📊 Clustering Results:")
    for result in results:
        if result.get("status") == "success":
            doc_id = result.get("doc_id", "unknown")
            cluster_result = result.get("cluster_result")
            if cluster_result:
                if cluster_result.is_outlier:
                    print(f"  📄 {doc_id[:20]}... -> OUTLIER (confidence: {cluster_result.confidence:.3f})")
                else:
                    print(f"  📄 {doc_id[:20]}... -> Cluster {cluster_result.cluster_id} (confidence: {cluster_result.confidence:.3f})")
    
    # Get clustering statistics
    print("\n📈 Clustering Statistics:")
    cluster_stats = await graphrag.get_clustering_statistics()
    if cluster_stats.get("clustering_enabled"):
        print(f"  Total documents processed: {cluster_stats.get('total_docs_processed', 0)}")
        print(f"  Outliers collected: {cluster_stats.get('outliers_collected', 0)}")
        print(f"  Model updates: {cluster_stats.get('model_updates', 0)}")
        print(f"  New clusters created: {cluster_stats.get('new_clusters_created', 0)}")
        print(f"  Total clusters: {cluster_stats.get('total_clusters', 0)}")
        print(f"  Outlier rate: {cluster_stats.get('outlier_rate', 0):.2%}")
        
        # Show cluster distribution
        distribution = cluster_stats.get("cluster_distribution", {})
        if distribution:
            print("  Cluster distribution:")
            for cluster_id, doc_count in distribution.items():
                print(f"    Cluster {cluster_id}: {doc_count} documents")
    
    # Demo clustering-aware queries
    print("\n🔍 Clustering-Aware Queries:")
    
    queries = [
        "Python programming",
        "Vietnamese food",
        "football match",
        "machine learning algorithms"
    ]
    
    for query in queries:
        print(f"\n  Query: '{query}'")
        result = await graphrag.query_with_clustering(query, top_k=5, use_clusters=True)
        
        if "error" not in result:
            entities = result.get("entities", [])
            relations = result.get("relations", [])
            clustering_info = result.get("clustering_info")
            
            print(f"    Found {len(entities)} entities, {len(relations)} relations")
            
            if clustering_info and "similar_clusters" in clustering_info:
                similar_clusters = clustering_info["similar_clusters"]
                print(f"    Similar clusters: {len(similar_clusters)}")
                for cluster in similar_clusters[:2]:  # Show top 2
                    print(f"      Cluster {cluster['cluster_id']}: similarity={cluster['similarity']:.3f}, keywords={cluster['keywords'][:3]}")
        else:
            print(f"    Error: {result['error']}")
    
    # Demo getting documents by cluster
    print("\n📋 Documents by Cluster:")
    all_clusters = await graphrag.clustering_manager.get_all_clusters()
    for cluster in all_clusters[:3]:  # Show first 3 clusters
        cluster_id = cluster.cluster_id
        docs = await graphrag.get_documents_by_cluster(cluster_id)
        print(f"  Cluster {cluster_id} ({cluster.keywords[:3]}): {len(docs)} documents")
        for doc in docs[:2]:  # Show first 2 docs
            print(f"    - {doc['doc_id'][:20]}... (confidence: {doc.get('cluster_confidence', 0):.3f})")
    
    # Cleanup
    await graphrag.cleanup()
    print("\n🧹 Cleanup completed!")


if __name__ == "__main__":
    asyncio.run(main()) 