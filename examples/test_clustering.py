"""
Test Clustering Functionality
Test đơn giản cho clustering system
"""

import asyncio
import os
from typing import List

from src.graphrag.core.graphrag_system import GraphRAGSystem
from src.graphrag.clients.embedding_client import create_embedding_client
from src.graphrag.utils.logger_config import setup_logger


async def test_clustering():
    """Test clustering functionality"""
    
    # Setup logger
    logger = setup_logger(name="Test-Clustering", log_level="INFO")
    logger.info("Bắt đầu test clustering...")
    
    # Khởi tạo embedding client
    embedding_client = create_embedding_client(
        client_type="sentence_transformers",
        model_name="all-MiniLM-L6-v2",
        device="cpu"
    )
    
    # Cấu hình
    global_config = {
        "save_interval": 100,
        "clustering": {
            "outlier_threshold": 3,
            "max_tokens": 2048,
            "batch_size": 4
        }
    }
    
    # Khởi tạo system (không cần LLM)
    working_dir = "test_clustering_data"
    system = GraphRAGSystem(
        working_dir=working_dir,
        embedding_client=embedding_client,
        global_config=global_config,
        llm_client=None
    )
    
    try:
        # Documents đơn giản để test
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing deals with human language.",
            "Computer vision processes visual information.",
            "Reinforcement learning learns through trial and error.",
            "Data science combines statistics and machine learning.",
            "Big data refers to large datasets for analysis.",
            "Database systems store and manage data.",
            "Web development creates websites and applications.",
            "Mobile development builds apps for smartphones.",
            "Cooking involves preparing food with heat.",
            "Travel means moving from one place to another.",
            "Fashion refers to clothing styles and trends."
        ]
        
        logger.info(f"Inserting {len(documents)} test documents...")
        
        # Insert documents
        results = await system.insert_documents_batch(documents, chunk_size=500, max_concurrent_docs=3)
        successful_inserts = sum(results)
        logger.info(f"Successfully inserted {successful_inserts}/{len(documents)} documents")
        
        if successful_inserts == 0:
            logger.error("No documents inserted, cannot test clustering")
            return
        
        # Test clustering
        logger.info("Testing clustering...")
        clustering_result = await system.cluster_documents(outlier_threshold=3)
        
        if "error" in clustering_result:
            logger.error(f"Clustering failed: {clustering_result['error']}")
            return
        
        logger.info(f"Clustering successful:")
        logger.info(f"  - Total documents: {clustering_result.get('total_documents', 0)}")
        logger.info(f"  - Clustered documents: {clustering_result.get('clustered_documents', 0)}")
        logger.info(f"  - Outlier documents: {clustering_result.get('outlier_documents', 0)}")
        logger.info(f"  - Number of clusters: {clustering_result.get('n_clusters', 0)}")
        
        # Test cluster info
        cluster_info = await system.get_cluster_info()
        logger.info(f"Cluster info: {cluster_info}")
        
        # Test get documents by cluster
        for cluster_id in range(clustering_result.get('n_clusters', 0)):
            docs = await system.get_documents_by_cluster(cluster_id)
            logger.info(f"Cluster {cluster_id}: {len(docs)} documents")
        
        # Test with new documents
        logger.info("Testing incremental clustering...")
        new_documents = [
            "Neural networks are computational models.",
            "Convolutional networks process grid-like data.",
            "Transformers use attention mechanisms."
        ]
        
        # Insert new documents
        new_results = await system.insert_documents_batch(new_documents, chunk_size=500, max_concurrent_docs=2)
        logger.info(f"Inserted {sum(new_results)}/{len(new_documents)} new documents")
        
        # Update clusters
        update_result = await system.update_clusters_with_new_data(new_documents)
        logger.info(f"Updated clusters: {update_result}")
        
        # Test get documents with same cluster
        all_docs = await system.doc_status_db.get_all()
        first_doc_id = None
        for doc_id, doc_info in all_docs.items():
            if doc_info.get("status") == "success":
                first_doc_id = doc_id
                break
        
        if first_doc_id:
            same_cluster_docs = await system.get_documents_with_same_cluster(first_doc_id)
            logger.info(f"Documents in same cluster as {first_doc_id}: {len(same_cluster_docs)}")
        
        logger.info("All clustering tests passed!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        await system.cleanup()
        logger.info("Test completed")


if __name__ == "__main__":
    asyncio.run(test_clustering()) 