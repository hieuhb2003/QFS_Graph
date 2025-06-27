"""
Demo Clustering Integration với GraphRAG System
Demo tích hợp clustering vào GraphRAG
"""

import asyncio
import os
from typing import List, Dict, Any

from src.graphrag.core.graphrag_system import GraphRAGSystem
from src.graphrag.clients.llm_client import create_llm_client
from src.graphrag.clients.embedding_client import create_embedding_client
from src.graphrag.utils.logger_config import setup_logger


async def demo_clustering_integration():
    """Demo tích hợp clustering với GraphRAG system"""
    
    # Setup logger
    logger = setup_logger(name="Demo-Clustering", log_level="INFO")
    logger.info("Bắt đầu demo clustering integration...")
    
    # Khởi tạo clients
    embedding_client = create_embedding_client(
        client_type="sentence_transformers",
        model_name="all-MiniLM-L6-v2",
        device="cpu"
    )
    
    llm_client = create_llm_client(
        client_type="openai",
        model_name="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Cấu hình clustering và summary
    global_config = {
        "save_interval": 100,
        "clustering": {
            "outlier_threshold": 5,
            "max_tokens": 4096,
            "batch_size": 8
        },
        "summary": {
            "max_workers": 2,
            "context_length": 2048
        }
    }
    
    # Khởi tạo GraphRAG system
    working_dir = "demo_clustering_data"
    system = GraphRAGSystem(
        working_dir=working_dir,
        embedding_client=embedding_client,
        global_config=global_config,
        llm_client=llm_client
    )
    
    try:
        # Dữ liệu mẫu - các documents về AI/ML
        documents = [
            "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models to enable computers to improve their performance on a specific task through experience. It involves training models on data to make predictions or decisions without being explicitly programmed for the task.",
            
            "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data. It has been particularly successful in computer vision, natural language processing, and speech recognition tasks.",
            
            "Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human language. It involves developing algorithms and models to understand, interpret, and generate human language in a meaningful way.",
            
            "Computer Vision is a field of artificial intelligence that enables computers to interpret and understand visual information from the world. It involves developing algorithms to process, analyze, and extract meaningful information from images and videos.",
            
            "Reinforcement Learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to achieve maximum cumulative reward. It is inspired by how humans and animals learn through trial and error.",
            
            "Data Science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data. It combines statistics, machine learning, and domain expertise.",
            
            "Big Data refers to extremely large datasets that may be analyzed computationally to reveal patterns, trends, and associations. It typically involves data that is too large, complex, or fast-changing to be processed by traditional data processing applications.",
            
            "Database Systems are software systems designed to store, retrieve, define, and manage data in a database. They provide efficient data storage, retrieval, and management capabilities for various applications.",
            
            "Web Development involves creating websites and web applications using various technologies and programming languages. It includes frontend development (user interface) and backend development (server-side logic and database management).",
            
            "Mobile Development is the process of creating software applications that run on mobile devices such as smartphones and tablets. It involves developing apps for iOS, Android, or cross-platform frameworks.",
            
            # Thêm một số outliers
            "Cooking is the art, science, and craft of using heat to prepare food for consumption. It involves various techniques such as baking, frying, grilling, and boiling to transform raw ingredients into delicious meals.",
            
            "Travel involves moving from one place to another for various purposes such as leisure, business, or exploration. It provides opportunities to experience different cultures, landscapes, and perspectives.",
            
            "Fashion refers to the styles and trends in clothing, accessories, and personal appearance that are popular at a particular time and place. It is influenced by culture, society, and individual preferences."
        ]
        
        logger.info(f"Inserting {len(documents)} documents...")
        
        # Insert documents với LLM extraction
        results = await system.insert_documents_batch_with_llm(documents, max_concurrent_docs=3)
        
        successful_inserts = sum(results)
        logger.info(f"Successfully inserted {successful_inserts}/{len(documents)} documents")
        
        # Lấy system stats
        stats = await system.get_system_stats()
        logger.info(f"System stats: {stats}")
        
        # Thực hiện clustering
        logger.info("Bắt đầu clustering documents...")
        clustering_result = await system.cluster_documents(outlier_threshold=5)
        
        if "error" not in clustering_result:
            logger.info(f"Clustering completed successfully:")
            logger.info(f"  - Total documents: {clustering_result.get('total_documents', 0)}")
            logger.info(f"  - Clustered documents: {clustering_result.get('clustered_documents', 0)}")
            logger.info(f"  - Outlier documents: {clustering_result.get('outlier_documents', 0)}")
            logger.info(f"  - Number of clusters: {clustering_result.get('n_clusters', 0)}")
            
            # Lấy cluster info
            cluster_info = await system.get_cluster_info()
            logger.info(f"Cluster info: {cluster_info}")
            
            # Tạo cluster summaries
            logger.info("Tạo cluster summaries...")
            summaries = await system.generate_cluster_summaries(max_workers=2)
            
            if "error" not in summaries:
                logger.info(f"Generated summaries for {len(summaries)} clusters:")
                for cluster_id, summary in summaries.items():
                    logger.info(f"  Cluster {cluster_id}: {summary[:100]}...")
                
                # Query cluster summaries
                logger.info("Querying cluster summaries...")
                query_results = await system.query_cluster_summaries("machine learning and artificial intelligence", top_k=3)
                
                logger.info(f"Query results: {len(query_results)} matches")
                for i, result in enumerate(query_results):
                    logger.info(f"  {i+1}. Cluster {result.get('cluster_id')}: {result.get('summary', '')[:100]}...")
            
            # Test với documents mới
            logger.info("Testing incremental clustering with new documents...")
            new_documents = [
                "Neural Networks are computational models inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information and can learn complex patterns through training.",
                
                "Convolutional Neural Networks (CNNs) are a specialized type of neural network designed for processing grid-like data such as images. They use convolutional layers to automatically learn spatial hierarchies of features.",
                
                "Transformers are a type of neural network architecture that has revolutionized natural language processing. They use self-attention mechanisms to process sequences of data and have achieved state-of-the-art results in many NLP tasks."
            ]
            
            # Insert documents mới
            new_results = await system.insert_documents_batch_with_llm(new_documents, max_concurrent_docs=2)
            logger.info(f"Inserted {sum(new_results)}/{len(new_documents)} new documents")
            
            # Update clusters với data mới
            update_result = await system.update_clusters_with_new_data(new_documents)
            logger.info(f"Updated clusters: {update_result}")
            
            # Lấy documents cùng cluster
            if successful_inserts > 0:
                # Lấy doc_id đầu tiên
                all_docs = await system.doc_status_db.get_all()
                first_doc_id = None
                for doc_id, doc_info in all_docs.items():
                    if doc_info.get("status") == "success":
                        first_doc_id = doc_id
                        break
                
                if first_doc_id:
                    same_cluster_docs = await system.get_documents_with_same_cluster(first_doc_id)
                    logger.info(f"Documents in same cluster as {first_doc_id}: {len(same_cluster_docs)} documents")
        
        else:
            logger.error(f"Clustering failed: {clustering_result['error']}")
        
        # Cleanup
        await system.cleanup()
        logger.info("Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in demo: {e}")
        await system.cleanup()


async def demo_clustering_only():
    """Demo chỉ clustering mà không cần LLM"""
    
    logger = setup_logger(name="Demo-Clustering-Only", log_level="INFO")
    logger.info("Bắt đầu demo clustering only...")
    
    # Khởi tạo clients (chỉ cần embedding)
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
    
    # Khởi tạo system (không có LLM)
    working_dir = "demo_clustering_only_data"
    system = GraphRAGSystem(
        working_dir=working_dir,
        embedding_client=embedding_client,
        global_config=global_config,
        llm_client=None  # Không cần LLM
    )
    
    try:
        # Documents đơn giản
        documents = [
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
            "Document về cooking recipes",
            "Document về travel destinations",
            "Document về fashion trends"
        ]
        
        # Insert documents (không cần LLM extraction)
        results = await system.insert_documents_batch(documents, chunk_size=500, max_concurrent_docs=3)
        logger.info(f"Inserted {sum(results)}/{len(documents)} documents")
        
        # Clustering
        clustering_result = await system.cluster_documents(outlier_threshold=3)
        logger.info(f"Clustering result: {clustering_result}")
        
        # Cleanup
        await system.cleanup()
        logger.info("Demo clustering only completed!")
        
    except Exception as e:
        logger.error(f"Error in demo clustering only: {e}")
        await system.cleanup()


if __name__ == "__main__":
    # Chạy demo chính
    asyncio.run(demo_clustering_integration())
    
    # Chạy demo clustering only
    # asyncio.run(demo_clustering_only()) 