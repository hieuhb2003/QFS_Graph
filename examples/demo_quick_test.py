"""
Demo Quick Test - GraphRAG System
Demo nhanh để test tất cả các chức năng cơ bản
"""

import asyncio
import os
from src.graphrag.core.graphrag_system import GraphRAGSystem
from src.graphrag.clients.llm_client import create_llm_client
from src.graphrag.clients.embedding_client import create_embedding_client
from src.graphrag.utils.logger_config import setup_logger


async def quick_test():
    """Test nhanh tất cả các chức năng"""
    
    logger = setup_logger(name="Quick-Test", log_level="INFO")
    logger.info("🚀 Bắt đầu Quick Test...")
    
    # Khởi tạo clients
    embedding_client = create_embedding_client(
        client_type="sentence_transformers",
        model_name="all-MiniLM-L6-v2",
        device="cpu"
    )
    
    # LLM client (optional)
    llm_client = None
    if os.getenv("OPENAI_API_KEY"):
        llm_client = create_llm_client(
            client_type="openai",
            model_name="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        logger.info("✅ Có LLM client")
    else:
        logger.info("⚠️ Không có LLM client")
    
    # Khởi tạo system
    system = GraphRAGSystem(
        working_dir="quick_test_data",
        embedding_client=embedding_client,
        global_config={"save_interval": 10},
        llm_client=llm_client
    )
    
    try:
        # Test documents
        documents = [
            "Apple is a technology company founded by Steve Jobs.",
            "Microsoft develops software and operating systems.",
            "Google specializes in search and internet services.",
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing deals with human language.",
            "Computer vision processes visual information.",
            "Tesla makes electric vehicles and clean energy products.",
            "Amazon is an e-commerce and cloud computing company.",
            "Cooking involves preparing food with heat.",
            "Travel means moving from one place to another."
        ]
        
        logger.info(f"📝 Processing {len(documents)} documents...")
        
        # Insert documents
        if llm_client:
            results = await system.insert_documents_batch_with_llm(documents, max_concurrent_docs=2)
        else:
            results = await system.insert_documents_batch(documents, chunk_size=300, max_concurrent_docs=2)
        
        successful = sum(results)
        logger.info(f"✅ Inserted {successful}/{len(documents)} documents")
        
        # Get stats
        stats = await system.get_system_stats()
        logger.info(f"📊 Stats: {stats}")
        
        # Query entities
        entities = await system.query_entities("technology", top_k=3)
        logger.info(f"🔍 Found {len(entities)} entities for 'technology'")
        
        # Clustering
        logger.info("🎯 Running clustering...")
        clustering_result = await system.cluster_documents(outlier_threshold=3)
        logger.info(f"✅ Clustering: {clustering_result}")
        
        # Get cluster info
        cluster_info = await system.get_cluster_info()
        logger.info(f"📋 Clusters: {cluster_info}")
        
        # Generate summaries (if LLM available)
        if llm_client:
            summaries = await system.generate_cluster_summaries(max_workers=2)
            if "error" not in summaries:
                logger.info(f"📝 Generated {len(summaries)} summaries")
            else:
                logger.error(f"❌ Summary error: {summaries['error']}")
        
        # Query summaries
        if llm_client:
            summary_results = await system.query_cluster_summaries("technology", top_k=2)
            logger.info(f"🔍 Found {len(summary_results)} summary matches")
        
        # Get documents by cluster
        for cluster_id in range(clustering_result.get('n_clusters', 0)):
            docs = await system.get_documents_by_cluster(cluster_id)
            logger.info(f"📦 Cluster {cluster_id}: {len(docs)} documents")
        
        # Cleanup
        await system.cleanup()
        logger.info("✅ Quick test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        await system.cleanup()


if __name__ == "__main__":
    asyncio.run(quick_test()) 