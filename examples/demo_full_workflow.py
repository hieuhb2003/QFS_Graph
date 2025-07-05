"""
Demo Full Workflow - GraphRAG System
Demo đầy đủ cho tất cả các luồng từ entity extraction đến clustering
"""

import asyncio
import os
import json
from typing import List, Dict, Any
from datetime import datetime

from src.graphrag.core.graphrag_system import GraphRAGSystem
from src.graphrag.clients.llm_client import create_llm_client
from src.graphrag.clients.embedding_client import create_embedding_client
from src.graphrag.utils.logger_config import setup_logger
from src.graphrag.utils.utils import ClusterInfo


def serialize_cluster_info(obj):
    """Helper function to serialize ClusterInfo objects and other complex types"""
    if isinstance(obj, ClusterInfo):
        return {
            'cluster_id': obj.cluster_id,
            'doc_hash_ids': obj.doc_hash_ids,
            'outlier_doc_hash_ids': obj.outlier_doc_hash_ids,
            'summary': obj.summary,
            'created_at': obj.created_at,
            'updated_at': obj.updated_at
        }
    # Handle other non-serializable objects by converting to string
    elif hasattr(obj, '__dict__'):
        return str(obj)
    else:
        return str(obj)


def safe_json_dumps(data, indent=2):
    """Safely serialize data to JSON, handling complex objects"""
    try:
        return json.dumps(data, indent=indent, default=serialize_cluster_info, ensure_ascii=False)
    except Exception as e:
        # Fallback to string representation
        return str(data)


async def demo_full_workflow():
    """Demo đầy đủ cho tất cả các luồng của GraphRAG system"""
    
    # Setup logger
    logger = setup_logger(name="Demo-Full-Workflow", log_level="INFO")
    logger.info("🚀 Bắt đầu demo full workflow GraphRAG system...")
    
    # Khởi tạo clients
    logger.info("📡 Khởi tạo clients...")
    
    # Embedding client
    embedding_client = create_embedding_client(
        client_type="sentence_transformers",
        model_name="BAAI/bge-m3",
        device="cuda:0"
    )
    
    # LLM client (có thể dùng OpenAI hoặc vLLM)
    # llm_client = None
    # if os.getenv("OPENAI_API_KEY"):
    #     llm_client = create_llm_client(
    #         client_type="openai",
    #         model_name="gpt-3.5-turbo",
    #         api_key=os.getenv("OPENAI_API_KEY")
    #     )
    #     logger.info("✅ Sử dụng OpenAI LLM client")
    # else:
    #     logger.warning("⚠️ Không có OpenAI API key, sẽ bỏ qua LLM extraction")
    llm_client = create_llm_client(
        client_type="vllm",
        model_name="qwen2.5-7b-it-gpu0",
        url ="http://localhost:9100/v1",
        api_key='0'
    )
    logger.info("✅ Sử dụng vLLM LLM client")
    # Cấu hình hệ thống
    global_config = {
        
        "save_interval": 50,
        "working_dir": "/home/hungpv/projects/next_work/our_method_data1_graph",
        "clustering": {
            "outlier_threshold": 10,
            "max_tokens": 4096,
            "batch_size": 8
        },
        "summary": {
            "max_workers": 5,
            "context_length": 60000
        },       
          "embedding_batch_num": 32,
        "embedding_dimension": 1024,  # Dimension của all-MiniLM-L6-v2
        "vector_db_storage_cls_kwargs": {
            "cosine_better_than_threshold": 0.5
        }
    }
    
    # Khởi tạo GraphRAG system
    working_dir = "/home/hungpv/projects/next_work/our_method_data1_graph"
    system = GraphRAGSystem(
        working_dir=working_dir,
        embedding_client=embedding_client,
        global_config=global_config,
        llm_client=llm_client
    )
    
    try:
        # ========================================
        # PHASE 1: DOCUMENT PROCESSING
        # ========================================
        logger.info("\n" + "="*60)
        logger.info("📄 PHASE 1: DOCUMENT PROCESSING")
        logger.info("="*60)
        
        with open("/home/hungpv/projects/next_work/data/data_1/data1_to_index.json", "r") as f:
            documents = json.load(f)
        # Dữ liệu mẫu đa dạng
        # documents = [
        #     # AI/ML Documents
        #     "Apple Inc. is a technology company that designs and manufactures consumer electronics, computer software, and online services. Founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976, Apple has become one of the world's most valuable companies. The company is known for its innovative products like the iPhone, iPad, Mac, and Apple Watch.",
            
        #     "Microsoft Corporation is a multinational technology company that develops, manufactures, licenses, supports, and sells computer software, consumer electronics, personal computers, and related services. Founded by Bill Gates and Paul Allen in 1975, Microsoft is best known for its Windows operating system and Office productivity suite.",
            
        #     "Google LLC is a technology company that specializes in internet-related services and products, which include online advertising technologies, search engine, cloud computing, software, and hardware. Founded by Larry Page and Sergey Brin in 1998, Google has become synonymous with web search and digital innovation.",
            
        #     "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models to enable computers to improve their performance on a specific task through experience. It involves training models on data to make predictions or decisions without being explicitly programmed for the task.",
            
        #     "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data. It has been particularly successful in computer vision, natural language processing, and speech recognition tasks.",
            
        #     "Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human language. It involves developing algorithms and models to understand, interpret, and generate human language in a meaningful way.",
            
        #     "Computer Vision is a field of artificial intelligence that enables computers to interpret and understand visual information from the world. It involves developing algorithms to process, analyze, and extract meaningful information from images and videos.",
            
        #     "Reinforcement Learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to achieve maximum cumulative reward. It is inspired by how humans and animals learn through trial and error.",
            
        #     # Business Documents
        #     "Tesla Inc. is an electric vehicle and clean energy company founded by Elon Musk in 2003. The company designs, develops, manufactures, leases, and sells electric vehicles, energy generation and storage systems, and offers services related to its products.",
            
        #     "Amazon.com Inc. is an American multinational technology company that focuses on e-commerce, cloud computing, digital streaming, and artificial intelligence. Founded by Jeff Bezos in 1994, Amazon has grown to become one of the world's largest companies.",
            
        #     "Netflix Inc. is a streaming service that offers a wide variety of award-winning TV shows, movies, anime, documentaries, and more on thousands of internet-connected devices. Founded by Reed Hastings and Marc Randolph in 1997.",
            
        #     # Science Documents
        #     "Quantum computing is a type of computation that harnesses the collective properties of quantum states to perform calculations. It uses quantum mechanical phenomena such as superposition and entanglement to process information in ways that classical computers cannot.",
            
        #     "Climate change refers to long-term shifts in global or regional climate patterns. It is primarily caused by human activities, particularly the burning of fossil fuels, which increases heat-trapping greenhouse gas levels in Earth's atmosphere.",
            
        #     "Renewable energy is energy that is collected from renewable resources, which are naturally replenished on a human timescale, such as sunlight, wind, rain, tides, waves, and geothermal heat. It is considered environmentally friendly and sustainable.",
            
        #     # Outliers
        #     "Cooking is the art, science, and craft of using heat to prepare food for consumption. It involves various techniques such as baking, frying, grilling, and boiling to transform raw ingredients into delicious meals.",
            
        #     "Travel involves moving from one place to another for various purposes such as leisure, business, or exploration. It provides opportunities to experience different cultures, landscapes, and perspectives.",
            
        #     "Fashion refers to the styles and trends in clothing, accessories, and personal appearance that are popular at a particular time and place. It is influenced by culture, society, and individual preferences."
        # ]
        
        logger.info(f"📝 Chuẩn bị {len(documents)} documents để xử lý...")
        import time
        start_time = time.time()
        # Insert documents với LLM extraction (nếu có LLM)
        if llm_client:
            logger.info("🔍 Inserting documents với LLM extraction...")
            results = await system.insert_documents_batch_with_llm(documents, max_concurrent_docs=3)
        else:
            logger.info("📄 Inserting documents với chunking (không có LLM)...")
            results = await system.insert_documents_batch(documents, chunk_size=4096, max_concurrent_docs=3)
        
        successful_inserts = sum(results)
        logger.info(f"✅ Successfully inserted {successful_inserts}/{len(documents)} documents")
        
        # Lấy system stats
        stats = await system.get_system_stats()
        logger.info(f"📊 System stats: {safe_json_dumps(stats)}")
        
        # # ========================================
        # # PHASE 2: ENTITY & RELATION QUERYING
        # # ========================================
        # logger.info("\n" + "="*60)
        # logger.info("🔍 PHASE 2: ENTITY & RELATION QUERYING")
        # logger.info("="*60)
        
        # # Query entities
        # logger.info("🔍 Querying entities...")
        # entity_queries = ["technology company", "artificial intelligence", "machine learning"]
        # for query in entity_queries:
        #     entities = await system.query_entities(query, top_k=3)
        #     logger.info(f"  Query '{query}': {len(entities)} entities found")
        #     for i, entity in enumerate(entities):
        #         logger.info(f"    {i+1}. {entity.get('entity_name', 'N/A')}: {entity.get('description', 'N/A')[:100]}...")
        
        # # Query relations
        # logger.info("🔗 Querying relations...")
        # relation_queries = ["founded", "develops", "specializes in"]
        # for query in relation_queries:
        #     relations = await system.query_relations(query, top_k=3)
        #     logger.info(f"  Query '{query}': {len(relations)} relations found")
        #     for i, relation in enumerate(relations):
        #         logger.info(f"    {i+1}. {relation.get('source_entity', 'N/A')} -> {relation.get('relation_description', 'N/A')} -> {relation.get('target_entity', 'N/A')}")
        
        # # Search entities by name
        # logger.info("🔍 Searching entities by name...")
        # entity_names = ["Apple", "Microsoft", "Google", "Tesla"]
        # for name in entity_names:
        #     results = await system.search_entities_by_name(name)
        #     logger.info(f"  Search '{name}': {len(results)} entities found")
        #     for i, result in enumerate(results):
        #         logger.info(f"    {i+1}. {result.get('entity_name', 'N/A')}: {result.get('description', 'N/A')[:100]}...")
        
        # # Get entity graph context
        # logger.info("🕸️ Getting entity graph context...")
        # context_entities = ["Apple", "technology"]
        # for entity_name in context_entities:
        #     context = await system.get_entity_graph_context(entity_name, max_depth=2)
        #     logger.info(f"  Context for '{entity_name}': {len(context.get('neighbors', []))} neighbors")
        
        # ========================================
        # PHASE 3: CLUSTERING
        # ========================================
        logger.info("\n" + "="*60)
        logger.info("🎯 PHASE 3: CLUSTERING")
        logger.info("="*60)
        
        # Thực hiện clustering
        logger.info("🎯 Bắt đầu clustering documents...")
        clustering_result = await system.cluster_documents(outlier_threshold=5)
        
        if "error" in clustering_result:
            logger.error(f"❌ Clustering failed: {clustering_result['error']}")
            return
        
        logger.info(f"✅ Clustering completed successfully:")
        logger.info(f"  📊 Total documents: {clustering_result.get('total_documents', 0)}")
        logger.info(f"  🎯 Clustered documents: {clustering_result.get('clustered_documents', 0)}")
        logger.info(f"  🔸 Outlier documents: {clustering_result.get('outlier_documents', 0)}")
        logger.info(f"  📦 Number of clusters: {clustering_result.get('n_clusters', 0)}")
        
        # Lấy cluster info chi tiết
        cluster_info = await system.get_cluster_info()
        logger.info(f"📋 Cluster info: {safe_json_dumps(cluster_info)}")
        
        # Hiển thị documents theo cluster
        logger.info("📋 Documents theo cluster:")
        for cluster_id in range(clustering_result.get('n_clusters', 0)):
            docs = await system.get_documents_by_cluster(cluster_id)
            logger.info(f"  Cluster {cluster_id}: {len(docs)} documents")
            for i, doc_id in enumerate(docs[:3]):  # Chỉ hiển thị 3 docs đầu
                logger.info(f"    {i+1}. {doc_id}")
            if len(docs) > 3:
                logger.info(f"    ... và {len(docs) - 3} documents khác")
        
        # ========================================
        # PHASE 4: CLUSTER SUMMARIES
        # ========================================
        logger.info("\n" + "="*60)
        logger.info("📝 PHASE 4: CLUSTER SUMMARIES")
        logger.info("="*60)
        
        if llm_client:
            # Tạo cluster summaries
            logger.info("📝 Tạo cluster summaries...")
            summaries = await system.generate_cluster_summaries(max_workers=3)
            
            if "error" not in summaries:
                logger.info(f"✅ Generated summaries for {len(summaries)} clusters:")
                for cluster_id, summary in summaries.items():
                    logger.info(f"  📦 Cluster {cluster_id}: {summary[:150]}...")
                
                # Query cluster summaries
                logger.info("🔍 Querying cluster summaries...")
                summary_queries = ["technology companies", "artificial intelligence", "business"]
                for query in summary_queries:
                    query_results = await system.query_cluster_summaries(query, top_k=2)
                    logger.info(f"  Query '{query}': {len(query_results)} matches")
                    for i, result in enumerate(query_results):
                        logger.info(f"    {i+1}. Cluster {result.get('cluster_id')}: {result.get('summary', '')[:100]}...")
            else:
                logger.error(f"❌ Summary generation failed: {summaries['error']}")
        else:
            logger.warning("⚠️ Bỏ qua summary generation (không có LLM client)")
        end_time = time.time()
        
        # # ========================================
        # # PHASE 5: INCREMENTAL CLUSTERING
        # # ========================================
        # logger.info("\n" + "="*60)
        # logger.info("🔄 PHASE 5: INCREMENTAL CLUSTERING")
        # logger.info("="*60)
        
        # # Thêm documents mới
        # new_documents = [
        #     "Neural Networks are computational models inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information and can learn complex patterns through training.",
            
        #     "Convolutional Neural Networks (CNNs) are a specialized type of neural network designed for processing grid-like data such as images. They use convolutional layers to automatically learn spatial hierarchies of features.",
            
        #     "Transformers are a type of neural network architecture that has revolutionized natural language processing. They use self-attention mechanisms to process sequences of data and have achieved state-of-the-art results in many NLP tasks.",
            
        #     "Meta Platforms Inc. (formerly Facebook) is a technology company that develops products and services for connecting people. Founded by Mark Zuckerberg in 2004, the company owns Facebook, Instagram, WhatsApp, and other social media platforms.",
            
        #     "NVIDIA Corporation is a technology company that designs graphics processing units (GPUs) for gaming and professional markets, as well as system on a chip units (SoCs) for mobile computing and automotive markets."
        # ]
        
        # logger.info(f"📝 Thêm {len(new_documents)} documents mới...")
        
        # # Insert documents mới
        # if llm_client:
        #     new_results = await system.insert_documents_batch_with_llm(new_documents, max_concurrent_docs=2)
        # else:
        #     new_results = await system.insert_documents_batch(new_documents, chunk_size=500, max_concurrent_docs=2)
        
        # logger.info(f"✅ Inserted {sum(new_results)}/{len(new_documents)} new documents")
        
        # # Update clusters với data mới
        # logger.info("🔄 Updating clusters với documents mới...")
        # update_result = await system.update_clusters_with_new_data(new_documents)
        # logger.info(f"✅ Updated clusters: {safe_json_dumps(update_result)}")
        
        # # ========================================
        # # PHASE 6: GRAPH ANALYSIS
        # # ========================================
        # logger.info("\n" + "="*60)
        # logger.info("🕸️ PHASE 6: GRAPH ANALYSIS")
        # logger.info("="*60)
        
        # # Lấy documents cùng cluster
        # logger.info("🔗 Finding documents in same cluster...")
        # all_docs = await system.doc_status_db.get_all()
        # first_doc_id = None
        # for doc_id, doc_info in all_docs.items():
        #     if doc_info.get("status") == "success":
        #         first_doc_id = doc_id
        #         break
        
        # if first_doc_id:
        #     same_cluster_docs = await system.get_documents_with_same_cluster(first_doc_id)
        #     logger.info(f"📋 Documents cùng cluster với {first_doc_id}: {len(same_cluster_docs)} documents")
        #     for i, doc_id in enumerate(same_cluster_docs[:5]):  # Chỉ hiển thị 5 docs đầu
        #         logger.info(f"  {i+1}. {doc_id}")
        #     if len(same_cluster_docs) > 5:
        #         logger.info(f"  ... và {len(same_cluster_docs) - 5} documents khác")
        
        # # Get entity neighbors
        # logger.info("🕸️ Getting entity neighbors...")
        # neighbor_entities = ["Apple", "technology", "artificial intelligence"]
        # for entity_name in neighbor_entities:
        #     neighbors = await system.get_entity_neighbors(entity_name)
        #     logger.info(f"  Neighbors of '{entity_name}': {len(neighbors)} connections")
        #     for i, (neighbor, edge_data, node_data) in enumerate(neighbors[:3]):  # Chỉ hiển thị 3 neighbors đầu
        #         logger.info(f"    {i+1}. {neighbor} - {edge_data.get('relation_description', 'N/A')}")
        
        # ========================================
        # PHASE 7: FINAL STATISTICS
        # ========================================
        logger.info("\n" + "="*60)
        logger.info("📊 PHASE 7: FINAL STATISTICS")
        logger.info("="*60)
        
        # Lấy final system stats
        final_stats = await system.get_system_stats()
        logger.info(f"📊 Final system statistics:")
        logger.info(f"  📄 Documents: {final_stats.get('document_status', {})}")
        logger.info(f"  🕸️ Graph: {final_stats.get('graph', {})}")
        logger.info(f"  🔍 Vector DBs: {final_stats.get('vector_dbs', {})}")
        logger.info(f"  📦 Chunks: {final_stats.get('chunks', 0)}")
        
        # Lấy cluster info cuối cùng
        final_cluster_info = await system.get_cluster_info()
        logger.info(f"🎯 Final clustering info:")
        logger.info(f"  📊 Total documents: {final_cluster_info.get('total_documents', 0)}")
        logger.info(f"  📦 Total clusters: {final_cluster_info.get('total_clusters', 0)}")
        logger.info(f"  🔸 Outlier documents: {final_cluster_info.get('outlier_documents', 0)}")
        
        # ========================================
        # PHASE 8: CLEANUP
        # ========================================
        logger.info("\n" + "="*60)
        logger.info("🧹 PHASE 8: CLEANUP")
        logger.info("="*60)
        
        # Cleanup
        await system.cleanup()
        logger.info("✅ Cleanup completed successfully")
        
        logger.info("\n" + "="*60)
        logger.info("🎉 DEMO FULL WORKFLOW COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info("📋 Summary:")
        logger.info(f"  📄 Processed {successful_inserts} documents")
        logger.info(f"  🎯 Created {clustering_result.get('n_clusters', 0)} clusters")
        logger.info(f"  📝 Generated {len(summaries) if 'summaries' in locals() and 'error' not in summaries else 0} summaries")
        logger.info(f"  🕸️ Built knowledge graph with {final_stats.get('graph', {}).get('nodes', 0)} nodes and {final_stats.get('graph', {}).get('edges', 0)} edges")
        logger.info("="*60)
        print(f"⏱️ Total time: {end_time - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"❌ Error in demo: {e}")
        import traceback
        traceback.print_exc()
        await system.cleanup()


async def demo_simple_workflow():
    """Demo đơn giản không cần LLM"""
    
    logger = setup_logger(name="Demo-Simple-Workflow", log_level="INFO")
    logger.info("🚀 Bắt đầu demo simple workflow (không có LLM)...")
    
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
    
    # Khởi tạo system (không có LLM)
    working_dir = "demo_simple_workflow_data"
    system = GraphRAGSystem(
        working_dir=working_dir,
        embedding_client=embedding_client,
        global_config=global_config,
        llm_client=None
    )
    
    try:
        # Documents đơn giản
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
        
        logger.info(f"📝 Processing {len(documents)} documents...")
        
        # Insert documents
        results = await system.insert_documents_batch(documents, chunk_size=500, max_concurrent_docs=3)
        successful_inserts = sum(results)
        logger.info(f"✅ Inserted {successful_inserts}/{len(documents)} documents")
        
        # Clustering
        logger.info("🎯 Performing clustering...")
        clustering_result = await system.cluster_documents(outlier_threshold=3)
        logger.info(f"✅ Clustering result: {clustering_result}")
        
        # Get cluster info
        cluster_info = await system.get_cluster_info()
        logger.info(f"📋 Cluster info: {cluster_info}")
        
        # Cleanup
        await system.cleanup()
        logger.info("✅ Simple workflow completed!")
        
    except Exception as e:
        logger.error(f"❌ Error in simple demo: {e}")
        await system.cleanup()


if __name__ == "__main__":
    # Chạy demo đầy đủ
    print("🎯 GraphRAG Full Workflow Demo")
    # print("="*50)
    # print("1. Full workflow với LLM (nếu có OpenAI API key)")
    # print("2. Simple workflow không cần LLM")
    # print("="*50)
    
    asyncio.run(demo_full_workflow())
    
    # choice = input("Chọn demo (1 hoặc 2): ").strip()
    
    # if choice == "1":
    #     asyncio.run(demo_full_workflow())
    # elif choice == "2":
    #     asyncio.run(demo_simple_workflow())
    # else:
    #     print("Chạy demo đầy đủ...")
    #     asyncio.run(demo_full_workflow()) 