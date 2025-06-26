#!/usr/bin/env python3
"""
Demo GraphRAG với Clustering Integration
Thể hiện toàn bộ flow: Insert documents -> Clustering -> Summary Generation
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
    
    print("🚀 GraphRAG with Clustering Integration Demo")
    print("=" * 60)
    
    # Configuration
    working_dir = "graphrag_clustering_data"
    global_config = {
        "embedding_dimension": 768,
        "chunk_size": 1000,
        "max_workers": 4
    }
    
    # Clustering configuration
    clustering_config = {
        "update_threshold": 3,  # Small threshold for demo
        "min_cluster_size": 2,
        "min_samples": 1,
        "model_name": "dynamic_bertopic_model",
        "max_tokens_per_batch": 3000,
        "max_concurrent_batches": 2
    }
    
    # Initialize clients
    print("\n📡 Initializing clients...")
    embedding_client = SentenceTransformersEmbeddingClient(
        model_name="BAAI/bge-m3",
        global_config=global_config
    )
    
    # Note: For demo, we'll use a mock LLM client if OpenAI key is not available
    llm_client = None
    if os.getenv("OPENAI_API_KEY"):
        llm_client = OpenAIClient(
            model_name="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY"),
            global_config=global_config
        )
        print("✅ OpenAI LLM client initialized")
    else:
        print("⚠️ No OpenAI API key found. Summary generation will be disabled.")
    
    # Initialize GraphRAG system with clustering
    print("\n🏗️ Initializing GraphRAG System...")
    graphrag = GraphRAGSystem(
        working_dir=working_dir,
        embedding_client=embedding_client,
        global_config=global_config,
        llm_client=llm_client,
        enable_clustering=True,
        clustering_config=clustering_config
    )
    
    print("✅ GraphRAG System initialized successfully!")
    
    # Sample documents for initial clustering training
    print("\n📚 Initializing clustering model...")
    initial_docs = [
        "Python is a programming language used for web development and data science.",
        "Machine learning algorithms require data preprocessing and feature engineering.",
        "Deep learning models use neural networks with multiple layers.",
        "Football is a popular sport played worldwide with millions of fans.",
        "Basketball games are exciting to watch with fast-paced action.",
        "Tennis matches require skill, strategy, and physical endurance."
    ]
    
    # Initialize clustering model
    clustering_success = await graphrag.initialize_clustering(initial_docs)
    if clustering_success:
        print("✅ Clustering model initialized successfully!")
    else:
        print("❌ Failed to initialize clustering model")
        return
    
    # Sample documents for processing
    print("\n📄 Processing documents...")
    documents = [
        # Technology documents (should cluster with existing tech docs)
        "JavaScript is used for frontend development and creating interactive web applications.",
        "Data science involves statistical analysis, machine learning, and data visualization.",
        "Cloud computing provides scalable infrastructure for modern applications.",
        
        # Sports documents (should cluster with existing sports docs)
        "Soccer matches are played on grass fields with 11 players per team.",
        "Volleyball is a team sport that requires coordination and communication.",
        "Championship finals attract millions of viewers worldwide.",
        
        # Food documents (should be outliers, create new cluster)
        "Vietnamese pho is a traditional soup dish made with rice noodles and beef.",
        "Italian pizza is made with fresh ingredients and wood-fired ovens.",
        "Sushi is a Japanese delicacy featuring fresh fish and vinegared rice.",
        "Thai curry is spicy and flavorful with coconut milk and herbs.",
        "Mexican tacos are filled with various ingredients and served with salsa.",
        
        # More technology documents
        "Version control with Git is essential for collaborative software development.",
        "Docker containers provide consistent environments for application deployment.",
        
        # More sports documents
        "Olympic games bring together athletes from around the world.",
        "Team strategy focuses on defensive play and counter-attacks."
    ]
    
    print(f"📝 Inserting {len(documents)} documents...")
    
    # Insert documents in batch
    results = await graphrag.insert_documents_batch_with_llm(
        documents=documents,
        max_concurrent_docs=3
    )
    
    # Analyze results
    success_count = sum(1 for r in results if r.get("status") == "success")
    print(f"✅ Successfully processed {success_count}/{len(documents)} documents")
    
    # Now perform clustering (lazy clustering - only when called)
    print("\n🔍 Performing clustering analysis...")
    clustering_results = await graphrag.process_clustering(documents)
    
    if clustering_results.get("status") == "success":
        print("✅ Clustering completed successfully!")
        
        # Show clustering results
        cluster_results = clustering_results.get("cluster_results", [])
        print(f"\n📊 Clustering Results:")
        for i, result in enumerate(cluster_results):
            doc_preview = documents[i][:50] + "..." if len(documents[i]) > 50 else documents[i]
            if result.is_outlier:
                print(f"  📄 {doc_preview} -> OUTLIER (confidence: {result.confidence:.3f})")
            else:
                print(f"  📄 {doc_preview} -> Cluster {result.cluster_id} (confidence: {result.confidence:.3f})")
        
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
    else:
        print(f"❌ Clustering failed: {clustering_results.get('error', 'Unknown error')}")
    
    # Generate cluster summaries if LLM client is available
    if llm_client:
        print("\n📝 Generating cluster summaries...")
        summary_results = await graphrag.generate_cluster_summaries()
        
        if summary_results.get("status") == "success":
            print("✅ Cluster summaries generated successfully!")
            
            summaries = summary_results.get("summaries", {})
            print(f"\n📋 Cluster Summaries:")
            for cluster_id, summary in summaries.items():
                print(f"\n  Cluster {cluster_id}:")
                print(f"    Keywords: {summary.keywords[:5]}")
                print(f"    Document count: {summary.doc_count}")
                print(f"    Processing time: {summary.processing_time:.2f}s")
                print(f"    Summary: {summary.summary[:200]}...")
        else:
            print(f"❌ Summary generation failed: {summary_results.get('error', 'Unknown error')}")
    else:
        print("\n⚠️ Skipping summary generation (no LLM client available)")
    
    # Demo clustering-aware queries
    print("\n🔍 Clustering-Aware Queries:")
    queries = [
        "programming languages",
        "sports and athletics", 
        "food and cuisine",
        "technology trends"
    ]
    
    for query in queries:
        print(f"\n  Query: '{query}'")
        result = await graphrag.query_with_clustering(query, top_k=5, use_clusters=True)
        
        if "error" not in result:
            entities = result.get("entities", [])
            relations = result.get("relations", [])
            clustering_info = result.get("clustering_info")
            
            print(f"    Found {len(entities)} entities, {len(relations)} relations")
            
            if clustering_info and "available_clusters" in clustering_info:
                available_clusters = clustering_info["available_clusters"]
                print(f"    Available clusters: {len(available_clusters)}")
                for cluster in available_clusters[:2]:  # Show top 2
                    print(f"      Cluster {cluster['cluster_id']}: keywords={cluster['keywords'][:3]}")
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
    
    # Get system statistics
    print("\n📊 System Statistics:")
    system_stats = await graphrag.get_system_stats()
    if "error" not in system_stats:
        print(f"  Total documents: {system_stats.get('total_documents', 0)}")
        print(f"  Total entities: {system_stats.get('total_entities', 0)}")
        print(f"  Total relations: {system_stats.get('total_relations', 0)}")
        print(f"  Graph nodes: {system_stats.get('graph_nodes', 0)}")
        print(f"  Graph edges: {system_stats.get('graph_edges', 0)}")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    await graphrag.cleanup()
    print("✅ Cleanup completed!")
    
    print("\n🎉 Demo completed successfully!")


if __name__ == "__main__":
    asyncio.run(main()) 