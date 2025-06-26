#!/usr/bin/env python3
"""
Simple test script for clustering functionality
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphrag.clients.embedding_client import SentenceTransformersEmbeddingClient
from graphrag.core.clustering_manager import DynamicClusteringManager


async def test_clustering():
    """Test clustering functionality"""
    
    print("🧪 Testing Clustering Functionality")
    
    # Initialize embedding client
    embedding_client = SentenceTransformersEmbeddingClient(
        model_name="BAAI/bge-m3",
        global_config={"embedding_dimension": 768}
    )
    
    # Initialize clustering manager
    clustering_manager = DynamicClusteringManager(
        embedding_client=embedding_client,
        working_dir="test_clustering_data",
        update_threshold=3,  # Small threshold for testing
        min_cluster_size=2,
        min_samples=1,
        similarity_threshold=0.5
    )
    
    # Initial documents for training
    initial_docs = [
        "Python is a programming language used for web development.",
        "Machine learning algorithms require data preprocessing.",
        "Deep learning models use neural networks.",
        "Football is a popular sport worldwide.",
        "Basketball games are exciting to watch.",
        "Tennis matches require skill and strategy."
    ]
    
    print("\n📊 Initializing clustering model...")
    success = await clustering_manager.initialize_model(initial_docs)
    if success:
        print("✅ Model initialized successfully!")
    else:
        print("❌ Failed to initialize model")
        return
    
    # Test documents (some should be outliers)
    test_docs = [
        # Technology (should cluster with existing)
        "JavaScript is used for frontend development.",
        "Data science involves statistical analysis.",
        
        # Sports (should cluster with existing)
        "Soccer matches are played on grass fields.",
        "Volleyball is a team sport.",
        
        # Food (should be outliers)
        "Vietnamese pho is a traditional soup dish.",
        "Italian pizza is made with fresh ingredients.",
        "Sushi is a Japanese delicacy.",
        
        # More technology
        "Cloud computing provides scalable infrastructure.",
        
        # More food (should trigger update)
        "Thai curry is spicy and flavorful."
    ]
    
    print(f"\n📄 Processing {len(test_docs)} test documents...")
    
    # Process each document
    for i, doc in enumerate(test_docs):
        print(f"\n  Document {i+1}: {doc[:50]}...")
        
        try:
            result = await clustering_manager.process_document(f"doc_{i}", doc)
            
            if result.is_outlier:
                print(f"    -> OUTLIER (confidence: {result.confidence:.3f})")
            else:
                print(f"    -> Cluster {result.cluster_id} (confidence: {result.confidence:.3f})")
                print(f"    -> Keywords: {result.keywords[:3]}")
                
        except Exception as e:
            print(f"    -> ERROR: {e}")
    
    # Get statistics
    print("\n📈 Clustering Statistics:")
    stats = await clustering_manager.get_cluster_statistics()
    for key, value in stats.items():
        if key != "model_path":  # Skip long path
            print(f"  {key}: {value}")
    
    # Get all clusters
    print("\n📋 All Clusters:")
    clusters = await clustering_manager.get_all_clusters()
    for cluster in clusters:
        print(f"  Cluster {cluster.cluster_id}: {cluster.keywords[:5]}")
    
    # Test cluster search
    print("\n🔍 Testing Cluster Search:")
    queries = ["programming", "food", "sports"]
    
    for query in queries:
        print(f"\n  Query: '{query}'")
        similar_clusters = await clustering_manager.search_similar_clusters(query, top_k=2)
        
        if similar_clusters:
            for cluster_id, similarity in similar_clusters:
                cluster_info = await clustering_manager.get_cluster_info(cluster_id)
                if cluster_info:
                    print(f"    Cluster {cluster_id}: similarity={similarity:.3f}, keywords={cluster_info.keywords[:3]}")
        else:
            print("    No similar clusters found")
    
    # Cleanup
    await clustering_manager.cleanup()
    print("\n🧹 Cleanup completed!")


if __name__ == "__main__":
    asyncio.run(test_clustering()) 