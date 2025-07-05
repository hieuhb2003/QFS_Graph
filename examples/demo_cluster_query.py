#!/usr/bin/env python3
"""
Demo script để test tính năng query cluster summaries với 2 mode:
- Retrieval mode: Chỉ lấy ra summaries phù hợp nhất từ vector DB
- Generation mode: Lấy summaries từ vector DB rồi gen câu trả lời bằng LLM
"""

import asyncio
import os
import sys
from pathlib import Path

# Thêm src vào path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphrag.utils.logger_config import setup_logger
from graphrag.core.graphrag_system import GraphRAGSystem
from graphrag.clients.embedding_client import create_embedding_client
from graphrag.clients.llm_client import create_llm_client


async def main():
    """Demo chính"""
    print("🚀 GraphRAG Cluster Summary Query Demo")
    print("=" * 50)
    
    # Setup logger
    logger = setup_logger(
        name="ClusterQueryDemo",
        log_level="INFO",
        log_dir="./logs"
    )
    
    # Tạo embedding client
    print("📡 Khởi tạo embedding client...")
    embedding_client = create_embedding_client(
        client_type="sentence_transformers",
        model_name="all-MiniLM-L6-v2",
        device="cpu"  # Sử dụng CPU cho demo
    )
    
    # Tạo LLM client (optional - chỉ cần cho generation mode)
    llm_client = None
    use_llm = input("🤖 Bạn có muốn sử dụng LLM cho generation mode không? (y/n): ").lower().strip()
    
    if use_llm == 'y':
        llm_type = input("Chọn LLM type (openai/vllm): ").lower().strip()
        
        if llm_type == "openai":
            api_key = input("Nhập OpenAI API key: ").strip()
            if api_key:
                llm_client = create_llm_client(
                    client_type="openai",
                    model_name="gpt-3.5-turbo",
                    api_key=api_key
                )
        elif llm_type == "vllm":
            url = input("Nhập vLLM URL (mặc định: http://localhost:8000/v1): ").strip()
            if not url:
                url = "http://localhost:8000/v1"
            llm_client = create_llm_client(
                client_type="vllm",
                model_name="llama2-7b-chat",
                url=url,
                api_key="dummy"
            )
    
    # Cấu hình hệ thống
    global_config = {
        "save_interval": 50,
        "clustering": {
            "outlier_threshold": 5,
            "max_tokens": 4096,
            "batch_size": 8
        },
        "summary": {
            "max_workers": 3,
            "context_length": 2048,
            "query_generation_prompt_template": """
You are an expert assistant. Based on the following cluster summaries, please provide a comprehensive answer to the user's question.

User Question: {query}

Relevant Cluster Summaries:
{summaries}

Requirements:
1. Answer the question based on the information in the cluster summaries
2. Provide specific details and examples from the summaries
3. If the information is not available in the summaries, clearly state that
4. Keep the answer well-structured and informative
5. Cite which clusters the information comes from when relevant
6. Use Vietnamese language for the answer

Answer:
"""
        }
    }
    
    # Khởi tạo hệ thống
    print("🔧 Khởi tạo GraphRAG system...")
    working_dir = "./demo_cluster_query_data"
    system = GraphRAGSystem(
        working_dir=working_dir,
        embedding_client=embedding_client,
        global_config=global_config,
        llm_client=llm_client
    )
    
    # Sample documents về technology
    sample_documents = [
        "Apple Inc. is a technology company that designs and manufactures consumer electronics, computer software, and online services. Founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976, Apple is known for its innovative products like the iPhone, iPad, and Mac computers.",
        
        "Microsoft Corporation is a multinational technology company that develops, manufactures, licenses, supports, and sells computer software, consumer electronics, and related services. Founded by Bill Gates and Paul Allen in 1975, Microsoft is best known for its Windows operating system and Office productivity suite.",
        
        "Google LLC is a technology company that specializes in internet-related services and products, including online advertising technologies, search engine, cloud computing, software, and hardware. Founded by Larry Page and Sergey Brin in 1998, Google is known for its search engine and Android operating system.",
        
        "Tesla, Inc. is an electric vehicle and clean energy company that designs, develops, manufactures, sells, and leases electric vehicles, energy generation and storage systems, and related services. Founded by Elon Musk in 2003, Tesla is a leader in electric vehicle technology.",
        
        "Amazon.com, Inc. is an American multinational technology company that focuses on e-commerce, cloud computing, digital streaming, and artificial intelligence. Founded by Jeff Bezos in 1994, Amazon is known for its online marketplace and AWS cloud services.",
        
        "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or decisions.",
        
        "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data. It has been particularly successful in image recognition, natural language processing, and speech recognition.",
        
        "Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human language. It involves developing algorithms and models to understand, interpret, and generate human language.",
        
        "Computer Vision is a field of artificial intelligence that trains computers to interpret and understand visual information from the world. It involves developing algorithms to process, analyze, and understand images and videos.",
        
        "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans. It encompasses various technologies including machine learning, deep learning, natural language processing, and computer vision."
    ]
    
    try:
        # Insert documents
        print(f"📄 Đang insert {len(sample_documents)} documents...")
        if llm_client:
            results = await system.insert_documents_batch_with_llm(sample_documents, max_concurrent_docs=3)
        else:
            results = await system.insert_documents_batch(sample_documents, chunk_size=1000, max_concurrent_docs=3)
        
        success_count = sum(results)
        print(f"✅ Đã xử lý thành công {success_count}/{len(sample_documents)} documents")
        
        # Thực hiện clustering
        print("🔍 Đang thực hiện clustering...")
        clustering_result = await system.cluster_documents(outlier_threshold=5)
        print(f"✅ Đã tạo {clustering_result.get('n_clusters', 0)} clusters")
        
        # Tạo cluster summaries (chỉ khi có LLM)
        if llm_client:
            print("📝 Đang tạo cluster summaries...")
            summaries = await system.generate_cluster_summaries(max_workers=3)
            print(f"✅ Đã tạo summaries cho {len(summaries)} clusters")
        else:
            print("⚠️ Bỏ qua tạo summaries vì không có LLM client")
        
        # Demo query với retrieval mode
        print("\n" + "="*50)
        print("🔍 DEMO RETRIEVAL MODE")
        print("="*50)
        
        retrieval_queries = [
            "technology companies",
            "artificial intelligence",
            "machine learning",
            "Apple and Microsoft"
        ]
        
        for query in retrieval_queries:
            print(f"\n❓ Query: {query}")
            result = await system.query_cluster_summaries(
                query=query, 
                mode="retrieval", 
                top_k=3
            )
            
            if "error" in result:
                print(f"❌ Lỗi: {result['error']}")
            else:
                print(f"✅ Tìm thấy {result['total_found']} clusters:")
                for i, cluster_result in enumerate(result['results'], 1):
                    print(f"  {i}. Cluster {cluster_result['cluster_id']} (Score: {cluster_result['score']:.3f})")
                    print(f"     Summary: {cluster_result['summary'][:100]}...")
        
        # Demo query với generation mode (chỉ khi có LLM)
        if llm_client:
            print("\n" + "="*50)
            print("🤖 DEMO GENERATION MODE")
            print("="*50)
            
            generation_queries = [
                "What are the main technology companies mentioned?",
                "Explain the relationship between AI and machine learning",
                "What are the key products of Apple and Microsoft?",
                "How does deep learning relate to computer vision?"
            ]
            
            for query in generation_queries:
                print(f"\n❓ Query: {query}")
                result = await system.query_cluster_summaries(
                    query=query, 
                    mode="generation", 
                    top_k=3
                )
                
                if "error" in result:
                    print(f"❌ Lỗi: {result['error']}")
                else:
                    print(f"✅ Answer: {result['answer']}")
                    print(f"📊 Sử dụng {result['total_found']} clusters:")
                    for cluster_result in result['used_summaries']:
                        print(f"  - Cluster {cluster_result['cluster_id']} (Score: {cluster_result['score']:.3f})")
        else:
            print("\n⚠️ Bỏ qua generation mode vì không có LLM client")
        
        # Hiển thị thống kê hệ thống
        print("\n" + "="*50)
        print("📊 SYSTEM STATISTICS")
        print("="*50)
        
        stats = await system.get_system_stats()
        print(f"📄 Documents: {stats.get('document_status', {}).get('success', 0)} successful")
        print(f"🔗 Graph: {stats.get('graph', {}).get('nodes', 0)} nodes, {stats.get('graph', {}).get('edges', 0)} edges")
        print(f"📊 Vector DBs: {stats.get('vector_dbs', {}).get('entities', 0)} entities, {stats.get('vector_dbs', {}).get('relations', 0)} relations")
        
        # Hiển thị cluster info
        cluster_info = await system.get_cluster_info()
        print(f"🔍 Clusters: {cluster_info.get('total_clusters', 0)} total, {cluster_info.get('outlier_documents', 0)} outliers")
        
    except Exception as e:
        logger.error(f"Lỗi trong demo: {e}")
        print(f"❌ Lỗi: {e}")
    
    finally:
        # Cleanup
        print("\n🧹 Đang cleanup...")
        await system.cleanup()
        print("✅ Demo hoàn thành!")


if __name__ == "__main__":
    asyncio.run(main()) 