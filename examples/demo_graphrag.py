import asyncio
import os
from typing import Dict, Any

from src.graphrag.utils.logger_config import setup_logger, get_logger
from src.graphrag.core.graphrag_system import GraphRAGSystem
from src.graphrag.clients.llm_client import create_llm_client
from src.graphrag.clients.embedding_client import create_embedding_client


async def demo_graphrag_system():
    """Demo hệ thống GraphRAG"""
    
    # Setup logger
    logger = setup_logger(
        name="GraphRAGDemo",
        log_level="INFO",
        log_dir="./logs"
    )
    
    # Cấu hình hệ thống
    working_dir = "./graphrag_data"
    global_config = {
        "working_dir": working_dir,
        "save_interval": 100,
        "max_retries": 3
    }
    
    # Tạo embedding client (có thể chọn một trong các options)
    # Option 1: Sentence Transformers (GPU acceleration)
    embedding_client = create_embedding_client(
        client_type="sentence_transformers",
        model_name="all-MiniLM-L6-v2",
        device="cuda"  # hoặc "cpu" nếu không có GPU
    )
    
    # Option 2: vLLM embedding (nếu có vLLM server)
    # embedding_client = create_embedding_client(
    #     client_type="vllm",
    #     model_name="llama2-7b-chat",
    #     url="http://localhost:8000/v1",
    #     api_key="dummy"
    # )
    
    # Option 3: OpenAI embedding
    # embedding_client = create_embedding_client(
    #     client_type="openai",
    #     model_name="text-embedding-ada-002",
    #     api_key="your-openai-api-key"
    # )
    
    # Tạo LLM client (có thể chọn một trong các options)
    # Option 1: vLLM client (self-hosted)
    llm_client = create_llm_client(
        client_type="vllm",
        model_name="llama2-7b-chat",
        url="http://localhost:8000/v1",
        api_key="dummy"
    )
    
    # Option 2: OpenAI client
    # llm_client = create_llm_client(
    #     client_type="openai",
    #     model_name="gpt-3.5-turbo",
    #     api_key="your-openai-api-key"
    # )
    
    # Option 3: Không sử dụng LLM (chỉ dùng regex)
    # llm_client = None
    
    # Khởi tạo hệ thống GraphRAG
    system = GraphRAGSystem(
        working_dir=working_dir,
        embedding_client=embedding_client,
        global_config=global_config,
        llm_client=llm_client
    )
    
    # Sample documents
    documents = [
        {
            "id": "doc1",
            "content": """
            Apple Inc. is a technology company that designs and manufactures consumer electronics. 
            The company was founded by Steve Jobs and Steve Wozniak in 1976. 
            Apple's most popular products include the iPhone, iPad, and Mac computers. 
            The iPhone is a line of smartphones designed by Apple Inc. and runs on iOS operating system.
            """
        },
        {
            "id": "doc2", 
            "content": """
            Microsoft Corporation is a multinational technology company. 
            Bill Gates and Paul Allen founded Microsoft in 1975. 
            Microsoft develops the Windows operating system and Office productivity software. 
            Windows is used by millions of people worldwide for personal and business computing.
            """
        },
        {
            "id": "doc3",
            "content": """
            Google LLC is a technology company that specializes in internet-related services. 
            Larry Page and Sergey Brin founded Google in 1998. 
            Google's main product is the Google Search engine, which helps users find information online. 
            The company also develops Android, a mobile operating system used by many smartphone manufacturers.
            """
        }
    ]
    
    try:
        # Insert documents
        logger.info("Starting batch document insertion...")
        
        # Chuẩn bị documents cho batch processing
        documents_batch = [
            {"doc_id": doc['id'], "content": doc['content']} 
            for doc in documents
        ]
        
        # Sử dụng batch processing thay vì tuần tự
        if llm_client:
            # Batch processing với LLM one-shot
            results = await system.insert_documents_batch_with_llm(
                documents=documents_batch,
                max_concurrent_docs=3  # Chạy song song tối đa 3 documents
            )
        else:
            # Batch processing với chunking
            results = await system.insert_documents_batch(
                documents=documents_batch,
                chunk_size=500,
                max_concurrent_docs=3  # Chạy song song tối đa 3 documents
            )
        
        # Kiểm tra kết quả
        success_count = sum(results)
        logger.info(f"Batch processing completed: {success_count}/{len(documents)} documents successful")
        
        for i, success in enumerate(results):
            if success:
                logger.info(f"Successfully processed document: {documents[i]['id']}")
            else:
                logger.error(f"Failed to process document: {documents[i]['id']}")
        
        # Query examples
        logger.info("\n" + "="*50)
        logger.info("QUERY EXAMPLES")
        logger.info("="*50)
        
        # Query entities
        logger.info("\n1. Querying entities...")
        entity_results = await system.query_entities("technology company", top_k=5)
        logger.info(f"Found {len(entity_results)} entities:")
        for i, result in enumerate(entity_results[:3]):
            logger.info(f"  {i+1}. {result.get('entity_name', 'N/A')}: {result.get('description', 'N/A')}")
        
        # Query relations
        logger.info("\n2. Querying relations...")
        relation_results = await system.query_relations("founded", top_k=5)
        logger.info(f"Found {len(relation_results)} relations:")
        for i, result in enumerate(relation_results[:3]):
            logger.info(f"  {i+1}. {result.get('source_entity', 'N/A')} -> {result.get('relation_description', 'N/A')} -> {result.get('target_entity', 'N/A')}")
        
        # Search entities by name
        logger.info("\n3. Searching entities by name...")
        apple_results = await system.search_entities_by_name("Apple")
        logger.info(f"Found {len(apple_results)} entities matching 'Apple':")
        for i, result in enumerate(apple_results[:2]):
            logger.info(f"  {i+1}. {result.get('entity_name', 'N/A')}: {result.get('description', 'N/A')}")
        
        # Get entity graph context
        logger.info("\n4. Getting entity graph context...")
        apple_context = await system.get_entity_graph_context("Apple")
        if "error" not in apple_context:
            logger.info(f"Apple entity context:")
            logger.info(f"  Description: {apple_context['entity']['data'].get('description', 'N/A')}")
            logger.info(f"  Neighbors: {apple_context['total_neighbors']}")
            for neighbor in apple_context['neighbors'][:3]:
                logger.info(f"    - {neighbor['name']}: {neighbor['relation']}")
        else:
            logger.warning(f"Could not get context for Apple: {apple_context['error']}")
        
        # Get system statistics
        logger.info("\n5. System statistics...")
        stats = await system.get_system_stats()
        logger.info(f"Document status: {stats.get('document_status', {})}")
        logger.info(f"Graph: {stats.get('graph', {})} nodes and edges")
        logger.info(f"Vector DBs: {stats.get('vector_dbs', {})} entities and relations")
        logger.info(f"Chunks: {stats.get('chunks', 0)}")
        
        # Get document status
        logger.info("\n6. Document status...")
        for doc in documents:
            status = await system.get_document_status(doc['id'])
            if status:
                logger.info(f"  {doc['id']}: {status.get('status', 'unknown')}")
                if status.get('status') == 'success':
                    logger.info(f"    Entities: {status.get('entities_count', 0)}, Relations: {status.get('relations_count', 0)}")
        
        logger.info("\n" + "="*50)
        logger.info("DEMO COMPLETED SUCCESSFULLY!")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Error in demo: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        await system.cleanup()
        logger.info("Demo completed!")


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_graphrag_system()) 