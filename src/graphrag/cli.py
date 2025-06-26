"""
Command Line Interface for GraphRAG system.
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path

from .utils.logger_config import setup_logger
from .core.graphrag_system import GraphRAGSystem
from .clients.llm_client import create_llm_client
from .clients.embedding_client import create_embedding_client


def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="GraphRAG - Knowledge Graph Construction from Documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo
  graphrag demo

  # Process a document
  graphrag process --input document.txt --output ./data

  # Query entities
  graphrag query --entities "Apple Inc" --data ./data
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demo")
    demo_parser.add_argument("--working-dir", default="./graphrag_data", help="Working directory")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process documents")
    process_parser.add_argument("--input", required=True, help="Input document file or directory")
    process_parser.add_argument("--output", default="./graphrag_data", help="Output directory")
    process_parser.add_argument("--embedding", default="sentence_transformers", 
                               choices=["sentence_transformers", "vllm", "openai"],
                               help="Embedding client type")
    process_parser.add_argument("--llm", default="vllm",
                               choices=["vllm", "openai", "none"],
                               help="LLM client type")
    process_parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Model name")
    process_parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    process_parser.add_argument("--batch", action="store_true", help="Use batch processing")
    process_parser.add_argument("--max-concurrent", type=int, default=5, help="Max concurrent documents for batch processing")
    process_parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for document processing")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the knowledge graph")
    query_parser.add_argument("--data", required=True, help="Data directory")
    query_parser.add_argument("--entities", help="Query entities")
    query_parser.add_argument("--relations", help="Query relations")
    query_parser.add_argument("--entity-name", help="Search entity by name")
    query_parser.add_argument("--graph-context", help="Get entity graph context")
    
    return parser


async def run_demo(args):
    """Run the demo."""
    from examples.demo_graphrag import demo_graphrag_system
    
    print("Running GraphRAG demo...")
    await demo_graphrag_system()


async def process_documents(args):
    """Process documents."""
    logger = setup_logger(name="GraphRAG-CLI", log_level="INFO")
    
    # Create embedding client
    embedding_kwargs = {
        "client_type": args.embedding,
        "model_name": args.model,
    }
    
    if args.embedding == "sentence_transformers":
        embedding_kwargs["device"] = args.device
    elif args.embedding == "vllm":
        embedding_kwargs.update({
            "url": "http://localhost:8000/v1",
            "api_key": "dummy"
        })
    elif args.embedding == "openai":
        embedding_kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
    
    embedding_client = create_embedding_client(**embedding_kwargs)
    
    # Create LLM client
    llm_client = None
    if args.llm != "none":
        llm_kwargs = {
            "client_type": args.llm,
            "model_name": args.model,
        }
        
        if args.llm == "vllm":
            llm_kwargs.update({
                "url": "http://localhost:8000/v1",
                "api_key": "dummy"
            })
        elif args.llm == "openai":
            llm_kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
        
        llm_client = create_llm_client(**llm_kwargs)
    
    # Initialize system
    system = GraphRAGSystem(
        working_dir=args.output,
        embedding_client=embedding_client,
        global_config={"save_interval": 100},
        llm_client=llm_client
    )
    
    # Process input
    input_path = Path(args.input)
    documents = []
    
    if input_path.is_file():
        # Single file
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        documents.append({
            "doc_id": input_path.stem,
            "content": content
        })
    elif input_path.is_dir():
        # Directory - collect all text files
        for file_path in input_path.glob("*.txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            documents.append({
                "doc_id": file_path.stem,
                "content": content
            })
    
    if not documents:
        logger.error("No documents found to process")
        return
    
    logger.info(f"Found {len(documents)} documents to process")
    
    # Process documents
    if args.batch:
        # Batch processing
        if llm_client:
            results = await system.insert_documents_batch_with_llm(
                documents=documents,
                max_concurrent_docs=args.max_concurrent
            )
        else:
            results = await system.insert_documents_batch(
                documents=documents,
                chunk_size=args.chunk_size,
                max_concurrent_docs=args.max_concurrent
            )
        
        success_count = sum(results)
        logger.info(f"Batch processing completed: {success_count}/{len(documents)} documents successful")
        
        for i, success in enumerate(results):
            if success:
                logger.info(f"Successfully processed {documents[i]['doc_id']}")
            else:
                logger.error(f"Failed to process {documents[i]['doc_id']}")
    else:
        # Sequential processing
        for doc in documents:
            if llm_client:
                success = await system.insert_document_with_llm(
                    doc_id=doc['doc_id'],
                    content=doc['content']
                )
            else:
                success = await system.insert_document(
                    doc_id=doc['doc_id'],
                    content=doc['content'],
                    chunk_size=args.chunk_size
                )
            
            if success:
                logger.info(f"Successfully processed {doc['doc_id']}")
            else:
                logger.error(f"Failed to process {doc['doc_id']}")
    
    # Cleanup
    await system.cleanup()


async def query_system(args):
    """Query the system."""
    logger = setup_logger(name="GraphRAG-CLI", log_level="INFO")
    
    # Initialize system (read-only)
    embedding_client = create_embedding_client(
        client_type="sentence_transformers",
        model_name="all-MiniLM-L6-v2",
        device="cpu"  # Use CPU for querying
    )
    
    system = GraphRAGSystem(
        working_dir=args.data,
        embedding_client=embedding_client,
        global_config={"save_interval": 100},
        llm_client=None  # No LLM needed for querying
    )
    
    # Perform queries
    if args.entities:
        logger.info(f"Querying entities: {args.entities}")
        results = await system.query_entities(args.entities, top_k=5)
        print(f"\nFound {len(results)} entities:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result.get('entity_name', 'N/A')}: {result.get('description', 'N/A')}")
    
    if args.relations:
        logger.info(f"Querying relations: {args.relations}")
        results = await system.query_relations(args.relations, top_k=5)
        print(f"\nFound {len(results)} relations:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result.get('source_entity', 'N/A')} -> {result.get('relation_description', 'N/A')} -> {result.get('target_entity', 'N/A')}")
    
    if args.entity_name:
        logger.info(f"Searching entities by name: {args.entity_name}")
        results = await system.search_entities_by_name(args.entity_name)
        print(f"\nFound {len(results)} entities matching '{args.entity_name}':")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result.get('entity_name', 'N/A')}: {result.get('description', 'N/A')}")
    
    if args.graph_context:
        logger.info(f"Getting graph context for: {args.graph_context}")
        context = await system.get_entity_graph_context(args.graph_context)
        if "error" not in context:
            print(f"\nEntity: {context['entity']['name']}")
            print(f"Description: {context['entity']['data'].get('description', 'N/A')}")
            print(f"Neighbors: {context['total_neighbors']}")
            for neighbor in context['neighbors'][:3]:
                print(f"  - {neighbor['name']}: {neighbor['relation']}")
        else:
            print(f"Error: {context['error']}")


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == "demo":
            asyncio.run(run_demo(args))
        elif args.command == "process":
            asyncio.run(process_documents(args))
        elif args.command == "query":
            asyncio.run(query_system(args))
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 