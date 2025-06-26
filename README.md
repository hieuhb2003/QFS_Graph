# GraphRAG System

A comprehensive Graph-based Retrieval-Augmented Generation (GraphRAG) system that integrates document processing, entity-relation extraction, knowledge graph construction, and dynamic topic clustering.

## Features

### Core Features

- **Document Processing**: Chunk-based document processing with configurable chunk sizes
- **Entity-Relation Extraction**: LLM-powered extraction of entities and relations from documents
- **Knowledge Graph Construction**: Automatic building of knowledge graphs using NetworkX
- **Vector Database Integration**: Multi-vector database support for efficient similarity search
- **Batch Processing**: Concurrent document processing with configurable concurrency limits

### 🆕 Dynamic Topic Clustering

- **BERTopic Integration**: Automatic topic discovery using BERTopic clustering
- **Incremental Learning**: Dynamic model updates when new outlier documents are detected
- **Outlier Buffer Management**: Intelligent buffering of outliers before model updates
- **Cluster-Aware Querying**: Enhanced search with clustering context
- **Real-time Processing**: Stream processing of new documents with immediate clustering
- **Document Nodes**: Document nodes in knowledge graph with "belong-to" relations
- **Map-Reduce Summaries**: LLM-powered cluster summaries using map-reduce approach

## Architecture

```
GraphRAG System
├── Document Processing Layer
│   ├── Chunking & Embedding
│   ├── Entity/Relation Extraction
│   └── Status Tracking
├── Dynamic Clustering Layer
│   ├── BERTopic Model Management
│   ├── Outlier Detection & Buffering
│   ├── Incremental Model Updates
│   └── Cluster Summary Generation
├── Storage Layer
│   ├── Vector Databases (NanoVectorDB)
│   ├── Knowledge Graph (NetworkX)
│   └── Document Status (JSON)
└── Query Layer
    ├── Entity/Relation Search
    ├── Graph Traversal
    └── Cluster-Aware Retrieval
```

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd next_work
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## Quick Start

### Basic Usage

```python
import asyncio
from graphrag.clients.embedding_client import SentenceTransformersEmbeddingClient
from graphrag.clients.llm_client import OpenAIClient
from graphrag.core.graphrag_system import GraphRAGSystem

async def main():
    # Initialize clients
    embedding_client = SentenceTransformersEmbeddingClient(
        model_name="BAAI/bge-m3",
        global_config={"embedding_dimension": 768}
    )

    llm_client = OpenAIClient(
        model_name="gpt-3.5-turbo",
        api_key="your-api-key"
    )

    # Initialize GraphRAG system with clustering
    clustering_config = {
        "update_threshold": 5,
        "min_cluster_size": 3,
        "min_samples": 2,
        "model_name": "dynamic_bertopic_model"
    }

    graphrag = GraphRAGSystem(
        working_dir="graphrag_data",
        embedding_client=embedding_client,
        global_config={"embedding_dimension": 768},
        llm_client=llm_client,
        enable_clustering=True,
        clustering_config=clustering_config
    )

    # Initialize clustering with initial documents
    initial_docs = [
        "Python is a popular programming language.",
        "Machine learning requires data preprocessing.",
        "Football matches are exciting events."
    ]
    await graphrag.initialize_clustering(initial_docs)

    # Process documents
    documents = [
        "Deep learning models need large datasets.",
        "The championship final was intense.",
        "Vietnamese pho is delicious traditional food."
    ]

    results = await graphrag.insert_documents_batch_with_llm(
        documents=documents,
        max_concurrent_docs=3
    )

    # Perform clustering analysis
    clustering_results = await graphrag.process_clustering(documents)
    print("Clustering results:", clustering_results)

    # Generate cluster summaries
    summary_results = await graphrag.generate_cluster_summaries()
    print("Summary results:", summary_results)

    # Query with clustering awareness
    query_result = await graphrag.query_with_clustering(
        "machine learning algorithms",
        top_k=10,
        use_clusters=True
    )

    print(query_result)

asyncio.run(main())
```

### Clustering Configuration

```python
# Advanced clustering configuration
clustering_config = {
    "update_threshold": 5,        # Update model when 5 outliers collected
    "min_cluster_size": 3,        # Minimum documents per cluster
    "min_samples": 2,             # HDBSCAN min_samples parameter
    "model_name": "dynamic_bertopic_model",
    "max_tokens_per_batch": 4000, # For summary generation
    "max_concurrent_batches": 3   # For summary generation
}

graphrag = GraphRAGSystem(
    working_dir="graphrag_data",
    embedding_client=embedding_client,
    global_config=global_config,
    llm_client=llm_client,
    enable_clustering=True,
    clustering_config=clustering_config
)
```

## How Clustering Works

### 1. Initial Model Training

- System trains BERTopic model on initial set of documents
- Creates base clusters for known topics
- Saves model for future use

### 2. Document Processing

- Each new document is processed through the clustering pipeline
- System predicts topic assignment using existing model
- Documents with high confidence are assigned to existing clusters
- Low-confidence documents are marked as outliers and buffered

### 3. Dynamic Updates

- When outlier buffer reaches threshold, model is updated
- New clusters are created from accumulated outliers
- Model is saved with new knowledge
- Buffer is cleared for next cycle

### 4. Document Graph Integration

- Document nodes are added to knowledge graph with type "Doc"
- Entity nodes have type "Entity"
- "belong-to" relations connect entities to their source documents
- These relations are not saved in vector DB (only in graph)

### 5. Cluster Summary Generation

- Uses map-reduce approach to handle large document sets
- Splits documents into batches within token limits
- Processes batches in parallel with LLM
- Combines batch summaries into final cluster summary

### 6. Enhanced Querying

- Queries can include clustering context
- System finds similar clusters to query
- Results include cluster information and keywords
- Enables topic-aware document retrieval

## API Reference

### Core Methods

#### Document Processing

- `insert_document(doc_id, content, chunk_size=1000)`: Process single document
- `insert_document_with_llm(doc_id, content)`: Process with LLM extraction and clustering
- `insert_documents_batch(documents, chunk_size=1000, max_concurrent_docs=5)`: Batch processing
- `insert_documents_batch_with_llm(documents, max_concurrent_docs=5)`: Batch with LLM and clustering

#### Clustering

- `initialize_clustering(initial_docs)`: Initialize clustering model
- `process_clustering(documents)`: Process documents through clustering pipeline
- `generate_cluster_summaries()`: Generate summaries for all clusters
- `get_documents_by_cluster(cluster_id)`: Get documents in specific cluster
- `get_clustering_statistics()`: Get clustering performance metrics
- `force_cluster_update()`: Force model update
- `query_with_clustering(query, top_k=10, use_clusters=True)`: Enhanced querying

#### Querying

- `query_entities(query, top_k=10)`: Search entities
- `query_relations(query, top_k=10)`: Search relations
- `get_entity_neighbors(entity_name)`: Get entity neighbors in graph
- `get_entity_graph_context(entity_name, max_depth=2)`: Get entity context

### Clustering Classes

#### DynamicClusteringManager

```python
from graphrag.core.clustering_manager import DynamicClusteringManager

# Initialize clustering manager
clustering_manager = DynamicClusteringManager(
    embedding_client=embedding_client,
    working_dir="data",
    update_threshold=10,
    min_cluster_size=5,
    min_samples=2
)

# Initialize model
await clustering_manager.initialize_model(initial_docs)

# Process document
result = await clustering_manager.process_document("doc_1", "document content")
```

#### ClusterSummaryGenerator

```python
from graphrag.core.cluster_summary_generator import ClusterSummaryGenerator

# Initialize summary generator
summary_generator = ClusterSummaryGenerator(
    llm_client=llm_client,
    max_tokens_per_batch=4000,
    max_concurrent_batches=3
)

# Generate summaries
summaries = await summary_generator.generate_cluster_summaries(
    cluster_documents=cluster_docs,
    cluster_keywords=cluster_keywords
)
```

## Examples

### Complete Demo

Run the comprehensive clustering demo:

```bash
python examples/demo_clustering_integration.py
```

This demo shows:

- Document insertion with LLM extraction
- Dynamic clustering with outlier detection
- Cluster summary generation
- Clustering-aware querying
- Document retrieval by cluster

### CLI Usage

```bash
# Basic document insertion
python -m graphrag.cli insert --documents doc1.txt doc2.txt

# With clustering
python -m graphrag.cli insert --documents doc1.txt doc2.txt --enable-clustering

# Query with clustering
python -m graphrag.cli query "machine learning" --use-clusters

# Generate summaries
python -m graphrag.cli summarize --clusters
```

## Performance Considerations

### Clustering Performance

- **Model Updates**: Triggered by outlier threshold (default: 10)
- **Batch Processing**: Documents processed concurrently with semaphore limits
- **Memory Usage**: BERTopic models can be memory-intensive for large document sets
- **Embedding Caching**: Embeddings are cached to avoid recomputation

### Summary Generation

- **Map-Reduce**: Handles large document sets efficiently
- **Token Limits**: Configurable batch sizes to stay within LLM limits
- **Parallel Processing**: Multiple batches processed concurrently
- **Error Handling**: Graceful degradation if LLM calls fail

### Storage

- **Model Persistence**: BERTopic models saved to disk
- **Graph Structure**: Document nodes and relations stored in NetworkX
- **Vector Storage**: Chunks, entities, and relations in vector DB
- **Status Tracking**: Document processing status in JSON storage

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch sizes or use smaller embedding models
2. **Slow Clustering**: Adjust update threshold or use GPU acceleration
3. **LLM Errors**: Check API keys and rate limits
4. **Model Loading**: Ensure BERTopic model files are accessible

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
