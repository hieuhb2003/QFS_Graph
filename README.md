# GraphRAG System

Hệ thống GraphRAG tích hợp với NetworkX và Vector Database để xử lý documents, extract entities và relations, và xây dựng knowledge graph.

## Tính năng chính

- **Document Processing**: Chunking và tracking trạng thái xử lý documents
- **LLM Integration**: Hỗ trợ nhiều LLM clients (vLLM, OpenAI) cho entity/relation extraction
- **Embedding Support**: Nhiều embedding clients (Sentence Transformers, vLLM, OpenAI)
- **Vector Database**: Lưu trữ entities và relations với semantic search
- **Knowledge Graph**: Xây dựng graph với NetworkX, merge entities, track relations
- **Logging System**: Comprehensive logging với file và console output
- **Async Processing**: Multi-threaded processing cho performance tốt

## Cấu trúc project

```
next_work/
├── db/                          # Database implementations
│   ├── __init__.py
│   ├── json_doc_status_impl.py  # Document status tracking
│   ├── json_kv_impl.py          # Key-value storage
│   ├── nano_vector_db_impl.py   # Vector database
│   └── networkx_impl.py         # Knowledge graph storage
├── logger_config.py             # Logging configuration
├── llm_client.py                # LLM clients (vLLM, OpenAI)
├── embedding_client.py          # Embedding clients (Sentence Transformers, vLLM, OpenAI)
├── llm_extractor.py             # Entity/relation extraction
├── operators.py                 # Document and query operators
├── graphrag_system.py           # Main system integration
├── demo_graphrag.py             # Demo script
├── utils.py                     # Utility functions
└── requirements.txt             # Dependencies
```

## Cài đặt

1. Clone repository:

```bash
git clone <repository-url>
cd next_work
```

2. Cài đặt dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Cài đặt CUDA support cho GPU acceleration:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Sử dụng

### 1. Khởi tạo hệ thống

```python
import asyncio
from logger_config import setup_logger
from graphrag_system import GraphRAGSystem
from llm_client import create_llm_client
from embedding_client import create_embedding_client

# Setup logger
logger = setup_logger(
    name="GraphRAG",
    log_level="INFO",
    log_dir="./logs"
)

# Tạo embedding client
embedding_client = create_embedding_client(
    client_type="sentence_transformers",
    model_name="all-MiniLM-L6-v2",
    device="cuda"  # hoặc "cpu"
)

# Tạo LLM client (optional)
llm_client = create_llm_client(
    client_type="vllm",
    model_name="llama2-7b-chat",
    url="http://localhost:8000/v1",
    api_key="dummy"
)

# Khởi tạo hệ thống
system = GraphRAGSystem(
    working_dir="./graphrag_data",
    embedding_client=embedding_client,
    global_config={"save_interval": 100},
    llm_client=llm_client
)
```

### 2. Insert documents

```python
# Insert với LLM extraction (one-shot)
success = await system.insert_document_with_llm(
    doc_id="doc1",
    content="Apple Inc. is a technology company that designs and manufactures consumer electronics..."
)

# Hoặc insert với chunking
success = await system.insert_document(
    doc_id="doc2",
    content="Microsoft Corporation is a multinational technology company...",
    chunk_size=1000
)

# Batch processing - xử lý nhiều documents song song
documents_batch = [
    "Apple Inc. is a technology company that designs and manufactures consumer electronics...",
    "Microsoft Corporation is a multinational technology company...",
    "Google LLC is a technology company that specializes in internet-related services..."
]

# Batch processing với LLM one-shot
results = await system.insert_documents_batch_with_llm(
    documents=documents_batch,
    max_concurrent_docs=5  # Chạy song song tối đa 5 documents
)

# Batch processing với chunking
results = await system.insert_documents_batch(
    documents=documents_batch,
    chunk_size=1000,
    max_concurrent_docs=5  # Chạy song song tối đa 5 documents
)
```

### 3. Query dữ liệu

```python
# Query entities
entities = await system.query_entities("technology company", top_k=10)

# Query relations
relations = await system.query_relations("founded", top_k=10)

# Search entities by name
apple_entities = await system.search_entities_by_name("Apple")

# Get entity graph context
context = await system.get_entity_graph_context("Apple")

# Get system statistics
stats = await system.get_system_stats()
```

### 4. Cleanup

```python
await system.cleanup()
```

## Cấu hình

### LLM Clients

#### vLLM Client (Self-hosted)

```python
llm_client = create_llm_client(
    client_type="vllm",
    model_name="llama2-7b-chat",
    url="http://localhost:8000/v1",
    api_key="dummy",
    max_retries=3,
    timeout=30
)
```

#### OpenAI Client

```python
llm_client = create_llm_client(
    client_type="openai",
    model_name="gpt-3.5-turbo",
    api_key="your-api-key",
    organization="your-org-id",  # optional
    max_retries=3,
    timeout=30
)
```

### Embedding Clients

#### Sentence Transformers (GPU acceleration)

```python
embedding_client = create_embedding_client(
    client_type="sentence_transformers",
    model_name="all-MiniLM-L6-v2",  # hoặc "all-mpnet-base-v2"
    device="cuda",  # hoặc "cpu"
    max_retries=3
)
```

#### vLLM Embedding

```python
embedding_client = create_embedding_client(
    client_type="vllm",
    model_name="llama2-7b-chat",
    url="http://localhost:8000/v1",
    api_key="dummy",
    max_retries=3,
    timeout=30
)
```

#### OpenAI Embedding

```python
embedding_client = create_embedding_client(
    client_type="openai",
    model_name="text-embedding-ada-002",
    api_key="your-api-key",
    organization="your-org-id",  # optional
    max_retries=3,
    timeout=30
)
```

### Logger Configuration

```python
logger = setup_logger(
    name="GraphRAG",
    log_level="INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_dir="./logs",
    log_file="graphrag.log"  # optional
)
```

## Demo

Chạy demo để xem hệ thống hoạt động:

```bash
python demo_graphrag.py
```

Demo sẽ:

1. Khởi tạo hệ thống với LLM và embedding clients
2. Insert sample documents
3. Thực hiện các query examples
4. Hiển thị system statistics
5. Cleanup resources

## Tính năng nâng cao

### Document Status Tracking

- Tracking trạng thái: pending, success, failed
- Lưu metadata: entities count, relations count, processing time
- Error tracking và retry logic

### Entity/Relation Extraction

- One-shot LLM extraction cho toàn bộ document
- Chunk-based extraction với parallel processing
- Format validation và error handling
- Fallback to regex nếu không có LLM client

### Knowledge Graph

- Merge entities với cùng normalized name
- Combine descriptions và chunk sources
- Store relations as edges với metadata
- Graph traversal và context retrieval

### Vector Database

- Semantic search cho entities và relations
- Metadata filtering và retrieval
- Batch processing và optimization
- Hash prefixing cho key management

## Performance

- **Async Processing**: Multi-threaded LLM calls và embedding
- **Batch Operations**: Batch embedding và database operations
- **Parallel Document Processing**: Multiple documents processed concurrently
- **Parallel Chunk Processing**: Chunks within each document processed in parallel
- **GPU Acceleration**: Support cho CUDA với Sentence Transformers
- **Memory Management**: Efficient data structures và cleanup
- **Concurrency Control**: Configurable limits for concurrent processing

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Giảm batch size hoặc sử dụng CPU
2. **LLM timeout**: Tăng timeout hoặc giảm max_tokens
3. **Embedding errors**: Check model name và device configuration
4. **Database errors**: Verify working directory permissions

### Logging

System sử dụng comprehensive logging:

- Console output cho real-time monitoring
- File logging cho debugging và analysis
- Different log levels cho different use cases
- Error tracking với stack traces

## Contributing

1. Fork repository
2. Create feature branch
3. Add tests cho new features
4. Update documentation
5. Submit pull request

## License

MIT License

Examples:

# Run demo

graphrag demo

# Process a document

graphrag process --input document.txt --output ./data

# Process multiple documents in batch (parallel)

graphrag process --input ./documents/ --output ./data --batch --max-concurrent 5

# Query entities

graphrag query --entities "Apple Inc" --data ./data
