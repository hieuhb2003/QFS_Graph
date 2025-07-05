# Cluster Summary Querying - GraphRAG System

## Tổng quan

Tính năng Cluster Summary Querying cho phép query trên cluster summaries với 2 mode:

- **Retrieval Mode**: Chỉ lấy ra summaries phù hợp nhất từ vector DB
- **Generation Mode**: Lấy summaries từ vector DB rồi gen câu trả lời bằng LLM

## Luồng hoạt động

```
Query → Vector DB (top_k) → Cluster Summaries → Mode Processing → Result
                                    ↓
                            Retrieval: Return summaries
                            Generation: LLM → Answer
```

## API Methods

### Query với Mode

```python
# Query cluster summaries với mode cụ thể
result = await system.query_cluster_summaries(
    query="artificial intelligence",
    mode="retrieval",  # hoặc "generation"
    top_k=5
)
```

### Retrieval Mode

```python
# Chỉ lấy ra summaries phù hợp nhất từ vector DB
result = await system.query_cluster_summaries(
    query="technology companies",
    mode="retrieval",
    top_k=3
)

# Kết quả:
{
    "mode": "retrieval",
    "query": "technology companies",
    "results": [
        {
            "cluster_id": 0,
            "summary": "Apple Inc. is a technology company...",
            "doc_hash_ids": ["doc-abc123", "doc-def456"],
            "score": 0.85
        },
        # ... more results
    ],
    "total_found": 3
}
```

### Generation Mode

```python
# Gen câu trả lời từ summaries từ vector DB
result = await system.query_cluster_summaries(
    query="What are the main technology companies?",
    mode="generation",
    top_k=3
)

# Kết quả:
{
    "mode": "generation",
    "query": "What are the main technology companies?",
    "answer": "Based on the cluster summaries, the main technology companies mentioned include...",
    "used_summaries": [
        {
            "cluster_id": 0,
            "summary": "Apple Inc. is a technology company...",
            "doc_hash_ids": ["doc-abc123"],
            "score": 0.85
        },
        # ... more summaries used
    ],
    "total_found": 3
}
```

## Cấu hình

### Prompt Templates

```python
global_config = {
    "summary": {
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
```

### LLM Client Setup

```python
# Cho generation mode, cần LLM client
llm_client = create_llm_client(
    client_type="openai",
    model_name="gpt-3.5-turbo",
    api_key="your-api-key"
)

# Hoặc vLLM
llm_client = create_llm_client(
    client_type="vllm",
    model_name="llama2-7b-chat",
    url="http://localhost:8000/v1",
    api_key="dummy"
)
```

## Demo Usage

### Chạy Demo

```bash
# Chạy demo riêng lẻ
python examples/demo_cluster_query.py

# Hoặc từ menu chính
python run_all_demos.py
# Chọn option 6: Cluster Summary Query Demo
```

### Demo Scenarios

1. **Retrieval Mode Examples**:

   - "technology companies"
   - "artificial intelligence"
   - "machine learning"
   - "Apple and Microsoft"

2. **Generation Mode Examples**:
   - "What are the main technology companies mentioned?"
   - "Explain the relationship between AI and machine learning"
   - "What are the key products of Apple and Microsoft?"
   - "How does deep learning relate to computer vision?"

## Implementation Details

### GraphRAGSystem Integration

```python
class GraphRAGSystem:
    async def query_cluster_summaries(self,
                                    query: str,
                                    top_k: int = 5,
                                    mode: str = "retrieval") -> Dict[str, Any]:
        """Query cluster summaries với 2 mode: retrieval và generation"""

        # 1. Query từ vector DB để lấy top_k results
        cluster_summaries = await self._cluster_summary_db.query(query, top_k)

        if mode == "retrieval":
            # Mode retrieval: chỉ trả về summaries
            results = []
            for summary in cluster_summaries:
                results.append({
                    "cluster_id": summary.get("cluster_id"),
                    "summary": summary.get("summary_text"),
                    "doc_hash_ids": summary.get("doc_hash_ids", []),
                    "score": summary.get("distance", 0.0)
                })

            return {
                "mode": "retrieval",
                "query": query,
                "results": results,
                "total_found": len(results)
            }

        elif mode == "generation":
            # Mode generation: gen câu trả lời bằng LLM
            # Tạo prompt và gọi LLM
            prompt = self.cluster_summary_generator.query_generation_prompt_template.format(
                query=query,
                summaries=summaries_text
            )

            answer = await self.llm_client.generate(prompt)

            return {
                "mode": "generation",
                "query": query,
                "answer": answer.strip(),
                "used_summaries": used_summaries,
                "total_found": len(used_summaries)
            }
```

## Error Handling

### Fallback Mechanisms

1. **No LLM Client**: Tự động fallback về retrieval mode
2. **No Summaries**: Trả về thông báo "Không tìm thấy thông tin"
3. **LLM Errors**: Trả về error message và used summaries

### Error Response Format

```python
{
    "error": "Error message",
    "mode": "generation",
    "query": "original query",
    "answer": "Error when generating answer: error message",
    "used_summaries": []
}
```

## Performance Considerations

### 1. Vector Search Optimization

- Sử dụng vector DB để lấy top_k results
- Semantic search với embeddings
- Efficient similarity computation

### 2. LLM Generation Optimization

- Parallel processing cho multiple queries
- Context length management
- Prompt optimization cho better responses

### 3. Memory Management

- Efficient data structures
- Cleanup unused resources
- Batch processing cho large datasets

## Use Cases

### 1. Information Retrieval

```python
# Tìm clusters liên quan đến topic
result = await system.query_cluster_summaries(
    query="machine learning algorithms",
    mode="retrieval",
    top_k=5
)
```

### 2. Question Answering

```python
# Trả lời câu hỏi từ cluster summaries
result = await system.query_cluster_summaries(
    query="How do neural networks work?",
    mode="generation",
    top_k=3
)
```

### 3. Content Analysis

```python
# Phân tích nội dung theo clusters
result = await system.query_cluster_summaries(
    query="technology trends",
    mode="generation",
    top_k=5
)
```

## Best Practices

### 1. Query Formulation

- Sử dụng từ khóa cụ thể cho retrieval mode
- Sử dụng câu hỏi rõ ràng cho generation mode
- Tránh queries quá dài hoặc mơ hồ

### 2. Top-k Selection

- `top_k=3-5` cho generation mode
- `top_k=5-10` cho retrieval mode
- Điều chỉnh theo độ phức tạp của query

### 3. LLM Configuration

- Sử dụng model phù hợp với task
- Tối ưu prompt templates
- Monitor response quality

## Troubleshooting

### Common Issues

1. **No Results**: Kiểm tra cluster summaries đã được tạo chưa
2. **Poor Quality**: Điều chỉnh prompt template hoặc top_k
3. **LLM Errors**: Kiểm tra API key và model availability
4. **Performance**: Giảm top_k hoặc sử dụng retrieval mode

### Debug Tips

```python
# Enable debug logging
logger = setup_logger(name="ClusterQuery", log_level="DEBUG")

# Check cluster summaries
cluster_info = await system.get_cluster_info()
print(f"Available clusters: {cluster_info['total_clusters']}")

# Test vector search directly
raw_results = await system._cluster_summary_db.query("test query", 5)
print(f"Raw search results: {len(raw_results)}")
```

## Future Enhancements

1. **Multi-modal Queries**: Hỗ trợ queries với images, audio
2. **Conversational Mode**: Hỗ trợ multi-turn conversations
3. **Advanced Filtering**: Filter theo metadata, time range
4. **Personalization**: User-specific query preferences
5. **Real-time Updates**: Live cluster updates và querying
