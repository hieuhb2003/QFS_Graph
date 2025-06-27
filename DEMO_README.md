# GraphRAG System - Demo Scripts

Tài liệu hướng dẫn sử dụng các demo scripts của GraphRAG system.

## 📋 Tổng quan

GraphRAG system cung cấp nhiều demo scripts để test và minh họa các chức năng khác nhau:

1. **Quick Test** - Test nhanh tất cả chức năng cơ bản
2. **Full Workflow** - Demo đầy đủ với LLM extraction
3. **Simple Workflow** - Demo không cần LLM
4. **Clustering Integration** - Demo clustering system
5. **GraphRAG Basic** - Demo cơ bản của GraphRAG

## 🚀 Cách chạy

### Chạy tất cả demo

```bash
python run_all_demos.py
```

### Chạy từng demo riêng lẻ

#### 1. Quick Test

```bash
python examples/demo_quick_test.py
```

- **Mô tả**: Test nhanh tất cả chức năng cơ bản
- **Thời gian**: ~2-3 phút
- **Yêu cầu**: Không cần LLM
- **Chức năng**: Document processing, entity querying, clustering, summaries

#### 2. Full Workflow

```bash
python examples/demo_full_workflow.py
```

- **Mô tả**: Demo đầy đủ với LLM extraction
- **Thời gian**: ~5-10 phút
- **Yêu cầu**: OpenAI API key
- **Chức năng**: Tất cả chức năng + LLM entity extraction

#### 3. Simple Workflow

```bash
python examples/demo_full_workflow.py
# Chọn option 2 khi được hỏi
```

- **Mô tả**: Demo không cần LLM
- **Thời gian**: ~3-5 phút
- **Yêu cầu**: Không cần LLM
- **Chức năng**: Document processing, clustering, basic queries

#### 4. Clustering Integration

```bash
python examples/demo_clustering_integration.py
```

- **Mô tả**: Demo clustering system
- **Thời gian**: ~3-5 phút
- **Yêu cầu**: Không cần LLM
- **Chức năng**: Clustering, incremental updates, outlier management

#### 5. GraphRAG Basic

```bash
python examples/demo_graphrag.py
```

- **Mô tả**: Demo cơ bản của GraphRAG
- **Thời gian**: ~2-3 phút
- **Yêu cầu**: Không cần LLM
- **Chức năng**: Basic document processing, entity storage

## 📊 Luồng hoạt động

### 1. Document Processing

```
Documents → Chunking → Vector Embeddings → Storage
```

### 2. Entity Extraction (với LLM)

```
Documents → LLM Processing → Entities & Relations → Knowledge Graph
```

### 3. Clustering

```
Documents → Embeddings → BERTopic → HDBSCAN → Clusters
```

### 4. Cluster Summaries (với LLM)

```
Clusters → LLM Map-Reduce → Cluster Summaries → Vector Storage
```

### 5. Querying

```
Query → Vector Search → Entities/Relations/Summaries → Results
```

## 🔧 Cấu hình

### Environment Variables

```bash
# Cho LLM functionality
export OPENAI_API_KEY="your-openai-api-key"

# Cho vLLM (optional)
export VLLM_MODEL_PATH="/path/to/model"
```

### System Configuration

```python
global_config = {
    "save_interval": 50,  # Save every 50 operations
    "clustering": {
        "outlier_threshold": 5,  # Outlier detection threshold
        "max_tokens": 4096,      # Max tokens for LLM
        "batch_size": 8          # Batch size for processing
    },
    "summary": {
        "max_workers": 3,        # Parallel workers for summaries
        "context_length": 2048   # Context length for LLM
    }
}
```

## 📁 Output Structure

Sau khi chạy demo, các thư mục sau sẽ được tạo:

```
demo_*_data/
├── chunks/              # Document chunks
├── entities/            # Entity embeddings
├── relations/           # Relation embeddings
├── summaries/           # Cluster summaries
├── graph/              # Knowledge graph
├── clusters/           # Clustering data
└── status/             # Processing status
```

## 🎯 Demo Scenarios

### Scenario 1: Technology Companies

- **Documents**: Apple, Microsoft, Google, Tesla, Amazon
- **Expected Clusters**: Technology companies, AI/ML topics
- **Entities**: Company names, founders, products
- **Relations**: Founded by, develops, specializes in

### Scenario 2: AI/ML Topics

- **Documents**: Machine learning, deep learning, NLP, computer vision
- **Expected Clusters**: AI/ML research areas
- **Entities**: AI concepts, algorithms, applications
- **Relations**: Subset of, uses, applies to

### Scenario 3: Outlier Detection

- **Documents**: Technology + Cooking, Travel, Fashion
- **Expected Clusters**: Technology cluster + outlier cluster
- **Outliers**: Non-technology topics

## 🔍 Query Examples

### Entity Queries

```python
# Find technology companies
entities = await system.query_entities("technology company", top_k=5)

# Find AI concepts
entities = await system.query_entities("artificial intelligence", top_k=3)
```

### Relation Queries

```python
# Find founding relationships
relations = await system.query_relations("founded", top_k=5)

# Find development relationships
relations = await system.query_relations("develops", top_k=3)
```

### Summary Queries

```python
# Find clusters about technology
summaries = await system.query_cluster_summaries("technology", top_k=2)

# Find clusters about AI
summaries = await system.query_cluster_summaries("artificial intelligence", top_k=2)
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**

   ```bash
   # Make sure you're in the project root
   cd /path/to/next_work
   python run_all_demos.py
   ```

2. **LLM Errors**

   ```bash
   # Check OpenAI API key
   echo $OPENAI_API_KEY

   # Or set it
   export OPENAI_API_KEY="your-key"
   ```

3. **Memory Issues**

   ```python
   # Reduce batch sizes in config
   "batch_size": 4,  # Instead of 8
   "max_workers": 2, # Instead of 3
   ```

4. **Clustering Issues**
   ```python
   # Adjust outlier threshold
   "outlier_threshold": 3,  # More strict
   # or
   "outlier_threshold": 10, # More lenient
   ```

### Performance Tips

1. **Use CPU for small datasets**

   ```python
   device="cpu"  # Faster for small data
   ```

2. **Use GPU for large datasets**

   ```python
   device="cuda"  # Better for large data
   ```

3. **Adjust save intervals**
   ```python
   "save_interval": 100,  # Less frequent saves
   ```

## 📈 Expected Results

### Quick Test Results

- ✅ 11 documents processed
- ✅ 3-4 clusters created
- ✅ 5-10 entities found
- ✅ 2-5 relations found
- ✅ Cluster summaries generated (if LLM available)

### Full Workflow Results

- ✅ 17 documents processed
- ✅ 4-6 clusters created
- ✅ 15-25 entities extracted
- ✅ 10-20 relations extracted
- ✅ Incremental clustering tested
- ✅ Graph analysis completed

## 🎉 Success Indicators

- ✅ Documents processed successfully
- ✅ Clusters created with reasonable sizes
- ✅ Entities and relations found
- ✅ Summaries generated (if LLM available)
- ✅ No errors in logs
- ✅ Cleanup completed successfully

## 📚 Next Steps

Sau khi chạy demo thành công, bạn có thể:

1. **Explore the codebase**: Xem source code trong `src/graphrag/`
2. **Customize the system**: Thay đổi config và prompts
3. **Add your own data**: Sử dụng documents của bạn
4. **Extend functionality**: Thêm features mới
5. **Read documentation**: Xem `README.md` và `CLUSTERING_README.md`

## 🤝 Support

Nếu gặp vấn đề:

1. Kiểm tra logs để tìm lỗi
2. Đảm bảo đã cài đặt đúng dependencies
3. Kiểm tra environment variables
4. Thử chạy quick test trước
5. Tạo issue trên GitHub nếu cần

---

**Happy exploring! 🚀**
