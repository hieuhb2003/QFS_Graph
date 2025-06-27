# GraphRAG Clustering System Documentation

## Tổng quan

Hệ thống Clustering được tích hợp vào GraphRAG để thực hiện phân cụm động các documents, tạo summaries cho từng cluster, và cập nhật knowledge graph với thông tin clustering.

## Kiến trúc tổng quan

```
Documents → Processing → Clustering → Summaries → Graph Updates
    ↓           ↓           ↓           ↓           ↓
  Insert    LLM Extract  BERTopic    LLM Map-    NetworkX
  Documents  Entities     Clustering   Reduce     Relations
```

## Luồng hoạt động chi tiết

### Phase 1: Document Processing

```
1. Insert Documents → Chunking → LLM Extraction → Entity/Relation Storage
2. Graph Update → Entity Nodes → Relation Edges → Document Nodes → Belong Relations
```

### Phase 2: Clustering

```
1. Get All Documents → Preprocessing → Embedding Creation
2. BERTopic Clustering → Topic Assignment → Outlier Detection
3. Cluster Info Creation → Save Model → Update Graph with Clusters
```

### Phase 3: Summary Generation

```
1. Get Cluster Documents → Content Mapping → Parallel Summary Generation
2. Map-Reduce Processing → LLM Summary Creation → Save to VDB
3. Update Clustering Manager → Save Cluster Info
```

### Phase 4: Graph Enhancement

```
1. Update Document Nodes → Add Cluster IDs
2. Create Same-Cluster Relations → Connect Documents in Same Cluster
3. Graph Traversal Ready → Query Capabilities
```

## Components chính

### 1. Data Structures (`src/graphrag/utils/utils.py`)

```python
@dataclass
class Document:
    doc_id: str
    content: str
    chunk_ids: List[str]
    entity_ids: List[str]
    relation_ids: List[str]

@dataclass
class ClusterInfo:
    cluster_id: int
    doc_hash_ids: List[str]        # Hash IDs của documents trong cluster
    outlier_doc_hash_ids: List[str] # Hash IDs của outliers
    summary: str = ""
    created_at: str = ""
    updated_at: str = ""
```

**Mục đích**: Định nghĩa cấu trúc dữ liệu cho documents và cluster information.

### 2. Clustering Manager (`src/graphrag/core/clustering_manager.py`)

#### Khởi tạo:

```python
clustering_manager = ClusteringManager(
    embedding_client=embedding_client,
    outlier_threshold=10,    # Ngưỡng outlier để tạo cluster mới
    max_tokens=8192,         # Số token tối đa
    batch_size=16            # Batch size cho embedding
)
```

#### Luồng Clustering:

**Initial Clustering:**

```python
async def cluster_documents(documents, doc_hash_ids):
    # 1. Preprocess documents
    processed_docs = self._preprocess_documents(documents)

    # 2. Create embeddings
    embeddings = await self._create_embeddings(processed_docs)

    # 3. Fit BERTopic model
    topics, probs = self.topic_model.fit_transform(processed_docs, embeddings)

    # 4. Update cluster assignments
    self._update_cluster_assignments(processed_docs, topics, doc_hash_ids)

    # 5. Create cluster info
    await self._create_cluster_info()

    # 6. Save model
    await self.save_clustering_model()
```

**Incremental Clustering:**

```python
async def update_clusters_with_new_data(new_documents, new_doc_hash_ids):
    # 1. Predict topics for new documents
    predicted_topics, _ = self.topic_model.transform(new_docs, new_embeddings)

    # 2. Classify as clustered or outliers
    for topic in predicted_topics:
        if topic == -1:  # Outlier
            outlier_docs.append(doc)
        else:  # Clustered
            clustered_docs.append(doc)

    # 3. Check outlier threshold
    if len(outliers) >= self.outlier_threshold:
        await self._create_new_clusters_from_outliers()

    # 4. Update cluster assignments
    self._update_cluster_assignments(new_docs, predicted_topics, new_doc_hash_ids)
```

**Outlier Management:**

```python
async def _create_new_clusters_from_outliers():
    # 1. Create new clustering model for outliers
    outlier_topic_model = BERTopic(umap_model=outlier_umap, hdbscan_model=outlier_hdbscan)

    # 2. Fit on outliers
    outlier_topics, _ = outlier_topic_model.fit_transform(self.outlier_docs, self.outlier_embeddings)

    # 3. Reclassify outliers
    for i, topic in enumerate(outlier_topics):
        if topic == -1:  # Still outlier
            remaining_outliers.append(self.outlier_docs[i])
        else:  # Now clustered
            new_clustered_outliers.append(self.outlier_docs[i])
```

### 3. Cluster Summary Generator (`src/graphrag/core/cluster_summary_generator.py`)

#### Luồng Map-Reduce Summary:

**Map Phase:**

```python
async def _map_phase(documents):
    # 1. Split documents into chunks based on context length
    chunks = self._split_documents_into_chunks(documents)

    # 2. Generate summary for each chunk
    for chunk in chunks:
        summary = await self._generate_chunk_summary(chunk)
        summaries.append(summary)

    return summaries
```

**Reduce Phase:**

```python
async def _reduce_phase(summaries):
    # 1. Create reduce prompt
    summaries_text = "\n\n".join([f"Summary {i+1}: {summary}" for i, summary in enumerate(summaries)])
    prompt = self.map_reduce_prompt_template.format(summaries=summaries_text)

    # 2. Generate final summary using LLM
    response = await self.llm_client.generate_text(prompt)

    return response.strip()
```

**Parallel Processing:**

```python
async def generate_cluster_summaries(cluster_documents, doc_content_map):
    # 1. Create semaphore for limiting workers
    semaphore = asyncio.Semaphore(self.max_workers)

    # 2. Create tasks for each cluster
    tasks = []
    for cluster_id, doc_hash_ids in cluster_documents.items():
        task = self._generate_single_cluster_summary(cluster_id, doc_hash_ids, doc_content_map, semaphore)
        tasks.append(task)

    # 3. Run parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return summaries
```

### 4. GraphRAG System Integration (`src/graphrag/core/graphrag_system.py`)

#### Khởi tạo với Clustering:

```python
def __init__(self, working_dir, embedding_client, global_config, llm_client=None):
    # ... existing initialization ...

    # Initialize clustering manager
    clustering_config = global_config.get('clustering', {})
    self.clustering_manager = ClusteringManager(
        embedding_client=embedding_client,
        outlier_threshold=clustering_config.get('outlier_threshold', 10),
        max_tokens=clustering_config.get('max_tokens', 8192),
        batch_size=clustering_config.get('batch_size', 16),
        model_save_path=os.path.join(working_dir, "clustering_models")
    )

    # Initialize cluster summary generator
    summary_config = global_config.get('summary', {})
    self.cluster_summary_generator = ClusterSummaryGenerator(
        llm_client=llm_client,
        max_workers=summary_config.get('max_workers', 4),
        context_length=summary_config.get('context_length', 4096)
    )

    # Initialize cluster summary VDB
    self._cluster_summary_db = NanoVectorDBStorage(
        namespace="cluster_summaries",
        embedding_func=self._embedding_wrapper,
        global_config=global_config,
        meta_fields=["cluster_id", "doc_hash_ids", "summary_text", "created_at"]
    )
```

#### Document Processing với Document Nodes:

```python
async def _update_graph(self, entities, relations):
    # ... existing entity and relation processing ...

    # Add document nodes and belong_to relations
    await self._add_document_nodes_and_belong_relations(entities, relations)

async def _add_document_nodes_and_belong_relations(self, entities, relations):
    # 1. Get unique doc_ids
    doc_ids = set()
    for entity in entities:
        doc_ids.add(entity.doc_id)

    # 2. Add document nodes
    for doc_id in doc_ids:
        doc_status = await self._doc_status_db.get_by_id(doc_id)
        if doc_status:
            doc_node_data = {
                "doc_id": doc_id,
                "content": doc_status.get("content", "")[:200] + "...",
                "type": "document",
                "cluster_id": None
            }
            await self._graph_db.upsert_node(doc_id, doc_node_data)

    # 3. Add belong_to relations
    for entity in entities:
        entity_normalized = normalize_entity_name(entity.entity_name)
        doc_id = entity.doc_id

        edge_data = {
            "relation_type": "belong_to",
            "entity_name": entity.entity_name,
            "doc_id": doc_id,
            "chunk_id": entity.chunk_id
        }
        await self._graph_db.upsert_edge(entity_normalized, doc_id, edge_data)
```

#### Clustering Workflow:

```python
async def cluster_documents(self, outlier_threshold=10):
    # 1. Get all successful documents
    all_docs = await self._doc_status_db.get_all()
    successful_docs = []
    doc_hash_ids = []

    for doc_id, doc_info in all_docs.items():
        if doc_info.get("status") == "success":
            content = doc_info.get("content", "")
            if content:
                successful_docs.append(content)
                doc_hash_ids.append(doc_id)

    # 2. Perform clustering
    clustering_result = await self.clustering_manager.cluster_documents(successful_docs, doc_hash_ids)

    # 3. Generate cluster summaries
    await self._generate_cluster_summaries(clustering_result)

    # 4. Update graph with cluster information
    await self._update_graph_with_clusters(clustering_result)

    return clustering_result
```

#### Graph Updates với Clusters:

```python
async def _update_graph_with_clusters(self, clustering_result):
    cluster_assignments = clustering_result.get("cluster_assignments", {})
    cluster_documents = clustering_result.get("cluster_documents", {})

    # 1. Update document nodes with cluster_id
    for doc_hash_id in cluster_assignments.keys():
        doc_status = await self._doc_status_db.get_by_id(doc_hash_id)
        if doc_status:
            doc_node_data = {
                "doc_id": doc_hash_id,
                "content": doc_status.get("content", "")[:200] + "...",
                "cluster_id": cluster_assignments.get(doc_hash_id, -1),
                "type": "document"
            }
            await self._graph_db.upsert_node(doc_hash_id, doc_node_data)

    # 2. Add has_same_cluster relations
    for cluster_id, doc_hash_ids in cluster_documents.items():
        if cluster_id == -1:  # Skip outlier cluster
            continue

        # Create relations between all documents in cluster
        for i, doc1 in enumerate(doc_hash_ids):
            for j, doc2 in enumerate(doc_hash_ids):
                if i < j:  # Avoid duplicates and self-relations
                    edge_data = {
                        "relation_type": "has_same_cluster",
                        "cluster_id": cluster_id,
                        "doc1": doc1,
                        "doc2": doc2
                    }
                    await self._graph_db.upsert_edge(doc1, doc2, edge_data)
```

#### Summary Storage:

```python
async def _save_cluster_summaries_to_vdb(self, summaries):
    summary_data = {}

    for cluster_id, summary in summaries.items():
        summary_key = f"cluster_summary_{cluster_id}"

        # Get doc_hash_ids of cluster
        doc_hash_ids = await self.clustering_manager.get_cluster_doc_ids(cluster_id)

        summary_data[summary_key] = {
            "content": summary,
            "cluster_id": cluster_id,
            "doc_hash_ids": doc_hash_ids,
            "summary_text": summary,
            "created_at": asyncio.get_event_loop().time()
        }

    if summary_data:
        await self._cluster_summary_db.upsert(summary_data)
```

## Data Flow

```
Documents (Input)
    ↓
Document Processing (Chunking + LLM Extraction)
    ↓
Entity/Relation Storage (Vector DB)
    ↓
Knowledge Graph (NetworkX)
    ↓
Clustering (BERTopic + UMAP + HDBSCAN)
    ↓
Cluster Info Storage (JSON + Pickle)
    ↓
Summary Generation (LLM Map-Reduce)
    ↓
Summary Storage (Vector DB)
    ↓
Graph Enhancement (Same-Cluster Relations)
    ↓
Query Interface (Ready for Use)
```

## API Methods

### Clustering Methods

```python
# Thực hiện clustering cho tất cả documents đã xử lý
await system.cluster_documents(outlier_threshold=10)

# Cập nhật clusters với documents mới
await system.update_clusters_with_new_data(new_documents)

# Lấy thông tin về clusters hiện tại
await system.get_cluster_info()

# Lấy hash doc IDs thuộc cluster cụ thể
await system.get_documents_by_cluster(cluster_id)

# Lấy hash doc IDs của cluster
await system.get_cluster_doc_ids(cluster_id)
```

### Summary Methods

```python
# Tạo summaries cho tất cả clusters
await system.generate_cluster_summaries(max_workers=4)

# Query cluster summaries từ vector DB
await system.query_cluster_summaries(query, top_k=5)
```

### Graph Methods

```python
# Lấy danh sách documents cùng cluster với document cho trước
await system.get_documents_with_same_cluster(doc_id)
```

## Cấu hình

### Clustering Configuration

```python
global_config = {
    "save_interval": 100,
    "clustering": {
        "outlier_threshold": 10,      # Ngưỡng outlier để tạo cluster mới
        "max_tokens": 8192,           # Số token tối đa cho mỗi document
        "batch_size": 16              # Batch size cho embedding
    },
    "summary": {
        "max_workers": 4,             # Số worker cho parallel summary generation
        "context_length": 4096        # Độ dài context tối đa cho LLM
    }
}
```

### Prompt Templates

```python
# Summary Prompt Template
summary_prompt_template = """
You are an expert document analyst. Please create a concise and accurate summary for the following documents:

Documents:
{documents}

Requirements:
1. Summarize the main content of the documents
2. Identify key topics and important concepts
3. Create a well-structured summary
4. Keep the summary under 300 words

Summary:
"""

# Map-Reduce Prompt Template
map_reduce_prompt_template = """
You are an expert document analyst. Please create a comprehensive summary from the following sub-summaries:

Sub-summaries:
{summaries}

Requirements:
1. Synthesize information from all sub-summaries
2. Remove duplicate information
3. Create a final well-structured summary
4. Keep the summary under 500 words
5. Ensure consistency and logical flow

Comprehensive Summary:
"""
```

## Key Features

1. **Incremental Processing**: Có thể thêm documents mới và update clusters
2. **Outlier Management**: Tự động tạo cluster mới khi đủ outliers
3. **Parallel Processing**: Xử lý song song cho summaries và embeddings
4. **Persistent Storage**: Lưu trữ clusters và summaries để tái sử dụng
5. **Graph Integration**: Tích hợp hoàn toàn với knowledge graph
6. **Flexible Configuration**: Cấu hình linh hoạt cho clustering và summary

## Dependencies

```txt
# Clustering dependencies
bertopic>=0.15.0
umap-learn>=0.5.0
hdbscan>=0.8.0
transformers>=4.20.0
```

## Testing

### Test Script

```bash
cd examples
python test_clustering.py
```

### Demo Script

```bash
cd examples
python demo_clustering_integration.py
```

## Storage Structure

```
working_dir/
├── clustering_models/
│   ├── bertopic_model/           # BERTopic model files
│   ├── clustering_data.pkl       # Clustering data
│   ├── cluster_history.json      # Clustering history
│   └── cluster_info.json         # Cluster information
├── cluster_summaries/            # Cluster summary VDB
├── entities/                     # Entity VDB
├── relations/                    # Relation VDB
├── chunks/                       # Chunk storage
├── doc_status/                   # Document status
└── knowledge_graph/              # NetworkX graph
```

## Performance Considerations

1. **Embedding Generation**: Sử dụng batch processing để tối ưu GPU/CPU usage
2. **Parallel Summary Generation**: Sử dụng semaphore để giới hạn concurrent LLM calls
3. **Memory Management**: Efficient data structures và cleanup
4. **Incremental Updates**: Chỉ update clusters khi cần thiết
5. **Caching**: Lưu trữ embeddings và models để tái sử dụng

## Troubleshooting

### Common Issues

1. **Out of Memory**: Giảm batch_size hoặc max_tokens
2. **LLM Timeout**: Tăng timeout hoặc giảm context_length
3. **Clustering Quality**: Điều chỉnh outlier_threshold và clustering parameters
4. **Summary Quality**: Tùy chỉnh prompt templates

### Logging

System sử dụng comprehensive logging cho clustering:

- Clustering progress và statistics
- Summary generation progress
- Error tracking và debugging
- Performance metrics

## Future Enhancements

1. **Advanced Clustering**: Hỗ trợ các algorithms clustering khác
2. **Dynamic Thresholds**: Tự động điều chỉnh outlier threshold
3. **Cluster Evolution**: Tracking cluster changes over time
4. **Interactive Clustering**: Manual cluster adjustment
5. **Multi-modal Clustering**: Hỗ trợ images và other data types
