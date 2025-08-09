# MCP VectorStore Server - Technical Documentation

## Overview

The MCP VectorStore Server is a Model Context Protocol (MCP) implementation that provides advanced document processing, vector storage, and search capabilities. It wraps the functionality from `vectorstore.py` into a standardized MCP interface, enabling seamless integration with MCP-compatible clients.

## Architecture

### Core Components

1. **MCP Server Layer** (`mcp_vectorstore_server.py`)
   - Handles MCP protocol communication
   - Manages tool registration and execution
   - Provides resource access

2. **Vector Store Engine** (`vectorstore.py`)
   - PDF document processing with LLMSherpa
   - FAISS vector database management
   - HuggingFace embeddings integration

3. **Search Integration**
   - Google search scraping
   - Wikipedia API integration
   - DuckDuckGo search API

4. **Utility Tools**
   - File reading operations
   - Mathematical calculations
   - Document management

### Data Flow

```
MCP Client → MCP Server → Tool Handler → Vector Store Engine → FAISS Database
                                    ↓
                              Search APIs (Google, Wikipedia, DuckDuckGo)
```

## Technical Specifications

### Dependencies

#### Core MCP
- `mcp>=1.0.0`: Model Context Protocol implementation

#### Vector Store & AI
- `langchain>=0.1.0`: LangChain framework
- `langchain-community>=0.0.20`: Community tools
- `faiss-cpu>=1.7.4`: Vector similarity search
- `sentence-transformers>=2.2.2`: Text embeddings
- `huggingface-hub>=0.19.0`: Model management

#### Document Processing
- `llmsherpa>=0.0.1`: PDF parsing and extraction
- `beautifulsoup4>=4.12.0`: HTML parsing
- `lxml>=4.9.0`: XML/HTML processing

#### Web & Search
- `requests>=2.31.0`: HTTP client
- `wikipedia-api>=0.6.0`: Wikipedia API
- `duckduckgo-search>=4.1.0`: DuckDuckGo search

#### Utilities
- `crewai-tools>=0.0.1`: AI tool framework
- `pydantic>=2.0.0`: Data validation
- `numpy>=1.24.0`: Numerical computing

### System Requirements

#### Minimum
- **Python**: 3.8+
- **RAM**: 4GB
- **Storage**: 2GB free space
- **Network**: Internet connection

#### Recommended
- **Python**: 3.9+
- **RAM**: 8GB+
- **Storage**: 5GB+ free space
- **GPU**: NVIDIA GPU with CUDA support

## Implementation Details

### MCP Server Implementation

```python
# Server initialization
server = mcp.server.Server("vectorstore-server")

# Tool registration
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    # Returns list of available tools with schemas

# Tool execution
@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    # Handles tool calls and returns results
```

### Vector Store Architecture

#### Document Processing Pipeline

1. **PDF Loading**: LLMSherpa extracts text and structure
2. **Text Splitting**: RecursiveCharacterTextSplitter chunks documents
3. **Embedding**: SentenceTransformers generates vector embeddings
4. **Storage**: FAISS indexes vectors for fast similarity search

#### Configuration Parameters

```python
# Text splitting configuration
chunk_size = 2000          # Characters per chunk
chunk_overlap = 200        # Overlap between chunks
separators = ["\n\n\n", "\n\n", "\n"]  # Split points

# Embedding configuration
model_name = "sentence-transformers/all-mpnet-base-v2"
device = "cuda"  # or "cpu"
normalize_embeddings = True
```

### Search Implementation

#### Vector Search
- **Algorithm**: FAISS similarity search
- **Distance Metric**: Cosine similarity
- **Results**: Top-k most similar documents

#### Web Search Integration

**Google Search**
- Custom scraping with BeautifulSoup
- User agent rotation
- Cookie management for rate limiting

**Wikipedia Search**
- Official Wikipedia API
- Summary extraction
- Error handling for missing pages

**DuckDuckGo Search**
- Official DuckDuckGo API
- Privacy-focused search
- Structured result parsing

## API Reference

### Tools

#### Vector Store Operations

##### `vectorstore_search`
```json
{
  "name": "vectorstore_search",
  "arguments": {
    "query": "string",
    "k": "integer (optional, default: 2)"
  }
}
```

**Implementation:**
```python
def vectorstore_search(self, query: str, k: int = 2) -> str:
    # 1. Generate query embedding
    # 2. Perform FAISS similarity search
    # 3. Retrieve and format results
    # 4. Return formatted text
```

##### `vectorstore_create`
```json
{
  "name": "vectorstore_create",
  "arguments": {
    "directory_path": "string"
  }
}
```

**Implementation:**
```python
def create_vector_store(self, directory_path: str) -> str:
    # 1. Scan directory for documents
    # 2. Process PDFs with LLMSherpa
    # 3. Split text into chunks
    # 4. Generate embeddings
    # 5. Store in FAISS index
```

#### Search Operations

##### `google_search`
```json
{
  "name": "google_search",
  "arguments": {
    "query": "string",
    "max_results": "integer (optional, default: 3)"
  }
}
```

**Implementation:**
```python
def get_google_content(text, max_results=3):
    # 1. Construct search URL
    # 2. Set headers and cookies
    # 3. Parse HTML with BeautifulSoup
    # 4. Extract titles, links, descriptions
    # 5. Return formatted results
```

### Resources

#### `vectorstore://info`
Returns JSON information about the vector store:

```json
{
  "num_documents": 150,
  "directory": "/path/to/documents",
  "embeddings_model": "sentence-transformers/all-mpnet-base-v2"
}
```

## Performance Optimization

### Memory Management

#### Chunk Size Optimization
- **Small documents**: 1000-1500 characters
- **Large documents**: 2000-3000 characters
- **Overlap**: 10-15% of chunk size

#### Batch Processing
```python
# Process documents in batches
batch_size = 10
for batch in document_batches:
    process_batch(batch)
    clear_memory()
```

### GPU Acceleration

#### CUDA Configuration
```python
# Enable GPU acceleration
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True}
```

#### Memory Optimization
```python
# Monitor GPU memory
import torch
torch.cuda.empty_cache()
```

### Search Performance

#### Index Optimization
- **Quantization**: Use FAISS IVF indices for large datasets
- **Compression**: Enable vector compression
- **Caching**: Cache frequently accessed embeddings

#### Query Optimization
- **Preprocessing**: Clean and normalize queries
- **Filtering**: Apply metadata filters
- **Ranking**: Implement custom ranking algorithms

## Security Considerations

### File System Access
- **Path validation**: Sanitize file paths
- **Permission checks**: Verify read permissions
- **Directory traversal**: Prevent path traversal attacks

### Network Security
- **Rate limiting**: Implement request throttling
- **User agents**: Rotate user agents for web scraping
- **Error handling**: Don't expose sensitive information

### Data Privacy
- **Local storage**: Keep embeddings local
- **No logging**: Don't log sensitive document content
- **Access control**: Implement proper access controls

## Error Handling

### Common Error Types

#### Import Errors
```python
try:
    import mcp.server
except ImportError:
    print("MCP library not installed")
    sys.exit(1)
```

#### File System Errors
```python
try:
    with open(filename, 'r') as f:
        content = f.read()
except FileNotFoundError:
    return "File not found"
except PermissionError:
    return "Permission denied"
```

#### Network Errors
```python
try:
    response = requests.get(url, timeout=10)
except requests.RequestException as e:
    return f"Network error: {str(e)}"
```

### Error Recovery

#### Graceful Degradation
- **LLMSherpa unavailable**: Fall back to cloud API
- **GPU unavailable**: Use CPU processing
- **Search API down**: Return cached results

#### Retry Logic
```python
import time
from functools import wraps

def retry_on_failure(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(delay)
            return None
        return wrapper
    return decorator
```

## Testing Strategy

### Unit Tests
- **Tool functionality**: Test each tool independently
- **Error handling**: Test error conditions
- **Edge cases**: Test boundary conditions

### Integration Tests
- **MCP protocol**: Test server-client communication
- **Vector store**: Test end-to-end document processing
- **Search APIs**: Test external service integration

### Performance Tests
- **Load testing**: Test with large document collections
- **Memory profiling**: Monitor memory usage
- **Response time**: Measure tool execution time

## Deployment

### Production Considerations

#### Environment Setup
```bash
# Use production Python environment
python3 -m venv /opt/mcp-vectorstore/venv
source /opt/mcp-vectorstore/venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export PYTHONPATH=/opt/mcp-vectorstore
export CUDA_VISIBLE_DEVICES=0
```

#### Process Management
```bash
# Use systemd for service management
sudo systemctl enable mcp-vectorstore
sudo systemctl start mcp-vectorstore
```

#### Monitoring
- **Logs**: Monitor application logs
- **Metrics**: Track performance metrics
- **Health checks**: Implement health check endpoints

### Scaling Considerations

#### Horizontal Scaling
- **Multiple instances**: Run multiple server instances
- **Load balancing**: Distribute requests across instances
- **Shared storage**: Use shared vector store storage

#### Vertical Scaling
- **More RAM**: Increase memory allocation
- **Better GPU**: Use more powerful GPUs
- **SSD storage**: Use fast storage for vector databases

## Maintenance

### Regular Tasks

#### Model Updates
- **Embedding models**: Update to newer versions
- **Dependencies**: Keep dependencies up to date
- **Security patches**: Apply security updates

#### Data Management
- **Vector store cleanup**: Remove outdated documents
- **Index optimization**: Rebuild indices periodically
- **Backup**: Backup vector store data

#### Performance Monitoring
- **Resource usage**: Monitor CPU, memory, GPU usage
- **Response times**: Track tool execution times
- **Error rates**: Monitor error frequencies

### Troubleshooting

#### Common Issues

**High Memory Usage**
```bash
# Check memory usage
ps aux | grep python
free -h

# Restart with more memory
export PYTHONMALLOC=malloc
python mcp_vectorstore_server.py
```

**Slow Search Performance**
```python
# Optimize search parameters
k = min(k, 10)  # Limit results
similarity_threshold = 0.7  # Filter low-quality matches
```

**GPU Issues**
```bash
# Check GPU status
nvidia-smi

# Use CPU fallback
export CUDA_VISIBLE_DEVICES=""
```

## Future Enhancements

### Planned Features

1. **Multi-modal Support**
   - Image embedding and search
   - Audio document processing
   - Video content analysis

2. **Advanced Search**
   - Semantic similarity search
   - Multi-language support
   - Contextual search

3. **Integration Improvements**
   - REST API endpoints
   - WebSocket support
   - GraphQL interface

4. **Performance Optimizations**
   - Distributed vector stores
   - Streaming document processing
   - Real-time indexing

### Architecture Evolution

#### Microservices
- **Service decomposition**: Split into microservices
- **API gateway**: Centralized API management
- **Service mesh**: Inter-service communication

#### Cloud Native
- **Containerization**: Docker containerization
- **Kubernetes**: Container orchestration
- **Cloud storage**: Cloud-based vector storage

## Conclusion

The MCP VectorStore Server provides a robust, scalable solution for document processing and search. Its modular architecture, comprehensive error handling, and performance optimizations make it suitable for both development and production environments.

The server successfully bridges the gap between traditional document processing and modern AI-powered search, providing users with powerful tools for information retrieval and knowledge management. 