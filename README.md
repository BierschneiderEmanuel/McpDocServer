# MCP VectorStore Server

A Model Context Protocol (MCP) server that provides advanced vector store operations for document search, PDF processing, and information retrieval. This server wraps the functionality from `vectorstore.py` into a standardized MCP interface.

## Features

- **Vector Store Operations**: Create, search, and manage document vector stores
- **PDF Processing**: Extract and index content from PDF documents using LLMSherpa
- **Semantic Search**: Advanced document search using HuggingFace embeddings
- **Web Search Integration**: Google, Wikipedia, and DuckDuckGo search capabilities
- **File Operations**: Read and process local files
- **Mathematical Calculations**: Built-in calculator functionality

## Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: Minimum 4GB RAM (8GB+ recommended for large document collections)
- **Storage**: At least 2GB free space for models and vector stores
- **Network**: Internet connection for downloading models and web searches

### Optional GPU Support

For improved performance with large document collections:
- **CUDA**: 11.8 or higher
- **GPU**: NVIDIA GPU with 4GB+ VRAM
- **cuDNN**: Compatible version for your CUDA installation

## Installation

### Step 1: Clone or Download the Repository

```bash
# If you have the files locally, navigate to the directory
cd /path/to/McpDocServer

# Or clone from a repository (if available)
# git clone <repository-url>
# cd McpDocServer
```

### Step 2: Create a Virtual Environment

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

### Step 4: Install LLMSherpa (Optional but Recommended)

For optimal PDF processing, install LLMSherpa locally:

```bash
# Install LLMSherpa
pip install llmsherpa

# Start the LLMSherpa server (in a separate terminal)
llmsherpa --port 5001
```

### Step 5: Download Embedding Models

The server will automatically download the required embedding model on first use, but you can pre-download it:

```bash
# Download the embedding model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"
```

## Configuration

### Environment Variables

Create a `.env` file in the project directory:

```bash
# LLMSherpa API URL (use local if available, otherwise cloud)
LLMSHERPA_API_URL=http://localhost:5001/api/parseDocument?renderFormat=all

# Vector store directory
VECTORSTORE_DIR=/path/to/your/documents

# User agent for web scraping
USER_AGENT=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36

# Optional: CUDA device for GPU acceleration
CUDA_VISIBLE_DEVICES=0
```

### Directory Structure

Prepare your document directory:

```
your_documents/
├── pdfs/
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
├── text_files/
│   ├── notes.txt
│   └── ...
└── other_documents/
    └── ...
```

## Usage

### Starting the MCP Server

```bash
# Make the server executable
chmod +x mcp_vectorstore_server.py

# Start the server on linux
python /home/em/McpDocServer/mcp_vectorstore_server.py
or windows with wsl
wsl -d Ubuntu-24.04 bash -c "/mnt/c/Users/emanu/Desktop/McpDocServer/start_mcp.sh"
```
### Using with MCP Clients

#### 0. Claude Desktop

Add to your MCP configuration:

```json
{
  "mcpServers": {
    "vectorstore": {
      "command": "python",
      "args": ["/home/em/McpDocServer/mcp_vectorstore_server.py"],
      "env": {
        "PYTHONPATH": "/home/em/McpDocServer/McpDocServer"
      }
    }
  }
}
```

#### 1. GitHub Copilot

1) Click on Configure Tools in the GitHub Copilot Chat window:<br>
2) Click on Add More Tools in the top search bar.<br>
3) Click on Add MCP Server in the top search bar.<br>
4) Click on command (stdio) in the top search bar.<br>
5) Enter command to run:<br>
6) python /home/em/McpDocServer/mcp_vectorstore_server.py<br>
   or on windows:
   wsl -d Ubuntu-24.04 /mnt/c/Users/emanu/Desktop/McpDocServer/start_mcp.sh
7) Enter mcp server id / name e.g. McpDocServer-19be5552<br>
8) Configure settings.json<br>

```json
{
    "security.workspace.trust.untrustedFiles": "open",
    "python.defaultInterpreterPath": "/mnt/c/Users/emanu/Desktop/LLM/venv/venv/bin/python",
    "terminal.integrated.inheritEnv": false,
    "git.openRepositoryInParentFolders": "never",
    "terminal.integrated.scrollback": 100000,
    "mcp": {
        "servers": {
            "McpDocServer-19be5552": {
                "type": "stdio",
                "command": "python",
                "args": [
                    "/mnt/c/Users/emanu/Desktop/McpDocServer/mcp_vectorstore_server.py"
                ]
            }
        }
    }
}
```
9) Check if the following tools are available in the mcp server tool list when you click on Configure Tools in the GitHub Copilot Chat window and scroll to bottom:<br>
&nbsp;&nbsp;&nbsp;&nbsp;vectorstore_search<br>
&nbsp;&nbsp;&nbsp;&nbsp;vectorstore_create<br>
&nbsp;&nbsp;&nbsp;&nbsp;vectorstore_info<br>
&nbsp;&nbsp;&nbsp;&nbsp;vectorstore_clear<br>
&nbsp;&nbsp;&nbsp;&nbsp;read_file<br>
&nbsp;&nbsp;&nbsp;&nbsp;google_search<br>
&nbsp;&nbsp;&nbsp;&nbsp;wikipedia_search<br>
&nbsp;&nbsp;&nbsp;&nbsp;duckduckgo_search<br>
&nbsp;&nbsp;&nbsp;&nbsp;calculate<br>
10) Select Agent mode in GitHub Copilot Chat window and use vectorstore_search to get information:<br>
use vectorstore_search to get information on unit testing<br>
11)Confirm tool call usage.

#### 2. Continue MCP CLient
```
name: McpDocServer
version: 1.0.1
schema: v1
mcpServers:
  - name: McpDocServer
    command: wsl -d Ubuntu-24.04
    args:
      - "/mnt/c/Users/emanu/Desktop/McpDocServer/start_mcp.sh"
    env: {}
    mcp_timeout: 180 # set timeout to 180 sec
    timeout: 9999
    connectionTimeout: 120000  # 120 seconds = 2 minutes
```

#### 3. Other MCP Clients

Configure your MCP client to use the server:

```bash
# Example with a generic MCP client
mcp-client --server python --args /path/to/McpDocServer/mcp_vectorstore_server.py
```

## Available Tools

### Vector Store Operations

#### `vectorstore_search`
Search the vector store for relevant documents.

**Parameters:**
- `query` (string, required): Search query
- `k` (integer, optional): Number of results (default: 2)

**Example:**
```json
{
  "name": "vectorstore_search",
  "arguments": {
    "query": "machine learning algorithms",
    "k": 5
  }
}
```

#### `vectorstore_create`
Create a new vector store from documents in a directory.

**Parameters:**
- `directory_path` (string, required): Path to directory containing documents

**Example:**
```json
{
  "name": "vectorstore_create",
  "arguments": {
    "directory_path": "/home/user/documents/research_papers"
  }
}
```

#### `vectorstore_info`
Get information about the current vector store.

**Example:**
```json
{
  "name": "vectorstore_info",
  "arguments": {}
}
```

#### `vectorstore_clear`
Clear all documents from the vector store.

**Example:**
```json
{
  "name": "vectorstore_clear",
  "arguments": {}
}
```

### File Operations

#### `read_file`
Read the contents of a file on the system.

**Parameters:**
- `filename` (string, required): Path to the file to read

**Example:**
```json
{
  "name": "read_file",
  "arguments": {
    "filename": "/home/user/documents/notes.txt"
  }
}
```

### Web Search Operations

#### `google_search`
Search Google for information.

**Parameters:**
- `query` (string, required): Search query
- `max_results` (integer, optional): Maximum number of results (default: 3)

**Example:**
```json
{
  "name": "google_search",
  "arguments": {
    "query": "latest AI developments 2024",
    "max_results": 5
  }
}
```

#### `wikipedia_search`
Search Wikipedia for information.

**Parameters:**
- `query` (string, required): Search query

**Example:**
```json
{
  "name": "wikipedia_search",
  "arguments": {
    "query": "artificial intelligence"
  }
}
```

#### `duckduckgo_search`
Search DuckDuckGo for information.

**Parameters:**
- `query` (string, required): Search query

**Example:**
```json
{
  "name": "duckduckgo_search",
  "arguments": {
    "query": "privacy-focused search engines"
  }
}
```

### Utility Operations

#### `calculate`
Perform mathematical calculations.

**Parameters:**
- `operation` (string, required): Mathematical operation to perform

**Example:**
```json
{
  "name": "calculate",
  "arguments": {
    "operation": "2 + 2 * 3"
  }
}
```

## Resources

The server provides the following resources:

### `vectorstore://info`
Returns information about the current vector store in JSON format.

**Example Response:**
```json
{
  "num_documents": 150,
  "directory": "/home/user/documents",
  "embeddings_model": "sentence-transformers/all-mpnet-base-v2"
}
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
**Problem:** `ModuleNotFoundError` for various packages
**Solution:** Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

#### 2. CUDA/GPU Issues
**Problem:** CUDA-related errors
**Solution:** Install CPU-only versions:
```bash
pip uninstall faiss-gpu torch
pip install faiss-cpu
```

#### 3. LLMSherpa Connection Issues
**Problem:** Cannot connect to LLMSherpa API
**Solution:** 
- Start LLMSherpa server: `llmsherpa --port 5001`
- Or use cloud API by updating the URL in the code

#### 4. Memory Issues
**Problem:** Out of memory errors with large documents
**Solution:**
- Reduce chunk size in the text splitter
- Use smaller embedding models
- Process documents in batches

#### 5. Permission Issues
**Problem:** Cannot read files or directories
**Solution:** Check file permissions:
```bash
chmod 644 /path/to/documents/*
chmod 755 /path/to/documents/
```

### Performance Optimization

#### For Large Document Collections

1. **Use GPU acceleration:**
   ```python
   # In vectorstore.py, ensure CUDA is enabled
   model_kwargs={'device': 'cuda'}
   ```

2. **Optimize chunk size:**
   ```python
   # Adjust in PDFVectorStoreTool.__init__
   chunk_size=1000,  # Smaller chunks for better performance
   chunk_overlap=100,
   ```

3. **Batch processing:**
   ```python
   # Process documents in smaller batches
   batch_size = 10
   ```

#### For Better Search Results

1. **Adjust similarity threshold:**
   ```python
   # In vectorstore_search method
   similarity_threshold = 0.7
   ```

2. **Use different embedding models:**
   ```python
   # Try different models for better results
   model_name="sentence-transformers/all-MiniLM-L6-v2"  # Faster
   model_name="sentence-transformers/all-mpnet-base-v2"  # Better quality
   ```

## Development

### Project Structure

```
McpDocServer/
├── mcp_vectorstore_server.py  # Main MCP server
├── vectorstore.py             # Original vectorstore implementation
├── requirements.txt           # Python dependencies
├── README.md                 # This documentation
└── .env                      # Environment variables (create this)
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Testing

```bash
# Run basic functionality tests
python -c "
from mcp_vectorstore_server import *
print('Server imports successfully')
"

# Test vector store operations
python -c "
from vectorstore import PDFVectorStoreTool
tool = PDFVectorStoreTool()
print(f'Vector store initialized with {tool.vectorstore_get_num_items()} documents')
"
```

## License

This project is provided as-is for educational and research purposes. Please ensure you comply with the licenses of all included dependencies.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the error logs
3. Ensure all dependencies are correctly installed
4. Verify your system meets the requirements

## Changelog

### Version 1.0.0
- Initial release
- MCP server implementation
- Vector store operations
- Web search integration
- File operations
- Mathematical calculations
