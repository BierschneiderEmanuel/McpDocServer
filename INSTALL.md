# Quick Installation Guide

## Prerequisites

- Python 3.8+
- 4GB+ RAM
- Internet connection

## Quick Start (Linux/macOS)

1. **Clone or download the files to your system**

2. **Run the automated setup:**
   ```bash
   chmod +x start_server.sh
   ./start_server.sh
   ```

   This script will:
   - Create a virtual environment
   - Install all dependencies
   - Check for LLMSherpa server
   - Start the MCP server

## Manual Installation

### Step 1: Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Optional: Start LLMSherpa
```bash
pip install llmsherpa
llmsherpa --port 5001
```

### Step 4: Start Server
```bash
python mcp_vectorstore_server.py
```

## Configuration

1. **Edit `config.json`** to customize settings
2. **Create `.env`** file for environment variables (optional)
3. **Update document directory** in `config.json`

## Testing

```bash
# Test basic functionality
python -c "from mcp_vectorstore_server import *; print('Server ready!')"

# Test vector store
python -c "from vectorstore import PDFVectorStoreTool; tool = PDFVectorStoreTool(); print(f'Documents: {tool.vectorstore_get_num_items()}')"
```

## Troubleshooting

- **Import errors**: Run `pip install -r requirements.txt`
- **CUDA errors**: Use CPU-only versions in requirements.txt
- **Permission errors**: Check file permissions
- **Memory issues**: Reduce chunk_size in config.json

## Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Configure your MCP client to use this server
3. Add your documents to the vector store
4. Start searching and processing documents!

## Support

- Check the troubleshooting section in README.md
- Ensure all dependencies are installed
- Verify Python version (3.8+)
- Check system resources (RAM, disk space) 