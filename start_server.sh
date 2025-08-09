#!/bin/bash

# MCP VectorStore Server Startup Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== MCP VectorStore Server ===${NC}"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed or not in PATH${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating one...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate

# Check if requirements are installed
if [ ! -f "venv/lib/python*/site-packages/mcp" ]; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# Check if vectorstore.py exists
if [ ! -f "vectorstore.py" ]; then
    echo -e "${RED}Error: vectorstore.py not found in current directory${NC}"
    exit 1
fi

# Check if LLMSherpa is running (optional)
if command -v curl &> /dev/null; then
    if curl -s http://localhost:5001 > /dev/null 2>&1; then
        echo -e "${GREEN}LLMSherpa server is running on port 5001${NC}"
    else
        echo -e "${YELLOW}Warning: LLMSherpa server not detected on port 5001${NC}"
        echo -e "${YELLOW}You can start it with: llmsherpa --port 5001${NC}"
    fi
fi

# Make server executable
chmod +x mcp_vectorstore_server.py

echo -e "${GREEN}Starting MCP VectorStore Server...${NC}"
echo -e "${BLUE}Press Ctrl+C to stop the server${NC}"
echo ""

# Start the server
python mcp_vectorstore_server.py 