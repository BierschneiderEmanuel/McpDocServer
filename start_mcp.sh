#!/bin/bash
source /mnt/c/Users/emanu/Desktop/LLM/venv/venv/bin/activate
export USER_AGENT="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
cd /mnt/c/Users/emanu/Desktop/McpDocServer/
python ./mcp_vectorstore_server.py

