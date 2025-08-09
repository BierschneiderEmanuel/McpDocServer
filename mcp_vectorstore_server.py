#!/usr/bin/env python3
"""
MCP VectorStore Server

A Model Context Protocol (MCP) server that provides vector store operations
for document search, PDF processing, and information retrieval.
https://www.youtube.com/watch?v=NqMU9cL2LfE
"""

import asyncio
import json
import logging
import mcp.server
import mcp.server.stdio
import mcp.types as types
import io
from contextlib import redirect_stdout
from mcp.server.models import InitializationOptions
from mcp.server.lowlevel import NotificationOptions
from vectorstore import (
    duckduckgo_tool,
    PDFVectorStoreTool,
    read_file_tool,
    get_google_content,
    get_wiki_content,
    calculate_tool
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MCP Server setup
server = mcp.server.Server("vectorstore-server")

vector_store_tool = None

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List all available tools."""
    return [
        types.Tool(
            name="vectorstore_search",
            description="Search the vector store for documents containing the query",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 2)"
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="vectorstore_create",
            description="Create a new vector store from documents in a directory",
            inputSchema={
                "type": "object",
                "properties": {
                    "directory_path": {
                        "type": "string",
                        "description": "Path to directory containing documents"
                    }
                },
                "required": ["directory_path"]
            }
        ),
        types.Tool(
            name="vectorstore_info",
            description="Get information about the current vector store",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        types.Tool(
            name="vectorstore_clear",
            description="Clear all documents from the vector store",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        types.Tool(
            name="read_file",
            description="Read the contents of a file on the system",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Path to the file to read"
                    }
                },
                "required": ["filename"]
            }
        ),
        types.Tool(
            name="google_search",
            description="Search Google for information",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 3)"
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="wikipedia_search",
            description="Search Wikipedia for information",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="duckduckgo_search",
            description="Search DuckDuckGo for information",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="calculate",
            description="Perform mathematical calculations",
            inputSchema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "Mathematical operation to perform"
                    }
                },
                "required": ["operation"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls."""
    try:
        if name == "vectorstore_search":
            query = arguments.get("query", "")
            k = arguments.get("k", 2)
            result = vector_store_tool.vectorstore_search(query, k)
            return [types.TextContent(type="text", text=result)]

        elif name == "vectorstore_create":
            directory_path = arguments.get("directory_path", "")
            result = vector_store_tool.create_vector_store(directory_path)
            return [types.TextContent(type="text", text=result)]

        elif name == "vectorstore_info":
            num_items = vector_store_tool.vectorstore_get_num_items()
            result = f"Vector store contains {num_items} documents"
            return [types.TextContent(type="text", text=result)]

        elif name == "vectorstore_clear":
            vector_store_tool.vectorstore_clear()
            return [types.TextContent(type="text", text="Vector store cleared successfully")]

        elif name == "read_file":
            filename = arguments.get("filename", "")
            result = read_file_tool._run(filename)
            return [types.TextContent(type="text", text=result)]

        elif name == "google_search":
            query = arguments.get("query", "")
            max_results = arguments.get("max_results", 3)
            # Capture the output from get_google_content
            f = io.StringIO()
            with redirect_stdout(f):
                get_google_content(query, max_results)
            result = f.getvalue()
            return [types.TextContent(type="text", text=result)]

        elif name == "wikipedia_search":
            query = arguments.get("query", "")
            result = get_wiki_content(query)
            return [types.TextContent(type="text", text=result)]

        elif name == "duckduckgo_search":
            query = arguments.get("query", "")
            result = duckduckgo_tool.invoke(query)
            return [types.TextContent(type="text", text=result)]

        elif name == "calculate":
            operation = arguments.get("operation", "")
            result = calculate_tool(operation)
            return [types.TextContent(type="text", text=result)]

        else:
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.error(f"Error in tool {name}: {str(e)}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]

@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """List available resources."""
    return [
        types.Resource(
            uri="vectorstore://info",
            name="Vector Store Information",
            description="Information about the current vector store",
            mimeType="application/json"
        )
    ]

@server.read_resource()
async def handle_read_resource(uri: str) -> bytes:
    """Read resource content."""
    if uri == "vectorstore://info":
        info = {
            "num_documents": vector_store_tool.vectorstore_get_num_items(),
            "directory": vector_store_tool.dir,
            "embeddings_model": "sentence-transformers/all-mpnet-base-v2"
        }
        return json.dumps(info, indent=2).encode()

    raise Exception(f"Unknown resource: {uri}")


# Function to initialize the vector store tool
def init_vector_store_tool():
    global vector_store_tool
    vector_store_tool = PDFVectorStoreTool()

async def main():
    """Main entry point."""

    init_task = asyncio.create_task(
        asyncio.to_thread(init_vector_store_tool)
    )

    # 2) Run your MCP server loop
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="vectorstore-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
