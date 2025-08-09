#!/usr/bin/env python3
"""
Test script for MCP VectorStore Server

This script tests the basic functionality of the MCP server
without requiring a full MCP client connection.
"""

import asyncio
import json
import sys
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import mcp.server
        import mcp.server.stdio
        import mcp.types as types
        print("✓ MCP modules imported successfully")
    except ImportError as e:
        print(f"✗ MCP import error: {e}")
        return False
    
    try:
        from vectorstore import PDFVectorStoreTool, ReadFile
        print("✓ Vectorstore modules imported successfully")
    except ImportError as e:
        print(f"✗ Vectorstore import error: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✓ Sentence transformers imported successfully")
    except ImportError as e:
        print(f"✗ Sentence transformers import error: {e}")
        return False
    
    return True

def test_vectorstore_initialization():
    """Test vector store initialization."""
    print("\nTesting vector store initialization...")
    
    try:
        from vectorstore import PDFVectorStoreTool
        tool = PDFVectorStoreTool()
        num_docs = tool.vectorstore_get_num_items()
        print(f"✓ Vector store initialized with {num_docs} documents")
        return True
    except Exception as e:
        print(f"✗ Vector store initialization error: {e}")
        return False

def test_file_operations():
    """Test file reading operations."""
    print("\nTesting file operations...")
    
    try:
        from vectorstore import ReadFile
        read_tool = ReadFile()
        
        # Test reading this file
        content = read_tool._run(__file__)
        if content and len(content) > 0:
            print("✓ File reading works correctly")
            return True
        else:
            print("✗ File reading returned empty content")
            return False
    except Exception as e:
        print(f"✗ File operations error: {e}")
        return False

def test_search_functions():
    """Test search functionality."""
    print("\nTesting search functions...")
    
    try:
        from vectorstore import get_wiki_content, duck_duck_go_search_tool
        
        # Test Wikipedia search
        wiki_result = get_wiki_content("Python programming")
        if wiki_result and len(wiki_result) > 0:
            print("✓ Wikipedia search works")
        else:
            print("✗ Wikipedia search failed")
            return False
        
        # Test DuckDuckGo search
        ddg_result = duck_duck_go_search_tool("test query")
        if ddg_result:
            print("✓ DuckDuckGo search works")
        else:
            print("✗ DuckDuckGo search failed")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Search functions error: {e}")
        return False

def test_calculator():
    """Test calculator functionality."""
    print("\nTesting calculator...")
    
    try:
        from vectorstore import calculate_tool
        
        result = calculate_tool("2 + 2")
        if result and "4" in result:
            print("✓ Calculator works correctly")
            return True
        else:
            print("✗ Calculator returned unexpected result")
            return False
    except Exception as e:
        print(f"✗ Calculator error: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        
        required_keys = ["server", "vectorstore", "llmsherpa", "search", "logging"]
        for key in required_keys:
            if key not in config:
                print(f"✗ Missing configuration key: {key}")
                return False
        
        print("✓ Configuration file is valid")
        return True
    except FileNotFoundError:
        print("✗ Configuration file not found")
        return False
    except json.JSONDecodeError as e:
        print(f"✗ Configuration file is invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def test_mcp_server_creation():
    """Test MCP server creation."""
    print("\nTesting MCP server creation...")
    
    try:
        import mcp.server
        server = mcp.server.Server("test-vectorstore-server")
        print("✓ MCP server created successfully")
        return True
    except Exception as e:
        print(f"✗ MCP server creation error: {e}")
        return False

def main():
    """Run all tests."""
    print("=== MCP VectorStore Server Test Suite ===\n")
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("MCP Server Creation", test_mcp_server_creation),
        ("Vector Store Initialization", test_vectorstore_initialization),
        ("File Operations", test_file_operations),
        ("Search Functions", test_search_functions),
        ("Calculator", test_calculator),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The server should work correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 