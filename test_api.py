"""Simple test script to verify API functionality."""

import asyncio
import httpx
import json
from typing import List, Dict, Any


async def test_health():
    """Test health endpoint."""
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/health")
        print(f"Health Check: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200


async def test_tool_filter():
    """Test tool filtering endpoint."""
    async with httpx.AsyncClient() as client:
        # Test conversation about finding Python files
        request_data = {
            "messages": [
                {
                    "role": "user",
                    "content": "I need to find all Python files in the project"
                },
                {
                    "role": "assistant", 
                    "content": "I'll help you find Python files in the project. Let me search for them."
                },
                {
                    "role": "user",
                    "content": "Also check if there are any test files"
                }
            ],
            "available_tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "find",
                        "description": "Find files and directories by name, type, or other attributes",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string"}
                            }
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "grep",
                        "description": "Search for patterns in files using regular expressions",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "pattern": {"type": "string"}
                            }
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "ls",
                        "description": "List directory contents",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"}
                            }
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "description": "Display text",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"}
                            }
                        }
                    }
                }
            ],
            "max_tools": 3,
            "include_reasoning": True
        }
        
        response = await client.post(
            "http://localhost:8000/api/v1/tools/filter",
            json=request_data,
            timeout=30.0
        )
        
        print(f"\nTool Filter Test: {response.status_code}")
        result = response.json()
        if response.status_code == 200:
            print(json.dumps(result, indent=2))
            
            # Verify expected tools are recommended
            tool_names = [tool["tool_name"] for tool in result["recommended_tools"]]
            print(f"\nRecommended tools: {tool_names}")
            
            # Check if 'find' is highly recommended
            if tool_names and tool_names[0] == "find":
                print("✓ 'find' tool correctly identified as most relevant")
            
            # Check processing time
            if result["metadata"]["processing_time_ms"] < 100:
                print("✓ Processing time under 100ms target")
        else:
            print(f"Error: {json.dumps(result, indent=2)}")
                
        return response.status_code == 200


async def test_tool_search():
    """Test tool search endpoint."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "http://localhost:8000/api/v1/tools/search",
            params={"query": "search files", "limit": 5}
        )
        
        print(f"\nTool Search Test: {response.status_code}")
        result = response.json()
        if response.status_code == 200:
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {json.dumps(result, indent=2)}")
            
        return response.status_code == 200


async def main():
    """Run all tests."""
    print("Starting API tests...\n")
    
    # Wait a bit for services to be ready
    await asyncio.sleep(2)
    
    tests = [
        ("Health Check", test_health),
        ("Tool Filter", test_tool_filter),
        ("Tool Search", test_tool_search)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*50}")
            print(f"Running: {test_name}")
            print('='*50)
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"Error in {test_name}: {e}")
            results.append((test_name, False))
    
    print(f"\n{'='*50}")
    print("Test Summary:")
    print('='*50)
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")


if __name__ == "__main__":
    asyncio.run(main())