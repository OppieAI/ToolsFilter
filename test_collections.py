"""Test script to verify collection naming with model names."""

import asyncio
import httpx

async def test_collections():
    """Test the collections endpoint."""
    async with httpx.AsyncClient() as client:
        # Test health check
        print("Testing health check...")
        response = await client.get("http://localhost:8000/health")
        print(f"Health status: {response.json()}")
        
        # Test collections endpoint
        print("\nTesting collections endpoint...")
        response = await client.get("http://localhost:8000/api/v1/collections")
        collections = response.json()
        
        print(f"\nTotal collections: {collections['total']}")
        for col in collections['collections']:
            print(f"\nCollection: {col['name']}")
            print(f"  Points: {col['points_count']}")
            print(f"  Vectors: {col['vectors_count']}")
            if 'metadata' in col:
                print(f"  Model: {col['metadata'].get('embedding_model', 'Unknown')}")
                print(f"  Dimension: {col['metadata'].get('embedding_dimension', 'Unknown')}")
                print(f"  Created: {col['metadata'].get('created_at', 'Unknown')}")
        
        # Test tools info
        print("\nTesting tools info endpoint...")
        response = await client.get("http://localhost:8000/api/v1/tools/info")
        info = response.json()
        
        print(f"\nCurrent collection info:")
        print(f"  Collection: {info['collection_info']['collection_name']}")
        print(f"  Total tools: {info['total_tools']}")
        print(f"  Indexed tools: {info['indexed_tools']}")
        if 'metadata' in info['collection_info']:
            print(f"  Metadata: {info['collection_info']['metadata']}")

if __name__ == "__main__":
    asyncio.run(test_collections())