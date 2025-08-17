"""Load testing script using Locust."""

from locust import HttpUser, task, between
import json
import random


class ToolFilterUser(HttpUser):
    """Simulated user for load testing."""
    
    wait_time = between(1, 3)
    
    def on_start(self):
        """Initialize test data."""
        self.test_messages = [
            [{"role": "user", "content": "Find Python files in the project"}],
            [{"role": "user", "content": "Search for TODO comments"}],
            [{"role": "user", "content": "List all configuration files"}],
            [{"role": "user", "content": "Show me error logs"}],
            [{"role": "user", "content": "Create a new directory"}]
        ]
        
        self.test_tools = [
            {
                "type": "function",
                "function": {
                    "name": "find",
                    "description": "Find files and directories"
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "grep",
                    "description": "Search for patterns in files"
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "ls",
                    "description": "List directory contents"
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "cat",
                    "description": "Display file contents"
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "mkdir",
                    "description": "Create directories"
                }
            }
        ]
    
    @task(3)
    def filter_tools(self):
        """Test the tool filter endpoint."""
        messages = random.choice(self.test_messages)
        
        request_data = {
            "messages": messages,
            "available_tools": self.test_tools,
            "max_tools": 5
        }
        
        with self.client.post(
            "/api/v1/tools/filter",
            json=request_data,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                # Check if we got recommendations
                if "recommended_tools" in data:
                    response.success()
                else:
                    response.failure("No recommended_tools in response")
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(1)
    def search_tools(self):
        """Test the tool search endpoint."""
        queries = ["find files", "search text", "list directory", "create folder"]
        query = random.choice(queries)
        
        with self.client.get(
            f"/api/v1/tools/search?query={query}&limit=5",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(1)
    def check_health(self):
        """Test the health endpoint."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure("API not healthy")
            else:
                response.failure(f"Got status code {response.status_code}")


if __name__ == "__main__":
    # Can be run with: locust -f load_test.py --host=http://localhost:8000
    print("Run with: locust -f tests/load_test.py --host=http://localhost:8000")