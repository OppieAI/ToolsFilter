"""
Data loading abstraction for the evaluation framework.
Supports multiple data sources through a common interface.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
import json
import random
from dataclasses import dataclass
from slugify import slugify

from .models import TestCase, TestSuite
from .config import DataConfig
from src.core.models import Tool


class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders.
    Defines the interface that all data loaders must implement.
    """
    
    def __init__(self, config: DataConfig):
        """
        Initialize the data loader with configuration.
        
        Args:
            config: Data configuration
        """
        self.config = config
        self._cache: Optional[TestSuite] = None
    
    @abstractmethod
    def load_raw_data(self) -> List[Dict[str, Any]]:
        """
        Load raw data from the source.
        
        Returns:
            List of raw data dictionaries
        """
        pass
    
    @abstractmethod
    def transform_to_test_case(self, raw_data: Dict[str, Any], index: int) -> TestCase:
        """
        Transform raw data into a TestCase.
        
        Args:
            raw_data: Raw data dictionary
            index: Index of the test case
            
        Returns:
            Transformed TestCase
        """
        pass
    
    def load_test_suite(self, force_reload: bool = False) -> TestSuite:
        """
        Load complete test suite with caching.
        
        Args:
            force_reload: Force reload even if cached
            
        Returns:
            TestSuite with all test cases
        """
        if self._cache is not None and not force_reload:
            return self._cache
        
        raw_data = self.load_raw_data()
        
        # Apply num_cases limit
        if self.config.num_cases and self.config.num_cases < len(raw_data):
            raw_data = raw_data[:self.config.num_cases]
        
        test_cases = []
        for idx, raw in enumerate(raw_data):
            try:
                test_case = self.transform_to_test_case(raw, idx)
                test_cases.append(test_case)
            except Exception as e:
                print(f"Error transforming test case {idx}: {e}")
                continue
        
        self._cache = TestSuite(
            name=f"{self.config.data_source}_{self.config.test_file}",
            test_cases=test_cases,
            metadata={
                "source": self.config.data_source,
                "file": self.config.test_file,
                "num_cases": len(test_cases)
            }
        )
        
        return self._cache
    
    def iterate_test_cases(self) -> Iterator[TestCase]:
        """
        Iterate over test cases without loading all into memory.
        Useful for large datasets.
        """
        raw_data = self.load_raw_data()
        
        count = 0
        for idx, raw in enumerate(raw_data):
            if self.config.num_cases and count >= self.config.num_cases:
                break
            
            try:
                test_case = self.transform_to_test_case(raw, idx)
                yield test_case
                count += 1
            except Exception as e:
                print(f"Error transforming test case {idx}: {e}")
                continue
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded data.
        
        Returns:
            Dictionary with statistics
        """
        test_suite = self.load_test_suite()
        
        total_expected_tools = sum(tc.num_expected_tools for tc in test_suite.test_cases)
        total_available_tools = sum(tc.num_available_tools for tc in test_suite.test_cases)
        
        return {
            "num_test_cases": test_suite.size,
            "avg_expected_tools": total_expected_tools / test_suite.size if test_suite.size > 0 else 0,
            "avg_available_tools": total_available_tools / test_suite.size if test_suite.size > 0 else 0,
            "min_expected_tools": min(tc.num_expected_tools for tc in test_suite.test_cases) if test_suite.test_cases else 0,
            "max_expected_tools": max(tc.num_expected_tools for tc in test_suite.test_cases) if test_suite.test_cases else 0,
            "min_available_tools": min(tc.num_available_tools for tc in test_suite.test_cases) if test_suite.test_cases else 0,
            "max_available_tools": max(tc.num_available_tools for tc in test_suite.test_cases) if test_suite.test_cases else 0,
        }


class ToolBenchDataLoader(BaseDataLoader):
    """
    Data loader for ToolBench datasets.
    Handles the specific format of ToolBench test data.
    """
    
    def load_raw_data(self) -> List[Dict[str, Any]]:
        """
        Load raw ToolBench data from JSON file.
        
        Returns:
            List of raw test cases
        """
        file_path = self.config.data_path / self.config.test_file
        
        if not file_path.exists():
            raise FileNotFoundError(f"Test file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def transform_to_test_case(self, raw_data: Dict[str, Any], index: int) -> TestCase:
        """
        Transform ToolBench raw data to TestCase.
        
        Args:
            raw_data: Raw ToolBench test case
            index: Test case index
            
        Returns:
            Transformed TestCase
        """
        # Extract query
        query = raw_data.get("query", "")
        query_id = raw_data.get("query_id", str(index))
        
        # Extract expected tools (relevant APIs)
        expected_apis = raw_data.get("relevant APIs", [])
        expected_tools = []
        
        for api_ref in expected_apis:
            if len(api_ref) >= 2:
                tool_name = api_ref[0]
                api_name = api_ref[1]
                raw_name = f"{tool_name}_{api_name}"
                # Normalize using slugify (same as in evaluator)
                expected_name = slugify(raw_name, separator="_")
                expected_tools.append(expected_name)
        
        # Convert available APIs to Tool format
        api_list = raw_data.get("api_list", [])
        available_tools = []
        
        for api in api_list:
            try:
                tool_dict = self._convert_api_to_tool_dict(api)
                available_tools.append(tool_dict)
            except Exception as e:
                print(f"Error converting API {api.get('api_name', 'Unknown')}: {e}")
                continue
        
        # Create metadata
        metadata = {
            "query_id": query_id,
            "category": raw_data.get("category", ""),
            "tool_names": raw_data.get("tool_names", []),
            "api_names": raw_data.get("api_names", [])
        }
        
        return TestCase(
            id=f"toolbench_{query_id}",
            query=query,
            expected_tools=expected_tools,
            available_tools=available_tools,
            metadata=metadata
        )
    
    def _convert_api_to_tool_dict(self, api: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a ToolBench API to tool dictionary format.
        
        Args:
            api: ToolBench API definition
            
        Returns:
            Tool dictionary in OpenAI function format
        """
        # Build properties from parameters
        properties = {}
        required = []
        
        # Process required parameters
        for param in api.get("required_parameters", []):
            param_name = param["name"]
            param_type = param.get("type", "string").lower()
            
            # Map ToolBench types to JSON Schema types
            json_type = self._map_parameter_type(param_type)
            
            properties[param_name] = {
                "type": json_type,
                "description": param.get("description", "")
            }
            
            if param.get("default"):
                properties[param_name]["default"] = param["default"]
            
            required.append(param_name)
        
        # Process optional parameters
        for param in api.get("optional_parameters", []):
            param_name = param["name"]
            param_type = param.get("type", "string").lower()
            
            json_type = self._map_parameter_type(param_type)
            
            properties[param_name] = {
                "type": json_type,
                "description": param.get("description", "")
            }
            
            if param.get("default"):
                properties[param_name]["default"] = param["default"]
        
        # Create function name
        tool_name = api.get("tool_name", "Unknown")
        api_name = api.get("api_name", "Unknown")
        raw_function_name = f"{tool_name}_{api_name}"
        function_name = slugify(raw_function_name, separator="_")
        
        # Create tool dictionary
        return {
            "type": "function",
            "name": function_name,
            "description": api.get("api_description", ""),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False
            },
            "strict": True
        }
    
    def _map_parameter_type(self, param_type: str) -> str:
        """
        Map ToolBench parameter type to JSON Schema type.
        
        Args:
            param_type: ToolBench parameter type
            
        Returns:
            JSON Schema type
        """
        type_mapping = {
            "string": "string",
            "number": "number",
            "integer": "number",
            "int": "number",
            "boolean": "boolean",
            "bool": "boolean",
            "enum": "string",
            "array": "array",
            "object": "object"
        }
        
        return type_mapping.get(param_type.lower(), "string")


class NoiseDataLoader(BaseDataLoader):
    """
    Data loader for generating noise tools.
    Used to create realistic noise for testing.
    """
    
    def __init__(self, config: DataConfig, source_loader: BaseDataLoader):
        """
        Initialize noise data loader.
        
        Args:
            config: Data configuration
            source_loader: Source data loader to extract noise from
        """
        super().__init__(config)
        self.source_loader = source_loader
        self._noise_pool: Optional[List[Dict[str, Any]]] = None
    
    def load_raw_data(self) -> List[Dict[str, Any]]:
        """
        Load raw data for noise generation.
        
        Returns:
            List of raw noise data
        """
        # Use source loader to get all available tools
        return self.source_loader.load_raw_data()
    
    def transform_to_test_case(self, raw_data: Dict[str, Any], index: int) -> TestCase:
        """
        Not used for noise loader.
        """
        raise NotImplementedError("NoiseDataLoader doesn't create test cases")
    
    def load_noise_pool(self, target_size: int = 1500) -> List[Dict[str, Any]]:
        """
        Load a pool of noise tools.
        
        Args:
            target_size: Target number of noise tools
            
        Returns:
            List of noise tool dictionaries
        """
        if self._noise_pool is not None:
            return self._noise_pool
        
        all_tools = []
        seen_tool_names = set()
        
        # Load tools from multiple test files if using ToolBench
        if isinstance(self.source_loader, ToolBenchDataLoader):
            test_files = ["G1_instruction.json", "G2_instruction.json", "G3_instruction.json"]
            
            for test_file in test_files:
                if len(all_tools) >= target_size:
                    break
                
                try:
                    # Temporarily change config to load different file
                    original_file = self.source_loader.config.test_file
                    self.source_loader.config.test_file = test_file
                    
                    raw_data = self.source_loader.load_raw_data()
                    
                    # Extract unique tools
                    for test_case in raw_data:
                        api_list = test_case.get("api_list", [])
                        
                        for api in api_list:
                            try:
                                tool_dict = self.source_loader._convert_api_to_tool_dict(api)
                                
                                if tool_dict["name"] not in seen_tool_names:
                                    all_tools.append(tool_dict)
                                    seen_tool_names.add(tool_dict["name"])
                                    
                                    if len(all_tools) >= target_size:
                                        break
                            except:
                                continue
                        
                        if len(all_tools) >= target_size:
                            break
                    
                    # Restore original file
                    self.source_loader.config.test_file = original_file
                    
                except Exception as e:
                    print(f"Error loading noise from {test_file}: {e}")
        
        # Shuffle for randomness
        random.seed(42)  # Fixed seed for reproducibility
        random.shuffle(all_tools)
        
        # Take target size
        self._noise_pool = all_tools[:target_size]
        
        return self._noise_pool
    
    def sample_noise_tools(self, num_tools: int) -> List[Dict[str, Any]]:
        """
        Sample random noise tools from the pool.
        
        Args:
            num_tools: Number of tools to sample
            
        Returns:
            List of sampled noise tools
        """
        noise_pool = self.load_noise_pool()
        
        if len(noise_pool) < num_tools:
            print(f"Warning: Only {len(noise_pool)} noise tools available, requested {num_tools}")
            return noise_pool.copy()
        
        return random.sample(noise_pool, num_tools)


class DataLoaderFactory:
    """
    Factory for creating data loaders based on configuration.
    """
    
    _loaders = {
        "toolbench": ToolBenchDataLoader,
    }
    
    @classmethod
    def register_loader(cls, name: str, loader_class: type):
        """
        Register a new data loader.
        
        Args:
            name: Name of the data source
            loader_class: Data loader class
        """
        cls._loaders[name] = loader_class
    
    @classmethod
    def create_loader(cls, config: DataConfig) -> BaseDataLoader:
        """
        Create a data loader based on configuration.
        
        Args:
            config: Data configuration
            
        Returns:
            Data loader instance
            
        Raises:
            ValueError: If data source is not supported
        """
        loader_class = cls._loaders.get(config.data_source)
        
        if loader_class is None:
            raise ValueError(
                f"Unsupported data source: {config.data_source}. "
                f"Available: {list(cls._loaders.keys())}"
            )
        
        return loader_class(config)