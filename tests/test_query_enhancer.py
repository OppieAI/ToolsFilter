"""Tests for the query enhancement functionality."""

import pytest
from src.services.query_enhancer import QueryEnhancer


class TestQueryEnhancer:
    """Test cases for QueryEnhancer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.enhancer = QueryEnhancer()
        
        # Sample conversations
        self.simple_conversation = [
            {"role": "user", "content": "I need to search for Python files"}
        ]
        
        self.error_conversation = [
            {"role": "user", "content": "I'm trying to connect to the database"},
            {"role": "assistant", "content": "I'll help you connect to the database"},
            {"role": "user", "content": "I'm getting an error: connection refused on port 5432"}
        ]
        
        self.code_conversation = [
            {"role": "user", "content": "Here's my code:\n```python\ndef hello():\n    print('world')\n```"},
            {"role": "assistant", "content": "I see your code. Let me help."},
            {"role": "user", "content": "Can you help me test this function?"}
        ]
        
        self.complex_conversation = [
            {"role": "user", "content": "How do I find all Python files in my project?"},
            {"role": "assistant", "content": "You can use the find command or grep"},
            {"role": "user", "content": "I tried `find . -name '*.py'` but got permission denied errors"},
            {"role": "assistant", "content": "Let me help you with that"},
            {"role": "user", "content": "Also need to search for test files specifically"}
        ]
    
    def test_basic_enhancement(self):
        """Test basic query enhancement."""
        result = self.enhancer.enhance_query(self.simple_conversation)
        
        assert "queries" in result
        assert "weights" in result
        assert "expanded" in result
        assert "metadata" in result
        
        # Check primary query extraction
        assert result["queries"]["primary"] == "I need to search for Python files"
        
        # Check intent extraction
        assert "search" in result["queries"].get("intent", "").lower()
        
        # Check expansion includes synonyms
        assert "find" in result["expanded"] or "locate" in result["expanded"]
    
    def test_error_context_extraction(self):
        """Test extraction of error context."""
        result = self.enhancer.enhance_query(self.error_conversation)
        
        # Check error context is extracted
        assert "error" in result["queries"]
        assert "connection refused" in result["queries"]["error"]
        
        # Check metadata flags error
        assert result["metadata"]["has_error"] is True
        
        # Check weights are adjusted for error context
        assert result["weights"]["error"] > 0.15  # Should be boosted
    
    def test_code_context_extraction(self):
        """Test extraction of code context."""
        result = self.enhancer.enhance_query(self.code_conversation)
        
        # Check code is extracted
        assert "code" in result["queries"]
        assert "def hello" in result["queries"]["code"] or "print" in result["queries"]["code"]
        
        # Check metadata flags code
        assert result["metadata"]["has_code"] is True
        
        # Check weights are adjusted for code context
        assert result["weights"]["code"] > 0.05  # Should be boosted
    
    def test_intent_extraction(self):
        """Test intent extraction from queries."""
        test_cases = [
            ("I want to create a new file", ["create"]),
            ("How do I delete old logs?", ["delete"]),
            ("Can you help me search and update the database?", ["search", "update"]),
            ("I need to test my API endpoints", ["test", "api"]),
            ("debug this error message", ["debug", "error"])
        ]
        
        for query, expected_actions in test_cases:
            result = self.enhancer.enhance_query([{"role": "user", "content": query}])
            intent = result["queries"].get("intent", "").lower()
            
            for action in expected_actions:
                assert action in intent or action in result["metadata"]["detected_actions"]
    
    def test_synonym_expansion(self):
        """Test query expansion with synonyms."""
        test_cases = [
            ("search files", ["find", "locate", "query"]),
            ("delete records", ["remove", "rm", "unlink"]),
            ("run tests", ["execute", "launch", "start"]),
            ("create database", ["make", "generate", "build"])
        ]
        
        for query, expected_synonyms in test_cases:
            result = self.enhancer.enhance_query([{"role": "user", "content": query}])
            expanded = result["expanded"].lower()
            
            # At least one synonym should be in expanded query
            assert any(syn in expanded for syn in expected_synonyms)
    
    def test_weight_calculation(self):
        """Test dynamic weight calculation."""
        # Simple conversation - primary should dominate
        simple_result = self.enhancer.enhance_query(self.simple_conversation)
        assert simple_result["weights"]["primary"] >= 0.4
        
        # Error conversation - error weight should be boosted
        error_result = self.enhancer.enhance_query(self.error_conversation)
        assert error_result["weights"]["error"] > 0.15
        
        # All weights should sum to approximately 1.0
        for result in [simple_result, error_result]:
            total = sum(result["weights"].values())
            assert 0.99 <= total <= 1.01  # Allow small floating point error
    
    def test_complex_conversation(self):
        """Test enhancement of complex multi-turn conversation."""
        result = self.enhancer.enhance_query(self.complex_conversation)
        
        # Should extract multiple aspects
        assert len(result["queries"]) >= 3
        
        # Should detect the error mention
        assert result["metadata"]["has_error"] is True
        
        # Should extract search intent
        assert "search" in result["queries"].get("intent", "").lower()
        
        # Should have reasonable topic extraction
        assert "file" in result["metadata"]["topics"] or "search" in result["metadata"]["topics"]
    
    def test_empty_conversation(self):
        """Test handling of empty conversation."""
        result = self.enhancer.enhance_query([])
        
        assert result["queries"] == {}
        assert result["expanded"] == ""
        assert result["metadata"]["message_count"] == 0
    
    def test_multimodal_content(self):
        """Test handling of multimodal message content."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image"},
                    {"type": "image", "url": "image.png"},
                    {"type": "text", "text": "and find similar patterns"}
                ]
            }
        ]
        
        result = self.enhancer.enhance_query(messages)
        
        # Should extract text parts
        assert "analyze" in result["queries"]["primary"].lower()
        assert "find" in result["queries"]["primary"].lower()
    
    def test_historical_tools(self):
        """Test inclusion of previously used tools."""
        used_tools = ["grep", "find", "sed", "awk", "curl"]
        
        result = self.enhancer.enhance_query(
            self.simple_conversation,
            used_tools=used_tools
        )
        
        # Should include last 3 tools
        assert "sed" in result["queries"]["historical"]
        assert "awk" in result["queries"]["historical"]
        assert "curl" in result["queries"]["historical"]
        assert "grep" not in result["queries"]["historical"]  # Not in last 3
    
    def test_action_extraction(self):
        """Test extraction of action verbs."""
        query = "I need to create, update, and delete database records"
        result = self.enhancer.enhance_query([{"role": "user", "content": query}])
        
        actions = result["metadata"]["detected_actions"]
        assert "create" in actions
        assert "update" in actions
        assert "delete" in actions
    
    def test_topic_extraction(self):
        """Test extraction of technical topics."""
        query = "I need to search files in the git repository and query the database"
        result = self.enhancer.enhance_query([{"role": "user", "content": query}])
        
        topics = result["metadata"]["topics"]
        assert any(topic in ["file", "git", "database", "search"] for topic in topics)