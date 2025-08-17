"""Tests for message parser service."""

import pytest
from src.services.message_parser import MessageParser


class TestMessageParser:
    """Test cases for MessageParser."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = MessageParser()
    
    def test_extract_user_intent_single_message(self):
        """Test extracting intent from a single user message."""
        messages = [
            {"role": "user", "content": "I need to find Python files"}
        ]
        
        intent = self.parser.extract_user_intent(messages)
        assert intent == "I need to find Python files"
    
    def test_extract_user_intent_multiple_messages(self):
        """Test extracting intent from multiple messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "I need to search for errors"}
        ]
        
        intent = self.parser.extract_user_intent(messages)
        assert intent == "I need to search for errors"
    
    def test_extract_conversation_context(self):
        """Test extracting full conversation context."""
        messages = [
            {"role": "user", "content": "Find Python files"},
            {"role": "assistant", "content": "I'll help you find Python files"}
        ]
        
        context, tools = self.parser.extract_conversation_context(messages)
        assert "user: Find Python files" in context
        assert "assistant: I'll help you find Python files" in context
        assert tools == []
    
    def test_extract_tool_calls(self):
        """Test extracting tool calls from messages."""
        messages = [
            {
                "role": "assistant",
                "content": "I'll search for files",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "find"}
                    }
                ]
            }
        ]
        
        _, tools = self.parser.extract_conversation_context(messages)
        assert "find" in tools
    
    def test_analyze_conversation_pattern(self):
        """Test conversation pattern analysis."""
        messages = [
            {"role": "user", "content": "Search for TODO comments in code"}
        ]
        
        analysis = self.parser.analyze_conversation_pattern(messages)
        
        assert analysis["has_search_intent"] is True
        assert analysis["has_code"] is True
        assert "search" in analysis["topics"]
    
    def test_multimodal_content(self):
        """Test handling multimodal content."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image", "image": "base64data"}
                ]
            }
        ]
        
        context, _ = self.parser.extract_conversation_context(messages)
        assert "What's in this image?" in context
    
    def test_empty_messages(self):
        """Test handling empty message list."""
        messages = []
        
        intent = self.parser.extract_user_intent(messages)
        context, tools = self.parser.extract_conversation_context(messages)
        
        assert intent == ""
        assert context == ""
        assert tools == []