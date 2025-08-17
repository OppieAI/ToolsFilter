"""Message parser for Claude and OpenAI formats."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class MessageFormat(Enum):
    """Supported message formats."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    UNKNOWN = "unknown"


class MessageParser:
    """Parser for different message formats."""
    
    @staticmethod
    def detect_format(messages: List[Dict[str, Any]]) -> MessageFormat:
        """
        Detect the message format.
        
        Args:
            messages: List of messages
            
        Returns:
            Detected message format
        """
        if not messages:
            return MessageFormat.UNKNOWN
        
        # Check first message structure
        first_msg = messages[0]
        
        # OpenAI format typically has 'role' and 'content'
        # May also have 'tool_calls', 'function_call'
        if "role" in first_msg and "content" in first_msg:
            # Check for OpenAI-specific fields
            if any(key in first_msg for key in ["tool_calls", "function_call", "name"]):
                return MessageFormat.OPENAI
            
            # Check content type
            content = first_msg.get("content")
            if isinstance(content, list) and content:
                # Anthropic uses list content for multimodal
                if any(isinstance(item, dict) and "type" in item for item in content):
                    return MessageFormat.ANTHROPIC
            
            # Default to OpenAI for simple role/content messages
            return MessageFormat.OPENAI
        
        return MessageFormat.UNKNOWN
    
    @staticmethod
    def extract_conversation_context(
        messages: List[Dict[str, Any]]
    ) -> Tuple[str, List[str]]:
        """
        Extract conversation context and tool usage.
        
        Args:
            messages: List of messages
            
        Returns:
            Tuple of (conversation_text, used_tools)
        """
        conversation_parts = []
        used_tools = []
        
        for message in messages:
            role = message.get("role", "unknown")
            content = message.get("content")
            
            # Extract text content
            text_content = ""
            if isinstance(content, str):
                text_content = content
            elif isinstance(content, list):
                # Handle multimodal content
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_parts.append(item)
                text_content = " ".join(text_parts)
            
            if text_content:
                conversation_parts.append(f"{role}: {text_content}")
            
            # Extract tool usage from OpenAI format
            if "tool_calls" in message and message["tool_calls"]:
                for tool_call in message["tool_calls"]:
                    if tool_call.get("type") == "function":
                        tool_name = tool_call.get("function", {}).get("name")
                        if tool_name:
                            used_tools.append(tool_name)
            
            # Extract tool usage from older OpenAI format
            if "function_call" in message and message["function_call"]:
                tool_name = message["function_call"].get("name")
                if tool_name:
                    used_tools.append(tool_name)
            
            # Extract tool usage from Anthropic format
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_use":
                        tool_name = item.get("name")
                        if tool_name:
                            used_tools.append(tool_name)
        
        conversation_text = "\n".join(conversation_parts)
        return conversation_text, list(set(used_tools))  # Remove duplicates
    
    @staticmethod
    def extract_user_intent(messages: List[Dict[str, Any]]) -> str:
        """
        Extract the primary user intent from messages.
        
        Args:
            messages: List of messages
            
        Returns:
            User intent text
        """
        # Find the last user message
        user_messages = [
            msg for msg in reversed(messages)
            if msg.get("role") == "user"
        ]
        
        if not user_messages:
            return ""
        
        last_user_msg = user_messages[0]
        content = last_user_msg.get("content", "")
        
        # Extract text from content
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            return " ".join(text_parts)
        
        return ""
    
    @staticmethod
    def extract_assistant_context(messages: List[Dict[str, Any]]) -> str:
        """
        Extract assistant's understanding and context.
        
        Args:
            messages: List of messages
            
        Returns:
            Assistant context text
        """
        # Get last few assistant messages for context
        assistant_messages = []
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                assistant_messages.append(msg)
                if len(assistant_messages) >= 3:  # Last 3 assistant messages
                    break
        
        context_parts = []
        for msg in reversed(assistant_messages):
            content = msg.get("content", "")
            
            # Extract text
            text_content = ""
            if isinstance(content, str):
                text_content = content
            elif isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                text_content = " ".join(text_parts)
            
            if text_content:
                # Truncate long responses
                if len(text_content) > 500:
                    text_content = text_content[:500] + "..."
                context_parts.append(text_content)
        
        return " ".join(context_parts)
    
    @staticmethod
    def analyze_conversation_pattern(
        messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze conversation patterns for better tool recommendation.
        
        Args:
            messages: List of messages
            
        Returns:
            Analysis results
        """
        analysis = {
            "message_count": len(messages),
            "user_messages": 0,
            "assistant_messages": 0,
            "has_code": False,
            "has_error": False,
            "has_file_operation": False,
            "has_search_intent": False,
            "topics": []
        }
        
        # Keywords for pattern detection
        code_keywords = ["code", "function", "class", "def", "import", "```"]
        error_keywords = ["error", "exception", "failed", "bug", "issue"]
        file_keywords = ["file", "directory", "folder", "path", "create", "delete", "edit"]
        search_keywords = ["find", "search", "look for", "where", "locate", "grep"]
        
        for message in messages:
            role = message.get("role", "")
            content_text = ""
            
            # Count message types
            if role == "user":
                analysis["user_messages"] += 1
            elif role == "assistant":
                analysis["assistant_messages"] += 1
            
            # Extract text content
            content = message.get("content", "")
            if isinstance(content, str):
                content_text = content.lower()
            elif isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", "").lower())
                content_text = " ".join(text_parts)
            
            # Pattern detection
            if any(keyword in content_text for keyword in code_keywords):
                analysis["has_code"] = True
            
            if any(keyword in content_text for keyword in error_keywords):
                analysis["has_error"] = True
            
            if any(keyword in content_text for keyword in file_keywords):
                analysis["has_file_operation"] = True
            
            if any(keyword in content_text for keyword in search_keywords):
                analysis["has_search_intent"] = True
        
        # Determine topics based on patterns
        if analysis["has_code"]:
            analysis["topics"].append("coding")
        if analysis["has_error"]:
            analysis["topics"].append("debugging")
        if analysis["has_file_operation"]:
            analysis["topics"].append("file_management")
        if analysis["has_search_intent"]:
            analysis["topics"].append("searching")
        
        return analysis