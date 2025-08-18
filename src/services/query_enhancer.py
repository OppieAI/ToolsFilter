"""Query enhancement service for improved tool matching."""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


class QueryEnhancer:
    """Enhances queries with multiple representations for better tool matching."""

    def __init__(self):
        """Initialize the query enhancer with synonym dictionaries."""
        # Technical synonyms for query expansion
        self.synonyms = {
            "search": ["find", "locate", "grep", "query", "lookup", "discover"],
            "file": ["document", "fs", "filesystem", "directory", "folder", "path"],
            "error": ["exception", "bug", "issue", "problem", "failure", "crash"],
            "run": ["execute", "launch", "start", "invoke", "trigger", "call"],
            "create": ["make", "generate", "build", "construct", "new", "add"],
            "delete": ["remove", "rm", "unlink", "destroy", "erase", "clear"],
            "update": ["modify", "change", "edit", "patch", "alter", "revise"],
            "list": ["ls", "show", "display", "enumerate", "get", "fetch"],
            "read": ["view", "cat", "open", "load", "retrieve", "access"],
            "write": ["save", "store", "persist", "output", "export", "dump"],
            "test": ["check", "verify", "validate", "assert", "confirm", "ensure"],
            "debug": ["troubleshoot", "diagnose", "trace", "inspect", "analyze"],
            "git": ["version", "vcs", "commit", "branch", "repository", "repo"],
            "api": ["endpoint", "rest", "http", "request", "webhook", "service"],
            "database": ["db", "sql", "query", "table", "schema", "collection"]
        }

        # Action verbs to identify intent
        self.action_verbs = {
            "search", "find", "locate", "create", "delete", "update", "modify",
            "run", "execute", "test", "debug", "analyze", "check", "verify",
            "get", "set", "list", "show", "read", "write", "save", "load",
            "connect", "disconnect", "start", "stop", "install", "uninstall",
            "compile", "build", "deploy", "push", "pull", "fetch", "merge"
        }

        # Error indicators
        self.error_patterns = [
            r"error[:\s]", r"exception[:\s]", r"traceback", r"stack trace",
            r"failed", r"failure", r"cannot", r"unable to", r"not found",
            r"undefined", r"null pointer", r"segfault", r"core dump",
            r"syntax error", r"type error", r"value error", r"key error",
            r"permission denied", r"access denied", r"denied"
        ]

        # Code indicators
        self.code_patterns = [
            r"```[\s\S]*?```",  # Markdown code blocks
            r"`[^`]+`",  # Inline code
            r"^\s{4,}.*$",  # Indented code (4+ spaces)
            r"def\s+\w+", r"class\s+\w+",  # Python definitions
            r"function\s+\w+", r"const\s+\w+",  # JavaScript
            r"import\s+", r"from\s+.*\s+import",  # Imports
        ]

    def enhance_query(self, messages: List[Dict[str, Any]],
                     used_tools: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Enhance query with multiple representations and weights.

        Args:
            messages: List of conversation messages
            used_tools: List of previously used tool names

        Returns:
            Dictionary containing:
            - queries: Different query representations
            - weights: Importance weights for each query type
            - expanded: Synonym-expanded version
            - metadata: Additional extracted information
        """
        # Extract different aspects of the conversation
        last_user_message = self._get_last_user_message(messages)
        full_context = self._get_full_conversation(messages)
        intent = self._extract_intent(last_user_message)
        error_context = self._extract_error_context(messages)
        code_context = self._extract_code_context(messages)

        # Create multiple query representations
        queries = {
            "primary": last_user_message,
            "intent": intent,
            "full_context": full_context,
            "error": error_context,
            "code": code_context,
            "historical": " ".join(used_tools[-3:]) if used_tools else ""
        }

        # Remove empty queries
        queries = {k: v for k, v in queries.items() if v}

        # Expand primary query with synonyms
        expanded_query = self._expand_with_synonyms(last_user_message)

        # Determine weights based on conversation characteristics
        weights = self._calculate_weights(queries, messages)

        # Extract metadata for additional context
        metadata = {
            "has_error": bool(error_context),
            "has_code": bool(code_context),
            "message_count": len(messages),
            "detected_actions": self._extract_actions(last_user_message),
            "topics": self._extract_topics(full_context)
        }

        return {
            "queries": queries,
            "weights": weights,
            "expanded": expanded_query,
            "metadata": metadata
        }

    def _get_last_user_message(self, messages: List[Dict[str, Any]]) -> str:
        """Extract the last user message from the conversation."""
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    # Handle multimodal content
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif isinstance(item, str):
                            text_parts.append(item)
                    return " ".join(text_parts)
        return ""

    def _get_full_conversation(self, messages: List[Dict[str, Any]]) -> str:
        """Extract full conversation text from all messages."""
        text_parts = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_parts.append(item)
        return " ".join(text_parts)

    def _extract_intent(self, text: str) -> str:
        """Extract action intent from the text."""
        if not text:
            return ""

        text_lower = text.lower()
        found_actions = []

        # Find action verbs
        for verb in self.action_verbs:
            if verb in text_lower:
                found_actions.append(verb)

        # Find question words
        question_words = ["how", "what", "where", "when", "why", "which"]
        for word in question_words:
            if word in text_lower:
                # Extract the phrase after the question word
                pattern = rf"\b{word}\s+(?:to\s+)?(\w+)"
                matches = re.findall(pattern, text_lower)
                found_actions.extend(matches)

        # Create intent string
        if found_actions:
            return " ".join(found_actions[:5])  # Limit to top 5 actions

        # Fallback: extract first verb-like word
        words = text_lower.split()
        for word in words:
            if word.endswith(("ing", "ed", "ify", "ize")):
                return word

        return ""

    def _extract_error_context(self, messages: List[Dict[str, Any]]) -> str:
        """Extract error-related context from messages."""
        error_texts = []
        full_text = self._get_full_conversation(messages).lower()

        # Check for error patterns
        for pattern in self.error_patterns:
            if re.search(pattern, full_text, re.IGNORECASE):
                # Extract surrounding context
                matches = re.finditer(pattern, full_text, re.IGNORECASE)
                for match in matches:
                    start = max(0, match.start() - 50)
                    end = min(len(full_text), match.end() + 100)
                    context = full_text[start:end]
                    error_texts.append(context)

        return " ".join(error_texts[:3])  # Limit to 3 error contexts

    def _extract_code_context(self, messages: List[Dict[str, Any]]) -> str:
        """Extract code-related context from messages."""
        code_texts = []
        full_text = self._get_full_conversation(messages)

        # Extract code blocks
        for pattern in self.code_patterns[:3]:  # Check main code patterns
            matches = re.findall(pattern, full_text, re.MULTILINE)
            code_texts.extend(matches[:2])  # Limit matches per pattern

        # Clean up code blocks
        cleaned = []
        for code in code_texts:
            # Remove markdown backticks
            code = re.sub(r"```\w*\n?", "", code)
            code = re.sub(r"```", "", code)
            code = re.sub(r"`", "", code)
            if code.strip():
                cleaned.append(code.strip())

        return " ".join(cleaned[:3])  # Limit to 3 code blocks

    def _expand_with_synonyms(self, query: str) -> str:
        """Expand query with technical synonyms."""
        if not query:
            return ""

        expanded_parts = [query]  # Start with original query
        query_lower = query.lower()

        # Add synonyms for found words
        for word, synonyms in self.synonyms.items():
            if word in query_lower:
                # Add a subset of synonyms
                expanded_parts.extend(synonyms[:3])

        # Also check for partial matches
        words = query_lower.split()
        for word in words:
            for key, synonyms in self.synonyms.items():
                if key in word or word in key:
                    expanded_parts.extend(synonyms[:2])
                    break

        # Remove duplicates while preserving order
        seen = set()
        unique_parts = []
        for part in expanded_parts:
            if part.lower() not in seen:
                seen.add(part.lower())
                unique_parts.append(part)

        return " ".join(unique_parts)

    def _calculate_weights(self, queries: Dict[str, str],
                          messages: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate dynamic weights based on conversation characteristics.

        Args:
            queries: Available query types
            messages: Conversation messages

        Returns:
            Weight dictionary for each query type
        """
        weights = {
            "primary": 0.4,  # Always important
            "intent": 0.3,   # Action understanding
            "full_context": 0.1,  # Background context
            "error": 0.1,    # Error handling
            "code": 0.05,    # Code context
            "historical": 0.05  # Previous tools
        }

        # Adjust weights based on what's available
        if "error" in queries and queries["error"]:
            # Boost error weight if errors detected
            weights["error"] = 0.25
            weights["primary"] = 0.35
            weights["intent"] = 0.25

        if "code" in queries and queries["code"]:
            # Boost code weight if code detected
            weights["code"] = 0.15
            weights["primary"] = 0.35

        # Short conversations rely more on primary query
        if len(messages) <= 2:
            weights["primary"] = 0.5
            weights["full_context"] = 0.05

        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}

    def _extract_actions(self, text: str) -> List[str]:
        """Extract action verbs from text."""
        if not text:
            return []

        text_lower = text.lower()
        found_actions = []

        for verb in self.action_verbs:
            if verb in text_lower:
                found_actions.append(verb)
        
        # Also check for technical terms that might be actions
        technical_action_terms = ["api", "endpoint", "database", "file", "git", "error", "exception"]
        for term in technical_action_terms:
            if term in text_lower and term not in found_actions:
                found_actions.append(term)

        return found_actions[:5]  # Return top 5 actions

    def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics from text using simple keyword frequency."""
        if not text:
            return []

        # Simple topic extraction based on technical terms
        technical_terms = set()
        text_lower = text.lower()

        # Check for known technical domains
        for category, terms in self.synonyms.items():
            if category in text_lower:
                technical_terms.add(category)
            for term in terms:
                if term in text_lower and len(term) > 2:
                    technical_terms.add(category)
                    break

        return list(technical_terms)[:5]  # Return top 5 topics
