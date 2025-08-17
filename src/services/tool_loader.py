"""Tool loader for sample MCP tools."""

from typing import List, Dict, Any
from src.core.models import Tool


# Sample tools for testing
SAMPLE_TOOLS = [
    {
        "name": "grep",
        "description": "Search for patterns in files using regular expressions",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regular expression pattern to search"},
                "file": {"type": "string", "description": "File to search in"},
                "recursive": {"type": "boolean", "description": "Search recursively in directories"}
            },
            "required": ["pattern"]
        },
        "category": "search"
    },
    {
        "name": "find",
        "description": "Find files and directories by name, type, or other attributes",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name pattern to match"},
                "type": {"type": "string", "enum": ["f", "d"], "description": "Type: f for file, d for directory"},
                "path": {"type": "string", "description": "Starting path for search"}
            },
            "required": ["name"]
        },
        "category": "search"
    },
    {
        "name": "ls",
        "description": "List directory contents with details",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to list"},
                "all": {"type": "boolean", "description": "Show hidden files"},
                "long": {"type": "boolean", "description": "Show detailed information"}
            }
        },
        "category": "file_management"
    },
    {
        "name": "cat",
        "description": "Display file contents",
        "parameters": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "File to display"}
            },
            "required": ["file"]
        },
        "category": "file_management"
    },
    {
        "name": "echo",
        "description": "Display a line of text or variable values",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to display"}
            },
            "required": ["text"]
        },
        "category": "output"
    },
    {
        "name": "cd",
        "description": "Change the current directory",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to change to"}
            },
            "required": ["path"]
        },
        "category": "navigation"
    },
    {
        "name": "pwd",
        "description": "Print the current working directory",
        "parameters": {
            "type": "object",
            "properties": {}
        },
        "category": "navigation"
    },
    {
        "name": "mkdir",
        "description": "Create directories",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to create"},
                "parents": {"type": "boolean", "description": "Create parent directories if needed"}
            },
            "required": ["path"]
        },
        "category": "file_management"
    },
    {
        "name": "rm",
        "description": "Remove files or directories",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to remove"},
                "recursive": {"type": "boolean", "description": "Remove directories recursively"},
                "force": {"type": "boolean", "description": "Force removal without confirmation"}
            },
            "required": ["path"]
        },
        "category": "file_management"
    },
    {
        "name": "cp",
        "description": "Copy files or directories",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Source path"},
                "destination": {"type": "string", "description": "Destination path"},
                "recursive": {"type": "boolean", "description": "Copy directories recursively"}
            },
            "required": ["source", "destination"]
        },
        "category": "file_management"
    },
    {
        "name": "mv",
        "description": "Move or rename files and directories",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Source path"},
                "destination": {"type": "string", "description": "Destination path"}
            },
            "required": ["source", "destination"]
        },
        "category": "file_management"
    },
    {
        "name": "touch",
        "description": "Create empty files or update file timestamps",
        "parameters": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "File to create or update"}
            },
            "required": ["file"]
        },
        "category": "file_management"
    },
    {
        "name": "chmod",
        "description": "Change file permissions",
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "description": "Permission mode (e.g., 755)"},
                "file": {"type": "string", "description": "File or directory to modify"}
            },
            "required": ["mode", "file"]
        },
        "category": "permissions"
    },
    {
        "name": "tail",
        "description": "Display the last lines of a file",
        "parameters": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "File to display"},
                "lines": {"type": "integer", "description": "Number of lines to show"},
                "follow": {"type": "boolean", "description": "Follow file updates"}
            },
            "required": ["file"]
        },
        "category": "file_management"
    },
    {
        "name": "head",
        "description": "Display the first lines of a file",
        "parameters": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "File to display"},
                "lines": {"type": "integer", "description": "Number of lines to show"}
            },
            "required": ["file"]
        },
        "category": "file_management"
    },
    {
        "name": "sed",
        "description": "Stream editor for filtering and transforming text",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Sed pattern/command"},
                "file": {"type": "string", "description": "File to process"}
            },
            "required": ["pattern"]
        },
        "category": "text_processing"
    },
    {
        "name": "awk",
        "description": "Pattern scanning and processing language",
        "parameters": {
            "type": "object",
            "properties": {
                "program": {"type": "string", "description": "AWK program"},
                "file": {"type": "string", "description": "File to process"}
            },
            "required": ["program"]
        },
        "category": "text_processing"
    },
    {
        "name": "sort",
        "description": "Sort lines in text files",
        "parameters": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "File to sort"},
                "numeric": {"type": "boolean", "description": "Sort numerically"},
                "reverse": {"type": "boolean", "description": "Reverse sort order"}
            }
        },
        "category": "text_processing"
    },
    {
        "name": "uniq",
        "description": "Report or omit repeated lines",
        "parameters": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "File to process"},
                "count": {"type": "boolean", "description": "Prefix lines with occurrence count"}
            }
        },
        "category": "text_processing"
    },
    {
        "name": "wc",
        "description": "Print line, word, and byte counts for files",
        "parameters": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "File to count"},
                "lines": {"type": "boolean", "description": "Count lines only"},
                "words": {"type": "boolean", "description": "Count words only"}
            }
        },
        "category": "text_processing"
    }
]


def get_sample_tools() -> List[Tool]:
    """Get sample tools in the correct format."""
    tools = []
    for tool_def in SAMPLE_TOOLS:
        tool = Tool.from_mcp(
            name=tool_def["name"],
            description=tool_def["description"],
            parameters=tool_def["parameters"],
            category=tool_def["category"]
        )
        tools.append(tool)
    return tools


def get_tools_by_category(category: str) -> List[Tool]:
    """Get tools filtered by category."""
    return [
        Tool.from_mcp(
            name=tool_def["name"],
            description=tool_def["description"],
            parameters=tool_def["parameters"],
            category=tool_def["category"]
        )
        for tool_def in SAMPLE_TOOLS
        if tool_def["category"] == category
    ]


def get_tool_categories() -> List[str]:
    """Get all available tool categories."""
    return list(set(tool["category"] for tool in SAMPLE_TOOLS))