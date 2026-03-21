"""Built-in tools and tool helpers."""

# ruff: noqa: F401

from .base import ToolFromFunction, tool
from .bash import BashTool, BashToolInput, bash_tool, create_bash_tool
from .edit import EditTool, EditToolInput, create_edit_tool, edit_tool
from .find import FindTool, FindToolInput, create_find_tool, find_tool
from .grep import GrepTool, GrepToolInput, create_grep_tool, grep_tool
from .ls import LsTool, LsToolInput, create_ls_tool, ls_tool
from .read import ReadTool, ReadToolInput, create_read_tool, read_tool
from .truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    GREP_MAX_LINE_LENGTH,
    TruncationResult,
    format_size,
    serialize_truncation,
    truncate_head,
    truncate_line,
    truncate_tail,
)
from .write import WriteTool, WriteToolInput, create_write_tool, write_tool

coding_tools = [read_tool, bash_tool, edit_tool, write_tool]
read_only_tools = [read_tool, grep_tool, find_tool, ls_tool]
all_tools = {
    "read": read_tool,
    "bash": bash_tool,
    "edit": edit_tool,
    "write": write_tool,
    "grep": grep_tool,
    "find": find_tool,
    "ls": ls_tool,
}


def create_coding_tools(cwd: str):
    return [
        create_read_tool(cwd),
        create_bash_tool(cwd),
        create_edit_tool(cwd),
        create_write_tool(cwd),
    ]


def create_read_only_tools(cwd: str):
    return [
        create_read_tool(cwd),
        create_grep_tool(cwd),
        create_find_tool(cwd),
        create_ls_tool(cwd),
    ]


def create_all_tools(cwd: str):
    return {
        "read": create_read_tool(cwd),
        "bash": create_bash_tool(cwd),
        "edit": create_edit_tool(cwd),
        "write": create_write_tool(cwd),
        "grep": create_grep_tool(cwd),
        "find": create_find_tool(cwd),
        "ls": create_ls_tool(cwd),
    }
