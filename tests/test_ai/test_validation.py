"""Tests for bampy.ai.validation."""

import pytest

from bampy.ai.validation import (
    ToolValidationError,
    parse_partial_json,
    schema_from_model,
    validate_tool_arguments,
    validate_tool_call,
)
from bampy.ai.types import Tool, ToolCall
from pydantic import BaseModel


class TestValidateToolArguments:
    def test_valid_arguments(self):
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        }
        result = validate_tool_arguments({"query": "test", "limit": 10}, schema)
        assert result["query"] == "test"
        assert result["limit"] == 10

    def test_optional_field_missing(self):
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        }
        result = validate_tool_arguments({"query": "test"}, schema)
        assert result["query"] == "test"
        assert result["limit"] is None

    def test_defaulted_field_uses_schema_default(self):
        schema = {
            "type": "object",
            "properties": {
                "wait_until": {
                    "type": "string",
                    "default": "domcontentloaded",
                },
            },
            "required": [],
        }
        result = validate_tool_arguments({}, schema)
        assert result["wait_until"] == "domcontentloaded"

    def test_non_nullable_null_is_treated_as_missing(self):
        schema = {
            "type": "object",
            "properties": {
                "wait_until": {
                    "type": "string",
                    "default": "domcontentloaded",
                },
            },
            "required": [],
        }
        result = validate_tool_arguments({"wait_until": None}, schema)
        assert result["wait_until"] == "domcontentloaded"

    def test_invalid_arguments_raises(self):
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        }
        with pytest.raises(ToolValidationError):
            validate_tool_arguments({}, schema)

    def test_validate_tool_call(self):
        tools = [
            Tool(
                name="search",
                description="Search",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            )
        ]
        tool_call = ToolCall(id="call_1", name="search", arguments={"query": "hello"})
        result = validate_tool_call(tools, tool_call)
        assert result == {"query": "hello"}

    def test_optional_integer_anyof_schema_accepts_integer(self):
        schema = {
            "type": "object",
            "properties": {
                "timeout_ms": {
                    "anyOf": [
                        {"type": "integer", "minimum": 1},
                        {"type": "null"},
                    ],
                    "default": None,
                },
            },
            "required": [],
        }

        result = validate_tool_arguments({"timeout_ms": 10000}, schema)
        assert result["timeout_ms"] == 10000


class TestSchemaFromModel:
    def test_generate_schema(self):
        class SearchParams(BaseModel):
            query: str
            limit: int = 10

        schema = schema_from_model(SearchParams)
        assert "properties" in schema
        assert "query" in schema["properties"]
        assert "limit" in schema["properties"]


class TestParsePartialJson:
    def test_complete_json(self):
        result = parse_partial_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_incomplete_json(self):
        result = parse_partial_json('{"key": "value"')
        assert result == {"key": "value"}

    def test_empty_string(self):
        result = parse_partial_json("")
        assert result == {}

    def test_totally_broken(self):
        result = parse_partial_json("not json at all <<<")
        assert result == {}
