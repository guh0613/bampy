"""Tool parameter validation using Pydantic JSON Schema.

Validates tool call arguments against the JSON Schema defined for each tool.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError, create_model

from bampy.ai.types import Tool, ToolCall


def validate_tool_arguments(
    arguments: dict[str, Any],
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Validate *arguments* against a JSON Schema dict.

    Returns the validated (and coerced) arguments, or raises
    ``ToolValidationError`` on failure.
    """
    try:
        arguments = _normalize_arguments(arguments, schema)
        # Build a dynamic Pydantic model from the schema
        model = _schema_to_model(schema)
        instance = model.model_validate(arguments)
        return instance.model_dump()
    except (ValidationError, Exception) as exc:
        raise ToolValidationError(str(exc), arguments=arguments, schema=schema) from exc


def validate_tool_call(
    tools: list[Tool],
    tool_call: ToolCall,
) -> dict[str, Any]:
    """Validate a tool call against the matching tool schema."""
    tool = next((tool for tool in tools if tool.name == tool_call.name), None)
    if tool is None:
        raise ToolValidationError(
            f'Tool "{tool_call.name}" not found',
            arguments=tool_call.arguments,
            schema={},
        )
    return validate_tool_arguments(tool_call.arguments, tool.parameters)


class ToolValidationError(Exception):
    """Raised when tool arguments fail validation."""

    def __init__(
        self,
        message: str,
        *,
        arguments: dict[str, Any],
        schema: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.arguments = arguments
        self.schema = schema


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _schema_to_model(schema: dict[str, Any]) -> type[BaseModel]:
    """Create a Pydantic model from a JSON Schema ``properties`` dict."""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    field_definitions: dict[str, Any] = {}

    for name, prop in properties.items():
        py_type = _json_type_to_python(prop)
        nullable = _schema_allows_null(prop)
        default = prop.get("default", None)
        if name in required:
            field_definitions[name] = (py_type, ...)
        else:
            field_definitions[name] = ((py_type | None) if nullable else py_type, default)

    return create_model("ToolArgs", **field_definitions)


_JSON_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def _json_type_to_python(prop: dict[str, Any]) -> type:
    """Map a JSON Schema property definition to a Python type."""
    for union_key in ("anyOf", "oneOf", "allOf"):
        variants = prop.get(union_key)
        if isinstance(variants, list):
            for variant in variants:
                if not isinstance(variant, dict):
                    continue
                variant_type = variant.get("type")
                if variant_type == "null":
                    continue
                return _json_type_to_python(variant)

    json_type = prop.get("type", "string")
    if isinstance(json_type, list):
        # e.g. ["string", "null"]
        non_null = [t for t in json_type if t != "null"]
        json_type = non_null[0] if non_null else "string"
    return _JSON_TYPE_MAP.get(json_type, Any)  # type: ignore[return-value]


def _schema_allows_null(prop: dict[str, Any]) -> bool:
    json_type = prop.get("type")
    if json_type == "null":
        return True
    if isinstance(json_type, list) and "null" in json_type:
        return True

    for union_key in ("anyOf", "oneOf", "allOf"):
        variants = prop.get(union_key)
        if not isinstance(variants, list):
            continue
        for variant in variants:
            if not isinstance(variant, dict):
                continue
            if _schema_allows_null(variant):
                return True
    return False


def _normalize_arguments(arguments: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties", {})
    normalized = dict(arguments)

    for name, prop in properties.items():
        if normalized.get(name) is not None:
            continue
        if name not in normalized:
            continue
        if _schema_allows_null(prop):
            continue
        normalized.pop(name, None)

    return normalized


def schema_from_model(model: type[BaseModel]) -> dict[str, Any]:
    """Generate a JSON Schema dict from a Pydantic model class."""
    return model.model_json_schema()


def parse_partial_json(s: str) -> dict[str, Any]:
    """Best-effort parse of possibly-incomplete JSON (streaming tool args).

    Falls back to empty dict if unparseable.
    """
    s = s.strip()
    if not s:
        return {}
    # Try as-is
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Try closing braces
    for suffix in ("}", '"}', '"}]', "]}"):
        try:
            return json.loads(s + suffix)
        except json.JSONDecodeError:
            continue
    return {}
