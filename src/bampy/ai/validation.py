"""Tool parameter validation using Pydantic JSON Schema.

Validates tool call arguments against the JSON Schema defined for each tool.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError, create_model


def validate_tool_arguments(
    arguments: dict[str, Any],
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Validate *arguments* against a JSON Schema dict.

    Returns the validated (and coerced) arguments, or raises
    ``ToolValidationError`` on failure.
    """
    try:
        # Build a dynamic Pydantic model from the schema
        model = _schema_to_model(schema)
        instance = model.model_validate(arguments)
        return instance.model_dump()
    except (ValidationError, Exception) as exc:
        raise ToolValidationError(str(exc), arguments=arguments, schema=schema) from exc


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
        if name in required:
            field_definitions[name] = (py_type, ...)
        else:
            field_definitions[name] = (py_type | None, None)

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
    json_type = prop.get("type", "string")
    if isinstance(json_type, list):
        # e.g. ["string", "null"]
        non_null = [t for t in json_type if t != "null"]
        json_type = non_null[0] if non_null else "string"
    return _JSON_TYPE_MAP.get(json_type, Any)  # type: ignore[return-value]


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
