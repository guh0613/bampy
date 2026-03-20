"""Helpers for working with agent messages and custom message converters."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, TypeAdapter, ValidationError

from bampy.ai.types import AssistantMessage, Message, ToolResultMessage, UserMessage

_MESSAGE_ADAPTER = TypeAdapter(Message)


@dataclass(slots=True)
class RegisteredMessageConverter:
    role: str
    converter: Any
    source_id: str | None = None


_registry: dict[str, RegisteredMessageConverter] = {}


def register_message_converter(
    role: str,
    converter: Any,
    *,
    source_id: str | None = None,
) -> None:
    """Register a converter for a custom agent message role."""
    _registry[role] = RegisteredMessageConverter(
        role=role,
        converter=converter,
        source_id=source_id,
    )


def unregister_message_converter(role: str) -> None:
    _registry.pop(role, None)


def unregister_message_converters(source_id: str) -> None:
    for role, entry in list(_registry.items()):
        if entry.source_id == source_id:
            _registry.pop(role, None)


def clear_message_converters() -> None:
    _registry.clear()


def message_role(message: Any) -> str | None:
    """Best-effort role lookup for Pydantic models, mappings, and custom objects."""
    if isinstance(message, Mapping):
        role = message.get("role")
        return str(role) if role is not None else None
    role = getattr(message, "role", None)
    return str(role) if role is not None else None


def message_timestamp(message: Any) -> float | None:
    """Best-effort timestamp lookup."""
    if isinstance(message, Mapping):
        value = message.get("timestamp")
    else:
        value = getattr(message, "timestamp", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def coerce_llm_message(message: Any) -> Message | None:
    """Convert mappings or Pydantic models into standard LLM message types."""
    if isinstance(message, (UserMessage, AssistantMessage, ToolResultMessage)):
        return message
    if isinstance(message, BaseModel):
        candidate = message.model_dump(exclude_none=False)
    elif isinstance(message, Mapping):
        candidate = dict(message)
    else:
        return None
    try:
        return _MESSAGE_ADAPTER.validate_python(candidate)
    except ValidationError:
        return None


def is_llm_message(message: Any) -> bool:
    return coerce_llm_message(message) is not None


def is_assistant_message(message: Any) -> bool:
    llm_message = coerce_llm_message(message)
    if llm_message is not None:
        return isinstance(llm_message, AssistantMessage)
    return message_role(message) == "assistant"


def clone_message(message: Any) -> Any:
    """Clone a message before emitting it to listeners."""
    if isinstance(message, BaseModel):
        return message.model_copy(deep=True)
    try:
        return deepcopy(message)
    except Exception:
        return message


def convert_message_to_llm(message: Any) -> list[Message]:
    """Convert one agent message into zero or more LLM-compatible messages."""
    llm_message = coerce_llm_message(message)
    if llm_message is not None:
        return [llm_message]

    role = message_role(message)
    if role is None:
        return []

    registered = _registry.get(role)
    if registered is None:
        return []

    converted = registered.converter(message)
    if converted is None:
        return []
    if isinstance(converted, (UserMessage, AssistantMessage, ToolResultMessage)):
        return [converted]
    if isinstance(converted, BaseModel):
        llm_message = coerce_llm_message(converted)
        return [llm_message] if llm_message is not None else []
    if isinstance(converted, Iterable) and not isinstance(converted, (str, bytes, bytearray)):
        result: list[Message] = []
        for item in converted:
            llm_item = coerce_llm_message(item)
            if llm_item is not None:
                result.append(llm_item)
        return result

    llm_message = coerce_llm_message(converted)
    return [llm_message] if llm_message is not None else []


def default_convert_to_llm(messages: list[Any]) -> list[Message]:
    """Default agent message conversion.

    Standard LLM messages pass through unchanged. Custom messages are converted
    only when a converter for their role is registered.
    """
    converted: list[Message] = []
    for message in messages:
        converted.extend(convert_message_to_llm(message))
    return converted

