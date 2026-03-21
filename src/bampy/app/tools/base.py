"""``@tool`` decorator and tool creation helpers.

Provides two ways to define tools:

1. **Decorator** — for simple tools::

       @tool(name="my_tool", description="Does something")
       async def my_tool(param: str) -> ToolResult:
           return ToolResult(content=[TextContent(text="done")])

2. **Class** — inherit from :class:`AgentTool` for complex tools that need
   state or streaming updates (use layer-2 ``AgentTool`` protocol directly).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

from pydantic import BaseModel, TypeAdapter, ValidationError, create_model

from bampy.agent.cancellation import CancellationToken
from bampy.agent.types import AgentToolResult, AgentToolUpdateCallback, ToolParameters
from bampy.ai.types import TextContent, ToolResultContentBlock

_TOOL_RESULT_BLOCK_ADAPTER = TypeAdapter(ToolResultContentBlock)


def _build_pydantic_model_from_fn(fn: Callable[..., Any]) -> type[BaseModel]:
    """Create a Pydantic model from a function's signature for JSON schema generation."""
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)

    fields: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        # Skip special parameters
        if name in ("self", "tool_call_id", "cancellation", "on_update"):
            continue
        annotation = hints.get(name, Any)
        if param.default is inspect.Parameter.empty:
            fields[name] = (annotation, ...)
        else:
            fields[name] = (annotation, param.default)

    model = create_model(f"{fn.__name__}_Params", **fields)
    return model


class ToolFromFunction:
    """Wraps a plain async function as an :class:`AgentTool`."""

    __slots__ = ("name", "label", "description", "parameters", "_fn")

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        label: str | None = None,
        description: str | None = None,
        parameters: ToolParameters | None = None,
    ) -> None:
        self.name = name or fn.__name__
        self.label = label or self.name
        self.description = description or (fn.__doc__ or "").strip() or self.name

        if parameters is not None:
            self.parameters = parameters
        else:
            self.parameters = _build_pydantic_model_from_fn(fn)

        self._fn = fn

    async def execute(
        self,
        tool_call_id: str,
        params: Any,
        cancellation: CancellationToken | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        # Unpack params
        if isinstance(params, BaseModel):
            kwargs = params.model_dump()
        elif isinstance(params, dict):
            kwargs = dict(params)
        else:
            kwargs = {}

        # Inject optional parameters if the function accepts them
        sig = inspect.signature(self._fn)
        param_names = set(sig.parameters.keys())
        if "tool_call_id" in param_names:
            kwargs["tool_call_id"] = tool_call_id
        if "cancellation" in param_names:
            kwargs["cancellation"] = cancellation
        if "on_update" in param_names:
            kwargs["on_update"] = on_update

        result = self._fn(**kwargs)
        if inspect.isawaitable(result):
            result = await result

        if isinstance(result, AgentToolResult):
            return result
        if isinstance(result, str):
            return AgentToolResult(content=[TextContent(text=result)])
        if isinstance(result, (list, tuple)):
            return AgentToolResult(content=_normalize_blocks(result))
        try:
            block = _TOOL_RESULT_BLOCK_ADAPTER.validate_python(result)
        except ValidationError:
            pass
        else:
            return AgentToolResult(content=[block])
        return AgentToolResult(content=[TextContent(text=str(result))])


def tool(
    name: str | None = None,
    *,
    label: str | None = None,
    description: str | None = None,
    parameters: ToolParameters | None = None,
) -> Callable[[Callable[..., Any]], ToolFromFunction]:
    """Decorator to create an :class:`AgentTool` from a function.

    Usage::

        @tool(name="greet", description="Say hello")
        async def greet(name: str) -> str:
            return f"Hello, {name}!"
    """

    def decorator(fn: Callable[..., Any]) -> ToolFromFunction:
        return ToolFromFunction(
            fn,
            name=name,
            label=label,
            description=description,
            parameters=parameters,
        )

    return decorator


def _normalize_blocks(values: list[Any] | tuple[Any, ...]) -> list[ToolResultContentBlock]:
    blocks: list[ToolResultContentBlock] = []
    for item in values:
        if isinstance(item, str):
            blocks.append(TextContent(text=item))
            continue
        try:
            blocks.append(_TOOL_RESULT_BLOCK_ADAPTER.validate_python(item))
        except ValidationError as exc:
            raise TypeError(f"Unsupported tool content block: {item!r}") from exc
    return blocks
