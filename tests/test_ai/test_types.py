"""Tests for bampy.ai.types."""

from pydantic import TypeAdapter

from bampy.ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    DoneEvent,
    ImageContent,
    Message,
    Model,
    ModelCost,
    SimpleStreamOptions,
    StartEvent,
    StopReason,
    TextContent,
    TextDeltaEvent,
    ThinkingContent,
    ThinkingLevel,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)


class TestContentTypes:
    def test_text_content(self):
        tc = TextContent(text="hello")
        assert tc.type == "text"
        assert tc.text == "hello"
        assert tc.text_signature is None

    def test_thinking_content(self):
        tc = ThinkingContent(thinking="let me think")
        assert tc.type == "thinking"
        assert tc.redacted is False

    def test_image_content(self):
        ic = ImageContent(data="abc123", mime_type="image/png")
        assert ic.type == "image"

    def test_tool_call(self):
        tc = ToolCall(id="tc_1", name="search", arguments={"query": "test"})
        assert tc.type == "tool_call"
        assert tc.arguments["query"] == "test"


class TestMessages:
    def test_user_message_string(self):
        msg = UserMessage(content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.timestamp > 0

    def test_user_message_blocks(self):
        msg = UserMessage(content=[TextContent(text="hi"), ImageContent(data="x", mime_type="image/png")])
        assert len(msg.content) == 2

    def test_assistant_message(self):
        msg = AssistantMessage(
            api="anthropic-messages",
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            content=[TextContent(text="hi")],
        )
        assert msg.role == "assistant"
        assert msg.stop_reason == StopReason.STOP

    def test_tool_result_message(self):
        msg = ToolResultMessage(
            tool_call_id="tc_1",
            tool_name="search",
            content=[TextContent(text="result")],
        )
        assert msg.role == "tool_result"
        assert not msg.is_error

    def test_message_discriminated_union(self):
        adapter = TypeAdapter(Message)
        data = {"role": "user", "content": "hello"}
        msg = adapter.validate_python(data)
        assert isinstance(msg, UserMessage)

    def test_message_serialization_roundtrip(self):
        original = UserMessage(content="hello")
        data = original.model_dump()
        restored = UserMessage.model_validate(data)
        assert restored.content == original.content


class TestModel:
    def test_model_defaults(self):
        m = Model(id="test", name="Test", api="test-api", provider="test")
        assert m.context_window == 128_000
        assert m.max_tokens == 16384
        assert m.reasoning is False
        assert m.input_types == ["text"]

    def test_model_cost(self):
        c = ModelCost(input=3.0, output=15.0)
        assert c.cache_read == 0.0


class TestContext:
    def test_context_with_tools(self):
        ctx = Context(
            system_prompt="You are helpful",
            messages=[UserMessage(content="hi")],
            tools=[Tool(name="search", description="Search", parameters={"type": "object"})],
        )
        assert ctx.system_prompt == "You are helpful"
        assert len(ctx.tools) == 1


class TestStreamOptions:
    def test_simple_stream_options(self):
        opts = SimpleStreamOptions(reasoning=ThinkingLevel.HIGH, temperature=0.5)
        assert opts.reasoning == ThinkingLevel.HIGH

    def test_simple_stream_options_accepts_max_reasoning(self):
        opts = SimpleStreamOptions(reasoning=ThinkingLevel.MAX)
        assert opts.reasoning == ThinkingLevel.MAX


class TestEvents:
    def test_start_event(self):
        msg = AssistantMessage(api="test", provider="test", model="test")
        event = StartEvent(partial=msg)
        assert event.type == "start"

    def test_text_delta_event(self):
        msg = AssistantMessage(api="test", provider="test", model="test")
        event = TextDeltaEvent(content_index=0, delta="hello", partial=msg)
        assert event.type == "text_delta"

    def test_done_event(self):
        msg = AssistantMessage(api="test", provider="test", model="test")
        event = DoneEvent(reason=StopReason.STOP, message=msg)
        assert event.type == "done"

    def test_event_discriminated_union(self):
        adapter = TypeAdapter(AssistantMessageEvent)
        msg = AssistantMessage(api="test", provider="test", model="test")
        event = DoneEvent(reason=StopReason.STOP, message=msg)
        data = event.model_dump()
        restored = adapter.validate_python(data)
        assert isinstance(restored, DoneEvent)


class TestUsage:
    def test_usage_defaults(self):
        u = Usage()
        assert u.input == 0
        assert u.cost.total == 0.0
