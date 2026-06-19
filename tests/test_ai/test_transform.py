"""Tests for bampy.ai.providers._transform."""

from bampy.ai.providers._transform import (
    sanitize_tool_call_id,
    transform_messages,
)
from bampy.ai.types import (
    AssistantMessage,
    ImageContent,
    Model,
    StopReason,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


def _target_model(
    *,
    model_id: str,
    provider: str,
    api: str,
    input_types: list[str] | None = None,
) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        provider=provider,
        api=api,
        input_types=input_types if input_types is not None else ["text", "image"],
    )


class TestSanitizeToolCallId:
    def test_valid_id_unchanged(self):
        assert sanitize_tool_call_id("toolu_abc123") == "toolu_abc123"

    def test_compound_id_split(self):
        result = sanitize_tool_call_id("call_abc|item_xyz")
        assert result == "call_abc"

    def test_invalid_chars_replaced(self):
        result = sanitize_tool_call_id("call@#$abc")
        assert result == "call___abc"

    def test_truncated_to_64(self):
        long_id = "a" * 100
        assert len(sanitize_tool_call_id(long_id)) == 64


class TestTransformMessages:
    def test_passthrough_user_messages(self):
        messages = [UserMessage(content="hello")]
        result = transform_messages(messages)
        assert len(result) == 1
        assert isinstance(result[0], UserMessage)

    def test_thinking_converted_for_different_model(self):
        messages = [
            AssistantMessage(
                api="anthropic-messages",
                provider="anthropic",
                model="claude-old",
                content=[ThinkingContent(thinking="reasoning here")],
            ),
        ]
        result = transform_messages(
            messages,
            target=_target_model(
                model_id="claude-new",
                provider="anthropic",
                api="anthropic-messages",
            ),
        )
        assert len(result) == 1
        msg = result[0]
        assert isinstance(msg, AssistantMessage)
        assert isinstance(msg.content[0], TextContent)
        assert msg.content[0].text == "reasoning here"

    def test_text_signature_dropped_for_different_model(self):
        messages = [
            AssistantMessage(
                api="google-generative-ai",
                provider="google",
                model="gemini-3-flash-preview",
                content=[TextContent(text="done", text_signature=b"sig")],
            ),
        ]
        result = transform_messages(
            messages,
            target=_target_model(
                model_id="claude-sonnet-4-6",
                provider="anthropic",
                api="anthropic-messages",
            ),
        )
        msg = result[0]
        assert isinstance(msg, AssistantMessage)
        assert isinstance(msg.content[0], TextContent)
        assert msg.content[0].text == "done"
        assert msg.content[0].text_signature is None

    def test_redacted_thinking_skipped_for_different_model(self):
        messages = [
            AssistantMessage(
                api="anthropic-messages",
                provider="anthropic",
                model="claude-old",
                content=[ThinkingContent(thinking="", redacted=True)],
            ),
        ]
        result = transform_messages(
            messages,
            target=_target_model(
                model_id="claude-new",
                provider="anthropic",
                api="anthropic-messages",
            ),
        )
        msg = result[0]
        assert isinstance(msg, AssistantMessage)
        assert len(msg.content) == 0

    def test_thinking_preserved_for_same_model(self):
        messages = [
            AssistantMessage(
                api="anthropic-messages",
                provider="anthropic",
                model="claude-same",
                content=[ThinkingContent(thinking="deep thought", thinking_signature="sig123")],
            ),
        ]
        result = transform_messages(
            messages,
            target=_target_model(
                model_id="claude-same",
                provider="anthropic",
                api="anthropic-messages",
            ),
        )
        msg = result[0]
        assert isinstance(msg, AssistantMessage)
        assert isinstance(msg.content[0], ThinkingContent)

    def test_synthetic_tool_results_inserted(self):
        messages = [
            AssistantMessage(
                api="test", provider="test", model="test",
                content=[ToolCall(id="tc1", name="search", arguments={})],
            ),
            # No ToolResultMessage for tc1!
            AssistantMessage(
                api="test", provider="test", model="test",
                content=[TextContent(text="continuation")],
            ),
        ]
        result = transform_messages(messages)
        # Should have synthetic result inserted between the two assistant messages
        assert len(result) == 3
        assert isinstance(result[0], AssistantMessage)
        assert isinstance(result[1], ToolResultMessage)
        assert result[1].is_error is True
        assert isinstance(result[2], AssistantMessage)

    def test_tool_result_resolves_pending(self):
        messages = [
            AssistantMessage(
                api="test", provider="test", model="test",
                content=[ToolCall(id="tc1", name="search", arguments={})],
            ),
            ToolResultMessage(tool_call_id="tc1", tool_name="search", content=[TextContent(text="done")]),
        ]
        result = transform_messages(messages)
        assert len(result) == 2  # No synthetic result needed

    def test_tool_call_ids_normalized(self):
        messages = [
            AssistantMessage(
                api="test", provider="test", model="test",
                content=[ToolCall(id="call@special", name="t", arguments={})],
            ),
            ToolResultMessage(tool_call_id="call@special", tool_name="t", content=[]),
        ]
        result = transform_messages(messages)
        # IDs should be sanitized
        tc = result[0].content[0]
        assert isinstance(tc, ToolCall)
        assert "@" not in tc.id
        tr = result[1]
        assert isinstance(tr, ToolResultMessage)
        assert "@" not in tr.tool_call_id

    def test_same_model_tool_call_ids_preserved(self):
        messages = [
            AssistantMessage(
                api="openai-responses",
                provider="openai",
                model="gpt-5.4",
                content=[ToolCall(id="call_1|fc_1", name="search", arguments={})],
            ),
            ToolResultMessage(tool_call_id="call_1|fc_1", tool_name="search", content=[]),
        ]
        result = transform_messages(
            messages,
            target=_target_model(
                model_id="gpt-5.4",
                provider="openai",
                api="openai-responses",
            ),
        )
        tc = result[0].content[0]
        assert isinstance(tc, ToolCall)
        assert tc.id == "call_1|fc_1"
        tr = result[1]
        assert isinstance(tr, ToolResultMessage)
        assert tr.tool_call_id == "call_1|fc_1"

    def test_cross_model_tool_call_drops_thought_signature(self):
        messages = [
            AssistantMessage(
                api="google-genai",
                provider="google",
                model="gemini-3-flash-preview",
                content=[
                    ToolCall(
                        id="call_1",
                        name="search",
                        arguments={},
                        thought_signature=b"opaque",
                    )
                ],
            ),
        ]
        result = transform_messages(
            messages,
            target=_target_model(
                model_id="claude-sonnet-4-6",
                provider="anthropic",
                api="anthropic-messages",
            ),
        )
        msg = result[0]
        assert isinstance(msg, AssistantMessage)
        tool_call = msg.content[0]
        assert isinstance(tool_call, ToolCall)
        assert tool_call.thought_signature is None

    def test_images_are_downgraded_for_text_only_target_without_mutating_history(self):
        user_message = UserMessage(
            content=[
                TextContent(text="before"),
                ImageContent(data="YQ==", mime_type="image/png"),
                ImageContent(data="Yg==", mime_type="image/jpeg"),
                TextContent(text="between"),
                ImageContent(data="Yw==", mime_type="image/webp"),
            ]
        )
        tool_result = ToolResultMessage(
            tool_call_id="call_1",
            tool_name="read",
            content=[
                ImageContent(data="ZA==", mime_type="image/png"),
                ImageContent(data="ZQ==", mime_type="image/png"),
            ],
        )

        result = transform_messages(
            [user_message, tool_result],
            target=_target_model(
                model_id="text-only",
                provider="custom",
                api="openai-completions",
                input_types=["text"],
            ),
        )

        transformed_user = result[0]
        transformed_tool = result[1]
        assert isinstance(transformed_user, UserMessage)
        assert isinstance(transformed_user.content, list)
        assert [block.text for block in transformed_user.content] == [
            "before",
            "(image omitted: model does not support images)",
            "between",
            "(image omitted: model does not support images)",
        ]
        assert isinstance(transformed_tool, ToolResultMessage)
        assert [block.text for block in transformed_tool.content] == [
            "(tool image omitted: model does not support images)"
        ]

        assert isinstance(user_message.content, list)
        assert sum(isinstance(block, ImageContent) for block in user_message.content) == 3
        assert sum(isinstance(block, ImageContent) for block in tool_result.content) == 2

    def test_images_are_preserved_for_multimodal_target(self):
        message = UserMessage(
            content=[
                TextContent(text="describe"),
                ImageContent(data="YQ==", mime_type="image/png"),
            ]
        )

        result = transform_messages(
            [message],
            target=_target_model(
                model_id="vision",
                provider="custom",
                api="openai-completions",
                input_types=["text", "image"],
            ),
        )

        transformed = result[0]
        assert isinstance(transformed, UserMessage)
        assert isinstance(transformed.content, list)
        assert isinstance(transformed.content[1], ImageContent)

    def test_error_assistant_message_skipped(self):
        messages = [
            AssistantMessage(
                api="test",
                provider="test",
                model="test",
                stop_reason=StopReason.ERROR,
                error_message="boom",
                content=[TextContent(text="partial")],
            ),
            UserMessage(content="retry"),
        ]
        result = transform_messages(messages)
        assert len(result) == 1
        assert isinstance(result[0], UserMessage)
