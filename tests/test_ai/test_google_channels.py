"""Exercise both Google channels through the real SDK with an offline transport."""

from __future__ import annotations

import asyncio
import base64
import json

import httpx
import pytest
from google import genai
from google.oauth2.credentials import Credentials

from bampy.agent.cancellation import CancellationToken
from bampy.ai import (
    AssistantMessage,
    Context,
    GeminiOptions,
    ImageContent,
    SimpleStreamOptions,
    StopReason,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    VertexAIOptions,
    get_model,
    stream,
    stream_simple,
)


def _chunk(parts=(), *, finish=None, usage=None, **candidate_fields):
    candidate = {"content": {"role": "model", "parts": list(parts)}, **candidate_fields}
    if finish:
        candidate["finishReason"] = finish
    chunk = {"candidates": [candidate], "responseId": "response-1"}
    if usage:
        chunk["usageMetadata"] = usage
    return chunk


def _sse(chunks):
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks).encode()


@pytest.fixture
def sdk(monkeypatch):
    """Replace HTTP only, so SDK request conversion and continuation still run."""
    for name in (
        "GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION", "GOOGLE_GENAI_USE_ENTERPRISE", "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_GEMINI_BASE_URL", "GOOGLE_VERTEX_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)

    class Harness:
        def __init__(self):
            self.replies = []
            self.requests = []
            self.client_kwargs = []
            self.transports = []

        def respond(self, request):
            self.requests.append(request)
            reply = self.replies.pop(0)
            if isinstance(reply, httpx.Response):
                return reply
            return httpx.Response(
                200, headers={"Content-Type": "text/event-stream"}, content=_sse(reply),
            )

        def client(self, **kwargs):
            self.client_kwargs.append(kwargs)
            kwargs["http_options"] = {
                **kwargs.get("http_options", {}),
                "client_args": {"transport": httpx.MockTransport(self.respond)},
                "async_client_args": {"transport": httpx.MockTransport(self.respond)},
            }
            client = real_client(**kwargs)
            self.transports.append((client._api_client._httpx_client, client._api_client._async_httpx_client))
            return client

        def assert_closed(self):
            assert self.transports
            assert all(sync.is_closed and async_.is_closed for sync, async_ in self.transports)

    real_client = genai.Client
    harness = Harness()
    monkeypatch.setattr(genai, "Client", harness.client)
    return harness


def _model(channel="google", model_id="gemini-4-argon"):
    model = get_model(model_id, provider=channel)
    assert model is not None
    return model


def _options(channel="google", **kwargs):
    if channel == "google-vertex":
        return VertexAIOptions(
            project="test-project", location="global",
            credentials=Credentials(token="test-token"), **kwargs,
        )
    return GeminiOptions(api_key="test-key", **kwargs)


@pytest.mark.parametrize("channel", ["google", "google-vertex"])
@pytest.mark.parametrize("model_id", [
    "gemini-4-argon", "gemini-3.8-flash", "gemini-3.7-flash",
    "gemini-3.6-flash", "gemini-3.5-flash-lite",
])
async def test_modern_gemini_request_and_tool_roundtrip(sdk, channel, model_id):
    signature = base64.b64encode(b"opaque-signature").decode()
    sdk.replies.extend([
        [_chunk([{
            "functionCall": {"id": "call:42", "name": "read", "args": {"path": "image.png"}},
            "thoughtSignature": signature,
        }], finish="STOP")],
        [_chunk([{"text": "Done"}], finish="STOP")],
    ])
    model = _model(channel, model_id=model_id)
    options = _options(channel, thinking_level="medium", temperature=0.2)
    context = Context(
        messages=[UserMessage(content="Read image.png")],
        tools=[Tool(name="read", description="Read a file", parameters={"type": "object"})],
    )
    first = await stream(model, context, options).result()
    assert first.stop_reason == StopReason.TOOL_USE
    call = first.content[0]
    assert isinstance(call, ToolCall)
    assert call.id == "call:42"
    assert call.thought_signature == b"opaque-signature"
    context.messages.extend([
        first,
        ToolResultMessage(
            tool_call_id=call.id, tool_name=call.name,
            content=[TextContent(text="file"), ImageContent(data="aGk=", mime_type="image/png")],
        ),
    ])
    second = await stream(model, context, options).result()
    assert second.stop_reason == StopReason.STOP
    body = json.loads(sdk.requests[1].content)
    generation = body["generationConfig"]
    assert generation["thinkingConfig"]["thinking_level"] == "MEDIUM"
    assert "thinking_budget" not in generation["thinkingConfig"]
    assert "temperature" not in generation
    fc_part = body["contents"][1]["parts"][0]
    fr = body["contents"][2]["parts"][0]["functionResponse"]
    assert fc_part["functionCall"]["id"] == fr["id"] == "call:42"
    assert fc_part["thoughtSignature"] == signature
    assert fr["parts"][0]["inline_data"]["mime_type"] == "image/png"
    if model_id == "gemini-4-argon":
        assert sdk.client_kwargs[0]["http_options"]["timeout"] == 1_200_000
    if channel == "google-vertex":
        assert sdk.requests[0].url.host == "aiplatform.googleapis.com"
        assert "/v1beta1/projects/test-project/locations/global/" in sdk.requests[0].url.path
        assert sdk.requests[0].headers["authorization"] == "Bearer test-token"
        assert body["toolConfig"]["functionCallingConfig"]["stream_function_call_arguments"] is False
    else:
        assert sdk.requests[0].url.host == "generativelanguage.googleapis.com"
        assert sdk.requests[0].url.path == f"/v1beta/models/{model_id}:streamGenerateContent"
        assert sdk.requests[0].headers["x-goog-api-key"] == "test-key"
    sdk.assert_closed()


@pytest.mark.parametrize("channel", ["google", "google-vertex"])
@pytest.mark.parametrize("model_id,minimal", [
    ("gemini-3.8-flash", "LOW"), ("gemini-3.7-flash", "LOW"),
    ("gemini-3.6-flash", "MINIMAL"), ("gemini-3.5-flash-lite", "MINIMAL"),
])
@pytest.mark.parametrize("reasoning,expected", [
    (None, None), ("minimal", "minimal"), ("low", "LOW"),
    ("medium", "MEDIUM"), ("high", "HIGH"), ("xhigh", "HIGH"), ("max", "HIGH"),
])
async def test_flash_simple_reasoning_wire_config(sdk, channel, model_id, minimal, reasoning, expected):
    sdk.replies.append([_chunk([{"text": "Hello"}], finish="STOP")])
    options = SimpleStreamOptions(
        api_key="test-key", reasoning=reasoning, temperature=0.5,
    )
    result = await stream_simple(
        _model(channel, model_id=model_id), Context(messages=[UserMessage(content="Hello")]), options,
    ).result()
    assert result.stop_reason == StopReason.STOP, result.error_message
    body = json.loads(sdk.requests[0].content)
    generation = body["generationConfig"]
    thinking = generation["thinkingConfig"]
    assert thinking.get("thinking_level") == (minimal if reasoning == "minimal" else expected)
    assert thinking["include_thoughts"] is True
    assert "thinking_budget" not in thinking
    assert "temperature" not in generation
    assert generation["maxOutputTokens"] == 65_536
    sdk.assert_closed()


@pytest.mark.parametrize("channel", ["google", "google-vertex"])
@pytest.mark.parametrize("model_id", [
    "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite",
])
async def test_new_flash_budget_fails_before_client_creation(sdk, channel, model_id):
    result = await stream(
        _model(channel, model_id=model_id), Context(messages=[UserMessage(content="Hello")]),
        _options(channel, thinking_budget=8192),
    ).result()
    assert result.stop_reason == StopReason.ERROR
    assert "thinking_level" in result.error_message
    assert not sdk.client_kwargs


@pytest.mark.parametrize("model_id", [
    "gemini-4-argon", "gemini-3.8-flash", "gemini-3.7-flash",
    "gemini-3.6-flash", "gemini-3.5-flash-lite",
])
async def test_new_gemini_rejects_assistant_prefills(sdk, model_id):
    model = _model(model_id=model_id)
    result = await stream(model, Context(messages=[
        UserMessage(content="Hello"),
        AssistantMessage(api=model.api, provider=model.provider, model=model.id,
                         content=[TextContent(text="Answer:")]),
    ]), _options()).result()
    assert result.stop_reason == StopReason.ERROR
    assert "prefills" in result.error_message
    assert not sdk.client_kwargs


@pytest.mark.parametrize("channel", ["google", "google-vertex"])
async def test_sdk_continuation_accumulates_usage_once(sdk, channel):
    sdk.replies.extend([
        [_chunk([{"text": "A"}], finish="CONTINUATION", continuationToken="dG9rZW4=", usage={
            "promptTokenCount": 100, "cachedContentTokenCount": 20,
            "candidatesTokenCount": 2, "thoughtsTokenCount": 10, "totalTokenCount": 112,
        })],
        [_chunk([{"text": "B"}], finish="STOP", usage={
            "promptTokenCount": 200, "cachedContentTokenCount": 100,
            "candidatesTokenCount": 4, "thoughtsTokenCount": 20, "totalTokenCount": 224,
        })],
    ])
    result = await stream(
        _model(channel), Context(messages=[UserMessage(content="Continue")]),
        _options(channel, max_tokens=131_072),
    ).result()
    assert result.stop_reason == StopReason.STOP, result.error_message
    assert "".join(block.text for block in result.content) == "AB"
    assert len(sdk.requests) == 2
    assert json.loads(sdk.requests[1].content)["continuationToken"] == "dG9rZW4="
    assert result.usage.input == 180
    assert result.usage.cache_read == 120
    assert result.usage.output == 36
    assert result.usage.total_tokens == 336
    assert result.usage.cost.total == pytest.approx((180 * 2 + 120 * 0.1 + 36 * 10) / 1_000_000)
    sdk.assert_closed()


async def test_vertex_adc_environment_and_simple_options(sdk, monkeypatch):
    import google.auth

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "adc-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "eu")
    monkeypatch.setenv("GOOGLE_API_KEY", "developer-key")
    monkeypatch.setattr(google.auth, "default", lambda **_: (Credentials(token="adc-token"), "adc-project"))
    sdk.replies.append([_chunk([{"text": "Hello"}], finish="STOP")])
    result = await stream_simple(
        _model("google-vertex"), Context(messages=[UserMessage(content="Hello")]),
        SimpleStreamOptions(reasoning="minimal"),
    ).result()
    assert result.stop_reason == StopReason.STOP, result.error_message
    request = sdk.requests[0]
    assert request.url.host == "aiplatform.eu.rep.googleapis.com"
    assert request.headers["authorization"] == "Bearer adc-token"
    assert "x-goog-api-key" not in request.headers
    thinking = json.loads(request.content)["generationConfig"]["thinkingConfig"]
    assert thinking["thinking_level"] == "LOW"
    sdk.assert_closed()


async def test_developer_channel_ignores_vertex_environment(sdk, monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "true")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    sdk.replies.append([_chunk([{"text": "Hello"}], finish="STOP")])
    result = await stream_simple(
        _model(), Context(messages=[UserMessage(content="Hello")]),
        SimpleStreamOptions(api_key="test-key", reasoning="medium"),
    ).result()
    assert result.stop_reason == StopReason.STOP, result.error_message
    assert sdk.requests[0].url.host == "generativelanguage.googleapis.com"
    assert sdk.client_kwargs[0]["enterprise"] is False
    sdk.assert_closed()


async def test_vertex_express_key_and_custom_endpoint(sdk):
    sdk.replies.append([_chunk([{"text": "Hello"}], finish="STOP")])
    model = _model("google-vertex").model_copy(update={
        "base_url": "https://gateway.example", "headers": {"X-Test": "model", "X-Model": "yes"},
    })
    result = await stream(model, Context(messages=[UserMessage(content="Hello")]), VertexAIOptions(
        api_key="express-key", timeout_ms=123_000, headers={"X-Test": "option"},
    )).result()
    assert result.stop_reason == StopReason.STOP, result.error_message
    request = sdk.requests[0]
    assert request.url.host == "gateway.example"
    assert request.headers["x-goog-api-key"] == "express-key"
    assert request.headers["x-test"] == "option"
    assert request.headers["x-model"] == "yes"
    assert sdk.client_kwargs[0]["http_options"]["timeout"] == 123_000
    sdk.assert_closed()


@pytest.mark.parametrize("base_url,api_version,path", [
    ("https://gateway.example/v1beta", None, "/v1beta/models/gemini-4-argon:streamGenerateContent"),
    ("https://gateway.example/proxy/v1beta/", "v1alpha", "/proxy/v1alpha/models/gemini-4-argon:streamGenerateContent"),
])
async def test_legacy_base_url_version_is_not_duplicated(sdk, base_url, api_version, path):
    sdk.replies.append([_chunk([{"text": "Hello"}], finish="STOP")])
    model = _model().model_copy(update={"base_url": base_url})
    result = await stream(model, Context(messages=[UserMessage(content="Hello")]),
                          _options(api_version=api_version)).result()
    assert result.stop_reason == StopReason.STOP, result.error_message
    assert sdk.requests[0].url.path == path
    sdk.assert_closed()


async def test_gemini25_retains_budget_and_temperature(sdk):
    sdk.replies.append([_chunk([{"text": "Hello"}], finish="STOP")])
    result = await stream_simple(
        _model(model_id="gemini-2.5-flash"), Context(messages=[UserMessage(content="Hello")]),
        SimpleStreamOptions(api_key="test-key", reasoning="medium", temperature=0.5),
    ).result()
    assert result.stop_reason == StopReason.STOP, result.error_message
    generation = json.loads(sdk.requests[0].content)["generationConfig"]
    assert generation["thinkingConfig"]["thinking_budget"] == 8192
    assert generation["temperature"] == 0.5
    sdk.assert_closed()


async def test_preserve_signed_parts_including_empty_trailer(sdk):
    sig = base64.b64encode(b"signature").decode()
    sdk.replies.append([
        _chunk([{"text": "Think", "thought": True, "thoughtSignature": sig}]),
        _chunk([{"text": " more", "thought": True}]),
        _chunk([{"text": "Answer"}]),
        _chunk([{"text": "", "thoughtSignature": sig}], finish="STOP"),
    ])
    result = await stream(_model(), Context(messages=[UserMessage(content="Hello")]), _options()).result()
    assert result.stop_reason == StopReason.STOP, result.error_message
    assert result.content == [
        ThinkingContent(thinking="Think", thinking_signature=b"signature"),
        ThinkingContent(thinking=" more"), TextContent(text="Answer"),
        TextContent(text="", text_signature=b"signature"),
    ]
    from bampy.ai.providers.gemini import _convert_messages

    contents = _convert_messages(_model(), Context(messages=[result]))
    assert len(contents[0].parts) == 4
    assert contents[0].parts[-1].thought_signature == b"signature"
    sdk.assert_closed()


@pytest.mark.parametrize("finish,expected", [
    ("MAX_TOKENS", StopReason.LENGTH), ("SAFETY", StopReason.ERROR),
    ("UNEXPECTED_TOOL_CALL", StopReason.ERROR),
])
async def test_tool_calls_do_not_override_failures_or_truncation(sdk, finish, expected):
    sdk.replies.append([_chunk([
        {"functionCall": {"id": "call", "name": "read", "args": {}}},
    ], finish=finish)])
    response = stream(_model(), Context(messages=[UserMessage(content="Hello")]), _options())
    events = [event async for event in response]
    result = await response.result()
    assert result.stop_reason == expected
    assert events[-1].type == ("error" if expected == StopReason.ERROR else "done")
    sdk.assert_closed()


async def test_prompt_block_is_error(sdk):
    sdk.replies.append([{"promptFeedback": {"blockReason": "SAFETY", "blockReasonMessage": "blocked"}}])
    result = await stream(_model(), Context(messages=[UserMessage(content="Hello")]), _options()).result()
    assert result.stop_reason == StopReason.ERROR
    assert "blocked" in result.error_message
    sdk.assert_closed()


async def test_unspecified_feedback_and_finish_are_not_terminal(sdk):
    first = _chunk([{"text": "Hello"}], finish="FINISH_REASON_UNSPECIFIED")
    first["promptFeedback"] = {"blockReason": "BLOCKED_REASON_UNSPECIFIED"}
    sdk.replies.append([first, _chunk([{"text": "!"}], finish="STOP")])
    result = await stream(_model(), Context(messages=[UserMessage(content="Hello")]), _options()).result()
    assert result.stop_reason == StopReason.STOP, result.error_message
    assert result.error_message is None
    assert result.content[0].text == "Hello!"
    sdk.assert_closed()


async def test_incomplete_stream_is_error(sdk):
    sdk.replies.append([_chunk([{"text": "partial"}])])
    result = await stream(_model(), Context(messages=[UserMessage(content="Hello")]), _options()).result()
    assert result.stop_reason == StopReason.ERROR
    assert "finish reason" in result.error_message
    sdk.assert_closed()


async def test_http_error_closes_clients(sdk):
    sdk.replies.append(httpx.Response(400, json={"error": {"code": 400, "message": "bad input"}}))
    result = await stream(_model(), Context(messages=[UserMessage(content="Hello")]), _options()).result()
    assert result.stop_reason == StopReason.ERROR
    assert "bad input" in result.error_message
    sdk.assert_closed()


async def test_cancellation_closes_stream_and_clients(sdk):
    started = asyncio.Event()
    closed = asyncio.Event()

    class PendingStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield _sse([_chunk([{"text": "partial"}])])
            started.set()
            await asyncio.Event().wait()

        async def aclose(self):
            closed.set()

    sdk.replies.append(httpx.Response(200, stream=PendingStream(), headers={"Content-Type": "text/event-stream"}))
    cancellation = CancellationToken()
    response = stream(_model(), Context(messages=[UserMessage(content="Hello")]), _options(cancellation=cancellation))
    await asyncio.wait_for(started.wait(), timeout=2)
    cancellation.cancel("test cancellation")
    result = await asyncio.wait_for(response.result(), timeout=2)
    assert result.stop_reason == StopReason.ABORTED
    assert result.error_message == "test cancellation"
    assert closed.is_set()
    sdk.assert_closed()


@pytest.mark.parametrize("options,expected", [
    (GeminiOptions(thinking_budget=8192), "thinking_budget"),
    (VertexAIOptions(max_tokens=131_072, api_version="v1"), "v1beta1"),
])
async def test_invalid_gemini4_config_fails_before_client_creation(sdk, options, expected):
    result = await stream(_model("google-vertex"), Context(messages=[UserMessage(content="Hello")]), options).result()
    assert result.stop_reason == StopReason.ERROR
    assert expected in result.error_message
    assert not sdk.client_kwargs


async def test_legacy_vertex_v1_url_rejects_long_decode(sdk):
    model = _model("google-vertex").model_copy(update={"base_url": "https://gateway.example/v1"})
    result = await stream(model, Context(messages=[UserMessage(content="Hello")]),
                          VertexAIOptions(max_tokens=131_072)).result()
    assert result.stop_reason == StopReason.ERROR
    assert "v1beta1" in result.error_message
    assert not sdk.client_kwargs


def test_vertex_credentials_are_not_serialized_or_displayed():
    options = VertexAIOptions(credentials=Credentials(token="private-token"))
    assert "credentials" not in options.model_dump()
    assert "private-token" not in repr(options)


def test_vertex_models_have_independent_channel_metadata():
    developer, vertex = _model(), _model("google-vertex")
    assert developer.api == "google-genai"
    assert vertex.api == vertex.provider == "google-vertex"
    assert not vertex.base_url
    assert vertex.context_window == 1_048_576
    assert vertex.max_tokens == 65_536
    assert vertex.reasoning_efforts == ["low", "medium", "high"]
    assert vertex.cost == developer.cost
    assert vertex.cost is not developer.cost
