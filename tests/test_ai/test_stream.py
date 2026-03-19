"""Tests for bampy.ai.stream."""

import asyncio

import pytest

from bampy.ai.stream import AssistantMessageEventStream, EventStream
from bampy.ai.types import (
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    StartEvent,
    StopReason,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
)


class TestEventStream:
    async def test_basic_push_and_iterate(self):
        stream: EventStream[int, int] = EventStream()
        stream.push(1)
        stream.push(2)
        stream.push(3)
        stream.end(99)

        events = []
        async for event in stream:
            events.append(event)
        assert events == [1, 2, 3]

    async def test_result(self):
        stream: EventStream[int, int] = EventStream()
        stream.push(1)
        stream.end(42)

        result = await stream.result()
        assert result == 42

    async def test_auto_complete(self):
        stream: EventStream[int, int] = EventStream(
            is_complete=lambda e: e == -1,
            extract_result=lambda e: e * 10,
        )
        stream.push(1)
        stream.push(2)
        stream.push(-1)

        events = []
        async for event in stream:
            events.append(event)
        assert events == [1, 2, -1]

        result = await stream.result()
        assert result == -10

    async def test_end_before_result_await(self):
        stream: EventStream[str, str] = EventStream()
        stream.push("a")
        stream.end("final")

        # Consume events first
        async for _ in stream:
            pass

        # Result should still be available
        result = await stream.result()
        assert result == "final"

    async def test_concurrent_producer_consumer(self):
        stream: EventStream[int, int] = EventStream()
        received: list[int] = []

        async def producer():
            for i in range(5):
                await asyncio.sleep(0.01)
                stream.push(i)
            stream.end(999)

        async def consumer():
            async for event in stream:
                received.append(event)

        await asyncio.gather(producer(), consumer())
        assert received == [0, 1, 2, 3, 4]
        assert await stream.result() == 999


class TestAssistantMessageEventStream:
    async def test_full_stream_lifecycle(self):
        stream = AssistantMessageEventStream()
        output = AssistantMessage(
            api="test", provider="test", model="test",
            content=[],
        )

        # Simulate producer
        async def producer():
            stream.push(StartEvent(partial=output))
            text = TextContent(text="")
            output.content.append(text)
            stream.push(TextStartEvent(content_index=0, content=text, partial=output))
            text.text = "hello"
            stream.push(TextDeltaEvent(content_index=0, delta="hello", partial=output))
            text.text = "hello world"
            stream.push(TextDeltaEvent(content_index=0, delta=" world", partial=output))
            stream.push(TextEndEvent(content_index=0, content=text, partial=output))
            stream.push(DoneEvent(reason=StopReason.STOP, message=output))

        asyncio.get_running_loop().create_task(producer())

        events = []
        async for event in stream:
            events.append(event)

        assert len(events) == 6
        assert isinstance(events[0], StartEvent)
        assert isinstance(events[1], TextStartEvent)
        assert isinstance(events[2], TextDeltaEvent)
        assert isinstance(events[3], TextDeltaEvent)
        assert isinstance(events[4], TextEndEvent)
        assert isinstance(events[5], DoneEvent)

        result = await stream.result()
        assert isinstance(result, AssistantMessage)

    async def test_error_stream(self):
        stream = AssistantMessageEventStream()
        output = AssistantMessage(
            api="test", provider="test", model="test",
            stop_reason=StopReason.ERROR,
            error_message="something went wrong",
        )
        stream.push(ErrorEvent(reason=StopReason.ERROR, error=output))

        events = []
        async for event in stream:
            events.append(event)
        assert len(events) == 1
        assert isinstance(events[0], ErrorEvent)

        result = await stream.result()
        assert result.stop_reason == StopReason.ERROR
