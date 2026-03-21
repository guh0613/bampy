"""Live integration tests for bampy.app.runtime.

These tests intentionally hit the real Gemini API through the app-layer
AgentSession/runtime orchestration.

Live tests require .env.dev at project root (or environment variables):
  GEMINI_API_KEY  - Google AI API key
  GEMINI_BASE_URL - optional custom proxy base URL
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bampy.ai import SimpleStreamOptions, get_model
from bampy.ai.types import AssistantMessage, Model, StopReason, TextContent, ToolResultMessage
from bampy.app import CompactionSettings, SessionManager, create_agent_session

# ---------------------------------------------------------------------------
# Load .env.dev if present
# ---------------------------------------------------------------------------

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env.dev"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())

_API_KEY = os.environ.get("GEMINI_API_KEY", "")
_BASE_URL = os.environ.get("GEMINI_BASE_URL", "")
_TEST_MODEL = "gemini-3.1-flash-lite-preview"

live = pytest.mark.live
requires_live_api = pytest.mark.skipif(not _API_KEY, reason="GEMINI_API_KEY not set")


def _model(
    model_id: str = _TEST_MODEL,
    *,
    context_window: int | None = None,
) -> Model:
    model = get_model(model_id, provider="google")
    assert model is not None, f"Model {model_id} not found in registry"
    updates: dict[str, object] = {}
    if _BASE_URL:
        updates["base_url"] = _BASE_URL
    if context_window is not None:
        updates["context_window"] = context_window
    if updates:
        model = model.model_copy(update=updates)
    return model


def _text(message: AssistantMessage | dict) -> str:
    content = message.content if hasattr(message, "content") else message.get("content", [])
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for block in content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def _write_live_project(tmp_path: Path) -> Path:
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    (project_dir / "CLAUDE.md").write_text(
        "Always prefer concise, factual answers.\n",
        encoding="utf-8",
    )
    (project_dir / "notes.txt").write_text(
        "Launch code: AURORA-17\nRegion hint: shanghai-edge\n",
        encoding="utf-8",
    )

    skill_dir = project_dir / ".bampy" / "skills" / "live-debug"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: live-debug\n"
        "description: Help with runtime live validation.\n"
        "---\n\n"
        "# Live Debug\n",
        encoding="utf-8",
    )

    extension_dir = project_dir / ".bampy" / "extensions"
    extension_dir.mkdir(parents=True)
    (extension_dir / "live_ext.py").write_text(
        "from bampy.agent import AgentToolResult\n"
        "from bampy.ai import TextContent\n"
        "from bampy.app import BeforeAgentStartEventResult, ToolDefinition, ToolResultEventResult\n\n"
        "def setup(api):\n"
        "    async def project_lookup(tool_call_id, params, cancellation, on_update, ctx):\n"
        "        del tool_call_id, cancellation, on_update, ctx\n"
        "        mapping = {\n"
        "            'deployment_region': 'shanghai-edge',\n"
        "            'owner': 'bampy-live',\n"
        "        }\n"
        "        value = mapping.get(params.get('key', ''), 'unknown')\n"
        "        return AgentToolResult(\n"
        "            content=[TextContent(text=value)],\n"
        "            details={'source': 'live_ext'},\n"
        "        )\n\n"
        "    def before_agent_start(event, ctx):\n"
        "        del ctx\n"
        "        return BeforeAgentStartEventResult(\n"
        "            system_prompt=event.system_prompt + '\\nUse project_lookup for stable project metadata when directly relevant.'\n"
        "        )\n\n"
        "    def tool_result(event, ctx):\n"
        "        del ctx\n"
        "        if event.tool_name == 'project_lookup':\n"
        "            return ToolResultEventResult(details={'source': 'live_ext', 'validated': True})\n"
        "        return None\n\n"
        "    api.on('before_agent_start', before_agent_start)\n"
        "    api.on('tool_result', tool_result)\n"
        "    api.register_tool(ToolDefinition(\n"
        "        name='project_lookup',\n"
        "        label='Project Lookup',\n"
        "        description='Lookup stable project metadata by key.',\n"
        "        parameters={\n"
        "            'type': 'object',\n"
        "            'properties': {'key': {'type': 'string'}},\n"
        "            'required': ['key'],\n"
        "        },\n"
        "        execute=project_lookup,\n"
        "        prompt_snippet='project_lookup: resolve stable project metadata keys such as deployment_region or owner.',\n"
        "    ))\n",
        encoding="utf-8",
    )

    return project_dir


@live
@requires_live_api
class TestLiveAgentSession:
    """Real end-to-end checks against the app-layer runtime."""

    async def test_end_to_end_tools_extensions_and_manual_compaction(self, tmp_path: Path):
        project_dir = _write_live_project(tmp_path)

        result = await create_agent_session(
            cwd=str(project_dir),
            model=_model(),
            thinking_level="off",
            session_manager=SessionManager.in_memory(str(project_dir)),
            stream_options=SimpleStreamOptions(api_key=_API_KEY),
            max_turns=12,
        )
        session = result.session

        try:
            assert len(result.extensions.errors) == 0
            assert len(result.extensions.extensions) == 1
            assert [skill.name for skill in result.skills.skills] == ["live-debug"]
            assert "project_lookup" in session.system_prompt
            assert "live-debug" in session.system_prompt
            assert "Always prefer concise" in session.system_prompt

            await session.prompt("Reply with exactly LIVE_OK and nothing else.")
            first_reply = _text(session.messages[-1])
            assert "live_ok" in first_reply.lower()

            await session.prompt(
                "Do not guess. Use the read tool to inspect notes.txt and tell me the launch code only."
            )
            read_results = [
                message
                for message in session.messages
                if isinstance(message, ToolResultMessage) and message.tool_name == "read"
            ]
            second_reply = _text(session.messages[-1])
            assert len(read_results) >= 1
            assert "aurora-17" in second_reply.lower()

            await session.prompt(
                "Use the project_lookup tool to get the deployment_region key, then answer in one short sentence."
            )
            lookup_results = [
                message
                for message in session.messages
                if isinstance(message, ToolResultMessage) and message.tool_name == "project_lookup"
            ]
            third_reply = _text(session.messages[-1])
            assert len(lookup_results) >= 1
            assert lookup_results[-1].details == {"source": "live_ext", "validated": True}
            assert "shanghai" in third_reply.lower()

            compaction_result = await session.compact()
            assert compaction_result is not None
            assert len(compaction_result.summary) > 0
            assert session.messages[0].role == "compaction_summary"

            await session.prompt("What deployment region did you just use? Answer briefly.")
            # Exact factual recall after summarization is model-dependent; the
            # runtime contract we care about is that the compacted session can
            # continue successfully.
            fourth_message = session.messages[-1]
            assert isinstance(fourth_message, AssistantMessage)
            assert fourth_message.stop_reason == StopReason.STOP
        finally:
            await session.close()

    async def test_auto_compaction_with_real_model(self, tmp_path: Path):
        project_dir = _write_live_project(tmp_path)
        result = await create_agent_session(
            cwd=str(project_dir),
            model=_model(context_window=600),
            thinking_level="off",
            session_manager=SessionManager.in_memory(str(project_dir)),
            stream_options=SimpleStreamOptions(api_key=_API_KEY),
            compaction_settings=CompactionSettings(
                enabled=True,
                reserve_tokens=250,
                keep_recent_tokens=120,
            ),
            auto_compaction=True,
            max_turns=12,
        )
        session = result.session
        events: list[str] = []
        session.subscribe(lambda event: events.append(event.type))

        try:
            await session.prompt(
                "In about 350 words, explain the difference between a stack and a queue. "
                "Include two tiny examples and keep it clear but reasonably detailed."
            )

            assert "auto_compaction_start" in events
            assert "auto_compaction_end" in events
            assert session.messages[0].role == "compaction_summary"

            await session.prompt(
                "What topic were we discussing just now? Answer in one short sentence."
            )
            followup_reply = _text(session.messages[-1])
            assert any(term in followup_reply.lower() for term in ("stack", "queue"))
        finally:
            await session.close()
