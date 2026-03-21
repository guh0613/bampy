"""Layer 3: Application layer.

Session persistence, extension system, tool decorator, context compaction, and
system prompt construction.
"""

# ruff: noqa: F401

# Session management
from .session import (
    SessionManager,
    SessionBackend,
    NDJSONBackend,
    InMemoryBackend,
    SessionHeader,
    SessionEntry,
    SessionMessageEntry,
    ModelChangeEntry,
    ThinkingLevelChangeEntry,
    CompactionEntry,
    BranchSummaryEntry,
    CustomEntry,
    CustomMessageEntry,
    LabelEntry,
    SessionInfoEntry,
    SessionContext,
    SessionTreeNode,
    SessionInfo,
    build_session_context,
)

# Custom message types
from .messages import (
    CompactionSummaryMessage,
    BranchSummaryMessage,
    CustomMessage,
    create_compaction_summary_message,
    create_branch_summary_message,
    create_custom_message,
    convert_app_messages_to_llm,
    register_app_message_converters,
)

# Extension system
from .extension import (
    ExtensionAPI,
    ExtensionRunner,
    ExtensionContext,
    ExtensionFactory,
    ExtensionEvent,
    Extension,
    RegisteredTool,
    RegisteredCommand,
    ToolDefinition,
    wrap_registered_tool,
    # Event types
    SessionStartEvent,
    SessionShutdownEvent,
    SessionCompactEvent,
    ContextEvent,
    BeforeAgentStartEvent,
    AgentStartEvent,
    AgentEndEvent,
    TurnStartEvent,
    TurnEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    MessageEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolExecutionEndEvent,
    ToolCallEvent,
    ToolResultEvent,
    InputEvent,
    # Event result types
    ContextEventResult,
    ToolCallEventResult,
    ToolResultEventResult,
    BeforeAgentStartEventResult,
    InputEventResult,
)

# Extension loader
from .loader import (
    load_extensions,
    discover_and_load_extensions,
    discover_extension_paths,
    LoadExtensionsResult,
)

# Tool decorator and built-in tools
from .tools import (
    tool,
    ToolFromFunction,
    BashTool,
    BashToolInput,
    EditTool,
    EditToolInput,
    FindTool,
    FindToolInput,
    GrepTool,
    GrepToolInput,
    LsTool,
    LsToolInput,
    ReadTool,
    ReadToolInput,
    WriteTool,
    WriteToolInput,
    bash_tool,
    coding_tools,
    create_all_tools,
    create_bash_tool,
    create_coding_tools,
    create_edit_tool,
    create_find_tool,
    create_grep_tool,
    create_ls_tool,
    create_read_only_tools,
    create_read_tool,
    create_write_tool,
    edit_tool,
    find_tool,
    grep_tool,
    ls_tool,
    read_only_tools,
    read_tool,
    write_tool,
)

# Compaction
from .compaction import (
    CompactionSettings,
    CompactionPreparation,
    CompactionResult,
    DEFAULT_COMPACTION_SETTINGS,
    estimate_tokens,
    estimate_context_tokens,
    should_compact,
    prepare_compaction,
    compact,
    generate_summary,
)

# System prompt
from .system_prompt import (
    BuildSystemPromptOptions,
    ContextFile,
    build_system_prompt,
    load_context_files,
)

# Skills
from .skills import (
    Skill,
    SkillCollision,
    SkillDiagnostic,
    LoadSkillsResult,
    load_skills,
    load_skills_from_dir,
    format_skills_for_prompt,
)

# Runtime orchestration
from .runtime import (
    AgentSession,
    AgentSessionEvent,
    AutoCompactionEndEvent,
    AutoCompactionStartEvent,
    CreateAgentSessionResult,
    create_agent_session,
)
