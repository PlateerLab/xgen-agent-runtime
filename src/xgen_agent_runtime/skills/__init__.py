"""Skill system — code-free capability units.

A Skill is a ``SKILL.md`` file (YAML frontmatter + markdown body) that
lets users extend an agent's capability without writing Python. The
executor loads skills at session start, exposes them through a Tool
wrapper, and routes LLM invocations back through the skill's prompt
body.

Phase 4 Week 7 ships the foundation:
- :class:`Skill` + :class:`SkillMetadata` + :class:`SkillContext` types
- :func:`parse_skill_file` / :func:`load_skills_dir` loaders
- :class:`SkillRegistry` in-memory store
- :func:`parse_frontmatter` YAML parser (stdlib + pyyaml)

Later phases add:
- ``SkillTool`` — exposes a skill as a ``Tool`` (Phase 4 Week 8)
- Bundled skills + CLI helpers (Phase 4 Week 9)
"""

from xgen_agent_runtime.skills.bundled_skills import (
    bundled_skill_ids,
    bundled_skills_dir,
    load_bundled_skills,
)
from xgen_agent_runtime.skills.fork import (
    ForkResult,
    SkillForkRunner,
    make_default_fork_runner,
)
from xgen_agent_runtime.skills.frontmatter import parse_frontmatter
from xgen_agent_runtime.skills.loader import (
    SKILL_FILENAME,
    SkillLoadError,
    SkillLoadReport,
    load_skills_dir,
    parse_skill_file,
)
from xgen_agent_runtime.skills.mcp_bridge import (
    SKILL_ID_PREFIX as MCP_SKILL_ID_PREFIX,
    SKILL_SOURCE_TAG as MCP_SKILL_SOURCE_TAG,
    mcp_prompts_to_skills,
    mcp_skill_id,
)
from xgen_agent_runtime.skills.registry import SkillRegistry
from xgen_agent_runtime.skills.shell_blocks import (
    ShellBlock,
    ShellRunOutcome,
    ShellRunSummary,
    execute_blocks,
    is_trusted_source,
    parse_blocks,
)
from xgen_agent_runtime.skills.skill_tool import (
    SkillTool,
    SkillToolProvider,
    build_skill_tool,
)
from xgen_agent_runtime.skills.types import (
    Skill,
    SkillContext,
    SkillMetadata,
    validate_execution_mode,
)
from xgen_agent_runtime.skills.watcher import SkillRegistryWatcher

__all__ = [
    "MCP_SKILL_ID_PREFIX",
    "MCP_SKILL_SOURCE_TAG",
    "Skill",
    "SkillContext",
    "SkillMetadata",
    "SkillRegistry",
    "SkillTool",
    "SkillToolProvider",
    "SkillLoadError",
    "SkillLoadReport",
    "SKILL_FILENAME",
    # Phase 10.3 — shell-block execution
    "ShellBlock",
    "ShellRunOutcome",
    "ShellRunSummary",
    "execute_blocks",
    "is_trusted_source",
    "parse_blocks",
    # Phase 10.4 — bundled skill catalog
    "bundled_skill_ids",
    "bundled_skills_dir",
    "load_bundled_skills",
    # Phase 10.5 — fork execution mode
    "ForkResult",
    "SkillForkRunner",
    "make_default_fork_runner",
    # Phase 10.7 — hot-reload watcher
    "SkillRegistryWatcher",
    # Loaders / parsers
    "load_skills_dir",
    "parse_skill_file",
    "parse_frontmatter",
    "mcp_prompts_to_skills",
    "mcp_skill_id",
    "validate_execution_mode",
    "build_skill_tool",
]
