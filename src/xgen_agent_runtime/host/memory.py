"""DB Memory port → conversation history preload for ``PipelineState.messages``.

DB Memory 노드(memory/db_memory_v*)의 ``memory`` 포트는
``List[BaseMessage]`` (langchain_core 메시지)를 내보낸다. geny-executor 는
Anthropic 형태의 ``{"role", "content"}`` 딕셔너리 리스트를 쓰므로 여기서
변환한다. System 메시지는 파이프라인이 시스템 프롬프트를 따로 관리하므로
걸러낸다 (agent_xgen 의 prepare_chat_history 와 동일한 규칙).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger("editor.geny_bridge.memory")

_ROLE_MAP = {"human": "user", "user": "user", "ai": "assistant", "assistant": "assistant"}


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Multimodal content blocks — keep the text parts only.
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(p for p in parts if p)
    return str(content)


def history_messages(memory_value: Any) -> List[Dict[str, Any]]:
    """Convert the Memory port value into Anthropic-shaped history messages.

    Accepts ``List[BaseMessage]`` (duck-typed on ``.type``/``.content``),
    tolerates the raw ``(messages, context_str)`` tuple shape of the memory
    node's execute(), and returns ``[]`` for anything unusable.
    """
    if memory_value is None:
        return []
    if isinstance(memory_value, tuple) and memory_value:
        memory_value = memory_value[0]
    if not isinstance(memory_value, list):
        logger.debug(
            "geny_bridge: memory port value is %s, expected list — ignoring",
            type(memory_value).__name__,
        )
        return []

    history: List[Dict[str, Any]] = []
    for msg in memory_value:
        msg_type = getattr(msg, "type", None)
        role = _ROLE_MAP.get(str(msg_type or "").lower())
        if role is None:
            continue  # system/tool/etc. — not part of the visible history
        text = _content_text(getattr(msg, "content", ""))
        if text:
            history.append({"role": role, "content": text})
    return history
