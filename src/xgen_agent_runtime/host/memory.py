"""DB Memory port → conversation history preload for ``PipelineState.messages``.

DB Memory 노드(memory/db_memory_v*)의 ``memory`` 포트는
``List[BaseMessage]`` (langchain_core 메시지)를 내보낸다. geny-executor 는
Anthropic 형태의 ``{"role", "content"}`` 딕셔너리 리스트를 쓰므로 여기서
변환한다. System 메시지는 파이프라인이 시스템 프롬프트를 따로 관리하므로
걸러낸다 (agent_xgen 의 prepare_chat_history 와 동일한 규칙).

이력이 **이미 Anthropic 형태의 평문 dict** 리스트
(``[{"role": "user"|"assistant", "content": str}, …]``)로 올 수도 있어 같은 함수가
그 모양도 받는다 — 객체/dict 가 섞여 있어도 항목별로 처리한다.
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

    Accepts ``List[BaseMessage]`` (duck-typed on ``.type``/``.content``) **or**
    plain dicts ``{"role": "user"|"assistant", "content": ...}`` (already
    Anthropic-shaped; unknown roles / empty content are dropped), tolerates the raw
    ``(messages, context_str)`` tuple shape of the memory node's execute(),
    and returns ``[]`` for anything unusable.
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
        if isinstance(msg, dict):
            # 평문 dict — role 키(없으면 langchain 직렬화의 type 키)로 판별.
            msg_type = msg.get("role", msg.get("type"))
            content = msg.get("content", "")
        else:
            msg_type = getattr(msg, "type", None)
            content = getattr(msg, "content", "")
        role = _ROLE_MAP.get(str(msg_type or "").lower())
        if role is None:
            continue  # system/tool/etc. — not part of the visible history
        text = _content_text(content)
        if text:
            history.append({"role": role, "content": text})
    return history


def history_context(memory_value: Any) -> str:
    """메모리 노드가 ``(최근 메시지, 컨텍스트 문자열)`` 튜플로 보낸 **문자열 쪽**.

    DB Memory 노드는 최근 N개(기본 2개)만 메시지로 넘기고, 그보다 앞선 대화 중 관련 있는 것은
    "[이전 대화 기록 - 참고용]" 문자열로 묶어 둘째 칸에 넣는다(옛 agent_xgen 은 이것을 시스템
    프롬프트 끝에 붙였다). :func:`history_messages` 는 첫째 칸만 읽으므로, 이 함수가 없으면 그
    앞선 대화는 모델에게 가지 않는다. 튜플이 아니거나 문자열이 비었으면 ``""``.
    """
    if isinstance(memory_value, tuple) and len(memory_value) >= 2:
        context = memory_value[1]
        if isinstance(context, str) and context.strip():
            return context.strip()
    return ""
