"""DocsContext port → pre-searched ``[DOC_n]`` context block (+ embedded tools).

A VectorDB retrieval/context node outputs a *retrieval config* dict —
``{"rag_service": <client>, "search_params": {...}, "status": "ready"}`` —
NOT pre-fetched text. Real search is delegated to the shared
``rag_context_builder`` helper (the same one agent_xgen uses) so both agent
nodes keep one retrieval/citation behavior, including ontology mode, hybrid
search and the strict ``[DOC_n]`` citation prompt.

VectorDB Context 통합 노드가 도구 모드로 연결되면 dict 에 ``tool`` 이 실려
온다 — 그 경우 사전 검색 대신 에이전트 도구로 등록해야 하므로 분리해서
돌려준다. 문자열/legacy 값은 그대로 컨텍스트 블록에 합류시킨다.
"""

from __future__ import annotations

import logging
from typing import Any, List, Tuple

logger = logging.getLogger("editor.geny_bridge.rag")


def _coerce_legacy(value: Any) -> str:
    """Old-style Context values: plain strings or ``{content|text: ...}`` dicts."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("content") or value.get("text") or "")
    return str(value)


def collect_rag(
    text: str,
    port_value: Any,
    *,
    context_builder: Any = None,
) -> Tuple[str, List[Any]]:
    """Resolve the Context port into ``(context_block, embedded_tools)``.

    ``context_block`` is ready to append to the user turn (agent_xgen folds
    RAG results into the user message, not the system prompt). Search
    failures degrade gracefully: the turn proceeds without that source.

    ``context_builder`` is the host-provided RAG builder ``(text, item) ->
    Optional[str]`` (server = the agent node's rag_context_builder; connector
    = None → those items are skipped). Injected so this module stays
    import-clean of xgen-workflow.
    """
    blocks: List[str] = []
    embedded_tools: List[Any] = []

    if port_value is None:
        return "", embedded_tools

    items = port_value if isinstance(port_value, (list, tuple)) else [port_value]
    for idx, item in enumerate(items):
        if item is None:
            continue
        if isinstance(item, dict) and item.get("tool") is not None:
            embedded_tools.append(item["tool"])
            continue
        if isinstance(item, dict) and "rag_service" in item and "search_params" in item:
            if context_builder is None:
                continue  # host 가 RAG 빌더를 안 줌(커넥터) — 이 소스는 스킵.
            try:
                result = context_builder(text, item)
                if result:
                    blocks.append(result)
            except Exception as exc:  # noqa: BLE001 - RAG must never kill the turn
                logger.error(
                    "geny_bridge: RAG search failed for context[%d]: %s", idx, exc, exc_info=True
                )
            continue
        legacy = _coerce_legacy(item)
        if legacy:
            blocks.append(legacy)

    return "\n\n".join(blocks), embedded_tools
