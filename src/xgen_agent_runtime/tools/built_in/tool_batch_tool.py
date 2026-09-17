"""ToolBatch — 같은 도구를 입력 목록으로 한 번에 실행한다.

왜 필요한가
-----------
도구 호출 하나는 모델 왕복 하나이고, 왕복마다 시스템 프롬프트·도구 정의·대화 전체가
다시 모델에 들어간다. 목록을 다루는 작업(상품 N개 검색, 문서 N개 조회, API N건 확인)을
항목마다 따로 부르면 비용이 N배가 된다. 실측 (2026-09-17 dev, claude-sonnet-4-6):
검색 도구를 품목마다 하나씩 불러 8회 측정에서 88회, 한 턴에 모델 왕복 약 19회였다.

모델이 스크립트를 짜서 우회할 수도 있지만, 그건 등록된 도구(직접 만든 도구·MCP·API
노드)를 스크립트에서 부를 수 없을 때 매번 파서·HTTP 코드를 새로 쓰게 만든다. 이 도구는
**어떤 등록 도구든** 입력만 목록으로 받아 한 왕복에 끝낸다.

계약
----
* 실행은 일반 호출과 같은 경로(``RegistryRouter.route``)를 지난다 — 입력 변환·스키마
  검증·권한·훅이 그대로 적용된다. 이 도구가 권한을 넓히지 않는다.
* 스테이지에서만 하던 검사도 그대로 한다 — 도구 허용 목록(``tool_allowed``)과 반복 실패
  차단(``repeat_guard``). 막힌 도구를 배치로 우회할 수 없다.
* 항목마다 ``tool.call_start``/``tool.call_complete`` 를 낸다 — UI·트레이스·도구 실행
  수 집계에 실제 실행이 그대로 보인다(항목 id 는 ``ToolBatch-<batch>-<i>``).
* ``concurrency_safe`` 인 도구만 병렬로, 나머지는 순서대로 실행한다.
* 결과는 항목별로 압축해 한 번에 돌려준다(항목당·전체 글자 상한).
* ToolBatch 안에서 ToolBatch 는 부를 수 없다.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List

from xgen_agent_runtime.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult

TOOL_BATCH_NAME = "ToolBatch"

MAX_ITEMS = 50
DEFAULT_CONCURRENCY = 8
MAX_CONCURRENCY = 16
PER_ITEM_CHARS = 2_000
TOTAL_CHARS = 40_000


def _text_of(result: ToolResult) -> str:
    if result.display_text is not None:
        return str(result.display_text)
    content = result.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        ]
        if parts:
            return "\n".join(parts)
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(content)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"…(+{len(text) - limit} chars)"


class ToolBatchTool(Tool):
    """Run one registered tool over a list of inputs in a single step."""

    @property
    def name(self) -> str:
        return TOOL_BATCH_NAME

    @property
    def description(self) -> str:
        return (
            "Run ONE registered tool over MANY inputs in a single step and get all results "
            "back together. Use it whenever the same tool must be called for several items "
            "(e.g. search each product, fetch each URL, look up each ID) instead of calling "
            "the tool once per item. Works for any tool you can call, including tools found "
            "via ToolSearch and tools you created. Inputs are validated exactly like normal "
            f"calls. Up to {MAX_ITEMS} inputs; results are returned in input order."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tool": {
                    "type": "string",
                    "description": "Name of the tool to run for every input.",
                },
                "inputs": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_ITEMS,
                    "items": {"type": "object"},
                    "description": "One argument object per call, same shape as calling the tool directly.",
                },
                "max_concurrency": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_CONCURRENCY,
                    "description": f"Parallel calls for concurrency-safe tools (default {DEFAULT_CONCURRENCY}).",
                },
            },
            "required": ["tool", "inputs"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        # 안쪽 도구의 성격을 알 수 없다 — 바깥은 보수적으로 직렬, 결과 상한은 이 도구가 관리.
        return ToolCapabilities(concurrency_safe=False, read_only=False, max_result_chars=0)

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        from xgen_agent_runtime.stages.s10_tool import repeat_guard
        from xgen_agent_runtime.stages.s10_tool.artifact.default.executors import (
            _apply_state_mutations_via_ctx,
            _emit_call_complete,
            _emit_call_start,
        )
        from xgen_agent_runtime.stages.s10_tool.artifact.default.routers import RegistryRouter

        tool_name = str((input or {}).get("tool") or "").strip()
        inputs = (input or {}).get("inputs") or []
        if tool_name == TOOL_BATCH_NAME:
            return ToolResult(
                content="ERROR invalid_input: ToolBatch cannot run ToolBatch.", is_error=True
            )
        registry = getattr(context, "tool_registry", None)
        if registry is None:
            return ToolResult(
                content="ERROR unavailable: no tool registry bound to this call.", is_error=True
            )
        target = registry.get(tool_name)
        if target is None:
            return ToolResult(
                content=f"ERROR unknown_tool: '{tool_name}' is not registered. Use ToolSearch to find the exact name.",
                is_error=True,
            )
        if (
            not isinstance(inputs, list)
            or not inputs
            or not all(isinstance(i, dict) for i in inputs)
        ):
            return ToolResult(
                content="ERROR invalid_input: 'inputs' must be a non-empty list of objects.",
                is_error=True,
            )
        inputs = inputs[:MAX_ITEMS]

        allowed = getattr(context, "tool_allowed", None)
        if callable(allowed) and not allowed(tool_name):
            return ToolResult(
                content=f"ERROR access_denied: '{tool_name}' is not allowed in this stage.",
                is_error=True,
            )
        state = getattr(context, "state_view", None)
        shared = getattr(state, "shared", None)
        shared = shared if isinstance(shared, dict) else None
        emit = getattr(state, "add_event", None)
        emit = emit if callable(emit) else None
        if shared is not None:
            blocked = repeat_guard.blocked_result({"tool_name": tool_name}, shared)
            if blocked is not None:
                return ToolResult(content=blocked["content"], is_error=True)

        # 호출 대상이 숨겨진(deferred) 도구여도 이름을 알면 부를 수 있게 활성화한다 — ToolSearch 와 같은 규칙.
        activate = getattr(registry, "activate", None)
        if callable(activate) and not getattr(registry, "is_exposed", lambda _n: True)(tool_name):
            try:
                activate(tool_name)
            except Exception:  # noqa: BLE001
                pass

        router = RegistryRouter(registry)
        try:
            safe = bool(target.capabilities(inputs[0]).concurrency_safe)
        except Exception:  # noqa: BLE001
            safe = False
        limit = int((input or {}).get("max_concurrency") or DEFAULT_CONCURRENCY)
        limit = max(1, min(MAX_CONCURRENCY, limit)) if safe else 1
        sem = asyncio.Semaphore(limit)

        batch_id = uuid.uuid4().hex[:8]
        calls = [
            {"tool_use_id": f"{TOOL_BATCH_NAME}-{batch_id}-{i}", "tool_name": tool_name, "tool_input": dict(a)}
            for i, a in enumerate(inputs)
        ]

        async def _one(tc: Dict[str, Any]) -> Dict[str, Any]:
            async with sem:
                _emit_call_start(emit, tc)
                t0 = time.monotonic()
                try:
                    res = await router.route(tool_name, tc["tool_input"], context)
                except Exception as exc:  # noqa: BLE001 — 항목 하나의 실패가 배치를 죽이지 않는다
                    res = ToolResult(content=f"ERROR {type(exc).__name__}: {exc}", is_error=True)
                _apply_state_mutations_via_ctx(res, tc, context)
                result = res.to_api_format(tc["tool_use_id"])
                if not isinstance(result.get("content"), str):
                    result["content"] = _text_of(res)
                _emit_call_complete(emit, tc, result, int((time.monotonic() - t0) * 1000))
                return result

        results = await asyncio.gather(*(_one(tc) for tc in calls))
        if shared is not None:
            # 한 배치는 모델의 한 번 시도다 — 같은 오류가 항목 N개에서 났다고 N번으로 세면
            # 모델이 고쳐 볼 기회도 없이 첫 배치에서 차단된다. 오류 문구마다 한 번만 센다.
            # 항목마다 입력이 다른 게 정상이므로 실행 오류는 같은 인자 반복만 센다.
            seen: set = set()
            observed = []
            for r in results:
                if r.get("is_error"):
                    err = repeat_guard.normalize_error(str(r.get("content") or ""))
                    if err in seen:
                        continue
                    seen.add(err)
                observed.append(r)
            flagged = repeat_guard.observe(calls, observed, shared, count_across_inputs=False)
            if flagged and emit is not None:
                emit("tool.repeat_failure", {"tools": [{"name": n, "count": c} for n, c in flagged]})

        per_item = max(200, min(PER_ITEM_CHARS, TOTAL_CHARS // len(results)))
        rows: List[Dict[str, Any]] = []
        ok = 0
        for i, (tc, res) in enumerate(zip(calls, results)):
            is_error = bool(res.get("is_error"))
            text = _clip(str(res.get("content") or "").strip(), per_item)
            row: Dict[str, Any] = {"i": i, "input": tc["tool_input"], "ok": not is_error}
            row["error" if is_error else "result"] = text
            ok += 0 if is_error else 1
            rows.append(row)
        body = {
            "tool": tool_name,
            "calls": len(rows),
            "ok": ok,
            "failed": len(rows) - ok,
            "parallel": limit if safe else 1,
            "results": rows,
        }
        # 전부 실패면 오류로 올린다 — 반복 실패 차단이 이 도구 단위로도 걸리게.
        return ToolResult(content=json.dumps(body, ensure_ascii=False), is_error=(ok == 0))
