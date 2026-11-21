"""한 턴의 요청들은 앞 요청을 그대로 머리에 둔다 — 생각 블록이 거부되지 않게.

Claude 5.5 세대(Sonnet 5.5 · Haiku 5.5 · Opus 5.5 …)는 되돌려 받은 생각 블록이 쓰일 때 앞에 있던 것
(system · 도구 목록 · 앞선 메시지)과 지금 요청이 같은지 본다. 다르면 400 "Invalid `signature` in
`thinking` block … Content that preceded this block when it was created is missing" 이다.

2026-11-20 홈 XGEN 에서 여러 단계 턴마다 났다. 원인 둘:

1. 턴 맥락(``<session-context>`` — 날짜 · 시각 · 기억 · 호스트 안내)을 매 요청 **가장 최근** user 메시지에
   붙였다. 도구 루프에서 그것은 새 도구 결과라, 앞 요청이 싣던 자리에서 블록이 빠졌다. 시각도 분마다 바뀌었다.
   → 턴의 첫 요청이 자리와 글을 정하고, 그 턴의 모든 요청이 같은 자리에 같은 글을 싣는다.
2. 문 도구(AppGuide 등)가 가족을 열면 도구 목록이 커진다 → 그 전에 쓴 생각 블록은 뺀다(앞쪽부터 빼는 것은
   API 가 허용한다). 그 뒤에 쓴 블록은 그대로 실린다.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import Any, Dict, List

from xgen_agent_runtime.core.shared_keys import SharedKeys
from xgen_agent_runtime.core.state import PipelineState
from xgen_agent_runtime.host import runner
from xgen_agent_runtime.llm_client.anthropic import AnthropicClient
from xgen_agent_runtime.stages.s03_system.artifact.default.builders import DateTimeBlock
from xgen_agent_runtime.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult
from xgen_agent_runtime.tools.registry import ToolRegistry


class _Tool(Tool):
    def __init__(self, name: str, reply: str) -> None:
        self._name, self._reply = name, reply

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"{self._name} tool"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {"q": {"type": "string"}}}

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(concurrency_safe=True)

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        return ToolResult(content=self._reply)


def _response(n: int, tool: str = "") -> List[SimpleNamespace]:
    blocks = [SimpleNamespace(type="thinking", thinking="", signature=f"sig-{n}")]
    if tool:
        blocks.append(SimpleNamespace(type="tool_use", id=f"toolu_{n}", name=tool, input={"q": str(n)}))
    else:
        blocks.append(SimpleNamespace(type="text", text="done"))
    return blocks


class _FakeSDK:
    """Anthropic SDK stand-in: records each request body, answers from a script."""

    def __init__(self, script: List[List[SimpleNamespace]]) -> None:
        self.script = script
        self.bodies: List[Dict[str, Any]] = []
        self.messages = self

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        body = {k: copy.deepcopy(kwargs.get(k)) for k in ("system", "tools", "messages")}
        self.bodies.append(json.loads(json.dumps(body, default=str)))
        blocks = self.script[min(len(self.bodies) - 1, len(self.script) - 1)]
        return SimpleNamespace(
            content=blocks,
            stop_reason="tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn",
            usage=SimpleNamespace(
                input_tokens=10, output_tokens=5, cache_creation_input_tokens=0, cache_read_input_tokens=0
            ),
            model=kwargs.get("model"),
            id=f"msg_{len(self.bodies)}",
        )


def _run(script, registry: ToolRegistry, state: PipelineState, text: str = "do the task") -> _FakeSDK:
    sdk = _FakeSDK(script)
    client = AnthropicClient(api_key="k")
    client._client = sdk
    pipe = runner.build_pipeline(
        name="t",
        provider="anthropic",
        model="claude-sonnet-5-5",
        api_key="k",
        llm_client=client,
        stream=False,
        enable_compaction=False,
        registry=registry,
        system_prompt="You are a test agent.",
    )
    runner.run_turn(pipe, text, state)
    return sdk


def _thinking_signatures(body: Dict[str, Any]) -> List[str]:
    return [
        b.get("signature")
        for m in body["messages"]
        if m["role"] == "assistant" and isinstance(m["content"], list)
        for b in m["content"]
        if b.get("type") == "thinking"
    ]


def _context_blocks(body: Dict[str, Any]) -> List[tuple]:
    found = []
    for i, m in enumerate(body["messages"]):
        if isinstance(m["content"], list):
            for b in m["content"]:
                if b.get("type") == "text" and str(b.get("text", "")).startswith("<session-context>"):
                    found.append((i, b["text"]))
    return found


def test_each_request_of_a_turn_starts_with_the_previous_request_unchanged(monkeypatch) -> None:
    # The clock moves between calls; the turn keeps the time its first request carried.
    ticks = iter(f"Current date: 2026-11-20 (Fri) 10:{m:02d} KST" for m in range(60))
    monkeypatch.setattr(DateTimeBlock, "render", lambda self, state: next(ticks))
    reg = ToolRegistry()
    reg.register(_Tool("Lookup", "result line\n" * 80), core=True)
    script = [_response(1, "Lookup"), _response(2, "Lookup"), _response(3, "Lookup"), _response(4)]

    sdk = _run(script, reg, PipelineState(session_id="s", model="claude-sonnet-5-5"))

    bodies = sdk.bodies
    assert len(bodies) == 4
    for prev, cur in zip(bodies, bodies[1:]):
        assert cur["system"] == prev["system"] and cur["tools"] == prev["tools"]
        assert cur["messages"][: len(prev["messages"])] == prev["messages"]
    # One context block, on the person's message, with the first request's clock.
    for body in bodies:
        assert _context_blocks(body) == [(0, _context_blocks(bodies[0])[0][1])]
    assert "10:00" in _context_blocks(bodies[-1])[0][1]
    assert _thinking_signatures(bodies[-1]) == ["sig-1", "sig-2", "sig-3"]


def test_thinking_from_before_a_tool_family_opened_is_left_out() -> None:
    reg = ToolRegistry()
    reg.register(_Tool("AppGuide", "how to build an app"), core=True)
    for name in ("AppCreate", "AppPublish", "AppStatus", "AppList", "AppDelete"):
        reg.register(_Tool(name, f"{name} ok"), core=False)
    script = [_response(1, "AppGuide"), _response(2, "AppStatus"), _response(3, "AppList"), _response(4)]

    sdk = _run(script, reg, PipelineState(session_id="s", model="claude-sonnet-5-5"))

    bodies = sdk.bodies
    names = [sorted(t["name"] for t in b["tools"]) for b in bodies]
    assert "AppCreate" not in names[0] and "AppCreate" in names[1]
    # sig-1 was written before the family opened; it never goes out again.
    assert [_thinking_signatures(b) for b in bodies] == [[], [], ["sig-2"], ["sig-2", "sig-3"]]
    # Requests under the same tool set still extend each other exactly.
    for prev, cur in zip(bodies[1:], bodies[2:]):
        assert cur["messages"][: len(prev["messages"])] == prev["messages"]


def test_a_new_turn_chooses_its_own_place_and_text() -> None:
    state = PipelineState(session_id="s", model="claude-sonnet-5-5")
    state.shared[SharedKeys.TURN_CONTEXT_PIN] = {"index": 0, "digest": "x", "text": "old"}
    state.begin_turn()
    assert SharedKeys.TURN_CONTEXT_PIN not in state.shared
