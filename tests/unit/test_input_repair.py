"""스키마 오류를 왕복 없이 되살리는 길 — 실측 dev 기록에서 뽑은 모양들.

근거(2026-08-23~09-22 dev, `agent_trace_spans`): `Invalid input` 258건 가운데
필수 필드 누락 166건, 배열·객체 자리에 JSON 문자열 12건. 필수 필드 누락 중
절반 가까이는 **모델이 다른 도구의 필드명을 그대로 쓴 것**이고(6도구·4도메인),
35건은 인자가 통째로 비어 온 것이다.

여기 테스트는 그 기록의 모양을 그대로 재현한다 — 도구 이름은 익명이지만
스키마 모양(``path`` vs ``file_path``, ``positive_prompt`` vs ``prompt``,
``target`` vs ``path``)은 실제 기록 그대로다.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from xgen_agent_runtime.stages.s10_tool.artifact.default.routers import RegistryRouter
from xgen_agent_runtime.tools.base import Tool, ToolContext, ToolResult
from xgen_agent_runtime.tools.errors import (
    UNPARSED_ARGUMENTS_KEY,
    coerce_input,
    describe_validation_failure,
    repair_missing_required,
)
from xgen_agent_runtime.tools.registry import ToolRegistry


def _ctx() -> ToolContext:
    return ToolContext(session_id="test", working_dir="/tmp")


class _SchemaTool(Tool):
    """주어진 스키마를 그대로 쓰고, 받은 입력을 돌려주는 도구."""

    def __init__(self, name: str, schema: Dict[str, Any]) -> None:
        self._name = name
        self._schema = schema
        self.seen: Dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "test tool"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return self._schema

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        self.seen = dict(input)
        return ToolResult(content="ok")


def _router(tool: Tool) -> RegistryRouter:
    registry = ToolRegistry()
    registry.register(tool)
    return RegistryRouter(registry)


_DOC_SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}, "to": {"type": "string"}},
    "required": ["path"],
}
_OPEN_SCHEMA = {
    "type": "object",
    "properties": {"target": {"type": "string"}},
    "required": ["target"],
}
_IMAGE_SCHEMA = {
    "type": "object",
    "properties": {"positive_prompt": {"type": "string"}},
    "required": ["positive_prompt"],
}


# ── 1. 별칭 자동 복구 ────────────────────────────────────


class TestAliasRepair:
    @pytest.mark.parametrize(
        "schema,sent,expected_key",
        [
            # DocRender/DocAnalyze/mcp_local_ReadFile: file_path → path
            (_DOC_SCHEMA, {"file_path": "/w/a.docx", "to": "md"}, "path"),
            # mcp_local_Open: path → target (이름은 안 닮았지만 후보가 하나)
            (_OPEN_SCHEMA, {"path": "/w/a.txt"}, "target"),
            # comfyui_test_anima: prompt → positive_prompt
            (_IMAGE_SCHEMA, {"prompt": "a cat"}, "positive_prompt"),
        ],
    )
    def test_moves_unknown_key_into_the_missing_required_slot(
        self, schema, sent, expected_key
    ):
        repaired = repair_missing_required(schema, sent)
        assert repaired is not None
        fixed, notes = repaired
        assert expected_key in fixed
        assert fixed[expected_key] == list(sent.values())[0]
        assert notes and expected_key in notes[0]

    def test_leaves_the_original_payload_alone(self):
        sent = {"file_path": "/w/a.docx"}
        repair_missing_required(_DOC_SCHEMA, sent)
        assert sent == {"file_path": "/w/a.docx"}

    def test_does_not_guess_when_the_value_does_not_fit(self):
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }
        assert repair_missing_required(schema, {"label": "high"}) is None

    def test_does_not_guess_between_two_equally_good_candidates(self):
        schema = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        # 둘 다 문자열이고 둘 다 이름이 안 닮았다 → 손대지 않는다.
        assert repair_missing_required(schema, {"src": "/a", "dst": "/b"}) is None

    def test_prefers_the_akin_name_when_several_fit(self):
        schema = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        repaired = repair_missing_required(schema, {"file_path": "/a", "note": "b"})
        assert repaired is not None
        assert repaired[0]["path"] == "/a"

    def test_known_key_is_never_reinterpreted(self):
        # BrowserNavigate 기록: url 은 스키마에 있는 키이고 action 이 빠졌다.
        schema = {
            "type": "object",
            "properties": {"url": {"type": "string"}, "action": {"type": "string"}},
            "required": ["action"],
        }
        assert repair_missing_required(schema, {"url": "http://x"}) is None

    @pytest.mark.asyncio
    async def test_router_runs_the_repaired_call_and_says_so(self):
        tool = _SchemaTool("DocRender", _DOC_SCHEMA)
        result = await _router(tool).route(
            "DocRender", {"file_path": "/w/a.docx", "to": "md"}, _ctx()
        )
        assert result.is_error is False
        assert tool.seen == {"path": "/w/a.docx", "to": "md"}
        assert "[input repaired]" in result.content
        assert "'file_path'" in result.content


# ── 2. 배열·객체 자리에 온 JSON 문자열 ──────────────────


class TestJsonStringContainer:
    def test_parses_a_json_array_string(self):
        schema = {
            "type": "object",
            "properties": {"inputs": {"type": "array", "items": {"type": "object"}}},
        }
        out = coerce_input(schema, {"inputs": '[{"url": "http://a"}]'})
        assert out["inputs"] == [{"url": "http://a"}]

    def test_parses_a_json_object_string(self):
        schema = {"type": "object", "properties": {"spec": {"type": "object"}}}
        out = coerce_input(schema, {"spec": '{"lang": "ko"}'})
        assert out["spec"] == {"lang": "ko"}

    def test_leaves_strings_alone_when_the_schema_wants_a_string(self):
        schema = {
            "type": "object",
            "properties": {"body": {"type": ["string", "array"]}},
        }
        out = coerce_input(schema, {"body": '["a"]'})
        assert out["body"] == '["a"]'

    def test_leaves_a_json_scalar_string_alone(self):
        schema = {"type": "object", "properties": {"spec": {"type": "object"}}}
        out = coerce_input(schema, {"spec": '"just text"'})
        assert out["spec"] == '"just text"'

    def test_leaves_broken_json_alone(self):
        schema = {"type": "object", "properties": {"spec": {"type": "object"}}}
        out = coerce_input(schema, {"spec": "{oops"})
        assert out["spec"] == "{oops"


# ── 3. 해석 못 한 인자를 모델 탓으로 돌리지 않는다 ──────


class TestUnparsedArguments:
    @pytest.mark.asyncio
    async def test_router_reports_a_parse_failure_not_a_missing_field(self):
        tool = _SchemaTool("Bash", {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        })
        raw = '{"command": "for f in *.py; do echo '  # 잘린 JSON
        result = await _router(tool).route("Bash", {UNPARSED_ARGUMENTS_KEY: raw}, _ctx())
        assert result.is_error is True
        message = result.content["error"]["message"]
        assert "not valid JSON" in message
        assert "required property" not in message
        assert tool.seen is None  # 실행되지 않았다

    def test_local_provider_carries_the_raw_text_instead_of_an_empty_dict(self):
        from xgen_agent_runtime.llm_client.openai_compatible import (
            OpenAICompatibleClient,
        )

        client = OpenAICompatibleClient.__new__(OpenAICompatibleClient)
        client.provider = "vllm"
        client._event_sink = None
        out = client._parse_tool_arguments('{"command": "echo hi; sleep')
        assert UNPARSED_ARGUMENTS_KEY in out

    def test_empty_arguments_stay_empty(self):
        from xgen_agent_runtime.llm_client.openai_compatible import (
            OpenAICompatibleClient,
        )

        client = OpenAICompatibleClient.__new__(OpenAICompatibleClient)
        client.provider = "vllm"
        client._event_sink = None
        assert client._parse_tool_arguments("") == {}
        assert client._parse_tool_arguments("{}") == {}


# ── 4. 고칠 수 없을 때는 최소한 제대로 말해 준다 ────────


class TestErrorMessage:
    def test_names_what_was_required_and_what_arrived(self):
        msg = describe_validation_failure(
            _DOC_SCHEMA, {"file_path": "/a", "to": "md"}, "'path' is a required property"
        )
        assert "required: path" in msg
        assert "you sent: file_path, to" in msg

    def test_empty_payload_is_said_plainly(self):
        msg = describe_validation_failure(
            _DOC_SCHEMA, {}, "'path' is a required property"
        )
        assert "you sent: (nothing)" in msg

    def test_other_errors_are_untouched(self):
        msg = describe_validation_failure(
            _DOC_SCHEMA, {"path": 3}, "3 is not of type 'string'"
        )
        assert msg == "3 is not of type 'string'"
