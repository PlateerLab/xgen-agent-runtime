"""이름을 바꾼 파라미터의 호환 다리 (4.49.0).

근거: 우리 카탈로그가 같은 "파일 경로" 개념을 여섯 이름으로 갈라 놓고 있었다 —
`path` 15개 / `file_path` 5개(Read·Write·Edit·NotebookEdit·SendUserFile) /
`source` 2 / `target`·`file`·`filename` 각 1. 참고 하네스(Claude Code)는
**파일 하나 = `file_path`, 디렉터리 = `path`** 로 나뉘어 있고, 모델은 그 규약으로
학습돼 있어서 우리 Doc*/Audio* 에도 `file_path` 를 보냈다(dev 실측 별칭 오류 19건).

그래서 "파일 하나를 받는" 도구를 `file_path` 로 통일하고, 옛 이름은 도구가
**선언**하는 별칭으로 계속 받는다. 스키마에는 정본만 싣는다(프리픽스 토큰).
"""

from __future__ import annotations

import pytest

from xgen_agent_runtime.tools.errors import apply_input_aliases


class TestApplyInputAliases:
    def test_moves_the_old_name_to_the_canonical_one(self):
        out = apply_input_aliases({"file_path": ("path",)}, {"path": "/w/a.docx", "to": "md"})
        assert out == {"file_path": "/w/a.docx", "to": "md"}

    def test_canonical_wins_when_both_arrive(self):
        out = apply_input_aliases(
            {"file_path": ("path",)}, {"file_path": "/new", "path": "/old"}
        )
        assert out["file_path"] == "/new"
        assert out["path"] == "/old"  # 건드리지 않는다

    def test_first_matching_alias_wins(self):
        out = apply_input_aliases({"file_path": ("path", "file")}, {"file": "/f", "path": "/p"})
        assert out["file_path"] == "/p"

    def test_no_aliases_returns_the_same_object(self):
        payload = {"path": "/a"}
        assert apply_input_aliases({}, payload) is payload

    def test_nothing_to_move_returns_the_same_object(self):
        payload = {"file_path": "/a"}
        assert apply_input_aliases({"file_path": ("path",)}, payload) is payload

    def test_original_payload_is_untouched(self):
        payload = {"path": "/a"}
        apply_input_aliases({"file_path": ("path",)}, payload)
        assert payload == {"path": "/a"}

    def test_non_dict_payload_is_passed_through(self):
        assert apply_input_aliases({"file_path": ("path",)}, "oops") == "oops"


class TestCatalogIsConsistent:
    """파일 하나를 받는 내장 도구는 전부 ``file_path`` 를 쓴다."""

    def _file_tools(self):
        from xgen_agent_runtime.tools.built_in import audio_tools, doc_tools, dev_tools

        out = []
        for mod in (doc_tools, audio_tools):
            for name in dir(mod):
                obj = getattr(mod, name)
                if isinstance(obj, type) and name.endswith("Tool") and not name.startswith("_"):
                    try:
                        out.append(obj())
                    except Exception:
                        pass
        out.append(dev_tools.LSPTool())
        return out

    def test_no_file_tool_still_requires_the_old_name(self):
        offenders = []
        for t in self._file_tools():
            req = set(t.input_schema.get("required") or [])
            if req & {"path", "file"}:
                offenders.append(t.name)
        assert offenders == [], f"아직 옛 이름을 요구하는 도구: {offenders}"

    def test_every_renamed_tool_declares_the_bridge(self):
        missing = []
        for t in self._file_tools():
            props = t.input_schema.get("properties") or {}
            if "file_path" in props and not (getattr(t, "input_aliases", {}) or {}):
                missing.append(t.name)
        assert missing == [], f"별칭 선언이 없는 도구: {missing}"

    def test_aliases_are_not_advertised_in_the_schema(self):
        """별칭을 스키마에 실으면 호출마다 토큰을 내고 모델에게 둘 다 가르친다."""
        leaked = []
        for t in self._file_tools():
            props = set(t.input_schema.get("properties") or {})
            for olds in (getattr(t, "input_aliases", {}) or {}).values():
                if props & set(olds):
                    leaked.append(t.name)
        assert leaked == [], f"스키마에 별칭이 새어 나간 도구: {leaked}"


class TestRouterUsesTheBridge:
    @pytest.mark.asyncio
    async def test_an_old_call_still_runs(self):
        from xgen_agent_runtime.stages.s10_tool.artifact.default.routers import RegistryRouter
        from xgen_agent_runtime.tools.built_in.doc_tools import DocRenderTool
        from xgen_agent_runtime.tools.base import ToolContext
        from xgen_agent_runtime.tools.registry import ToolRegistry

        tool = DocRenderTool()
        reg = ToolRegistry()
        reg.register(tool)
        # 옛 이름으로 불러도 "필수 필드 누락" 으로 거절당하지 않는다.
        result = await RegistryRouter(reg).route(
            "DocRender",
            {"path": "/nonexistent/a.docx", "to": "md"},
            ToolContext(session_id="t", working_dir="/tmp"),
        )
        payload = result.content
        if isinstance(payload, dict):
            msg = payload.get("error", {}).get("message", "")
            assert "required property" not in msg, msg
