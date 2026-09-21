"""ForgeTool 실패는 구조화 오류 + 무엇이 있는지 말한다 — 같은 값을 네 번 다시 내지 않도록."""

from __future__ import annotations

import asyncio
import os
from typing import Any, List

from xgen_agent_runtime.host.forged_tools import (
    ForgeTool,
    _missing_script_hint,
    _script_candidates,
)
from xgen_agent_runtime.tools._xgeny_sandbox import ExecResult
from xgen_agent_runtime.tools.base import ToolContext


class _Sandbox:
    def __init__(self, files: List[str], *, exists: bool = False):
        self.files, self._exists = files, exists

    async def ensure(self):
        return None

    async def exists(self, path):
        return self._exists

    async def exec(self, argv, **kw):
        assert "find" in " ".join(argv)
        return ExecResult(0, ("\n".join(f"./{f}" for f in self.files) + "\n").encode(), b"")


class _Store:
    def get(self, name):
        return None


def _ctx(ws: str, sandbox: Any) -> ToolContext:
    return ToolContext(session_id="i1", working_dir=ws, allowed_paths=[ws], sandbox=sandbox)


class TestCandidates:
    def test_name_pieces_and_similarity(self):
        files = ["samsung_stock_search.py", "tools/fx.py", "notes.md", "search_stock.py", "unrelated.py"]
        assert _script_candidates("search_samsung_stock", files)[:2] == ["samsung_stock_search.py", "search_stock.py"]
        assert _script_candidates("fx", files) == ["tools/fx.py"]
        assert _script_candidates("zzz", files) == []

    def test_hint_lists_candidates_or_files_or_asks_to_create(self):
        hint = asyncio.run(_missing_script_hint(_Sandbox(["samsung_stock_search.py"]), "search_samsung_stock"))
        assert "파일 경로" in hint and "함수 이름이 아닙니다" in hint and "samsung_stock_search.py" in hint
        hint2 = asyncio.run(_missing_script_hint(_Sandbox(["a.py", "b.py"]), "zzz"))
        assert "workspace 의 스크립트: a.py, b.py" in hint2
        hint3 = asyncio.run(_missing_script_hint(_Sandbox([]), "zzz"))
        assert "먼저 workspace 에 파일을 만든 뒤" in hint3
        assert asyncio.run(_missing_script_hint(object(), "x")).endswith("등록하세요.")


class TestForgeToolErrors:
    def test_missing_script_is_a_structured_input_error_with_hint(self, tmp_path):
        ws = str(tmp_path)
        tool = ForgeTool(workflow_id="wf", workspace_dir=ws, registry=None, store=_Store())
        res = asyncio.run(tool.execute(
            {"name": "samsung_stock_search", "description": "d", "entrypoint": "search_samsung_stock"},
            _ctx(ws, _Sandbox(["samsung_stock_search.py", "tools/fx.py"])),
        ))
        assert res.is_error
        assert res.content.startswith("ERROR invalid_input: 스크립트를 찾을 수 없습니다: search_samsung_stock")
        assert "samsung_stock_search.py" in res.content

    def test_validation_failures_are_structured_too(self, tmp_path):
        ws = str(tmp_path)
        tool = ForgeTool(workflow_id="wf", workspace_dir=ws, registry=None, store=_Store())
        res = asyncio.run(tool.execute({"name": "x", "description": "d", "entrypoint": ""}, _ctx(ws, _Sandbox([]))))
        assert res.is_error and res.content.startswith("ERROR invalid_input: 도구를 만들 수 없습니다")
        res2 = asyncio.run(tool.execute({"name": "x", "description": "d", "entrypoint": "/etc/passwd"}, _ctx(ws, _Sandbox([]))))
        assert res2.content.startswith("ERROR invalid_input:")

    def test_schema_says_file_path_not_function_and_array_deps(self, tmp_path):
        tool = ForgeTool(workflow_id="wf", workspace_dir=str(tmp_path), registry=None, store=_Store())
        props = tool.input_schema["properties"]
        assert "Not a function name" in props["entrypoint"]["description"]
        assert "JSON array" in props["dependencies"]["description"]
        assert os.path.basename(str(tmp_path))  # tmp_path 사용(경로 규약 확인용)
