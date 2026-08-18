"""문서 엔진 임포트 폴백 — XGEN 이름(xgen_edit2docs)과 레거시(edit2docs) 모두.

패키지 이관 회귀: 엔진이 xgen_edit2docs 로 설치돼 있어도 doc_tools 가 옛
이름만 찾아 모든 문서 도구가 "not installed" 로 죽었다 (2026-08-18 177 실측).
"""

from __future__ import annotations

import sys
import types

import pytest

from xgen_agent_runtime.tools.built_in.doc_tools import _load_edit2docs


def _fake_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__version__ = "0.0-test"
    return mod


def test_prefers_xgen_name(monkeypatch: pytest.MonkeyPatch) -> None:
    xgen = _fake_module("xgen_edit2docs")
    legacy = _fake_module("edit2docs")
    monkeypatch.setitem(sys.modules, "xgen_edit2docs", xgen)
    monkeypatch.setitem(sys.modules, "edit2docs", legacy)
    assert _load_edit2docs() is xgen, "XGEN 이름이 우선이어야 한다"


def test_falls_back_to_legacy_name(monkeypatch: pytest.MonkeyPatch) -> None:
    legacy = _fake_module("edit2docs")
    monkeypatch.setitem(sys.modules, "edit2docs", legacy)
    monkeypatch.setitem(sys.modules, "xgen_edit2docs", None)  # import → ImportError
    assert _load_edit2docs() is legacy


def test_missing_both_raises_actionable_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "xgen_edit2docs", None)
    monkeypatch.setitem(sys.modules, "edit2docs", None)
    with pytest.raises(RuntimeError) as exc:
        _load_edit2docs()
    msg = str(exc.value)
    assert "xgen-edit2docs" in msg, "설치 힌트에 XGEN 패키지명이 있어야 한다"
