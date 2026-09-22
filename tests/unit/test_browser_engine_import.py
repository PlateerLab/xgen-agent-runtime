"""브라우저 엔진 로더가 개명된 패키지를 찾는다 (4.42.0).

an-web 은 2026-08-06 에 ``xgen-an-web`` 으로 개명됐다(모듈 ``an_web`` → ``xgen_an_web``).
로더가 옛 이름만 import 해서, 엔진이 설치된 배포에서도 Browser* 도구가 "not installed"
로 죽었다(dev 2026-09-22: xgen-an-web 0.11.0 설치돼 있는데 BrowserNavigate 첫 호출 즉사).
"""

from __future__ import annotations

import sys
import types

import pytest

from xgen_agent_runtime.tools.built_in import browser_tools


def _fake_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)

    class ANWebEngine:  # noqa: D401 - 자리표시
        pass

    mod.ANWebEngine = ANWebEngine  # type: ignore[attr-defined]
    return mod


def test_prefers_the_renamed_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "xgen_an_web", _fake_module("xgen_an_web"))
    monkeypatch.setitem(sys.modules, "an_web", _fake_module("an_web"))
    engine = browser_tools._load_an_web()
    assert engine is sys.modules["xgen_an_web"].ANWebEngine


def test_falls_back_to_the_old_name(monkeypatch):
    monkeypatch.setitem(sys.modules, "xgen_an_web", None)  # import 가 ImportError 를 낸다
    monkeypatch.setitem(sys.modules, "an_web", _fake_module("an_web"))
    assert browser_tools._load_an_web() is sys.modules["an_web"].ANWebEngine


def test_install_hint_when_neither_exists(monkeypatch):
    monkeypatch.setitem(sys.modules, "xgen_an_web", None)
    monkeypatch.setitem(sys.modules, "an_web", None)
    with pytest.raises(RuntimeError, match="an-web engine is not installed"):
        browser_tools._load_an_web()
