"""토큰 추정은 **네트워크를 타지 않는다.**

예전에는 tiktoken 이 셌다. 그 라이브러리는 인코딩 표(BPE)를 첫 호출 때 인터넷에서
내려받는데 타임아웃조차 없다 — 폐쇄망에서 턴이 거기서 멎었다(o200k 실패 → cl100k 로
한 번 더 시도했으므로 두 배로). 이 파일은 그게 다시 들어오지 못하게 막는다.
"""
from __future__ import annotations

from pathlib import Path

from xgen_agent_runtime.host import token_budget
from xgen_agent_runtime.host.token_estimator import (
    HANGUL_LEGACY,
    HANGUL_MODERN,
    estimate_tokens,
)

_HOST = Path(token_budget.__file__).parent


class TestItNeverReachesOut:
    def test_no_module_in_the_budget_path_imports_a_downloader(self):
        for name in ("token_budget.py", "token_estimator.py"):
            source = (_HOST / name).read_text("utf-8")
            for banned in ("import tiktoken", "import requests", "import urllib"):
                assert banned not in source, f"{name}: {banned}"


class TestTheEstimate:
    def test_nothing_is_zero(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens(None) == 0

    def test_korean_costs_more_per_character_than_english(self):
        assert estimate_tokens("안녕하세요반갑습니다오늘도좋은하루") > estimate_tokens(
            "greetings friends have a nice day")

    def test_the_legacy_weight_only_moves_korean(self):
        korean = "회의는 오후 세시에 시작합니다"
        english = "the meeting starts at three"
        assert estimate_tokens(korean, hangul=HANGUL_LEGACY) > estimate_tokens(
            korean, hangul=HANGUL_MODERN)
        assert estimate_tokens(english, hangul=HANGUL_LEGACY) == estimate_tokens(
            english, hangul=HANGUL_MODERN)


class TestTruncationStaysLanguageAware:
    """자를 때 고정 4자/토큰을 쓰면 한국어 문서가 한도의 다섯 배로 잘린다."""

    def test_korean_truncation_lands_under_the_limit(self):
        text = "배포는 승인된 그 시점의 사본을 연다. 원본을 고쳐도 외부는 그대로다. " * 40
        out, cut = token_budget.truncate_text_to_token_budget(text, 200)
        assert cut is True
        assert token_budget.count_text_tokens(out) <= 200

    def test_english_truncation_lands_under_the_limit(self):
        text = "The applier must not perform any network call on the event loop. " * 60
        out, cut = token_budget.truncate_text_to_token_budget(text, 200)
        assert cut is True
        assert token_budget.count_text_tokens(out) <= 200

    def test_mixed_text_too(self):
        text = ("XGEN 플랫폼의 deploy 파이프라인은 approval 결재를 통과한 snapshot 만 "
                "외부에 노출합니다. rollback 은 pointer 를 되돌리면 됩니다. ") * 30
        out, cut = token_budget.truncate_text_to_token_budget(text, 150)
        assert cut is True
        assert token_budget.count_text_tokens(out) <= 150

    def test_short_text_is_left_alone(self):
        text = "짧은 문장입니다."
        out, cut = token_budget.truncate_text_to_token_budget(text, 1000)
        assert (out, cut) == (text, False)
