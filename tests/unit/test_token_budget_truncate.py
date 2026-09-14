"""4.24.1 — truncate_text_to_token_budget 결과가 같은 추정기로 다시 세어도 한도 안에 든다.

예전에는 raw 토큰 수로 잘라서 provider 보정계수(vLLM ×1.20 · 안전마진 ×1.05)를 곱해 다시
세면 늘 한도를 넘었다. 호출부(agents/xgen · geny 입력 clamp)는 그 초과분을 사용자 텍스트에서
깎아, 긴 RAG 에서 질문이 '?' 한 글자만 남는 일이 있었다.
"""

from __future__ import annotations

import pytest

from xgen_agent_runtime.host.token_budget import count_text_tokens, truncate_text_to_token_budget

LONG_KO = "제1조(정의) 이 규정에서 사용하는 용어의 뜻은 다음과 같다. 부실자산관련자란 원인을 제공한 자를 말한다. " * 3000


@pytest.mark.parametrize("provider", ["vllm", "openai", "anthropic", "google", None])
@pytest.mark.parametrize("budget", [500, 8_000, 21_917])
def test_truncated_text_fits_budget_under_same_estimator(provider, budget):
    truncated, cut = truncate_text_to_token_budget(LONG_KO, budget, provider=provider, model="m")

    assert cut is True
    assert count_text_tokens(truncated, provider, "m") <= budget


def test_truncation_keeps_both_ends_and_marks_omission():
    truncated, cut = truncate_text_to_token_budget(LONG_KO + "끝문장", 2_000, provider="vllm", model="m")

    assert cut is True
    assert truncated.startswith("제1조(정의)")
    assert truncated.endswith("끝문장")
    assert "중간 내용 생략됨" in truncated


def test_text_within_budget_is_returned_unchanged():
    text = "짧은 질문입니다."
    assert truncate_text_to_token_budget(text, 1_000, provider="vllm", model="m") == (text, False)
