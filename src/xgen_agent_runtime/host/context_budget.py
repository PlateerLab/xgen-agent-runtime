"""agents/geny 컨텍스트 예산 — 호출 전 입력측(clamp) 강제.

geny-executor 파이프라인의 Stage 2(proactive 80% 요약-압축)와 Stage 4(토큰
예산 guard → compact 후 1회 재검사)는 **대화 이력**은 줄일 수 있지만, 압축기가
보존하는 최근-메시지 창 안에 있는 **이번 턴의 입력 자체**(사용자 텍스트 +
RAG 블록)는 줄이지 못한다 — 단일 입력이 윈도우를 넘으면 어떤 압축으로도
회복 불가라 guard 가 턴을 거절한다.

그래서 입력측은 호출 **전에** 여기서 맞춘다 (agent_xgen 의
``_fit_input_to_budget`` 와 같은 우선순위 — 손실 최소화):

    1) RAG/참조문서 블록을 남는 예산에 맞춰 토큰 인지 truncate
       (사용자 텍스트 보존)
    2) 그래도 초과면 사용자 텍스트를 truncate (최후 수단, 중간 생략 마커)

대화 이력(preload)은 건드리지 않는다 — 그건 파이프라인 Stage 2/4 의 몫이고,
같은 것을 두 층에서 자르면 어느 쪽이 잘랐는지 알 수 없게 된다.

토큰 계산·윈도우 해석은 agent_xgen 과 **한 소스**(helper/token_budget)를
쓴다 — 두 에이전트 노드의 예산 의미가 갈라지면 안 된다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

logger = logging.getLogger("editor.geny_bridge.context_budget")

#: 도구 스키마 직렬화가 불가능할 때 도구 1개당 보수적 토큰 추정.
_FALLBACK_TOKENS_PER_TOOL = 400


@dataclass
class BudgetFit:
    """``fit_input_to_budget`` 의 결과."""

    text: str
    rag_block: str
    clamped: bool
    #: 진단용 — provider 컨텍스트 오류와 대조할 압축 전 구성요소 추정치.
    window: int = 0
    budget: int = 0
    total_before: int = 0


def resolve_window(
    provider: str,
    model: str,
    override: int = 0,
    base_url: Optional[str] = None,
    vllm_probe: Any = None,
) -> int:
    """모델의 전체 컨텍스트 윈도우(토큰) — agent_xgen 과 동일 해석 체인.

    override(노드 파라미터) → vLLM 서버 실측 → 어드민 카탈로그 max_context →
    모델 접두 테이블 → provider 기본값. 실패 시 보수적 기본값.
    """
    try:
        from xgen_agent_runtime.host.token_budget import resolve_context_window

        return int(
            resolve_context_window(
                provider,
                model,
                override,
                base_url=base_url,
                vllm_probe=vllm_probe,
            )
        )
    except Exception as exc:  # noqa: BLE001 — 예산 해석 실패가 실행을 막으면 안 된다
        logger.warning("context_budget: 윈도우 해석 실패(%s/%s): %s", provider, model, exc)
        return int(override) if override and override > 0 else 0


def _estimate_tools_tokens(registry: Any, provider: str, model: str) -> int:
    """executor ToolRegistry 의 스키마가 프롬프트에 차지할 토큰 근사.

    ``registry.to_api_format()`` (name/description/input_schema) 직렬화를
    카운트한다. deferred 노출 모델에서는 실제 전송분이 더 작을 수 있으나
    과대추정은 안전한 방향이다. 직렬화 실패 시 도구당 상수로 강등.
    """
    if registry is None:
        return 0
    try:
        from xgen_agent_runtime.host.token_budget import count_text_tokens

        import json

        schemas = registry.to_api_format()
        blob = json.dumps(schemas, ensure_ascii=False, default=str)
        return count_text_tokens(blob, provider, model)
    except Exception:  # noqa: BLE001
        try:
            return _FALLBACK_TOKENS_PER_TOOL * len(registry)
        except Exception:  # noqa: BLE001
            return 0


def fit_input_to_budget(
    *,
    text: str,
    rag_block: str,
    system_prompt: str,
    history: Optional[List[Any]],
    registry: Any,
    provider: str,
    model: str,
    max_tokens: int,
    window: int,
    reserved_tokens: int = 0,
) -> BudgetFit:
    """이번 턴의 입력측(text + rag_block)을 유효 예산 안으로 맞춘다.

    예산 = window − max_tokens(출력 예약) − 안전마진 − reserved_tokens.
    구성요소:
        overhead = system_prompt + tool 스키마
        고정분   = 대화 이력(preload; 파이프라인 압축의 몫이라 여기선 계정만)
        가변분   = rag_block → text 순으로 truncate

    ``reserved_tokens`` — fit **이후** 턴 중에 system 으로 주입되는 것들의
    선예약분. 대표적으로 내장 메모리(Pinned Facts/Relevant Knowledge)는
    Stage 2 가 최대 10k자(≈3k 토큰)를 주입하는데, fit 시점의 system_prompt
    에는 아직 없다 — 예약 없이 꽉 채우면 그만큼 provider 400 으로 넘친다.

    fail-open: 어떤 예외도 입력을 그대로 돌려준다 — 예산 로직의 버그가
    턴을 죽여선 안 된다 (agent_xgen 과 동일 원칙).
    """
    result = BudgetFit(text=text, rag_block=rag_block, clamped=False, window=window)
    if not window or window <= 0:
        return result
    try:
        from xgen_agent_runtime.host.token_budget import (
            count_messages_tokens,
            count_text_tokens,
            effective_input_budget,
            truncate_text_to_token_budget,
        )

        budget = effective_input_budget(window, max_tokens)
        budget = max(1, budget - max(0, int(reserved_tokens)))
        sys_t = count_text_tokens(system_prompt, provider, model)
        tools_t = _estimate_tools_tokens(registry, provider, model)
        hist_t = count_messages_tokens(history, provider, model) if history else 0
        text_t = count_text_tokens(text, provider, model)
        rag_t = count_text_tokens(rag_block, provider, model) if rag_block else 0

        total = sys_t + tools_t + hist_t + text_t + rag_t
        result.budget = budget
        result.total_before = total
        if total <= budget:
            return result

        logger.warning(
            "[GENY_BUDGET] 입력 %d > 예산 %d (window=%d sys=%d tools=%d hist=%d text=%d rag=%d) — clamp",
            total,
            budget,
            window,
            sys_t,
            tools_t,
            hist_t,
            text_t,
            rag_t,
        )

        overhead = sys_t + tools_t + hist_t
        # ── 1) RAG 블록 truncate (사용자 텍스트 보존) ──
        new_rag = rag_block
        if rag_block:
            rag_budget = max(0, budget - overhead - text_t)
            new_rag, cut = truncate_text_to_token_budget(
                rag_block,
                rag_budget,
                provider=provider,
                model=model,
            )
            if cut:
                result.rag_block = new_rag
                result.clamped = True
                rag_t = count_text_tokens(new_rag, provider, model)

        # ── 2) 그래도 초과면 사용자 텍스트 truncate (최후 수단) ──
        available_for_text = budget - overhead - rag_t
        if text and text_t > available_for_text:
            if available_for_text <= 0:
                # system/tools/이력만으로 예산 초과 — 입력 공간이 없다.
                # 파이프라인의 이력 압축(Stage 2/4)이 hist 를 줄일 수 있으므로
                # 최소한의 입력은 남긴다 (완전 소거는 질문 자체를 지운다).
                logger.error(
                    "[GENY_BUDGET] overhead(%d)+hist(%d)+rag(%d) 만으로 예산(%d) 초과 — "
                    "이력 압축(Stage 2/4)에 위임하고 입력은 최소 보존",
                    sys_t + tools_t,
                    hist_t,
                    rag_t,
                    budget,
                )
            new_text, cut = truncate_text_to_token_budget(
                text,
                max(1024, available_for_text),
                provider=provider,
                model=model,
            )
            if cut:
                result.text = new_text
                result.clamped = True
    except Exception as exc:  # noqa: BLE001 — fail-open
        logger.warning("[GENY_BUDGET] 예산 적용 중 오류(무시하고 진행): %s", exc, exc_info=True)
    return result


#: 입력이 잘렸을 때 스트림 앞에 내보내는 사용자 안내 (agent_xgen 과 동일 원칙 —
#: 축약은 실패가 아니라 부분 반영이므로 안내 톤).
CLAMP_NOTICE = (
    "[안내: 입력이 모델 컨텍스트보다 커서 참조 내용/입력 일부를 생략해 반영했습니다. "
    "더 정확한 결과가 필요하면 입력이나 참조 문서를 줄여 주세요.]\n\n"
)
