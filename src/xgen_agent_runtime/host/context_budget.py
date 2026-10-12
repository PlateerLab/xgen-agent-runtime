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

대화 이력(preload)은 API 백엔드에서는 건드리지 않는다. 파이프라인 압축(Stage 2/4/6)이
요약 체크포인트까지 줄일 수 있으므로 그 크기만큼만 센다(4.89.0). 압축이 없는 CLI 백엔드는
가장 오래된 이력부터 뺀다.

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
    #: 이력을 잘라야 했을 때(압축이 없는 CLI 백엔드)의 새 이력. ``None`` 이면 그대로.
    history: Optional[List[Any]] = None
    #: 맞춘 뒤의 메모리 노드 이전 대화 문자열(잘리지 않았으면 받은 그대로).
    history_context: str = ""
    #: 사용자 입력(text·rag)이 잘렸는가. 이력·이전 대화 문자열만 줄었으면 False: 사용자에게 "입력 일부를
    #: 생략했다" 고 알릴 일이 아니다.
    input_clamped: bool = False
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

    Stage 3 이 실제로 보내는 것(``to_api_format(exposed_only=True)``: core + 열린 도구)을
    센다. 직렬화 실패 시 도구당 상수로 강등.
    """
    if registry is None:
        return 0
    try:
        from xgen_agent_runtime.host.token_budget import count_text_tokens

        import json

        try:
            # What Stage 3 actually sends: core + activated tools. Deferred tools stay out of the
            # request until ToolSearch promotes them: counting the whole catalogue (hierarchy
            # mode, the default) used to clamp the user's message on small windows for nothing.
            schemas = registry.to_api_format(exposed_only=True)
        except TypeError:
            schemas = registry.to_api_format()
        blob = json.dumps(schemas, ensure_ascii=False, default=str)
        return count_text_tokens(blob, provider, model)
    except Exception:  # noqa: BLE001
        try:
            return _FALLBACK_TOKENS_PER_TOOL * len(registry)
        except Exception:  # noqa: BLE001
            return 0


#: 파이프라인 압축이 이력을 최소한 이만큼은 남긴다. 입력 맞춤은 이력을 이 크기까지 줄어들 수 있는
#: 것으로 본다. 이력 때문에 사용자의 새 질문을 자르지 않는다.
_COMPACTED_HISTORY_FLOOR = 5_000
#: 이전 대화 문자열을 이보다 작게 줄일 바에는 뺀다(머리글만 남는다).
_MIN_HISTORY_CONTEXT_TOKENS = 256


def _compacted_history_floor(window: int) -> int:
    """압축 뒤 이력(체크포인트)이 차지할 수 있는 최대 토큰: 요약 + 원문으로 남기는 사용자 말 + 머리글.

    5,000 고정이었을 때는 큰 창에서 체크포인트(사용자 말만 창의 8%, 최대 2만)가 그보다 커서, 입력 맞춤이
    받아들인 질문과 RAG 를 압축으로도 담지 못했다.
    """
    try:
        from xgen_agent_runtime.stages.s02_context.artifact.default.compactors import (
            checkpoint_tokens_bound,
        )

        return max(_COMPACTED_HISTORY_FLOOR, checkpoint_tokens_bound(window))
    except Exception:  # noqa: BLE001 (fail-open)
        return _COMPACTED_HISTORY_FLOOR


#: 내장 메모리 주입(Stage 2 → 시스템 프롬프트의 Relevant Knowledge)이 창에서 차지할 수 있는 몫.
#: 주입 상한은 글자(기본 10,000자)라 8k~32k 창에서는 그것만으로 창의 3분의 1을 넘었다.
MEMORY_INJECT_WINDOW_SHARE = 0.10
#: 주입 예약의 상한(창 비율): 작은 창에서 예약이 입력 자리를 다 먹지 않게.
_MEMORY_RESERVE_WINDOW_SHARE = 0.15
#: 주입 블록의 머리글·구분선 몫(토큰).
_MEMORY_INJECT_FRAME_TOKENS = 600
#: 메모리 노드가 넘긴 이전 대화 문자열이 창에서 차지할 수 있는 몫. 노드 기본값은 100,000자다.
HISTORY_CONTEXT_WINDOW_SHARE = 0.25


def memory_inject_cap_chars(window: int, configured: int) -> int:
    """창에 맞춘 내장 메모리 주입 상한(글자). 창을 모르면 설정값 그대로."""
    from xgen_agent_runtime.core.text_tokens import chars_for_tokens

    configured = max(0, int(configured))
    if not window or window <= 0:
        return configured
    return min(configured, chars_for_tokens(int(window * MEMORY_INJECT_WINDOW_SHARE)))


def memory_reserved_tokens(window: int, configured_chars: int) -> int:
    """fit 이후 시스템 프롬프트에 붙을 내장 메모리 주입의 선예약분(토큰).

    주입 상한(:func:`memory_inject_cap_chars`)을 글자당 1토큰으로 보수적으로 환산하고 머리글 몫을
    더한다. 예전에는 창과 무관하게 3,000토큰이었다. 큰 창에서는 실제 주입(최대 10,000자, 한국어면
    거의 같은 수의 토큰)보다 작았고, 8k 창에서는 입력 자리의 절반이었다.
    """
    from xgen_agent_runtime.core.text_tokens import CONSERVATIVE_CHARS_PER_TOKEN

    cap = memory_inject_cap_chars(window, configured_chars)
    tokens = int(cap / CONSERVATIVE_CHARS_PER_TOKEN) + _MEMORY_INJECT_FRAME_TOKENS
    if window and window > 0:
        tokens = min(tokens, int(window * _MEMORY_RESERVE_WINDOW_SHARE))
    return max(0, tokens)


def fit_history_context_tokens(context: str, tokens: int, provider: str, model: str) -> str:
    """이전 대화 문자열을 ``tokens`` 안으로(머리글과 가장 최근 대화가 남게 가운데를 줄인다)."""
    if not context:
        return context
    try:
        from xgen_agent_runtime.host.token_budget import truncate_text_to_token_budget

        fitted, _cut = truncate_text_to_token_budget(
            context,
            max(_MIN_HISTORY_CONTEXT_TOKENS, int(tokens)),
            provider=provider,
            model=model,
            front_ratio=0.2,
            omit_notice="\n\n[... 더 오래된 대화 생략 ...]\n\n",
        )
        return fitted
    except Exception:  # noqa: BLE001 (fail-open)
        logger.warning("[GENY_BUDGET] 이전 대화 문자열 맞춤 실패: 그대로 둔다", exc_info=True)
        return context


def fit_history_context(context: str, window: int, provider: str, model: str) -> str:
    """메모리 노드의 이전 대화 문자열을 창의 :data:`HISTORY_CONTEXT_WINDOW_SHARE` 안으로. 창을 모르면 그대로.

    입력 맞춤(:func:`fit_input_to_budget`)이 남는 자리에 맞춰 한 번 더 줄인다.
    """
    if not context or not window or window <= 0:
        return context
    return fit_history_context_tokens(
        context, int(window * HISTORY_CONTEXT_WINDOW_SHARE), provider, model
    )


def _next_history_start(history: List[Any], start: int, keep_last: int = 2) -> int:
    """``history[start:]`` 에서 가장 오래된 메시지 하나를 뺀 다음의 시작 위치. 더 뺄 수 없으면 ``start``.

    새 첫 메시지는 user 여야 한다. 앞쪽 assistant·도구 결과는 함께 빠진다(도구 결과가 짝을 잃지
    않게). 마지막 ``keep_last`` 개는 남긴다.
    """
    end = len(history)
    if end - start <= keep_last:
        return start
    nxt = start + 1
    while end - nxt > keep_last and not (
        isinstance(history[nxt], dict)
        and history[nxt].get("role") == "user"
        and not _tool_result_only(history[nxt])
    ):
        nxt += 1
    return nxt


def _tool_result_only(message: Any) -> bool:
    content = message.get("content") if isinstance(message, dict) else None
    return (
        isinstance(content, list)
        and bool(content)
        and all(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
    )


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
    history_compressible: bool = True,
    history_context: str = "",
) -> BudgetFit:
    """이번 턴의 입력측을 유효 예산 안으로 맞춘다.

    예산 = window − 출력 예약(:func:`core.compaction.output_reserve`, Stage 4 guard 와 같은 값)
    − 안전마진 − reserved_tokens. 예전에는 출력 예약을 max_tokens 로만 잡아 guard 가 요구하는 것보다
    2,048토큰 더 채웠고, RAG 를 꽉 채운 턴은 provider 가 받았을 요청을 guard 가 거절했다.

    구성요소:
        overhead = system_prompt + tool 스키마(실제로 보내는 것)
        이력     = preload 된 지난 대화
        가변분   = history_context → rag_block → text 순으로 줄인다

    이력의 몫(4.89.0):
        ``history_compressible=True`` (API 백엔드): 파이프라인 압축(Stage 2/4/6)이 이력을 요약
        체크포인트까지 줄일 수 있다. 그래서 RAG·질문의 자리를 정할 때 이력은 체크포인트가 차지할 수
        있는 최대 크기(:func:`_compacted_history_floor`)만큼만 센다. 예전에는 이력 전체를 세어, 압축할 수
        있는 지난 대화 때문에 압축할 수 없는 **이번 질문**을 1,024토큰까지 잘랐다.
        ``history_compressible=False`` (CLI 백엔드: 런타임 압축이 없고 지난 대화가 첫 메시지 하나로
        접혀 CLI 도 줄이지 못한다): 넘치면 **가장 오래된 이력부터** 빼고(마지막 두 메시지는 남김),
        그다음 가변분을 줄인다. 잘린 이력은 ``BudgetFit.history`` 로 돌려준다.

    ``history_context``: 메모리 노드가 넘긴 이전 대화 문자열. 시스템 프롬프트 끝에 붙어 압축이 줄일 수
    없으므로 여기서 가장 먼저 줄인다(맞춘 값은 ``BudgetFit.history_context``).

    ``reserved_tokens``: fit **이후** 턴 중에 system 으로 주입되는 것들(내장 메모리 등)의 선예약분.

    fail-open: 어떤 예외도 입력을 그대로 돌려준다. 예산 로직의 버그가 턴을 죽여선 안 된다.
    """
    result = BudgetFit(
        text=text,
        rag_block=rag_block,
        clamped=False,
        window=window,
        history_context=history_context,
    )
    if not window or window <= 0:
        return result
    try:
        from xgen_agent_runtime.core.compaction import output_reserve
        from xgen_agent_runtime.host.token_budget import (
            budget_margin,
            count_message_tokens,
            count_messages_tokens,
            count_text_tokens,
            effective_input_budget,
            truncate_text_to_token_budget,
        )

        budget = min(
            effective_input_budget(window, max_tokens),
            window - output_reserve(window, max_tokens) - budget_margin(window),
        )
        budget = max(1, budget - max(0, int(reserved_tokens)))
        sys_t = count_text_tokens(system_prompt, provider, model)
        tools_t = _estimate_tools_tokens(registry, provider, model)
        hist_t = count_messages_tokens(history, provider, model) if history else 0
        text_t = count_text_tokens(text, provider, model)
        rag_t = count_text_tokens(rag_block, provider, model) if rag_block else 0
        ctx_t = count_text_tokens(history_context, provider, model) if history_context else 0

        total = sys_t + tools_t + hist_t + text_t + rag_t + ctx_t
        result.budget = budget
        result.total_before = total
        if total <= budget:
            return result

        if history_compressible:
            hist_need = min(hist_t, _compacted_history_floor(window))
        else:
            # 오래된 이력부터 뺀다. 이번 턴의 입력보다 먼저. 메시지별 토큰은 한 번만 센다.
            msgs = list(history or [])
            per = [count_message_tokens(m, provider, model) for m in msgs]
            start = 0
            while sys_t + tools_t + hist_t + text_t + rag_t + ctx_t > budget:
                nxt = _next_history_start(msgs, start)
                if nxt == start:
                    break
                hist_t -= sum(per[start:nxt])
                start = nxt
            if start:
                result.history = msgs[start:]
                result.clamped = True
            hist_need = hist_t

        if sys_t + tools_t + hist_need + text_t + rag_t + ctx_t <= budget:
            if history_compressible:
                logger.info(
                    "[GENY_BUDGET] 입력 %d > 예산 %d: 이력(%d)은 파이프라인 압축에 맡기고 입력은 그대로",
                    total,
                    budget,
                    hist_t,
                )
            return result

        logger.warning(
            "[GENY_BUDGET] 입력 %d > 예산 %d (window=%d sys=%d tools=%d hist=%d ctx=%d text=%d rag=%d), clamp",
            total,
            budget,
            window,
            sys_t,
            tools_t,
            hist_t,
            ctx_t,
            text_t,
            rag_t,
        )

        overhead = sys_t + tools_t + hist_need
        # ── 0) 메모리 노드의 이전 대화 문자열: 가장 오래된 쪽부터 ──
        if history_context:
            room = budget - overhead - text_t - rag_t
            if ctx_t > room:
                new_ctx = (
                    fit_history_context_tokens(history_context, room, provider, model)
                    if room >= _MIN_HISTORY_CONTEXT_TOKENS
                    else ""
                )
                result.history_context = new_ctx
                result.clamped = True
                ctx_t = count_text_tokens(new_ctx, provider, model) if new_ctx else 0
        overhead += ctx_t

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
                result.input_clamped = True
                rag_t = count_text_tokens(new_rag, provider, model)

        # ── 2) 그래도 초과면 사용자 텍스트 truncate (최후 수단) ──
        available_for_text = budget - overhead - rag_t
        if text and text_t > available_for_text:
            if available_for_text <= 0:
                # system/tools 만으로 예산 초과: 입력 공간이 없다. 최소한의 입력은 남긴다
                # (완전 소거는 질문 자체를 지운다).
                logger.error(
                    "[GENY_BUDGET] overhead(%d)+hist(%d)+rag(%d) 만으로 예산(%d) 초과: 입력은 최소 보존",
                    sys_t + tools_t,
                    hist_need,
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
                result.input_clamped = True
    except Exception as exc:  # noqa: BLE001 (fail-open)
        logger.warning("[GENY_BUDGET] 예산 적용 중 오류(무시하고 진행): %s", exc, exc_info=True)
    return result


#: 입력이 잘렸을 때 스트림 앞에 내보내는 사용자 안내 (agent_xgen 과 동일 원칙 —
#: 축약은 실패가 아니라 부분 반영이므로 안내 톤).
CLAMP_NOTICE = (
    "[안내: 입력이 모델 컨텍스트보다 커서 참조 내용/입력 일부를 생략해 반영했습니다. "
    "더 정확한 결과가 필요하면 입력이나 참조 문서를 줄여 주세요.]\n\n"
)
