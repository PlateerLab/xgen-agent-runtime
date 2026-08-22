"""Provider별 LLM 파라미터 사전 검증 — 실행 전에 잘못된 설정을 정확한 메시지로 차단.

지금까지는 prepare_llm_components 가 범위 밖 temperature 를 조용히 잘라(clamp)
실행했다 — Anthropic 에 1.5 를 넣으면 사용자가 설정한 값과 다른 값으로 실행되거나
(bedrock), provider 원문 400 에러가 그대로 노출됐다(anthropic). 여기서 실행 전에
검증해 ERROR104 규약 메시지로 차단한다 (에러 코드 정의: ERROR.md).

에러 메시지는 다른 에러 코드와 동일하게 `[ERROR104: ...]` 문자열로 그대로
채팅 출력에 내보낸다 — agent_xgen(스트리밍/논스트리밍 양쪽), agent_geny 공용.
"""

import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

# provider별 temperature 허용 범위 (provider API 명세 기준).
# - OpenAI / vLLM(OpenAI 호환) / Google Gemini: 0 ~ 2
# - Anthropic / AWS Bedrock(Converse) / Claude Code(Anthropic 백엔드): 0 ~ 1
PROVIDER_TEMPERATURE_RANGES = {
    "openai": (0.0, 2.0),
    "deepseek": (0.0, 2.0),
    "vllm": (0.0, 2.0),
    "google": (0.0, 2.0),
    "anthropic": (0.0, 1.0),
    "bedrock": (0.0, 1.0),
    "vertex": (0.0, 2.0),
    "claude_code": (0.0, 1.0),
    # codex CLI 는 temperature 를 받지 않는다(드롭) — 범위 미등록 = 검증 생략.
}

PROVIDER_LABELS = {
    "openai": "OpenAI",
    "deepseek": "DeepSeek (OpenAI 호환)",
    "vllm": "vLLM (OpenAI 호환)",
    "google": "Google (Gemini)",
    "anthropic": "Anthropic",
    "bedrock": "AWS Bedrock",
    "vertex": "Google Vertex AI",
    "claude_code": "Claude Code",
    "codex": "OpenAI Codex",
}


def temperature_range_for(provider: str) -> Optional[Tuple[float, float]]:
    """provider 의 temperature 허용 범위. 미등록 provider 는 None(검증 생략)."""
    return PROVIDER_TEMPERATURE_RANGES.get((provider or "").strip().lower())


def _format_range(bounds: Tuple[float, float]) -> str:
    lo, hi = bounds
    fmt = lambda v: str(int(v)) if float(v).is_integer() else str(v)  # noqa: E731
    return f"{fmt(lo)} ~ {fmt(hi)}"


def validate_agent_params(
    provider: str,
    *,
    temperature: Any = None,
    param_label: str = "Temperature(답변 창의성 수준)",
) -> Optional[str]:
    """실행 전 provider별 파라미터 검증.

    위반 시 그대로 사용자에게 내보낼 `[ERROR104: ...]` 메시지를 반환하고,
    통과하면 None. 값이 비어 있으면(None/"") provider 기본값을 쓰는 경우이므로
    검증하지 않는다. 미등록 provider(커스텀 엔드포인트 등)도 검증하지 않는다.
    """
    normalized = (provider or "").strip().lower()
    label = PROVIDER_LABELS.get(normalized, provider or "지정되지 않은 provider")

    if temperature is not None and temperature != "":
        try:
            value = float(temperature)
        except (TypeError, ValueError):
            return (
                f"[ERROR104: {param_label} 값 '{temperature}'을(를) 숫자로 해석할 수 없습니다. "
                f"노드 설정에서 숫자 값으로 수정해 주세요.]"
            )
        bounds = temperature_range_for(normalized)
        if bounds is not None and not (bounds[0] <= value <= bounds[1]):
            range_text = _format_range(bounds)
            hint = ""
            if bounds[1] == 1.0:
                hint = (
                    " (OpenAI/vLLM/Google은 0 ~ 2를 허용하지만 이 provider는 0 ~ 1만 허용합니다.)"
                )
            return (
                f"[ERROR104: {param_label} {value}은(는) {label}이(가) 지원하지 않는 값입니다. "
                f"{label}의 허용 범위는 {range_text} 입니다.{hint} "
                f"노드 설정에서 값을 범위 안으로 수정해 주세요.]"
            )

    return None
