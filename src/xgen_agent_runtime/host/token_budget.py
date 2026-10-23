"""토큰 예산(budget) 유틸 — agent_xgen 입력 토큰 한계 & compaction 의 단일 진실.

설계 배경 (TOKEN/COMPACT 개선):
    기존 코드는 입력 크기를 *글자 수* (MAX_CONTEXT_CHARS, functions.py) 로만 근사했다.
    토큰≠글자이고 모델별 실제 컨텍스트 윈도우(128K/200K/1M)도 무시했으며, system
    prompt·tool 스키마·memory·이미지를 한도 계산에 합산하지 않았다. 이 모듈은:

      1) 문자 부류 기반 *토큰* 추정 (+ provider 보정계수 + 안전마진) — 항상 과대추정해
         provider 측 400(context length exceeded) 을 사전 차단한다.
      2) 모델별 실제 컨텍스트 윈도우 테이블 (+ 사용자 override).
      3) 유효 입력 예산 = 윈도우 − 출력 max_tokens − 안전마진.
      4) 토큰 인지 truncation / 대화 compaction(드롭+요약 하이브리드, 동일 모델 사용).

    **네트워크를 타지 않는다.** 예전에는 tiktoken 으로 셌는데, 그 라이브러리는 인코딩
    표(BPE)를 첫 호출 때 인터넷에서 내려받는다 — 타임아웃조차 없다. 폐쇄망에서는 그 한
    줄이 턴을 멈춰 세웠다(o200k → cl100k 로 두 번 시도했으므로 두 배로). 지금은
    :mod:`.token_estimator` 로 센다 — 실제 tiktoken 대조 평균 절대 오차 8.6%(옛 글자/4
    fallback 은 48%). 정확도보다 "안전한 과대추정"이 우선인 것은 그대로다.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from xgen_agent_runtime.core.text_tokens import estimate_tokens, estimate_tokens_cached  # noqa: F401

logger = logging.getLogger("token-budget")


# ─────────────────────────────────────────────────────────────────────────
# Provider 보정계수 & 안전마진
#   tiktoken(o200k_base) 은 OpenAI 기준. 타 provider 는 토크나이저가 달라 토큰 수가
#   더 많을 수 있으므로 보정계수로 *과대추정* 한다 (안전한 방향). 추가로 전역 안전마진.
# ─────────────────────────────────────────────────────────────────────────
# 계수 표의 원본은 core.text_tokens 다(압축 판단도 같은 표를 쓴다). 여기 이름은 예전 호출부를 위한 별칭.
from xgen_agent_runtime.core.text_tokens import (  # noqa: E402
    DEFAULT_PROVIDER_FACTOR as _DEFAULT_PROVIDER_FACTOR,
    PROVIDER_TOKEN_FACTOR as _PROVIDER_TOKEN_FACTOR,
    SAFETY_MULTIPLIER as _SAFETY_MULTIPLIER,
)

# 이미지 1장당 근사 토큰 비용 (멀티모달). provider/해상도별로 다르나 보수적 상수.
_IMAGE_TOKEN_COST = 1200


# ─────────────────────────────────────────────────────────────────────────
# 모델별 컨텍스트 윈도우 (토큰). 접두 매칭(longest-prefix wins).
#   - 값은 모델의 *전체* 컨텍스트 윈도우(입력+출력). 유효 입력 예산은 여기서 출력
#     max_tokens 와 안전마진을 뺀 값(effective_input_budget 참조).
#   - 이 테이블은 *fallback* — 1순위는 vLLM 서버 실측(max_model_len), 2순위는
#     llm_model_catalog.capabilities.max_context (어드민 카탈로그 DB).
#     둘 다 없을 때만 접두 매칭 → provider 기본값 순으로 내려온다.
#   - agent 노드의 context_window 파라미터는 언제나 최우선 override.
# ─────────────────────────────────────────────────────────────────────────
_MODEL_CONTEXT_WINDOW: List[Tuple[str, int]] = [
    # OpenAI. gpt-5 계열은 창이 400k 여도 **입력은 272k** 까지다(나머지는 출력 몫): 창 값을 400k 로
    # 두면 유효 예산이 실제 입력 한도를 넘어 압축보다 provider 400 이 먼저 난다. 그래서 입력 한도를
    # 창으로 쓴다. gpt-5.5 이후·gpt-6 은 1.05M 창에 출력 128k(2026-10-12 OpenAI 모델 문서 확인).
    ("gpt-4.1", 1_047_576),
    ("gpt-4o", 128_000),
    ("gpt-4-turbo", 128_000),
    ("gpt-4", 128_000),
    ("gpt-3.5", 16_385),
    ("gpt-35-turbo", 16_385),  # Azure 배포 이름
    ("o1", 200_000),
    ("o3", 200_000),
    ("o4", 200_000),
    ("gpt-5", 272_000),
    ("gpt-5-chat", 128_000),
    ("gpt-5.5", 922_000),
    ("gpt-6", 922_000),
    # Anthropic / Bedrock / Vertex Claude. Opus 4.6 이후·Sonnet 4.6 이후·Haiku 5.5·Fable·Mythos 는
    # 1M 이 기본이다(베타 헤더 불필요). 그 밖은 200k(2026-10-12 Anthropic 문서 확인).
    ("claude-3-5", 200_000),
    ("claude-3-7", 200_000),
    ("claude-3", 200_000),
    ("claude-sonnet-4", 200_000),
    ("claude-sonnet-4-6", 1_000_000),
    ("claude-opus-4", 200_000),
    ("claude-opus-4-6", 1_000_000),
    ("claude-opus-4-7", 1_000_000),
    ("claude-opus-4-8", 1_000_000),
    ("claude-haiku-4", 200_000),
    ("claude-4", 200_000),
    ("claude-opus-5", 1_000_000),
    ("claude-sonnet-5", 1_000_000),
    ("claude-haiku-5", 1_000_000),
    ("claude-fable", 1_000_000),
    ("claude-mythos", 1_000_000),
    ("claude", 200_000),
    # Google Gemini / Gemma (Gemma 는 gemini 접두에 걸리지 않으므로 provider 기본값 1M 으로 새지 않게)
    ("gemini-1.5", 1_048_576),
    ("gemini-2.0", 1_048_576),
    ("gemini-2.5", 1_048_576),
    ("gemini", 1_048_576),
    ("gemma-4", 131_072),
    ("gemma-3", 131_072),
    ("gemma-3-1b", 32_768),
    ("gemma-3-270m", 32_768),
    ("gemma-3n", 32_768),
    ("gemma-2", 8_192),
    # 모르는 Gemma 판. 작게 잡으면 질문이 잘리고(8k 창 + max_tokens 8k 는 입력 자리가 없다), 크게 잡으면
    # provider 가 한도를 말해 줄 때 넘침 복구가 창을 고친다. 중간값.
    ("gemma", 32_768),
]

# provider 기준 기본값 (모델 매칭 실패 시).
_PROVIDER_DEFAULT_WINDOW = {
    "openai": 128_000,
    # Azure 는 모델 이름이 배포 이름이라 표에 안 잡히는 일이 흔하다. 현행 배포 모델의 바닥.
    "azure": 128_000,
    "azure_foundry": 128_000,
    "anthropic": 200_000,
    # BedrockClient 는 AnthropicClient 이고 Claude 만 부른다. 이름에 모델명이 없는 별칭(sonnet)·추론 프로필
    # ARN 도 Claude 다.
    "bedrock": 200_000,
    "google": 1_048_576,
    "vertex": 1_048_576,
    "vllm": 32_768,
    # OpenRouter 는 모델마다 창이 다르다(4k~1M). 호스트는 카탈로그의 context_length 를 먼저 쓰고,
    # 그것이 없을 때만 이 값 — 요즘 도구 호출 모델의 바닥.
    "openrouter": 128_000,
    # CLI 백엔드: 모델 이름이 별칭(sonnet/opus)이라 표에 안 잡힌다. 런타임은 이 값으로 첫 메시지에
    # 접어 넣는 지난 대화의 양만 맞춘다. 작게 잡으면 더 일찍 덜어낼 뿐 실패하지 않는다.
    "claude_code": 200_000,
    "codex": 272_000,
}
# 완전 미상(provider 도 모름)일 때 보수적 기본값.
DEFAULT_CONTEXT_WINDOW = 32_000

# 안전마진: 윈도우의 2%, 최소 1024 토큰.
_BUDGET_MARGIN_RATIO = 0.02
_BUDGET_MARGIN_MIN = 1024


# 카탈로그(capabilities) 조회 캐시 — (provider, model) → (조회 시각, capabilities|None).
# 실패(None)도 캐시해 DB 장애 시 실행마다 재시도하지 않는다.
_CATALOG_WINDOW_TTL_SEC = 300
_catalog_caps_cache: Dict[Tuple[str, str], Tuple[float, Optional[Dict[str, Any]]]] = {}

# agent provider 표기 → llm_model_catalog.provider 정규화 (catalog_reader 와 동일 규칙)
_PROVIDER_TO_CATALOG_ID = {"gemini": "google"}


def _catalog_capabilities(
    provider: Optional[str], model: Optional[str]
) -> Optional[Dict[str, Any]]:
    """`llm_model_catalog.capabilities` 조회 (TTL 캐시, fail-open).

    public 모델(openai/anthropic/google/bedrock)의 윈도우·이미지 입력 같은 능력의 관리 진실은
    어드민 카탈로그 DB. DB/컨테이너 불가 환경(테스트 등)에서는 조용히 None → 정적 fallback.
    """
    if not provider or not model:
        return None
    key = (provider, model)
    cached = _catalog_caps_cache.get(key)
    now = time.time()
    if cached and now - cached[0] < _CATALOG_WINDOW_TTL_SEC:
        return cached[1]

    caps: Optional[Dict[str, Any]] = None
    try:
        from app_container import app_container  # noqa: PLC0415 — lazy (테스트/도구 환경 방어)

        app_db = app_container.get_app_db()
        catalog_provider = _PROVIDER_TO_CATALOG_ID.get(provider.lower(), provider.lower())
        rows = app_db.config_db_manager.execute_query(
            "SELECT capabilities FROM llm_model_catalog "
            "WHERE provider = %s AND model_id = %s LIMIT 1",
            (catalog_provider, model),
        )
        if rows:
            raw = rows[0].get("capabilities")
            if isinstance(raw, str):
                raw = json.loads(raw)
            caps = raw if isinstance(raw, dict) else None
    except Exception as e:
        logger.debug(
            "[TOKEN_BUDGET] 카탈로그 capabilities 조회 실패(%s/%s): %s", provider, model, e
        )

    _catalog_caps_cache[key] = (now, caps)
    return caps


def _catalog_max_context(provider: Optional[str], model: Optional[str]) -> Optional[int]:
    """`llm_model_catalog.capabilities.max_context` (없으면 None)."""
    max_context = (_catalog_capabilities(provider, model) or {}).get("max_context")
    if isinstance(max_context, (int, float)) and max_context > 0:
        win = int(max_context)
        logger.info(
            "[TOKEN_BUDGET] 카탈로그 max_context: %s/%s → %s tokens", provider, model, f"{win:,}"
        )
        return win
    return None


#: 공급자가 넘침 오류로 알려 준 실제 창: (provider, base_url, model) → (토큰, 배운 시각).
#: 예전에는 그 턴에만 썼다(state.context_window_budget). 다음 턴은 다시 큰 창을 믿고 같은 넘침과 압축을
#: 되풀이했다(2026-10-23 점검). 배포가 창을 늘렸을 수 있으므로 영원히 믿지 않는다.
_LEARNED_WINDOWS: Dict[Tuple[str, str, str], Tuple[int, float]] = {}
_LEARNED_LOCK = threading.Lock()
LEARNED_WINDOW_TTL_S = 6 * 3600.0


def _window_key(
    provider: Optional[str], model: Optional[str], base_url: Optional[str]
) -> Tuple[str, str, str]:
    return (
        (provider or "").lower().strip(),
        (base_url or "").strip().rstrip("/").lower(),
        (model or "").strip(),
    )


def learn_context_window(
    provider: Optional[str], model: Optional[str], base_url: Optional[str], limit: int
) -> None:
    """공급자가 넘침 오류에 적은 창을 기억한다(더 작은 값만, :data:`LEARNED_WINDOW_TTL_S` 동안)."""
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return
    if limit <= 0:
        return
    key = _window_key(provider, model, base_url)
    now = time.time()
    with _LEARNED_LOCK:
        old = _LEARNED_WINDOWS.get(key)
        if old is None or now - old[1] > LEARNED_WINDOW_TTL_S or limit < old[0]:
            _LEARNED_WINDOWS[key] = (limit, now)


def learned_context_window(
    provider: Optional[str], model: Optional[str], base_url: Optional[str]
) -> int:
    """기억해 둔 창(없거나 만료면 0)."""
    key = _window_key(provider, model, base_url)
    with _LEARNED_LOCK:
        entry = _LEARNED_WINDOWS.get(key)
        if entry is None:
            return 0
        if time.time() - entry[1] > LEARNED_WINDOW_TTL_S:
            _LEARNED_WINDOWS.pop(key, None)
            return 0
        return entry[0]


def resolve_context_window(
    provider: Optional[str],
    model: Optional[str],
    override: Optional[int] = None,
    base_url: Optional[str] = None,
    vllm_probe: Any = None,
) -> int:
    """모델의 전체 컨텍스트 윈도우(토큰)를 결정.

    우선순위:
        1) override (agent 노드 context_window 파라미터, >0)
        2) vLLM: 서버 /v1/models 실측 max_model_len (base_url 필요) — 서빙 엔진이 진실
        3) llm_model_catalog.capabilities.max_context — public 모델 관리 진실(DB)
        4) 모델 접두 매칭 테이블 (fallback)
        5) provider 기본값 > 전역 기본값

    그 값이 공급자가 최근 넘침 오류로 알려 준 창보다 크면 공급자의 값이 이긴다(:func:`learn_context_window`):
    그보다 큰 요청은 거절된다.
    """
    window = _resolve_context_window(provider, model, override, base_url, vllm_probe)
    learned = learned_context_window(provider, model, base_url)
    if 0 < learned < window:
        logger.info(
            "[TOKEN_BUDGET] %s/%s: 공급자가 알려 준 창 %d 을 쓴다(해석값 %d)",
            provider,
            model,
            learned,
            window,
        )
        return learned
    return window


def _resolve_context_window(
    provider: Optional[str],
    model: Optional[str],
    override: Optional[int],
    base_url: Optional[str],
    vllm_probe: Any,
) -> int:
    if override and override > 0:
        return int(override)

    prov = (provider or "").lower().strip()

    if prov == "vllm" and base_url and vllm_probe is not None:
        # 라이브 max_model_len 프로브는 host 주입(서버=agent 노드 헬퍼, 커넥터=None).
        try:
            live = vllm_probe(base_url, model)
            if live:
                return int(live)
        except Exception as e:
            logger.debug("[TOKEN_BUDGET] vLLM max_model_len 실측 실패(%s): %s", base_url, e)

    catalog_win = _catalog_max_context(prov, model)
    if catalog_win:
        return catalog_win

    name = (model or "").lower().strip()
    if prov == "claude_code":
        # CLI 가 쓰는 창은 모델 이름이 아니라 CLI 와 요금제가 정한다(1M 은 [1m] 별칭으로 고를 때). 정식 ID 를
        # 넘겨도 표의 API 창을 쓰지 않는다. 런타임은 이 값으로 첫 메시지에 접는 지난 대화만 맞춘다.
        return 1_000_000 if name.endswith("[1m]") else _PROVIDER_DEFAULT_WINDOW["claude_code"]
    if prov == "codex":
        # Codex CLI 는 모델의 API 창(gpt-5.5 1.05M)과 무관하게 자기 창(272k 입력)으로 돈다.
        return _PROVIDER_DEFAULT_WINDOW["codex"]
    if name and prov != "vllm":
        # 자체 서빙(vLLM)은 표를 쓰지 않는다: 같은 모델도 --max-model-len 을 줄여 띄우는 일이 흔해 모델의
        # 원래 창은 과대다. 서버 실측(위) → 카탈로그 → vLLM 기본값 순이다.
        # 이름 앞이나 구분자(. / :) 뒤에서 시작하는 접두만: 'apac.anthropic.claude-…',
        # 'meta.llama3-1-70b-…', 'Qwen/…'. 그냥 포함으로 보면 'o1' 같은 짧은 접두가 엉뚱한 이름 속에서
        # 걸린다. 가장 긴 접두가 이긴다.
        segments = [name] + [name[i + 1 :] for i, ch in enumerate(name) if ch in "./:"]
        best_len, best_win = -1, None
        for prefix, win in _MODEL_CONTEXT_WINDOW:
            if len(prefix) > best_len and any(_prefix_matches(seg, prefix) for seg in segments):
                best_len, best_win = len(prefix), win
        if best_win is not None:
            return best_win

    return _PROVIDER_DEFAULT_WINDOW.get(prov, DEFAULT_CONTEXT_WINDOW)


def _prefix_matches(name: str, prefix: str) -> bool:
    """``name`` starts with ``prefix`` and does not continue its number: 'gpt-5.5' is not a prefix
    of 'gpt-5.50', 'claude-opus-4-6' is not one of 'claude-opus-4-60'."""
    if not name.startswith(prefix):
        return False
    rest = name[len(prefix) :]
    return not (rest and rest[0].isdigit() and prefix[-1:].isdigit())


# ─────────────────────────────────────────────────────────────────────────
# 이미지 입력 — 모델이 그림을 볼 수 있는가. Read 가 이미지를 모델에게 건넬지("보여 준다"),
# "이 모델은 못 본다" 고 말할지를 가른다. 못 보는 모델에 그림을 보내면 provider 가 400 으로
# 턴을 깨므로 모르면 **못 본다** 쪽이다.
#   1) 노드 파라미터 override (bool)
#   2) llm_model_catalog.capabilities.supports_vision (bool) — 배포마다 다르다(같은 이름도
#      서빙한 체크포인트가 글 전용일 수 있다). 자체 서빙(vLLM) 모델은 여기서만 켠다.
#   3) 공개 API 모델 계열의 접두 테이블 (fallback, longest-prefix wins)
# ─────────────────────────────────────────────────────────────────────────
_MODEL_VISION: List[Tuple[str, bool]] = [
    ("claude-3-5-haiku", False),
    ("claude", True),
    ("gpt-4o", True),
    ("gpt-4.1", True),
    ("gpt-4-turbo", True),
    ("gpt-5", True),
    ("o1-mini", False),
    ("o3-mini", False),
    ("o1", True),
    ("o3", True),
    ("o4", True),
    ("gemini", True),
]
#: provider 자체가 이미지 입력을 받는 CLI 백엔드 (모델 이름이 sonnet/opus 같은 별칭이다).
_PROVIDER_VISION = {"claude_code": True}


def resolve_supports_vision(
    provider: Optional[str],
    model: Optional[str],
    override: Any = None,
) -> bool:
    """모델이 이미지 입력을 받는가 — 모르면 False."""
    if isinstance(override, bool):
        return override
    caps = _catalog_capabilities((provider or "").lower().strip(), model) or {}
    for key in ("supports_vision", "vision"):
        if isinstance(caps.get(key), bool):
            return bool(caps[key])
    prov = (provider or "").lower().strip()
    if prov in _PROVIDER_VISION:
        return _PROVIDER_VISION[prov]
    name = (model or "").lower().strip()
    if not name:
        return False
    if "-vl" in name or "vl-" in name:
        return True
    # 이름 앞이나 구분자(. / :) 뒤에서 시작하는 것만 — 'apac.anthropic.claude-…', 'openai/gpt-4.1'.
    # 그냥 포함으로 보면 'o1'·'o3' 같은 짧은 접두가 엉뚱한 이름 속에서 걸린다.
    segments = [name] + [name[i + 1 :] for i, ch in enumerate(name) if ch in "./:"]
    best_len, best = -1, None
    for prefix, vision in _MODEL_VISION:
        if any(seg.startswith(prefix) for seg in segments) and len(prefix) > best_len:
            best_len, best = len(prefix), vision
    return bool(best)


def budget_margin(context_window: int) -> int:
    """추정 오차에 남기는 안전마진: 창의 2%, 최소 1,024토큰."""
    return max(_BUDGET_MARGIN_MIN, int(max(0, int(context_window or 0)) * _BUDGET_MARGIN_RATIO))


def effective_input_budget(
    context_window: int,
    reserved_output_tokens: int,
) -> int:
    """유효 입력 예산(토큰) = 윈도우 − 출력 예약 − 안전마진.

    최소 1 이상을 보장(음수 방지). 출력 예약이 윈도우보다 크면 윈도우의 절반만 입력에 할당.
    """
    margin = max(_BUDGET_MARGIN_MIN, int(context_window * _BUDGET_MARGIN_RATIO))
    reserved = max(0, int(reserved_output_tokens))
    if reserved >= context_window:
        # 비정상 설정 방어: 출력이 윈도우 이상이면 입력에 윈도우 절반만 허용.
        return max(1, context_window // 2)
    return max(1, context_window - reserved - margin)


# ─────────────────────────────────────────────────────────────────────────
# 토큰 카운팅
# ─────────────────────────────────────────────────────────────────────────
def _raw_token_count(text: str) -> int:
    """보정 전 원시 토큰 수. 네트워크를 타지 않는다(:mod:`.token_estimator`)."""
    if not text:
        return 0
    return estimate_tokens_cached(text)


def count_text_tokens(
    text: Optional[str],
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> int:
    """텍스트의 추정 토큰 수 (provider 보정 + 안전마진 포함, 과대추정)."""
    if not text:
        return 0
    raw = _raw_token_count(text)
    factor = _PROVIDER_TOKEN_FACTOR.get((provider or "").lower(), _DEFAULT_PROVIDER_FACTOR)
    return int(raw * factor * _SAFETY_MULTIPLIER) + 1


def _message_text_and_images(message: Any) -> Tuple[str, int]:
    """LangChain BaseMessage(또는 dict) 의 content 에서 (텍스트, 이미지수) 추출.

    content 가 멀티모달 list 인 경우 text 파트만 합치고 image_url 파트 수를 센다.
    """
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, str):
        return content, 0
    if isinstance(content, list):
        texts: List[str] = []
        images = 0
        for part in content:
            if isinstance(part, dict):
                ptype = part.get("type")
                if ptype == "text":
                    texts.append(str(part.get("text", "")))
                elif ptype in ("image_url", "image"):
                    images += 1
                else:
                    # 알 수 없는 dict 파트 — 문자열화해 보수적으로 포함.
                    texts.append(str(part))
            elif isinstance(part, str):
                texts.append(part)
        return "\n".join(texts), images
    if content is None:
        return "", 0
    return str(content), 0


def count_message_tokens(
    message: Any,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> int:
    """단일 메시지의 추정 토큰 (텍스트 + 이미지 비용 + role overhead)."""
    text, images = _message_text_and_images(message)
    tokens = count_text_tokens(text, provider, model)
    tokens += images * _IMAGE_TOKEN_COST
    tokens += 4  # role/구분자 overhead 근사
    return tokens


def count_messages_tokens(
    messages: Optional[List[Any]],
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> int:
    """메시지 리스트의 추정 토큰 합."""
    if not messages:
        return 0
    return sum(count_message_tokens(m, provider, model) for m in messages)


def estimate_tools_tokens(
    tools: Optional[List[Any]],
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> int:
    """tool(함수) 스키마가 프롬프트에 차지하는 토큰의 근사.

    name + description + args schema 를 직렬화해 카운트. 정확한 wire 포맷은 provider 마다
    다르나 근사로 충분(과대추정 방향).
    """
    if not tools:
        return 0
    import json  # noqa: PLC0415

    total = 0
    for t in tools:
        try:
            name = getattr(t, "name", "") or ""
            desc = getattr(t, "description", "") or ""
            args = getattr(t, "args", None)
            schema_str = ""
            if args is not None:
                try:
                    schema_str = json.dumps(args, ensure_ascii=False, default=str)
                except Exception:
                    schema_str = str(args)
            total += count_text_tokens(f"{name}\n{desc}\n{schema_str}", provider, model)
            total += 8  # 함수 정의 wrapper overhead 근사
        except Exception:
            total += 64  # 알 수 없는 도구 — 보수적 상수
    return total


# ─────────────────────────────────────────────────────────────────────────
# 토큰 인지 truncation
# ─────────────────────────────────────────────────────────────────────────
def truncate_text_to_token_budget(
    text: str,
    max_tokens: int,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    front_ratio: float = 0.5,
    omit_notice: str = "\n\n[... 중간 내용 생략됨 (토큰 한도 초과) ...]\n\n",
) -> Tuple[str, bool]:
    """텍스트를 max_tokens 이하로 토큰 인지 truncate. (결과, 잘렸는지) 반환.

    앞 front_ratio, 뒤 (1-front_ratio) 비율로 유지하고 중간을 생략한다. 자르는 단위는
    글자이지만, **그 텍스트에서 잰 글자/토큰 비율**로 환산하므로 언어가 섞여 있어도
    치우치지 않는다(한글은 글자당 토큰이 영어의 다섯 배다 — 고정 4자/토큰으로 자르면
    한국어 문서가 한도의 다섯 배로 잘린다).
    """
    if max_tokens <= 0:
        return "", bool(text)
    if not text:
        return text, False

    current = count_text_tokens(text, provider, model)
    if current <= max_tokens:
        return text, False

    notice_tokens = count_text_tokens(omit_notice, provider, model)
    # max_tokens 는 count_text_tokens(보정계수·안전마진 포함) 기준이다. raw 토큰으로 자를 때는
    # 그 배율을 역산해야 결과를 다시 세었을 때 한도 안에 든다 — 역산하지 않으면 vLLM(×1.26)
    # 에서 잘라 낸 결과가 늘 한도를 넘고, 호출부가 남은 초과분을 사용자 텍스트에서 깎는다.
    scale = (
        _PROVIDER_TOKEN_FACTOR.get((provider or "").lower(), _DEFAULT_PROVIDER_FACTOR)
        * _SAFETY_MULTIPLIER
    )
    keep = max(1, int((max_tokens - notice_tokens) / scale) - 1)

    # 앞과 뒤의 글 밀도가 전체 평균과 다를 수 있다(영어 본문 + 한국어 머리말): 잰 값에 비례해 줄여 맞춘다.
    for _ in range(8):
        front_budget = max(1, int(keep * front_ratio))
        back_budget = max(1, keep - front_budget)
        truncated = _cut_middle(text, front_budget, back_budget, omit_notice)
        counted = count_text_tokens(truncated, provider, model)
        if counted <= max_tokens or keep <= 1:
            return truncated, True
        keep = max(1, int(keep * min(0.9, max_tokens / counted * 0.95)))
    return truncated, True


def _cut_middle(text: str, front_budget: int, back_budget: int, omit_notice: str) -> str:
    """앞 front_budget, 뒤 back_budget raw 토큰만 남기고 중간을 생략한다.

    토큰 경계를 아는 인코더가 없으므로 글자로 자르되, **이 텍스트에서 잰** 글자/토큰
    비율을 쓴다. 고정 상수(옛 4자/토큰)를 쓰면 한국어에서 다섯 배로 잘린다.
    """
    total_tokens = estimate_tokens(text)
    if total_tokens <= 0:
        return text
    chars_per_token = len(text) / total_tokens
    front_chars = max(1, int(front_budget * chars_per_token))
    back_chars = max(1, int(back_budget * chars_per_token))
    if front_chars + back_chars >= len(text):
        return text
    return text[:front_chars] + omit_notice + text[-back_chars:]


# ─────────────────────────────────────────────────────────────────────────
# 대화 compaction (하이브리드: 최근 verbatim + 오래된 것 요약, 동일 모델)
# ─────────────────────────────────────────────────────────────────────────
_SUMMARY_PROMPT = (
    "다음은 사용자와 AI의 이전 대화입니다. 이후 대화에 필요한 핵심 사실·결정·맥락만"
    " 간결한 불릿으로 요약하세요. 추측하지 말고 대화에 있는 내용만 담으세요.\n\n{conversation}"
)


def _stringify_messages_for_summary(messages: List[Any]) -> str:
    parts: List[str] = []
    for m in messages:
        role = getattr(m, "type", None) or getattr(m, "role", None) or "msg"
        text, _ = _message_text_and_images(m)
        if text:
            parts.append(f"[{role}] {text}")
    return "\n".join(parts)


def compact_chat_history(
    messages: List[Any],
    target_tokens: int,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    llm: Any = None,
    keep_recent: int = 6,
) -> Tuple[List[Any], Optional[str], bool]:
    """대화 메시지를 target_tokens 이하로 compaction.

    전략(하이브리드):
      1) 최근 keep_recent 개는 항상 verbatim 보존(메시지 리스트로 반환).
      2) 그보다 오래된 메시지가 예산 초과 원인이면 동일 모델(llm)로 1개의 *요약 텍스트*로
         압축. (llm 없거나 실패 시 드롭 마커 텍스트)
      3) 그래도 초과면 최근 메시지도 오래된 순으로 추가 드롭.

    요약을 messages 리스트에 SystemMessage 로 끼워넣지 않고 *텍스트로 분리 반환* 한다 —
    Anthropic/Bedrock 등은 대화 중간 system 메시지를 싫어하므로, 호출자가 요약을
    system_prompt(대화 컨텍스트)에 안전하게 배치하도록 한다.

    Returns:
        (recent_messages, summary_text_or_None, changed)
    """
    if not messages:
        return messages, None, False
    if count_messages_tokens(messages, provider, model) <= target_tokens:
        return messages, None, False

    recent = list(messages[-keep_recent:]) if keep_recent > 0 else []
    older = messages[:-keep_recent] if keep_recent > 0 else list(messages)
    changed = False
    summary_text: Optional[str] = None

    if older:
        changed = True
        raw_summary = None
        if llm is not None:
            try:
                convo = _stringify_messages_for_summary(older)
                resp = llm.invoke(_SUMMARY_PROMPT.format(conversation=convo))
                raw_summary = getattr(resp, "content", None) or str(resp)
            except Exception as e:
                logger.warning("[TOKEN_BUDGET] 대화 요약 실패 — 드롭+마커로 대체: %s", e)
                raw_summary = None
        if raw_summary:
            summary_text = f"[이전 대화 요약]\n{raw_summary}"
        else:
            summary_text = f"[이전 메시지 {len(older)}개가 토큰 한도로 생략되었습니다.]"

    # 요약 텍스트 토큰도 예산에 포함해 최근 메시지를 추가 드롭한다.
    summary_t = count_text_tokens(summary_text, provider, model) if summary_text else 0
    while (summary_t + count_messages_tokens(recent, provider, model)) > target_tokens and recent:
        del recent[0]  # 가장 오래된 recent 부터 제거
        changed = True

    return recent, summary_text, changed
