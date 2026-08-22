"""토큰 예산(budget) 유틸 — agent_xgen 입력 토큰 한계 & compaction 의 단일 진실.

설계 배경 (TOKEN/COMPACT 개선):
    기존 코드는 입력 크기를 *글자 수* (MAX_CONTEXT_CHARS, functions.py) 로만 근사했다.
    토큰≠글자이고 모델별 실제 컨텍스트 윈도우(128K/200K/1M)도 무시했으며, system
    prompt·tool 스키마·memory·이미지를 한도 계산에 합산하지 않았다. 이 모듈은:

      1) tiktoken 기반 *토큰* 추정 (+ provider 보정계수 + 안전마진) — 항상 과대추정해
         provider 측 400(context length exceeded) 을 사전 차단한다.
      2) 모델별 실제 컨텍스트 윈도우 테이블 (+ 사용자 override).
      3) 유효 입력 예산 = 윈도우 − 출력 max_tokens − 안전마진.
      4) 토큰 인지 truncation / 대화 compaction(드롭+요약 하이브리드, 동일 모델 사용).

    tiktoken 이 없는(폐쇄망 등) 환경에서도 동작하도록 글자/4 fallback 으로 graceful
    degrade 한다. 정확도보다 "안전한 과대추정"이 우선.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("token-budget")

# ─────────────────────────────────────────────────────────────────────────
# tiktoken lazy load (없으면 char/4 fallback)
# ─────────────────────────────────────────────────────────────────────────
_ENCODER = None
_ENCODER_TRIED = False


def _get_encoder():
    """tiktoken o200k_base 인코더 (best-effort, 캐시). 실패 시 None → char fallback."""
    global _ENCODER, _ENCODER_TRIED
    if _ENCODER_TRIED:
        return _ENCODER
    _ENCODER_TRIED = True
    try:
        import tiktoken  # noqa: PLC0415
        try:
            _ENCODER = tiktoken.get_encoding("o200k_base")
        except Exception:
            _ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception as e:  # tiktoken 미설치 등
        logger.warning("[TOKEN_BUDGET] tiktoken 사용 불가 — 글자/4 추정으로 fallback: %s", e)
        _ENCODER = None
    return _ENCODER


# ─────────────────────────────────────────────────────────────────────────
# Provider 보정계수 & 안전마진
#   tiktoken(o200k_base) 은 OpenAI 기준. 타 provider 는 토크나이저가 달라 토큰 수가
#   더 많을 수 있으므로 보정계수로 *과대추정* 한다 (안전한 방향). 추가로 전역 안전마진.
# ─────────────────────────────────────────────────────────────────────────
_PROVIDER_TOKEN_FACTOR = {
    "openai": 1.0,
    "anthropic": 1.15,
    "bedrock": 1.15,
    "google": 1.20,
    "vertex": 1.20,
    "vllm": 1.20,
}
_DEFAULT_PROVIDER_FACTOR = 1.20
_SAFETY_MULTIPLIER = 1.05  # 모든 추정에 5% 가산

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
    # OpenAI
    ("gpt-4.1", 1_047_576),
    ("gpt-4o", 128_000),
    ("gpt-4-turbo", 128_000),
    ("gpt-4", 128_000),
    ("gpt-3.5", 16_385),
    ("o1", 200_000),
    ("o3", 200_000),
    ("o4", 200_000),
    ("gpt-5", 400_000),
    # Anthropic / Bedrock Claude
    ("claude-3-5", 200_000),
    ("claude-3-7", 200_000),
    ("claude-3", 200_000),
    ("claude-sonnet-4", 200_000),
    ("claude-opus-4", 200_000),
    ("claude-haiku-4", 200_000),
    ("claude-4", 200_000),
    ("claude", 200_000),
    # Google Gemini
    ("gemini-1.5", 1_048_576),
    ("gemini-2.0", 1_048_576),
    ("gemini-2.5", 1_048_576),
    ("gemini", 1_048_576),
]

# provider 기준 기본값 (모델 매칭 실패 시).
_PROVIDER_DEFAULT_WINDOW = {
    "openai": 128_000,
    "anthropic": 200_000,
    "bedrock": 200_000,
    "google": 1_048_576,
    "vertex": 1_048_576,
    "vllm": 32_768,
}
# 완전 미상(provider 도 모름)일 때 보수적 기본값.
DEFAULT_CONTEXT_WINDOW = 32_000

# 안전마진: 윈도우의 2%, 최소 1024 토큰.
_BUDGET_MARGIN_RATIO = 0.02
_BUDGET_MARGIN_MIN = 1024


# 카탈로그(capabilities.max_context) 조회 캐시 — (provider, model) → (조회 시각, 값|None).
# 실패(None)도 캐시해 DB 장애 시 실행마다 재시도하지 않는다.
_CATALOG_WINDOW_TTL_SEC = 300
_catalog_window_cache: Dict[Tuple[str, str], Tuple[float, Optional[int]]] = {}

# agent provider 표기 → llm_model_catalog.provider 정규화 (catalog_reader 와 동일 규칙)
_PROVIDER_TO_CATALOG_ID = {"gemini": "google"}


def _catalog_max_context(provider: Optional[str], model: Optional[str]) -> Optional[int]:
    """`llm_model_catalog.capabilities.max_context` 조회 (TTL 캐시, fail-open).

    public 모델(openai/anthropic/google/bedrock)의 윈도우 관리 진실은 어드민 카탈로그 DB.
    DB/컨테이너 불가 환경(테스트 등)에서는 조용히 None → 정적 fallback.
    """
    if not provider or not model:
        return None
    key = (provider, model)
    cached = _catalog_window_cache.get(key)
    now = time.time()
    if cached and now - cached[0] < _CATALOG_WINDOW_TTL_SEC:
        return cached[1]

    win: Optional[int] = None
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
            caps = rows[0].get("capabilities")
            if isinstance(caps, str):
                caps = json.loads(caps)
            max_context = (caps or {}).get("max_context")
            if isinstance(max_context, (int, float)) and max_context > 0:
                win = int(max_context)
                logger.info(
                    "[TOKEN_BUDGET] 카탈로그 max_context: %s/%s → %s tokens",
                    catalog_provider, model, f"{win:,}",
                )
    except Exception as e:
        logger.debug("[TOKEN_BUDGET] 카탈로그 max_context 조회 실패(%s/%s): %s", provider, model, e)

    _catalog_window_cache[key] = (now, win)
    return win


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
    """
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
    if name:
        best_len, best_win = -1, None
        for prefix, win in _MODEL_CONTEXT_WINDOW:
            if name.startswith(prefix) and len(prefix) > best_len:
                best_len, best_win = len(prefix), win
        if best_win is not None:
            return best_win
        # 접두가 아니라 포함이라도 잡아본다 (예: 'apac.anthropic.claude-...').
        best_len, best_win = -1, None
        for prefix, win in _MODEL_CONTEXT_WINDOW:
            if prefix in name and len(prefix) > best_len:
                best_len, best_win = len(prefix), win
        if best_win is not None:
            return best_win

    return _PROVIDER_DEFAULT_WINDOW.get((provider or "").lower(), DEFAULT_CONTEXT_WINDOW)


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
    """보정 전 원시 토큰 수 (tiktoken 또는 글자/4 fallback)."""
    if not text:
        return 0
    enc = _get_encoder()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    # fallback: 영문 ~4자/토큰, 한중일은 더 조밀하나 보수적으로 4 사용 후 보정계수가 흡수.
    return (len(text) + 3) // 4


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

    앞 front_ratio, 뒤 (1-front_ratio) 비율로 유지하고 중간을 생략(기존 동작 계승하되
    글자→토큰 기준). tiktoken 이 있으면 토큰 단위로 정확히 자르고, 없으면 글자 비례로 자른다.
    """
    if max_tokens <= 0:
        return "", bool(text)
    if not text:
        return text, False

    current = count_text_tokens(text, provider, model)
    if current <= max_tokens:
        return text, False

    enc = _get_encoder()
    notice_tokens = count_text_tokens(omit_notice, provider, model)
    keep = max(1, max_tokens - notice_tokens)
    front_budget = max(1, int(keep * front_ratio))
    back_budget = max(1, keep - front_budget)

    if enc is not None:
        try:
            ids = enc.encode(text)
            # 보정계수를 역산하지 않고 raw 토큰 기준으로 자르되, 보수적으로 keep 사용.
            front_ids = ids[:front_budget]
            back_ids = ids[-back_budget:] if back_budget < len(ids) else []
            return (enc.decode(front_ids) + omit_notice + enc.decode(back_ids)), True
        except Exception:
            pass

    # fallback: 글자 비례 (토큰≈글자/4 가정 역산).
    approx_front_chars = front_budget * 4
    approx_back_chars = back_budget * 4
    if approx_front_chars + approx_back_chars >= len(text):
        return text[: approx_front_chars + approx_back_chars], True
    return (text[:approx_front_chars] + omit_notice + text[-approx_back_chars:]), True


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
