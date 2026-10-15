"""텍스트 토큰 추정: 런타임의 모든 크기 판단이 쓰는 **한 함수**.

압축 시점(Stage 2·4·6 의 판단), 호출 전 입력 맞춤(host), 도구 결과 상한이 모두 이 추정을 쓴다.
예전에는 압축 판단만 ``글자 수 ÷ 4`` 를 써서 한국어를 2~3배 적게 셌고, 그래서 80% 압축과 호출
직전 확인이 제때 켜지지 않아 공급자가 먼저 "컨텍스트 초과" 로 거절했다(2026-10-12 조사).
같은 대화를 두 추정기가 다르게 세면 한쪽이 맞춘 크기를 다른 쪽이 넘쳤다고 본다. 그래서 하나다.

**모델도 데이터 파일도 네트워크도 쓰지 않는다.** 문자를 부류로 나누고 연속 구간 단위로 센다
(라틴은 여러 글자가 한 토큰, 한글은 음절마다, 숫자는 두세 자리씩). 계수는 실제 토크나이저와
대조해 맞췄다(o200k 평균 절대 오차 9.0%, cl100k 8.7%). ``host.token_estimator`` 와
``xgen_sdk.tokens`` 가 같은 함수다. 계수를 고치면 함께 고친다.

공급자 보정계수: 토크나이저가 OpenAI 계열과 다른 공급자는 같은 글이 더 많은 토큰이 된다.
추정이 모자라면 공급자 400 으로 턴이 죽고, 넘치면 조금 일찍 압축할 뿐이라 **과대추정** 쪽으로 맞춘다.
"""

from __future__ import annotations

import threading
import unicodedata
from collections import OrderedDict
from typing import Any

__all__ = [
    "estimate_tokens",
    "estimate_tokens_cached",
    "HANGUL_MODERN",
    "HANGUL_LEGACY",
    "PROVIDER_TOKEN_FACTOR",
    "DEFAULT_PROVIDER_FACTOR",
    "SAFETY_MULTIPLIER",
    "provider_token_factor",
    "CONSERVATIVE_CHARS_PER_TOKEN",
    "chars_for_tokens",
]

#: 한글 음절당 토큰: 최신 토크나이저(o200k 계열). 기본값.
HANGUL_MODERN = 0.75
#: 한글 음절당 토큰: 구형(cl100k 계열).
HANGUL_LEGACY = 1.28

#: 라틴 문자 연속 구간에서 토큰 하나가 먹는 글자 수(최소 1토큰).
_LATIN_CHARS_PER_TOKEN = 7.0
#: 숫자 연속 구간: BPE 는 보통 두세 자리씩 묶는다.
_DIGIT_CHARS_PER_TOKEN = 3.0
#: 한자·가나는 글자당 한 토큰에 가깝다.
_CJK_PER_CHAR = 1.0
#: 공백은 **첫 칸이 뒤 단어에 붙으므로** 나머지만 센다.
_SPACE_PER_EXTRA = 0.25
#: 줄바꿈은 대체로 한 토큰.
_NEWLINE_PER_CHAR = 1.0
#: 문장부호·기호(ASCII)는 앞뒤와 자주 뭉친다.
_PUNCT_PER_CHAR = 0.6
#: 이모지 등 비-ASCII 기호는 여러 바이트라 토큰을 더 먹는다.
_SYMBOL_PER_CHAR = 2.0

_SPACE = "sp"
_NEWLINE = "nl"
_DIGIT = "num"
_LATIN = "lat"
_HANGUL = "han"
_CJK = "cjk"
_PUNCT = "pun"
_SYMBOL = "sym"


def _class_of(ch: str) -> str:
    """문자 하나의 부류. 여기서만 유니코드를 본다."""
    code = ord(ch)
    if ch in " \t":
        return _SPACE
    if ch in "\r\n":
        return _NEWLINE
    if "0" <= ch <= "9":
        return _DIGIT
    if ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
        return _LATIN
    # 한글 음절 + 자모
    if 0xAC00 <= code <= 0xD7A3 or 0x1100 <= code <= 0x11FF or 0x3130 <= code <= 0x318F:
        return _HANGUL
    # 한자
    if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
        return _CJK
    # 가나
    if 0x3040 <= code <= 0x30FF:
        return _CJK
    if code < 0x80:
        return _PUNCT
    # 키릴·그리스·아랍 등 다른 문자 체계는 라틴과 비슷하게 뭉친다.
    if unicodedata.category(ch).startswith("L"):
        return _LATIN
    return _SYMBOL


def estimate_tokens(text: Any, *, hangul: float = HANGUL_MODERN) -> int:
    """``text`` 의 토큰 수 추정(올림). 빈 값이면 0.

    문자열이 아니면 ``str()`` 로 바꿔 센다. 호출부마다 변환을 되풀이하지 않게.

    Args:
        hangul: 한글 음절당 토큰. 구형 계열이면 :data:`HANGUL_LEGACY`.
    """
    if text is None:
        return 0
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return 0

    total = 0.0
    index = 0
    length = len(text)
    while index < length:
        kind = _class_of(text[index])
        end = index + 1
        while end < length and _class_of(text[end]) == kind:
            end += 1
        run = end - index

        if kind == _LATIN:
            total += max(1.0, run / _LATIN_CHARS_PER_TOKEN)
        elif kind == _DIGIT:
            total += max(1.0, run / _DIGIT_CHARS_PER_TOKEN)
        elif kind == _HANGUL:
            total += run * hangul
        elif kind == _CJK:
            total += run * _CJK_PER_CHAR
        elif kind == _SPACE:
            total += (run - 1) * _SPACE_PER_EXTRA
        elif kind == _NEWLINE:
            total += run * _NEWLINE_PER_CHAR
        elif kind == _PUNCT:
            total += run * _PUNCT_PER_CHAR
        else:
            total += run * _SYMBOL_PER_CHAR

        index = end

    return max(1, int(total + 0.5))


# ─────────────────────────────────────────────────────────────────────────
# 공급자 보정: 과대추정 쪽(모자라면 400, 넘치면 조금 일찍 압축할 뿐)
# ─────────────────────────────────────────────────────────────────────────

#: 공급자별 보정계수. o200k 계열(OpenAI)이 1.0 이다.
PROVIDER_TOKEN_FACTOR = {
    "openai": 1.0,
    "azure": 1.0,
    "azure_foundry": 1.0,
    "anthropic": 1.15,
    "bedrock": 1.15,
    "claude_code": 1.15,
    "google": 1.20,
    "vertex": 1.20,
    "vllm": 1.20,
}
#: 모르는 공급자.
DEFAULT_PROVIDER_FACTOR = 1.20
#: 모든 추정에 더하는 여유.
SAFETY_MULTIPLIER = 1.05


def provider_token_factor(provider: Any) -> float:
    """``provider`` 의 보정계수(여유 포함 전)."""
    return PROVIDER_TOKEN_FACTOR.get(str(provider or "").lower().strip(), DEFAULT_PROVIDER_FACTOR)


#: 긴 글의 추정 캐시. 같은 글은 같은 값이다. (해시, 길이)로 찾는다. 파이썬 문자열은 해시를 객체에
#: 저장하므로 두 번째부터는 상수 시간이다. 대화가 커지면 매 반복 같은 메시지를 다시 세던 비용이 사라진다.
#: 최근에 쓴 것부터 남긴다(LRU). 들어온 순서로 비우면 긴 대화를 처음부터 훑을 때마다 상한을 넘는 순간
#: 모든 조회가 빗나갔다(2026-10-12 검토: 긴 글 5,000개에서 한 번 훑는 데 1.9초).
_CACHE_MIN_CHARS = 512
_CACHE_MAX_ENTRIES = 16_384
_cache: "OrderedDict[tuple, int]" = OrderedDict()
_cache_lock = threading.Lock()


def estimate_tokens_cached(text: Any) -> int:
    """:func:`estimate_tokens` 와 같은 값. 긴 글은 캐시에서 찾는다."""
    if text is None:
        return 0
    if not isinstance(text, str):
        text = str(text)
    if len(text) < _CACHE_MIN_CHARS:
        return estimate_tokens(text) if text else 0
    key = (hash(text), len(text))
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit
    value = estimate_tokens(text)
    with _cache_lock:
        _cache[key] = value
        while len(_cache) > _CACHE_MAX_ENTRIES:
            _cache.popitem(last=False)
    return value


#: 글자 수 → 토큰 상한 환산에 쓰는 글자/토큰 비. 한국어는 음절당 0.75 토큰 × 보정
#: 1.2 ≈ 글자당 0.95 토큰이라 글자 하나를 토큰 하나로 본다. "이 토큰 예산이면 몇 글자까지 넣어도
#: 되나" 를 물을 때 쓴다(영어에는 넉넉하게 작은 값이다).
CONSERVATIVE_CHARS_PER_TOKEN = 1.0


def chars_for_tokens(tokens: int) -> int:
    """``tokens`` 토큰 예산에 안전하게 들어가는 글자 수(어느 언어든)."""
    return max(0, int(int(tokens) * CONSERVATIVE_CHARS_PER_TOKEN))
