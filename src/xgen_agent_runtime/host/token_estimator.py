"""토큰 추정 — **모델도 데이터 파일도 네트워크도 쓰지 않는다.**

⚠ 이 파일은 ``xgen_sdk.tokens`` 와 **같은 함수**다. 런타임은 의도적으로 SDK 에 기대지
않으므로(독립 패키지) 사본을 둔다 — 계수를 고치면 양쪽을 함께 고친다.

왜 자체 추정인가
----------------
예전에는 ``tiktoken`` 으로 셌다. 그런데 tiktoken 은 인코딩 표(BPE)를 **첫 호출 때
인터넷에서 내려받는다** — ``requests.get(url)`` 에 타임아웃조차 없다. 폐쇄망에서는 그
한 줄이 이벤트 루프 위에서 멎고, 그동안 그 파드의 모든 요청이 같이 선다(2026-09-22
실측: 캐시 디렉터리 미설정이라 파드마다·재시작마다 다시 시도한다).

토큰 수를 알아야 하는 이유는 두 가지뿐이다 — **사용량 집계**와 **컨텍스트 예산**. 둘 다
정확한 토큰화가 아니라 **좋은 추정**이면 충분하다. 실제 사용량은 모델이 보고하는 값이
정본이고(``usage``), 이 추정은 그 값이 없을 때만 쓴다.

어떻게 세는가
-------------
문자를 부류로 나누고 **연속 구간(run)** 단위로 센다. BPE 가 실제로 하는 일과 결이 같다:
라틴 문자는 여러 글자가 한 토큰으로 뭉치고, 한글은 음절마다 쪼개지고, 숫자는 두세 자리씩
묶이고, 공백은 뒤따르는 단어에 붙는다.

계수는 실제 tiktoken 과 **대조해 맞췄다**(한국어 산문·영어 산문·혼합·코드·JSON·로그·짧은
발화 20종). 평균 절대 오차:

    o200k_base  (GPT-4o/5, Gemini, 최신 계열)   9.0%
    cl100k_base (GPT-3.5/4 계열)                8.7%

두 인코딩의 차이는 **한글 가중치 하나뿐**이다 — 최신 토크나이저는 한글을 음절당 0.75
토큰으로, 구형은 1.28 토큰으로 쓴다. 그래서 기본값은 최신 계열이고, 구형이 필요하면
:data:`HANGUL_LEGACY` 를 넘긴다.

예전 fallback 이던 ``len(text)/4`` 는 한국어에서 평균 **+42%** 틀렸다. 이 추정은 그 자리를
대신한다.
"""

from __future__ import annotations

import unicodedata
from typing import Any

__all__ = [
    "estimate_tokens",
    "HANGUL_MODERN",
    "HANGUL_LEGACY",
]

#: 한글 음절당 토큰 — 최신 토크나이저(o200k 계열). 기본값.
HANGUL_MODERN = 0.75
#: 한글 음절당 토큰 — 구형(cl100k 계열).
HANGUL_LEGACY = 1.28

#: 라틴 문자 연속 구간에서 토큰 하나가 먹는 글자 수(최소 1토큰).
_LATIN_CHARS_PER_TOKEN = 7.0
#: 숫자 연속 구간 — BPE 는 보통 두세 자리씩 묶는다.
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

    문자열이 아니면 ``str()`` 로 바꿔 센다 — 호출부마다 변환을 되풀이하지 않게.

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
