"""LLM 호출의 시간 상한 — **한 곳에서 정한다.**

무엇이 있었나 (2026-09-23 안정성 감사 F2)
-----------------------------------------
어느 클라이언트도 SDK 에 ``timeout`` 을 넘기지 않았다. 그래서 SDK 기본값이 그대로
적용됐다 — anthropic·openai 모두 **읽기 600초, 재시도 2회**. 그 위에 파이프라인의
API 스테이지가 **자기 재시도(최대 3회)** 를 또 돌렸고, 타임아웃도 "복구 가능" 으로
분류돼 거기서도 다시 시도했다. 곱하면:

    스테이지 4번 × SDK 3번 × 600초  ≈  **LLM 호출 하나에 최악 2시간**

폐쇄망에서 LLM 은 사내 게이트웨이·vLLM 을 거친다. 그것이 연결은 받는데 응답을 흘리지
않으면(과부하, 버퍼링 프록시) 턴 하나가 그 시간 동안 실행 스레드를 붙잡는다. 스테이지의
``timeout_ms`` 설정은 받는 클라이언트가 하나도 없어 **무효**였다.

어떻게 바꿨나
-------------
* **재시도는 한 층만.** SDK 재시도를 0 으로 두고 스테이지가 유일하게 재시도한다.
* **소켓 연결은 짧게.** 연결이 안 되는 것을 오래 기다릴 이유가 없다.
* **스트림은 두 가지를 본다** — 첫 내용이 올 때까지(추론형 모델은 생각이 길다)와,
  그 뒤 내용 사이의 무응답. 둘 다 스테이지가 직접 감시한다. SDK 의 읽기 타임아웃은
  그보다 넉넉히 둬서 감시보다 먼저 끊지 않게 한다.
* **타임아웃 재시도는 1회.** 멎어 있는 게이트웨이에 같은 요청을 네 번 보내 봐야
  네 번 기다릴 뿐이다.

값은 환경변수로 바꿀 수 있고 **호출 시점에** 읽는다(재시작 없이 운영 조정).
"""

from __future__ import annotations

import os
from typing import Any, Dict


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        value = float(raw)
    except ValueError:
        return float(default)
    return value if value > 0 else float(default)


def connect_timeout_s() -> float:
    """TCP 연결(이름 해석 포함)까지. 안 붙는 것을 오래 기다릴 이유가 없다."""
    return _env_float("XGEN_LLM_CONNECT_TIMEOUT_S", 10.0)


def first_chunk_timeout_s() -> float:
    """요청을 보내고 **첫 내용**(글자·생각·도구 호출)이 오기까지.

    추론형 모델은 첫 글자 전에 오래 생각할 수 있다 — 그래서 무응답 상한보다 길다.
    """
    return _env_float("XGEN_LLM_FIRST_CHUNK_TIMEOUT_S", 180.0)


def idle_timeout_s() -> float:
    """첫 내용 이후, 내용과 내용 사이의 무응답 상한."""
    return _env_float("XGEN_LLM_IDLE_TIMEOUT_S", 120.0)


def request_timeout_s() -> float:
    """스트리밍하지 않는 호출 한 번의 전체 상한(요약·압축 같은 보조 호출)."""
    return _env_float("XGEN_LLM_REQUEST_TIMEOUT_S", 600.0)


def timeout_retries() -> int:
    """타임아웃으로 끝난 호출을 몇 번 더 시도하는가. 기본 1."""
    raw = os.getenv("XGEN_LLM_TIMEOUT_RETRIES")
    if raw is None or not str(raw).strip():
        return 1
    try:
        return max(0, int(raw))
    except ValueError:
        return 1


#: SDK 자체 재시도. 0 — 재시도는 API 스테이지 한 층만 한다(곱해지지 않게).
SDK_MAX_RETRIES = 0


def sdk_read_timeout_s() -> float:
    """SDK 소켓 읽기 상한 — 스테이지 감시보다 **넉넉히 길게.**

    짧으면 스테이지가 판정하기 전에 SDK 가 먼저 끊어, 무엇 때문에 끊겼는지(첫 응답
    대기인지 중간 무응답인지)를 스테이지가 말하지 못한다.
    """
    return max(first_chunk_timeout_s(), idle_timeout_s(), request_timeout_s()) + 30.0


def httpx_timeout() -> Any:
    """anthropic·openai SDK 가 받는 ``httpx.Timeout``."""
    import httpx

    connect = connect_timeout_s()
    return httpx.Timeout(sdk_read_timeout_s(), connect=connect, pool=connect)


def sdk_client_kwargs() -> Dict[str, Any]:
    """anthropic·openai 계열 SDK 클라이언트 생성자에 그대로 붙이는 인자."""
    return {"timeout": httpx_timeout(), "max_retries": SDK_MAX_RETRIES}


def genai_timeout_ms() -> int:
    """google-genai ``HttpOptions.timeout`` (밀리초)."""
    return int(sdk_read_timeout_s() * 1000)


__all__ = [
    "SDK_MAX_RETRIES",
    "connect_timeout_s",
    "first_chunk_timeout_s",
    "genai_timeout_ms",
    "httpx_timeout",
    "idle_timeout_s",
    "request_timeout_s",
    "sdk_client_kwargs",
    "sdk_read_timeout_s",
    "timeout_retries",
]
