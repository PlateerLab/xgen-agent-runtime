"""span 수명 — 켜짐 판정, tracer, 현재 span 으로 열고 닫기.

무엇을 계측하는지(GenAI 이름·속성)는 모른다 — 그건 ``genai`` 의 몫이다. 여기서는 "span 하나를
안전하게 열고 닫는다" 만 책임진다: OTel 이 없거나 꺼졌거나 고장 나도 실행은 그대로 간다.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator, Mapping

from xgen_agent_runtime.observability.attributes import ERROR_TYPE

logger = logging.getLogger(__name__)

try:  # pragma: no cover - 설치 여부에 따라 한쪽만 탄다
    from opentelemetry import trace as _otel_trace
    from opentelemetry.trace import SpanKind
    from opentelemetry.trace import Status as _Status
    from opentelemetry.trace import StatusCode as _StatusCode
except ImportError:  # pragma: no cover
    _otel_trace = None  # type: ignore[assignment]
    SpanKind = None  # type: ignore[assignment,misc]
    _Status = None  # type: ignore[assignment]
    _StatusCode = None  # type: ignore[assignment]

#: 끄는 스위치. 호스트가 같은 일을 자기 방식으로 계측하고 있으면 0 으로 둔다.
ENV_SWITCH = "XGEN_AGENT_RUNTIME_OTEL"

TRACER_NAME = "xgen_agent_runtime"

_OFF_VALUES = frozenset({"0", "false", "no", "off"})


def spans_enabled() -> bool:
    """이 프로세스에서 런타임이 span 을 내는가.

    호스트는 이 값으로 자기 쪽 같은 span(사건으로 만든 도구 span, 턴 span 의 토큰)을 끈다.
    둘 다 내면 같은 호출이 trace 에 두 번 나온다.
    """
    if _otel_trace is None:
        return False
    return os.environ.get(ENV_SWITCH, "1").strip().lower() not in _OFF_VALUES


def get_tracer() -> Any:
    """전역 provider 의 tracer. 테스트가 이 함수만 바꿔 끼운다."""
    return _otel_trace.get_tracer(TRACER_NAME)


def client_kind() -> Any:
    return SpanKind.CLIENT if SpanKind is not None else None


def internal_kind() -> Any:
    return SpanKind.INTERNAL if SpanKind is not None else None


class SpanHandle:
    """열린 span 하나. 비활성이면 ``span`` 이 None 이라 모든 메서드가 no-op 이다."""

    __slots__ = ("span",)

    def __init__(self, span: Any = None) -> None:
        self.span = span

    @property
    def recording(self) -> bool:
        return self.span is not None

    def set(self, attributes: Mapping[str, Any]) -> None:
        """None·빈 문자열은 건너뛴다 — "모름" 을 값으로 남기지 않는다."""
        if self.span is None:
            return
        try:
            for key, value in attributes.items():
                if value is None or value == "":
                    continue
                self.span.set_attribute(key, value)
        except Exception:  # noqa: BLE001 — 관측이 실행을 깨선 안 된다
            logger.debug("span attribute 기록 실패(무시)", exc_info=True)

    def fail(self, error_type: str) -> None:
        """오류로 표시한다. 분류 코드만 — 자유 텍스트는 PII 경로다."""
        if self.span is None:
            return
        try:
            self.span.set_attribute(ERROR_TYPE, error_type or "error")
            self.span.set_status(_Status(_StatusCode.ERROR, error_type or "error"))
        except Exception:  # noqa: BLE001
            logger.debug("span 오류 표시 실패(무시)", exc_info=True)


@contextmanager
def current_span(name: str, kind: Any, attributes: Mapping[str, Any]) -> Iterator[SpanHandle]:
    """``name`` span 을 현재 span 으로 열고 블록이 끝나면 닫는다.

    블록 안의 예외는 그대로 올라가고 span 에는 오류로 남는다. 취소(``CancelledError``)와 제너레이터
    종료는 오류가 아니다 — 사용자가 멈춘 턴이 빨갛게 보이면 안 된다.
    """
    if not spans_enabled():
        yield SpanHandle()
        return
    try:
        cm = get_tracer().start_as_current_span(
            name, kind=kind, record_exception=False, set_status_on_exception=False
        )
        raw = cm.__enter__()
    except Exception:  # noqa: BLE001 — span 을 못 열어도 실행은 간다
        logger.debug("span 시작 실패(무시): %s", name, exc_info=True)
        yield SpanHandle()
        return
    handle = SpanHandle(raw)
    handle.set(attributes)
    exc_info: tuple = (None, None, None)
    try:
        yield handle
    except BaseException as exc:
        exc_info = (type(exc), exc, exc.__traceback__)
        if isinstance(exc, Exception):
            handle.fail(type(exc).__name__)
        raise
    finally:
        try:
            cm.__exit__(*exc_info)
        except Exception:  # noqa: BLE001
            logger.debug("span 종료 실패(무시): %s", name, exc_info=True)


__all__ = [
    "ENV_SWITCH",
    "SpanHandle",
    "client_kind",
    "current_span",
    "get_tracer",
    "internal_kind",
    "spans_enabled",
]
