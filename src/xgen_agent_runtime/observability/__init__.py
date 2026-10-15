"""OpenTelemetry 계측 — 런타임이 **직접 하는 일**(LLM 호출 1건, 도구 실행 1건)을 span 으로 남긴다.

왜 런타임이 내는가
------------------
호스트(xgen-workflow)는 턴이 끝난 뒤 받는 사용량 합계와 도구 시작/끝 사건으로만 span 을
만들 수 있었다. 그 방식으로는 두 가지가 끝내 안 된다.

* **LLM 호출 1건의 구간.** 사건은 큐를 지나 늦게 닿아 실제 호출 구간이 아니다. 그래서 trace 에
  ``chat`` span 이 없고 ``POST /v1/chat/completions`` 같은 HTTP span 만 남아 병목을 읽을 수 없었다.
* **부모 관계.** 사건으로 연 span 은 현재 context 가 아니라서, 도구가 부르는 HTTP·MCP 호출이
  그 도구 아래로 가지 못하고 턴 밖에 흩어졌다.

호출과 실행이 일어나는 자리가 여기이므로 span 도 여기서 연다. span 은 현재 context 의 자식이 되고,
그 안의 HTTP 자동 계측 span 은 이 span 의 자식이 된다.

구성
----
* ``spans``      — span 수명(켜짐 판정·tracer·현재 span 으로 열고 닫기). GenAI 를 모른다.
* ``attributes`` — 속성 키. OpenTelemetry GenAI semantic conventions + ``xgen.*`` 확장.
* ``genai``      — ``chat {model}`` / ``execute_tool {tool}`` span 과 응답·결과의 속성 기록.

규약
----
* 토큰(``gen_ai.usage.*``)은 **호출 span 에만** 붙인다. 호스트는 :func:`spans_enabled` 가 참이면
  턴 합계를 ``gen_ai.usage.*`` 로 다시 쓰지 않는다 — 더할 때 두 번 세지 않게.
* 프롬프트·응답·도구 입출력 **본문은 남기지 않는다.** 길이·개수만.
* ``opentelemetry-api`` 는 선택 의존성이다. 없거나 ``XGEN_AGENT_RUNTIME_OTEL=0`` 이면 전 경로가
  no-op 이다. 호스트가 SDK 를 설정하지 않았으면 API 가 no-op span 을 준다.
* 관측이 실행을 깨선 안 된다 — span 조작 실패는 삼킨다.
"""

from xgen_agent_runtime.observability.genai import (
    llm_call_span,
    record_llm_response,
    record_tool_result,
    tool_span,
)
from xgen_agent_runtime.observability.spans import ENV_SWITCH, SpanHandle, spans_enabled

__all__ = [
    "ENV_SWITCH",
    "SpanHandle",
    "llm_call_span",
    "record_llm_response",
    "record_tool_result",
    "spans_enabled",
    "tool_span",
]
