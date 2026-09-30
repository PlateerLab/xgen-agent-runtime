"""Gemini 는 functionResponse 를 **함수 이름**으로 가리킨다 — 정규 tool_result 에는 id 뿐이다.

에이전트 루프도, 호스트도 tool_result 를 ``{type, tool_use_id, content}`` 로 만든다. 번역기가 이름을
``tool_result.name`` 에서만 읽어서, Gemini 로 가는 도구 결과는 늘 빈 이름이었다(4.74.1 에서 수정 —
앞선 tool_use 의 같은 id 에서 이름을 찾는다).
"""

from xgen_agent_runtime.llm_client.translators import canonical_messages_to_google


def test_the_result_is_named_after_its_call():
    contents = canonical_messages_to_google([
        {"role": "user", "content": "서울 날씨"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "Seoul"}},
            {"type": "tool_use", "id": "call_2", "name": "get_time", "input": {}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call_2", "content": "09:00"},
            {"type": "tool_result", "tool_use_id": "call_1", "content": "맑음"},
        ]},
    ])
    responses = [p["functionResponse"] for p in contents[-1]["parts"]]
    assert [(r["name"], r["id"], r["response"]["result"]) for r in responses] == [
        ("get_time", "call_2", "09:00"), ("get_weather", "call_1", "맑음")]


def test_an_explicit_name_still_wins():
    contents = canonical_messages_to_google([
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "name": "given", "content": "ok"}]},
    ])
    assert contents[0]["parts"][0]["functionResponse"]["name"] == "given"
