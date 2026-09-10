"""CLI 백엔드의 도구 결과는 **사람이 읽는 텍스트**여야 한다.

회귀 배경: tool_result 의 content 는 보통 블록 리스트로 온다.
그걸 json.dumps 해서 넘기는 바람에, 모델도 화면도 봉투를 읽었다 —

    [{"type": "text", "text": "(no output)"}]

짧은 결과일수록 봉투가 내용보다 크고, 에러 한 줄은 그 안에 파묻힌다.
CLI 백엔드로 도는 **모든** 턴의 **모든** 도구 결과가 이 모양이었다.
"""
from __future__ import annotations

from xgen_agent_runtime.host.runner import _stringify_content, _tool_end_event
from xgen_agent_runtime.llm_client.translators._cli import StreamJsonAccumulator


def test_text_blocks_are_unwrapped():
    assert _stringify_content([{"type": "text", "text": "(no output)"}]) == "(no output)"
    assert _stringify_content("plain") == "plain"
    assert _stringify_content(None) == ""
    assert _stringify_content({"type": "text", "text": "one"}) == "one"


def test_multiple_blocks_join_as_lines():
    out = _stringify_content(
        [{"type": "text", "text": "line1"}, {"type": "text", "text": "line2"}]
    )
    assert out == "line1\nline2"


def test_non_text_blocks_are_named_not_dropped():
    """조용히 버리면 '결과가 비었다' 로 보인다 — 종류라도 남긴다."""
    out = _stringify_content([{"type": "image", "source": {}}, {"type": "text", "text": "ok"}])
    assert out == "[image]\nok"


def test_the_whole_cli_path_yields_readable_text():
    """CLI 스트림 한 줄 → agent_event 까지, 봉투가 남지 않는다."""
    acc = StreamJsonAccumulator("claude-sonnet-4-6")
    acc.feed(
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "mcp__xgen__Bash", "input": {}}
                ],
            },
        }
    )
    events = acc.feed(
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "tool_use_id": "t1",
                        "type": "tool_result",
                        "content": [{"type": "text", "text": "python3: not found"}],
                        "is_error": True,
                    }
                ],
            },
        }
    )
    result = next(e for e in events if e["type"] == "tool_result")
    event = _tool_end_event(
        "Bash", _stringify_content(result["content"]), is_error=bool(result["is_error"])
    )
    assert event["type"] == "tool_error"
    assert event["error"] == "python3: not found"
