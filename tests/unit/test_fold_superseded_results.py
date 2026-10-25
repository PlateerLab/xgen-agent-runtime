"""같은 것을 더 새로 보여 준 결과가 뒤에 있으면 앞 결과의 큰 필드를 요청 사본에서 접는다.

기기 브라우저 도구는 페이지를 옮길 때마다 화면 글(스냅샷)을 결과에 싣는다. 같은 탭·같은 주소를 다시 읽으면 앞
스냅샷은 낡은 사본이고 ref 도 이미 만료됐다 — 그대로 두면 매 요청에 수천~수만 자가 쌓인다.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Dict, List

from xgen_agent_runtime.core.message_repair import EXTERNAL_RESULT_LABEL, fold_superseded_results
from xgen_agent_runtime.host.device_tools import BROWSER_SNAPSHOT_FOLD_NOTE, browser_snapshot_fold_spec

NAV = "mcp_local_BrowserNavigate"
SNAP = "mcp_local_BrowserSnapshot"
SPEC = browser_snapshot_fold_spec([NAV, SNAP, "mcp_local_BrowserInteract", "mcp_local_BrowserGuide"])


def _call(call_id: str, name: str) -> Dict[str, Any]:
    return {"role": "assistant", "content": [{"type": "tool_use", "id": call_id, "name": name, "input": {}}]}


def _result(call_id: str, payload: Any, **extra: Any) -> Dict[str, Any]:
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call_id, "content": text, **extra}]}


def _page(snapshot: str, *, page_id: str = "p1", url: str = "https://bo.example.com/goods/reg", data: bool = False) -> Dict[str, Any]:
    result = {"data": {"snapshot": snapshot, "origin": url}} if data else {"navigated": "goto", "snapshot": snapshot}
    return {"workflow_id": "wf", "page_id": page_id, "url": url, "title": "상품 등록", "generation": 3, "result": result}


def _payload(messages: List[Dict[str, Any]], index: int) -> Dict[str, Any]:
    content = messages[index]["content"][0]["content"]
    return json.loads(content if isinstance(content, str) else content[0]["text"])


def test_spec_covers_the_device_browser_tools_but_not_the_guide() -> None:
    assert SPEC["names"] == ["mcp_local_BrowserInteract", NAV, SNAP]
    assert SPEC["key"] == ["page_id", "url"]
    assert browser_snapshot_fold_spec(["mcp_local_ReadFile", "mcp_github_BrowserX"]) is None


def test_an_older_snapshot_of_the_same_tab_and_address_is_folded() -> None:
    messages = [
        _call("a", NAV), _result("a", _page("- textbox \"상품명\" [ref=e1]\n" * 50)),
        _call("b", SNAP), _result("b", _page("- textbox \"상품명\" [ref=e7]", data=True)),
    ]
    before = deepcopy(messages)
    out = fold_superseded_results(messages, SPEC)
    assert messages == before  # 기록은 그대로
    older = _payload(out, 1)
    assert older["result"] == {"navigated": "goto", "snapshot": BROWSER_SNAPSHOT_FOLD_NOTE}
    assert older["page_id"] == "p1" and older["generation"] == 3  # 나머지는 그대로
    assert _payload(out, 3) == _page("- textbox \"상품명\" [ref=e7]", data=True)
    assert out[0] is messages[0] and out[3] is messages[3]


def test_another_address_or_tab_is_kept_because_only_it_may_hold_what_was_read() -> None:
    messages = [
        _call("a", NAV), _result("a", _page("목록: A-001, A-002", url="https://bo.example.com/goods/list")),
        _call("b", NAV), _result("b", _page("목록 2", page_id="p2")),
        _call("c", NAV), _result("c", _page("등록 폼")),
    ]
    assert fold_superseded_results(messages, SPEC) == messages


def test_only_the_newest_of_many_keeps_its_snapshot() -> None:
    messages = []
    for i in range(4):
        messages += [_call(f"c{i}", SNAP), _result(f"c{i}", _page(f"화면 {i}", data=True))]
    out = fold_superseded_results(messages, SPEC)
    snaps = [_payload(out, 2 * i + 1)["result"]["data"]["snapshot"] for i in range(4)]
    assert snaps == [BROWSER_SNAPSHOT_FOLD_NOTE] * 3 + ["화면 3"]
    assert fold_superseded_results(out, SPEC) == out  # 다시 접어도 같다


def test_results_without_a_snapshot_do_not_count_as_newer() -> None:
    # 페이지가 그대로인 클릭은 스냅샷을 싣지 않는다 — 앞 스냅샷의 ref 가 아직 산다.
    click = {"workflow_id": "wf", "page_id": "p1", "url": "https://bo.example.com/goods/reg", "generation": 3, "result": {"success": True}}
    messages = [_call("a", NAV), _result("a", _page("폼")), _call("b", "mcp_local_BrowserInteract"), _result("b", click)]
    assert fold_superseded_results(messages, SPEC) == messages


def test_errors_other_tools_and_text_that_is_not_json_are_left_alone() -> None:
    messages = [
        _call("a", NAV), _result("a", _page("폼"), is_error=True),
        _call("b", "mcp_local_ReadFile"), _result("b", _page("폼")),
        _call("c", NAV), _result("c", "browser_timeout: page did not load"),
        _call("d", NAV), _result("d", _page("폼")),
    ]
    assert fold_superseded_results(messages, SPEC) == messages


def test_bridge_prefixed_names_and_list_content_keep_their_shape() -> None:
    newer = _result("b", _page("새 화면"))
    older = _result("a", _page("옛 화면"))
    older["content"][0]["content"] = [{"type": "text", "text": older["content"][0]["content"]}]
    messages = [_call("a", f"mcp__connector__{NAV}"), older, _call("b", NAV), newer]
    out = fold_superseded_results(messages, SPEC)
    folded = out[1]["content"][0]["content"]
    assert isinstance(folded, list) and folded[0]["type"] == "text"
    assert json.loads(folded[0]["text"])["result"]["snapshot"] == BROWSER_SNAPSHOT_FOLD_NOTE


def test_a_malformed_spec_changes_nothing() -> None:
    messages = [_call("a", NAV), _result("a", _page("폼")), _call("b", NAV), _result("b", _page("폼"))]
    for spec in (None, "x", {}, {"names": [NAV]}, {"names": [NAV], "key": ["page_id"]}):
        assert fold_superseded_results(messages, spec) == messages


def test_stage_six_folds_the_request_copy_including_this_loops_results() -> None:
    from xgen_agent_runtime.core.shared_keys import SharedKeys
    from xgen_agent_runtime.core.state import PipelineState
    from xgen_agent_runtime.stages.s06_api import APIStage

    state = PipelineState()
    state.messages = [{"role": "user", "content": "상품 등록 화면 열어 줘"}, _call("a", NAV), _result("a", _page("옛 화면"))]
    state.shared[SharedKeys.FOLDED_TOOL_RESULTS] = SPEC
    before = deepcopy(state.messages)
    stage = APIStage()
    kwargs = stage._call_kwargs(
        stage.resolve_model_config(state), state, extra_messages=[_call("b", SNAP), _result("b", _page("새 화면", data=True))]
    )
    assert state.messages == before
    assert _payload(kwargs["messages"], 2)["result"]["snapshot"] == BROWSER_SNAPSHOT_FOLD_NOTE
    assert _payload(kwargs["messages"], 4)["result"]["data"]["snapshot"] == "새 화면"


def test_results_labeled_as_external_data_fold_and_keep_the_label() -> None:
    # Stage 10 이 기기 도구 결과 앞에 머리말을 붙인다(10/25 dev 실측: 이것 때문에 하나도 접히지 않았다).
    old, new = _page("- link \"메일\" [ref=e1]\n" * 50), _page("- link \"메일\" [ref=e9]", data=True)

    def labeled(payload: Dict[str, Any]) -> str:
        return EXTERNAL_RESULT_LABEL + json.dumps(payload, ensure_ascii=False)

    messages = [_call("a", NAV), _result("a", labeled(old)), _call("b", SNAP), _result("b", labeled(new))]
    out = fold_superseded_results(messages, SPEC)
    folded = out[1]["content"][0]["content"]
    assert folded.startswith(EXTERNAL_RESULT_LABEL)
    assert json.loads(folded[len(EXTERNAL_RESULT_LABEL) :])["result"]["snapshot"] == BROWSER_SNAPSHOT_FOLD_NOTE
    assert out[3] is messages[3]


def test_a_label_carried_as_its_own_text_block_stays_in_front() -> None:
    def labeled(call_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        content = [
            {"type": "text", "text": EXTERNAL_RESULT_LABEL},
            {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
        ]
        return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call_id, "content": content}]}

    old, new = _page("- textbox \"상품명\" [ref=e1]\n" * 50), _page("- textbox \"상품명\" [ref=e7]")
    out = fold_superseded_results([_call("a", NAV), labeled("a", old), _call("b", NAV), labeled("b", new)], SPEC)
    label, body = out[1]["content"][0]["content"]
    assert label == {"type": "text", "text": EXTERNAL_RESULT_LABEL}
    assert json.loads(body["text"])["result"]["snapshot"] == BROWSER_SNAPSHOT_FOLD_NOTE
