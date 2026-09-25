"""작업 공간에서 못 찾은 파일 — **기계가 둘인 대화**에서는 "없다" 가 아니라 "여기엔 없다" 다.

커넥터(데스크톱) 대화에는 기계가 둘이다: 에이전트의 sandbox 와 사용자 PC. 사용자가 "작업 폴더의 X" 라고
하면 대개 PC 쪽 파일인데, 모델은 눈앞의 sandbox 도구(Read·Glob·Bash)로 먼저 찾고, 못 찾으면 **PC 는 보지
않은 채** "파일이 없습니다" 라고 답한다.

실측
----
* 2026-09-26 dev (gpt-4.1, runtime 4.61.0, trace 46792): "작업 폴더의 dex_note_0926.md 첫 줄이 뭐야?" →
  sandbox ``Read`` 한 번 "File not found" → PC 확인 없이 "파일이 존재하지 않습니다". 파일은 PC 작업 폴더에
  있었다. 시스템 프롬프트의 "sandbox 에 없다고 없는 게 아니다" 문단은 이미 들어가 있었다 — 부탁은 안 통했다.
* 2026-09-24 dev (gpt-4.1, trace 46746): 같은 모양으로 sandbox 에서 ``rm -rf`` 한 뒤 "삭제했다" (없는 경로라
  조용히 성공).
* 로컬 실험실(qwen3.8-27b): "작업 폴더의 X" 요청의 첫 행동이 sandbox 였던 비율이 높았다(23%, 설계 시나리오).

규칙
----
* 이번 턴 도구 목록에 사용자 PC 의 문(``…LocalControl``)이 있을 때만 — 즉 PC 가 실제로 연결된 대화에서만.
* sandbox 파일 도구의 결과가 "없음" 이면(``File not found`` · ``No files matching`` · ``No such file or
  directory``) 결과 끝에 한 줄: 여기(sandbox)엔 없다, 없다고 답하기 전에 PC 도 확인하라 — 문 이름을 댄다.
* 턴당 ``MAX_NOTES`` 번까지 — 코딩 중의 정상적인 "없음" 이 매번 안내로 불어나지 않게.

오류를 고치는 법까지 말해 주는 도구 결과가 첫 시도 성공을 올린다는 근거(SWE-agent ACI, Anthropic
"Writing tools for agents")와 같은 방향이다. 도메인 규칙 없음 — 도구 이름·결과 모양만 본다.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from xgen_agent_runtime.tools.gates import split_prefix

#: sandbox(에이전트 작업 공간)의 파일·셸 도구 — 사용자 PC 도구(``mcp_local_*``)는 대상이 아니다.
SANDBOX_FILE_TOOLS = frozenset({"Read", "Edit", "Write", "Glob", "NotebookEdit", "Bash"})
MAX_NOTES = 2
NOTES_KEY = "tool.second_machine_notes"

_NOT_FOUND = re.compile(r"File not found|No files matching|No such file or directory", re.I)

NOTE = (
    "[Not in your sandbox] That path is not in YOUR sandbox. This conversation is also connected to "
    "the user's own computer, where the files they talk about usually are. Before saying it does not "
    "exist, look there: call {gate}, then ListDir / ReadFile / Shell on the user's computer."
)


def local_gate(names: Iterable[str]) -> Optional[str]:
    """이번 턴에 사용자 PC 로 가는 문이 있으면 그 이름(접두 포함), 없으면 None."""
    for name in names:
        prefix, base = split_prefix(str(name))
        if prefix and base == "LocalControl":
            return str(name)
    return None


def _text(result: Dict[str, Any]) -> str:
    content = result.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and isinstance(b.get("text"), str)
        )
    return ""


def annotate(
    tool_calls: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    registry_names: Iterable[str],
    state_shared: Dict[str, Any],
) -> int:
    """못 찾은 sandbox 결과에 'PC 도 확인' 안내를 붙인다. 붙인 수를 돌려준다(``results`` 제자리 수정)."""
    gate = local_gate(registry_names)
    if not gate:
        return 0
    used = int(state_shared.get(NOTES_KEY, 0))
    calls = {str(tc.get("tool_use_id") or ""): tc for tc in tool_calls}
    added = 0
    for result in results:
        if used >= MAX_NOTES:
            break
        tc = calls.get(str(result.get("tool_use_id") or "")) or {}
        if str(tc.get("tool_name") or "") not in SANDBOX_FILE_TOOLS:
            continue
        text = _text(result)
        if not text or not _NOT_FOUND.search(text) or "[Not in your sandbox]" in text:
            continue
        if isinstance(result.get("content"), str):
            result["content"] = f"{result['content']}\n\n{NOTE.format(gate=gate)}"
            used += 1
            added += 1
    state_shared[NOTES_KEY] = used
    return added
