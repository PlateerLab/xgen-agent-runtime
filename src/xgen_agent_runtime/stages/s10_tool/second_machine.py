"""기계가 둘인 대화 — sandbox 도구의 "없음"·"접근 불가" 는 "여기엔 없음" 이다.

사용자가 기기(데스크톱·휴대폰·웹 브라우저)의 폴더를 대화에 붙이면 기계가 둘이 된다: 에이전트의 sandbox 와
사용자의 기기. 사용자가 "작업 폴더의 X" 라고 하면 대개 기기 쪽 파일인데, 모델은 눈앞의 sandbox 도구
(Read·Glob·Bash)로 먼저 찾고, 못 찾으면 **기기는 보지 않은 채** "파일이 없습니다" 라고 답한다. 거꾸로
기기 폴더의 경로를 sandbox 도구에 그대로 넘기기도 한다.

실측
----
* 2026-09-26 dev (gpt-4.1, runtime 4.61.0, trace 46792): "작업 폴더의 dex_note_0926.md 첫 줄이 뭐야?" →
  sandbox ``Read`` 한 번 "File not found" → 기기 확인 없이 "파일이 존재하지 않습니다". 파일은 PC 작업 폴더에
  있었다. 시스템 프롬프트의 "sandbox 에 없다고 없는 게 아니다" 문단은 이미 들어가 있었다 — 부탁은 안 통했다.
* 2026-09-24 dev (gpt-4.1, trace 46746): 같은 모양으로 sandbox 에서 ``rm -rf`` 한 뒤 "삭제했다" (없는 경로라
  조용히 성공).
* 2026-09-30 감사: 이 안내는 옛 입구(``…LocalControl``)가 있을 때만 붙었다. 지금 앱은 폴더를 붙이고 그 입구를
  보내지 않으므로, 기계가 둘인 대화 **전부**에서 안내가 한 번도 붙지 않았다.

규칙
----
* 이번 턴에 기기 폴더 도구(``mcp_<기기>_ReadFile`` 등)가 있거나, 옛 앱의 입구(``…LocalControl``)가 있을 때만.
* sandbox 도구가 받은 경로가 **연결된 기기 폴더 안**이면 → 그 경로는 기기 것이다, 기기 도구를 쓰라(항상).
* sandbox 도구의 결과가 "없음"(``File not found`` · ``No files matching`` · ``No such file or directory``)
  이면 → 여기(sandbox)엔 없다, 없다고 답하기 전에 기기도 보라. 턴당 ``MAX_NOTES`` 번까지 — 코딩 중의
  정상적인 "없음" 이 매번 안내로 불어나지 않게.

오류를 고치는 법까지 말해 주는 도구 결과가 첫 시도 성공을 올린다는 근거(SWE-agent ACI, Anthropic
"Writing tools for agents")와 같은 방향이다. 도메인 규칙 없음 — 도구 이름·경로·결과 모양만 본다.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from xgen_agent_runtime.tools.gates import split_prefix

#: sandbox(에이전트 작업 공간)의 파일·셸 도구 — 사용자 기기 도구(``mcp_local_*`` 등)는 대상이 아니다.
SANDBOX_FILE_TOOLS = frozenset({"Read", "Edit", "Write", "Glob", "Grep", "NotebookEdit", "Bash"})
MAX_NOTES = 2
NOTES_KEY = "tool.second_machine_notes"

_NOT_FOUND = re.compile(
    r"File not found|Directory not found|Path not found|No files matching|No such file or directory"
    r"|outside allowed directories",
    re.I,
)

#: sandbox 도구 입력에서 경로를 담는 키.
_PATH_KEYS = ("file_path", "path", "notebook_path")

#: 옛 앱(폴더를 보내지 않는 커넥터)의 안내 — 입구 도구를 부르라고 한다.
NOTE = (
    "[Not in your sandbox] That path is not in YOUR sandbox. This conversation is also connected to "
    "the user's own computer, where the files they talk about usually are. Before saying it does not "
    "exist, look there: call {gate}, then ListDir / ReadFile / Shell on the user's computer."
)

#: 폴더가 연결된 대화의 "없음" 안내.
FOLDER_NOTE = (
    "[Not in your sandbox] That path is not in YOUR sandbox. This conversation also has folders "
    "connected on the user's {device}: {folders}. If the user means their own files, look there with "
    "{tools} before saying it does not exist."
)

#: 사용자 기기 접속(host.user_pc) 경로 — sandbox 도구가 연결 폴더의 경로를 받았을 때. 경로가 그 기기
#: 폴더의 것이라는 확실한 사실만 말한다("없음" 결과에 기기를 보라고 미는 안내는 이 경로에 없다).
USER_PC_WRONG_MACHINE_NOTE = (
    '[Wrong machine] {path} is inside the connected folder "{name}" on the user\'s device{device}. '
    'Your sandbox tools cannot reach it; UserPc with folder="{name}" can.'
)

#: sandbox 도구가 기기 폴더의 경로를 받았을 때.
WRONG_MACHINE_NOTE = (
    '[Wrong machine] {path} is inside the folder "{name}" on the user\'s {device}. Your sandbox tools '
    "cannot reach it. Use {tools} for that path."
)


def local_gate(names: Iterable[str]) -> Optional[str]:
    """이번 턴에 옛 앱의 사용자 PC 입구가 있으면 그 이름(접두 포함), 없으면 None."""
    for name in names:
        prefix, base = split_prefix(str(name))
        if prefix and base == "LocalControl":
            return str(name)
    return None


def device_file_tools(names: Iterable[str]) -> Optional[str]:
    """이번 턴의 기기 폴더 도구를 사람이 읽는 목록으로 (예: ``mcp_local_ListDir / …``). 없으면 None."""
    from xgen_agent_runtime.host.local_folders import device_tool, is_folder_tool

    found: Dict[str, List[str]] = {}
    for name in names:
        text = str(name)
        if not is_folder_tool(text):
            continue
        dev = device_tool(text)
        if dev is None:
            continue
        found.setdefault(dev[0], []).append(text)
    if not found:
        return None
    order = ("ListDir", "ReadFile", "SearchFiles", "Shell", "CopyToWorkspace")
    picked: List[str] = []
    for tools in found.values():
        by_base = {split_prefix(t)[1]: t for t in tools}
        picked.extend(by_base[b] for b in order if b in by_base)
    return " / ".join(picked) if picked else None


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


def _append(result: Dict[str, Any], note: str) -> bool:
    content = result.get("content")
    if isinstance(content, str):
        result["content"] = f"{content}\n\n{note}"
        return True
    if isinstance(content, list):
        content.append({"type": "text", "text": note})
        return True
    return False


def _norm(path: str) -> str:
    return str(path or "").replace("\\", "/").rstrip("/")


def _inside(path: str, folder: str) -> bool:
    p, f = _norm(path), _norm(folder)
    return bool(f) and (p == f or p.startswith(f + "/"))


def _device_hit(
    tool_name: str, tool_input: Any, facts: Dict[str, Any]
) -> Optional[Tuple[str, str]]:
    """sandbox 도구 입력이 연결된 기기 폴더를 가리키면 (그 경로, 폴더 이름)."""
    folders = [f for f in (facts.get("folders") or []) if isinstance(f, dict) and f.get("path")]
    if not folders or not isinstance(tool_input, dict):
        return None
    candidates: List[str] = []
    for key in _PATH_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
    for path in candidates:
        for f in folders:
            if _inside(path, str(f["path"])):
                return path, str(f.get("name") or f["path"])
    if tool_name == "Bash":
        command = str(tool_input.get("command") or "")
        for f in folders:
            root = _norm(str(f["path"]))
            if root and len(root) > 1 and root in command.replace("\\", "/"):
                return root, str(f.get("name") or f["path"])
    return None


def annotate(
    tool_calls: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    registry_names: Iterable[str],
    state_shared: Dict[str, Any],
) -> int:
    """sandbox 도구 결과에 '기기 쪽을 보라' 안내를 붙인다. 붙인 수를 돌려준다(``results`` 제자리 수정)."""
    from xgen_agent_runtime.host.local_folders import SHARED_FOLDERS_KEY

    names = list(registry_names)
    facts_any = state_shared.get(SHARED_FOLDERS_KEY)
    if isinstance(facts_any, dict) and facts_any.get("user_pc"):
        return _annotate_user_pc(tool_calls, results, names, facts_any)
    # 옛 앱(폴더 목록을 보내지 않는 커넥터)은 입구가 있고 폴더 도구는 그 뒤에 숨어 있다 — 입구를 가리킨다.
    gate = local_gate(names)
    device_tools = None if gate else device_file_tools(names)
    if not device_tools and not gate:
        return 0
    facts = state_shared.get(SHARED_FOLDERS_KEY)
    facts = facts if isinstance(facts, dict) else {}
    used = int(state_shared.get(NOTES_KEY, 0))
    calls = {str(tc.get("tool_use_id") or ""): tc for tc in tool_calls}
    added = 0
    for result in results:
        tc = calls.get(str(result.get("tool_use_id") or "")) or {}
        tool_name = str(tc.get("tool_name") or "")
        if tool_name not in SANDBOX_FILE_TOOLS:
            continue
        text = _text(result)
        if "[Not in your sandbox]" in text or "[Wrong machine]" in text:
            continue
        if device_tools:
            hit = _device_hit(tool_name, tc.get("tool_input"), facts)
            if hit is not None:
                # 경로가 기기 것이라는 건 확실한 사실이다 — 횟수 제한 없이 매번 말한다.
                path, folder = hit
                note = WRONG_MACHINE_NOTE.format(
                    path=path,
                    name=folder,
                    device=facts.get("device") or "device",
                    tools=device_tools,
                )
                if _append(result, note):
                    added += 1
                continue
        if used >= MAX_NOTES or not text or not _NOT_FOUND.search(text):
            continue
        if device_tools:
            folder_list = (
                ", ".join(
                    f"{f.get('name')} ({f.get('path')})"
                    for f in (facts.get("folders") or [])
                    if isinstance(f, dict)
                )
                or "the connected folders"
            )
            note = FOLDER_NOTE.format(
                device=facts.get("device") or "device", folders=folder_list, tools=device_tools
            )
        else:
            note = NOTE.format(gate=gate)
        if _append(result, note):
            used += 1
            added += 1
    state_shared[NOTES_KEY] = used
    return added


def _annotate_user_pc(
    tool_calls: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    names: List[str],
    facts: Dict[str, Any],
) -> int:
    """사용자 기기 접속 경로 — sandbox 도구가 연결 폴더의 경로를 받았을 때만 사실 한 줄을 붙인다."""
    from xgen_agent_runtime.host.user_pc import TOOL_NAME

    if TOOL_NAME not in names:
        return 0
    calls = {str(tc.get("tool_use_id") or ""): tc for tc in tool_calls}
    added = 0
    for result in results:
        tc = calls.get(str(result.get("tool_use_id") or "")) or {}
        tool_name = str(tc.get("tool_name") or "")
        if tool_name not in SANDBOX_FILE_TOOLS:
            continue
        if "[Wrong machine]" in _text(result):
            continue
        hit = _device_hit(tool_name, tc.get("tool_input"), facts)
        if hit is None:
            continue
        path, name = hit
        device = next(
            (
                str(f.get("device") or "")
                for f in facts.get("folders") or []
                if isinstance(f, dict) and str(f.get("name") or "") == name
            ),
            "",
        )
        note = USER_PC_WRONG_MACHINE_NOTE.format(
            path=path, name=name, device=f' "{device}"' if device else ""
        )
        if _append(result, note):
            added += 1
    return added
