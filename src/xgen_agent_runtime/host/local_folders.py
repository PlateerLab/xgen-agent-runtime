"""대화에 연결된 **사용자 기기의 폴더** — 기기 도구의 표면과 안내를 정하는 한 곳.

사용자는 채팅 헤더의 [폴더] 로 이 대화에 폴더를 붙인다(데스크톱·모바일 앱, 그리고 웹
브라우저). 폴더는 **대화에 붙는다**: 연결된 폴더가 있는 대화에서만 폴더 도구(파일·셸·
열기…)가 보이고, 해제하면 다음 턴부터 사라진다.

기기는 자기 도구를 MCP 서버 이름 하나로 올린다 — 데스크톱은 ``local``(모델에게
``mcp_local_<Tool>``), 모바일은 ``mobile``(``mcp_mobile_<Tool>``), 웹 브라우저는 ``web``
(``mcp_web_<Tool>``). 폴더에 묶이는 도구는 기기마다 다르다(:data:`FOLDER_TOOLS_BY_SERVER`):
데스크톱은 PC 조작 전부가 폴더 연결로 열리고, 모바일은 파일을 다루는 도구만 폴더에 묶인다
(알림·위치 같은 휴대폰 기능은 모바일 설정의 도구 그룹이 따로 정한다). 웹 브라우저는 사용자가
고른 폴더의 파일 도구만 있다 — 터미널이 없고, 그 XGEN 화면이 열려 있는 동안만 닿는다.

여기서 정하는 것은 둘이다.

  표면   이번 턴 모델에게 어떤 기기 도구를 보여 주는가 (:func:`filter_device_tools`).
         서버 SDK 경로(turn_executor)와 CLI 브리지(workflow)가 같은 함수를 쓴다 — 두
         경로가 따로 판단하면 한쪽만 틀린다.
  안내   이번 턴의 연결 상태를 모델에게 말하는 한 블록 (:func:`turn_note`). 매 턴 최신
         사용자 메시지 옆에 붙고(volatile) 기록에 남지 않는다 — 옛 상태가 쌓이지 않는다.

**판정의 정본은 기기다.** 앱은 호출마다 그 대화(interaction_id)에 연결된 폴더인지 다시
확인하고 밖이면 거부한다. 여기는 무엇을 보여 주고 무엇을 말할지만 정한다.

클라이언트 세대 — ``local_folders`` 요청 필드:

  None   옛 앱·웹(필드를 모른다). 예전 규칙 그대로(앱의 전역 스위치가 표면을 정한다).
  []     새 앱, 이 대화에 연결된 폴더 없음. 폴더 도구를 뺀다.
  [..]   연결됨. 폴더 도구를 첫 화면에 바로 보여 준다(옛 LocalControl 입구는 뺀다).

혼동 방지: 폴더를 붙였다 뗐다 하면 지난 턴의 도구 호출이 기록에 남는다. 이번 턴에 폴더
도구가 없으면 그 호출·결과를 요청 사본에서 **평문 한 줄**로 바꾼다
(:data:`SharedKeys.RETIRED_TOOL_CALLS` → Stage 6) — 모델이 없는 도구를 다시 부르거나 옛
경로를 지금도 있는 것으로 믿지 않게.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

#: 앱별 폴더 도구. 연결된 폴더가 있어야 보인다. 같은 서버의 다른 도구(데스크톱 브라우저·
#: 로컬 MCP 관리, 모바일 알림·위치…)는 여기 들지 않는다 — 각자 자기 설정이 정한다.
FOLDER_TOOLS_BY_SERVER: Dict[str, frozenset] = {
    # 데스크톱: 설정의 [로컬 컨트롤] 탭이 없어지고 폴더 연결이 PC 조작의 유일한 문이다.
    "local": frozenset(
        {
            "ReadFile",
            "WriteFile",
            "ListDir",
            "Search",
            "Shell",
            "ShellJob",
            "Open",
            "Clipboard",
            "Notify",
        }
    ),
    # 모바일: 파일을 다루는 도구만. 터미널은 없다.
    "mobile": frozenset(
        {
            "ReadFile",
            "WriteFile",
            "ListDir",
            "DeleteFile",
            "Search",
            "OpenFile",
            "TakePhoto",
        }
    ),
    # 웹 브라우저: 사용자가 고른 폴더(File System Access)의 파일 도구만. 터미널은 없다.
    "web": frozenset(
        {
            "ReadFile",
            "WriteFile",
            "ListDir",
            "DeleteFile",
            "Search",
        }
    ),
}

#: 터미널이 없는 기기 — 턴 안내가 파일 도구만 쓰라고 말한다.
_NO_TERMINAL_SERVERS = frozenset({"mobile", "web"})

#: 앱 도구 서버 이름들.
DEVICE_SERVERS = tuple(FOLDER_TOOLS_BY_SERVER)

#: 옛 입구 — 전역 스위치 시절 "이 PC 가 두 번째 기계" 임을 알리던 도구. 새 앱에서는
#: 폴더 목록과 턴 안내가 그 일을 한다.
LEGACY_GATES: Dict[str, str] = {"local": "LocalControl"}

#: CLI 브리지가 붙이는 서버 접두(``mcp__connector__mcp_local_Shell``).
_BRIDGE_PREFIX = re.compile(r"^mcp__[A-Za-z0-9-]+__")
_DEVICE_NAME = re.compile(r"^mcp_(" + "|".join(re.escape(s) for s in DEVICE_SERVERS) + r")_(.+)$")

#: 기록 속 옛 폴더 도구 호출을 평문으로 바꿀 때 붙는 이유 한 마디.
RETIRE_REASON = "device folder no longer connected"


@dataclass(frozen=True)
class LocalFolder:
    """대화에 연결된 폴더 하나.

    ``path`` 는 **모델이 쓰는 경로**다. 데스크톱은 실제 절대 경로, 모바일은 앱이 정한
    가상 루트(``/Documents``) — 실제 위치(content:// 트리 URI, iOS 북마크)는 기기만 안다.
    """

    id: str
    name: str
    path: str


def parse_local_folders(raw: Any) -> Optional[List[LocalFolder]]:
    """요청의 ``local_folders`` → 폴더 목록. **None 은 None 으로 남긴다**(옛 클라이언트).

    잘못된 항목(경로 없음)은 버린다. 같은 경로가 두 번 오면 한 번만 둔다.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    if not isinstance(raw, (list, tuple)):
        return []
    out: List[LocalFolder] = []
    seen = set()
    for item in raw:
        if isinstance(item, LocalFolder):
            folder = item
        elif isinstance(item, dict):
            path = str(item.get("path") or "").strip()
            if not path:
                continue
            name = str(item.get("name") or "").strip() or _basename(path)
            folder = LocalFolder(id=str(item.get("id") or path), name=name, path=path)
        elif isinstance(item, str) and item.strip():
            path = item.strip()
            folder = LocalFolder(id=path, name=_basename(path), path=path)
        else:
            continue
        if folder.path in seen:
            continue
        seen.add(folder.path)
        out.append(folder)
    return out


def _basename(path: str) -> str:
    parts = [p for p in re.split(r"[\\/]", path) if p]
    return parts[-1] if parts else path


def device_tool(name: str) -> Optional[Tuple[str, str]]:
    """``mcp_local_Shell`` · ``mcp__connector__mcp_mobile_ReadFile`` → (서버, 도구). 아니면 None."""
    m = _DEVICE_NAME.match(_BRIDGE_PREFIX.sub("", str(name or "")))
    return (m.group(1), m.group(2)) if m else None


def is_legacy_gate(name: str) -> bool:
    found = device_tool(name)
    return bool(found) and LEGACY_GATES.get(found[0]) == found[1]


def is_folder_tool(name: str) -> bool:
    """폴더에 묶이는 기기 도구인가 (옛 입구 제외)."""
    found = device_tool(name)
    return bool(found) and found[1] in FOLDER_TOOLS_BY_SERVER.get(found[0], ())


def _tool_name(tool: Any) -> str:
    if isinstance(tool, dict):
        return str(tool.get("name") or "")
    return str(getattr(tool, "name", "") or "")


def filter_device_tools(
    tools: Sequence[Any],
    folders: Optional[Sequence[LocalFolder]],
    *,
    name_of: Callable[[Any], str] = _tool_name,
) -> List[Any]:
    """이번 턴에 보여 줄 기기 도구만 남긴다.

    - ``folders is None`` (옛 클라이언트): 손대지 않는다.
    - 폴더 없음: 폴더 도구와 옛 입구를 뺀다. 브라우저 등 다른 기기 도구는 남는다.
    - 폴더 있음: 옛 입구만 뺀다 — 폴더 목록이 곧 지도다.
    """
    items = list(tools or [])
    if folders is None:
        return items
    kept: List[Any] = []
    for tool in items:
        name = name_of(tool)
        if is_legacy_gate(name):
            continue
        if not folders and is_folder_tool(name):
            continue
        kept.append(tool)
    return kept


def folder_tool_is_turn_one(name: str, folders: Optional[Sequence[LocalFolder]]) -> bool:
    """폴더가 연결된 대화에서 폴더 도구는 첫 화면에 스키마까지 나간다.

    사용자가 폴더를 붙인 것 자체가 "이 대화에서 내 파일을 다뤄라" 는 뜻이라, 검색으로
    찾게 할 이유가 없다.
    """
    return bool(folders) and is_folder_tool(name)


def retired_device_tool_names() -> List[str]:
    """이번 턴에 폴더 도구가 없을 때 기록에서 평문으로 바꿀 도구 이름(서버 접두 없는 모양)."""
    names = {
        f"mcp_{server}_{tool}" for server, tools in FOLDER_TOOLS_BY_SERVER.items() for tool in tools
    }
    names |= {f"mcp_{server}_{gate}" for server, gate in LEGACY_GATES.items()}
    return sorted(names)


def retired_calls_spec() -> Dict[str, Any]:
    """``SharedKeys.RETIRED_TOOL_CALLS`` 값 — Stage 6 이 요청 사본에 적용한다."""
    return {"names": retired_device_tool_names(), "reason": RETIRE_REASON}


def retire_device_tool_calls(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """기록 속 폴더 도구 호출·결과를 평문 한 줄로 바꾼 사본 (core.message_repair 위임)."""
    from xgen_agent_runtime.core.message_repair import retire_tool_calls_by_name

    return retire_tool_calls_by_name(messages, retired_calls_spec())


# ── 턴 안내 ────────────────────────────────────────────────────────


_PLATFORM_LABEL = {
    "darwin": "Mac",
    "mac": "Mac",
    "macos": "Mac",
    "win32": "Windows PC",
    "windows": "Windows PC",
    "linux": "Linux PC",
    "android": "Android phone",
    "ios": "iPhone",
    "web": "web browser",
}


def device_label(platform: Optional[str]) -> str:
    key = str(platform or "").strip().lower()
    return _PLATFORM_LABEL.get(key, "device")


def turn_note(
    folders: Optional[Sequence[LocalFolder]],
    *,
    available_tools: Iterable[str] = (),
    platform: Optional[str] = None,
) -> str:
    """이번 턴의 폴더 연결 상태 — 매 턴 새로 쓰는 한 블록. 옛 클라이언트(None)면 빈 문자열.

    ``available_tools`` 는 이번 턴에 실제로 보이는 기기 도구 이름(접두 무관)이다.
    폴더는 연결돼 있는데 폴더 도구가 하나도 없으면 앱이 지금 서버에 닿지 않는 것이다.
    """
    if folders is None:
        return ""
    device = device_label(platform)
    found = [d for d in (device_tool(n) for n in available_tools) if d]
    folder_found = [(s, t) for s, t in found if t in FOLDER_TOOLS_BY_SERVER.get(s, ())]
    head = "# Folders on the user's device"
    if not folders:
        return (
            f"{head}\n"
            "No folder on the user's device is connected to this conversation, so there "
            "are no device file or shell tools this turn. Anything earlier turns read or "
            "wrote on the device is history only — it is not reachable now. If the task "
            "needs the user's own files, ask the user to connect a folder with the "
            "[폴더] button in the chat header."
        )
    lines = [head, f"This conversation is connected to these folders on the user's {device}:"]
    lines += [f"- {f.name}: {f.path}" for f in folders]
    if not folder_found:
        if str(platform or "").strip().lower() == "web":
            lines.append(
                "The browser is not reachable right now, so the device tools are unavailable "
                "this turn. Tell the user to keep this XGEN page open in the browser and allow "
                "access to the folder, then try again; do not guess file contents."
            )
        else:
            lines.append(
                "The app is not reachable right now, so the device tools are unavailable this "
                "turn. Tell the user to open the app (it must stay open and signed in) and try "
                "again; do not guess file contents."
            )
        return "\n".join(lines)
    prefixes = sorted({f"mcp_{s}_*" for s, _ in folder_found})
    lines.append(
        f"Device tools ({', '.join(prefixes)}) work only inside these folders. They run on "
        f"the user's {device}, not in your server sandbox: the sandbox cannot see these files "
        "and the device cannot see the sandbox. Use paths under the folders above."
    )
    if any(t == "Shell" for _, t in folder_found):
        lines.append(
            "Shell and ShellJob run terminal commands with the working directory inside a "
            "connected folder."
        )
    elif any(s in _NO_TERMINAL_SERVERS for s, _ in folder_found):
        lines.append("This device has no terminal; use the file tools.")
    lines.append(
        "Paths from earlier turns that are not under these folders are history only — they "
        "are no longer reachable."
    )
    return "\n".join(lines)


__all__ = [
    "DEVICE_SERVERS",
    "FOLDER_TOOLS_BY_SERVER",
    "LEGACY_GATES",
    "LocalFolder",
    "RETIRE_REASON",
    "device_label",
    "device_tool",
    "filter_device_tools",
    "folder_tool_is_turn_one",
    "is_folder_tool",
    "is_legacy_gate",
    "parse_local_folders",
    "retire_device_tool_calls",
    "retired_calls_spec",
    "retired_device_tool_names",
    "turn_note",
]
