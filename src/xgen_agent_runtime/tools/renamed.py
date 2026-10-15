"""이름을 바꾼 도구 — 옛 이름으로 불러도 지금 이름의 도구가 돈다.

도구 이름을 바꿔도 모델이 옛 이름을 부를 길은 남는다: 직전 턴을 되풀이하는 기록, 에이전트
기억에 적힌 요령, 사람이 붙여 넣은 예시. 그때 "Unknown tool" 로 한 번 헛돌게 하지 않고 지금
이름으로 보낸다. **표면에는 새 이름만 있다** — 옛 이름을 도구 목록에 다시 올리지 않는다.

MCP 접두(``mcp__connector__``·``mcp_local_``)는 그대로 둔다: 같은 서버의 새 이름이 있을 때만
보낸다.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from xgen_agent_runtime.tools.gates import split_prefix

#: 옛 이름 → 지금 이름.
RENAMED_TOOLS: Dict[str, str] = {
    # 2026-09-28 아티팩트 → 앱.
    "ArtifactGuide": "AppGuide",
    "ArtifactCreate": "AppCreate",
    "ArtifactPublish": "AppPublish",
    "ArtifactStatus": "AppStatus",
    "ArtifactList": "AppList",
    "ArtifactDelete": "AppDelete",
}


def current_name(name: str, exists: Callable[[str], bool]) -> Optional[str]:
    """옛 이름이면 지금 이름(접두 포함), 아니거나 지금 이름의 도구가 없으면 None."""
    prefix, base = split_prefix(name)
    new = RENAMED_TOOLS.get(base)
    if not new:
        return None
    full = prefix + new
    return full if exists(full) else None
