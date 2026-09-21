"""첨부 경로를 **에이전트가 그대로 쓸 수 있는 절대 경로**로 바꾼다.

## 왜

첨부는 워크스페이스 상대 경로(``uploads/users_1/<대화>/<파일>``)로 들어온다. 그런데 파일
도구의 스키마는 절대 경로를 요구한다. 그러면 모델이 스스로 절대화해야 하고, 그때 기준을
틀리게 잡는다 — 2026-09-21 실측: 작업 폴더가 ``…/<wf>/workspace`` 인데 안내 문구가 그 값을
"workspace path" 라고 부르는 바람에, 모델이 그 단어를 폴더 이름과 겹쳐 읽고 한 조각을 건너뛴
``…/<wf>/uploads/…`` 로 읽으려다 샌드박스 가드에 막혔다. 파일은 제자리에 있었는데 열지 못했다.

그래서 **기준을 모델에게 맡기지 않는다.** 턴을 시작하는 쪽이 이미 작업 폴더를 알고 있으므로,
거기서 한 번 붙여서 내려보낸다. 상대 경로는 출처 기록용으로 그대로 남긴다(기억·로그의 계약).
"""

from __future__ import annotations

import posixpath
from typing import Any, Dict, List, Sequence

#: 절대 경로가 실린 자리. 렌더러·프롬프트는 이 값이 있으면 이것만 말한다.
ABS_KEY = "path"
#: 워크스페이스 상대 경로 — 출처 기록(기억·첨부 원장)은 계속 이 값을 쓴다.
REL_KEY = "workspace_path"


def session_path(base: str, relative: str) -> str:
    """작업 폴더 + 워크스페이스 상대 경로 → 절대 경로. 못 만들면 빈 문자열."""
    root = str(base or "").replace("\\", "/").strip()
    rel = str(relative or "").replace("\\", "/").strip()
    if not rel:
        return ""
    if posixpath.isabs(rel):
        return posixpath.normpath(rel)
    if not root or not posixpath.isabs(root):
        return ""
    if rel.startswith("./"):
        rel = rel[2:]
    return posixpath.normpath(posixpath.join(root, rel))


def absolutize_attachments(attachments: Sequence[Any], base: str) -> List[Any]:
    """첨부 서술자마다 ``path``(절대)를 채운다. 원본은 바꾸지 않는다.

    ``base`` 를 모르면(로컬 실행·샌드박스 미부착) 그대로 돌려준다 — 그때는 렌더러가 상대
    경로를 말하고, 도구는 작업 폴더 기준으로 푼다. 둘 다 같은 자리를 가리킨다.
    """
    root = str(base or "").replace("\\", "/").strip()
    out: List[Any] = []
    for item in attachments or []:
        if not isinstance(item, dict):
            out.append(item)
            continue
        absolute = session_path(root, item.get(REL_KEY) or item.get(ABS_KEY) or "")
        if not absolute or item.get(ABS_KEY) == absolute:
            out.append(item)
            continue
        out.append({**item, ABS_KEY: absolute})
    return out


def absolutize_input(raw: Any, base: str) -> Any:
    """``{"attachments": [...]}`` 모양의 턴 입력을 같은 규칙으로 바꾼다."""
    if not isinstance(raw, dict):
        return raw
    changed: Dict[str, Any] = dict(raw)
    touched = False
    for key in ("attachments", "images", "files"):
        items = raw.get(key)
        if isinstance(items, list) and items:
            changed[key] = absolutize_attachments(items, base)
            touched = True
    return changed if touched else raw


__all__ = [
    "ABS_KEY",
    "REL_KEY",
    "absolutize_attachments",
    "absolutize_input",
    "session_path",
]
