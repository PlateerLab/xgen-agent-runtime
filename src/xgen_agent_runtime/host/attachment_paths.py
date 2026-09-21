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

import logging
import posixpath
from typing import Any, Dict, List, Sequence

#: 절대 경로가 실린 자리. 렌더러·프롬프트는 이 값이 있으면 이것만 말한다.
ABS_KEY = "path"
#: 워크스페이스 상대 경로 — 출처 기록(기억·첨부 원장)은 계속 이 값을 쓴다.
REL_KEY = "workspace_path"

logger = logging.getLogger(__name__)


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
        existing = str(item.get(ABS_KEY) or "")
        if existing.startswith("/"):
            # 호스트가 이미 절대 경로를 실었다(예: 대화 스코프 첨부 — 그 파일은 에이전트
            # 작업 폴더가 아니라 그 대화만의 트리에 있다). 여기서 다시 계산하면 엉뚱한
            # 자리를 가리킨다 — 기준을 아는 쪽이 이미 붙였으면 그것이 맞다.
            out.append(item)
            continue
        absolute = session_path(root, item.get(REL_KEY) or existing)
        if not absolute:
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
    "hydrate_sandbox_images",
    "session_path",
]


async def hydrate_sandbox_images(
    attachments: Sequence[Any], sandbox: Any, *, max_bytes: int, turn_max_bytes: int
) -> List[Any]:
    """세션에만 있는 이미지 첨부의 바이트를 읽어 프로바이더가 쓸 수 있게 만든다.

    배포된 고정본의 첨부는 **그 대화만의 샌드박스**에 있고 서빙 파드에는 사본이 없다.
    그래서 호스트는 경로만 싣고 바이트는 여기서 읽는다 — 턴이 열려 세션을 들고 있는
    지금이 유일하게 읽을 수 있는 자리다. 파일 첨부는 읽지 않는다(에이전트가 자기 파일
    도구로 연다). 읽기 실패는 그 첨부 하나만 포기한다 — 턴은 계속된다.
    """
    import os
    import tempfile

    from xgen_agent_runtime.tools._xgeny_sandbox import sb_read_bytes

    if sandbox is None:
        return list(attachments or [])
    out: List[Any] = []
    total = 0
    for item in attachments or []:
        if not isinstance(item, dict):
            out.append(item)
            continue
        kind = str(item.get("kind") or item.get("type") or "").lower()
        path = str(item.get(ABS_KEY) or "")
        needs_bytes = (
            kind in ("image", "img", "picture")
            and path.startswith("/")
            and not (item.get("data") or item.get("base64") or item.get("url"))
            and not item.get("local_path")
        )
        if not needs_bytes:
            out.append(item)
            continue
        try:
            data = await sb_read_bytes(sandbox, path)
        except Exception:  # noqa: BLE001 — 첨부 하나가 턴을 깨지 않는다
            logger.warning("세션 이미지 첨부를 읽지 못했습니다: %s", path, exc_info=True)
            out.append(item)
            continue
        if len(data) > max_bytes or total + len(data) > turn_max_bytes:
            logger.warning(
                "세션 이미지 첨부가 예산을 넘어 건너뜁니다: %s (%d bytes)", path, len(data)
            )
            out.append(item)
            continue
        total += len(data)
        fd, local = tempfile.mkstemp(prefix="xgen-chat-attach-", suffix=_suffix_of(path))
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
        except OSError:
            logger.warning("세션 이미지 첨부를 내려 두지 못했습니다: %s", path, exc_info=True)
            out.append(item)
            continue
        out.append({**item, "local_path": local, "size": len(data)})
    return out


def _suffix_of(path: str) -> str:
    tail = str(path or "").rsplit("/", 1)[-1]
    return f".{tail.rsplit('.', 1)[-1]}" if "." in tail else ""
