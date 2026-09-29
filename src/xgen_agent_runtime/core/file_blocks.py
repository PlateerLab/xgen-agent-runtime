"""첨부(``file`` 블록)를 모델이 읽을 한 줄 포인터로.

첨부 바이트는 메시지에 싣지 않는다 — 파일은 에이전트 작업 폴더에 저장되고, 모델은 경로로 연다.
그래서 이 한 줄이 첨부의 전부다. 요청 번역(모든 백엔드)·단기 기억 창·대화 기록 노트가 같은
문구를 쓰도록 한 곳에 둔다.

**"이번 턴" 이라고 쓰지 않는다.** 같은 블록이 다음 턴과 새 대화에서도 다시 실린다. 예전 문구
``[Current-turn attachment: …]`` 가 지난 턴에 그대로 재생되자, 모델이 경로를 보고도 "지금 대화에서는
첨부 원문을 다시 열 수 없다" 고 답했다(2026-09-29 dev, gpt-5.4). 파일은 작업 폴더에 남아 있었다.
"""

from __future__ import annotations

from typing import Any, Mapping

#: 포인터 줄의 머리 — 사람의 말과 가를 때(대화 노트 제목) 쓴다.
POINTER_PREFIX = "[Attached file:"


def file_block_pointer(block: Mapping[str, Any]) -> str:
    """``file`` 블록 → ``[Attached file: 이름 (mime). … at: 경로]``.

    ``path`` 는 호스트가 작업 폴더를 알 때 채운 절대 경로다
    (:mod:`xgen_agent_runtime.host.attachment_paths`). 없으면 작업 폴더 기준 상대 경로를 쓰고
    기준을 밝힌다 — 기준 없는 경로는 모델이 엉뚱한 루트를 지어내게 했다(2026-09-21).
    """
    name = block.get("name") or "unnamed"
    mime = block.get("mime_type") or "application/octet-stream"
    absolute = block.get("path")
    if absolute and str(absolute).startswith("/"):
        return f"{POINTER_PREFIX} {name} ({mime}). Saved in your working folder — read it at: {absolute}]"
    relative = block.get("workspace_path") or absolute
    if relative:
        return (
            f"{POINTER_PREFIX} {name} ({mime}). Saved in your working folder — read it at: {relative} "
            "(relative to your working folder)]"
        )
    return f"{POINTER_PREFIX} {name} ({mime})]"


def file_pointers(content: Any) -> list[str]:
    """message content(str | 블록 목록)에 든 첨부마다 포인터 한 줄."""
    if not isinstance(content, list):
        return []
    return [
        file_block_pointer(b) for b in content if isinstance(b, Mapping) and b.get("type") == "file"
    ]


__all__ = ["POINTER_PREFIX", "file_block_pointer", "file_pointers"]
