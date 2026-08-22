"""파일시스템/식별자 안전 id — 순수 헬퍼 (서버·커넥터 공용).

memory_vault(③, 서버 전용) 안에 있던 순수 조각을 여기로 빼서 conversation_archive
같은 ② 모듈이 서버 결합 없이 쓰게 한다. 관련: ``xgeny-shared-host-extraction``.
"""

from __future__ import annotations

import re

_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_.-]")


def _safe_id(value: str) -> str:
    """파일시스템/식별자 성분으로 안전한 id (경로 탈출 방지)."""
    cleaned = _SAFE_ID_RE.sub("_", str(value or "").strip())
    return cleaned.strip("._") or "unknown"
