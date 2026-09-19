"""완료 직전 산출물 대조 — 모델이 "만들었다" 고 한 파일을 실제로 읽어 보여 준다.

배경 (Harness-Bench 68과제, 2026-09-19): 잃은 점수의 23% 가 "요구 조건을 다시 대조하지
않고 완료 선언" 이었다 — 정확히 3행이어야 하는 CSV 에 5행, 헤더 오타, 없는 파일을 있다고
보고. 공개 논문에서도 1위 실패 증상(계약·형식 위반 36%). 도구·복구 실패는 0건이었으니
남은 건 이 마지막 대조다.

방식: 모델이 완료하려는 순간(마커 또는 도구 호출 없는 응답), 이 턴에서 **모델이
주장한 파일**(Write/Edit 로 쓴 경로 + 마지막 답변에 언급한 경로)을 실제로 읽어 짧은
요약(존재 여부·행 수·헤더·JSON 유효성)을 한 번 보여 주고 "요청의 명시 조건과
대조하라" 고만 한다. 어떤 조건이 맞는지는 모델이 요청문을 보고 판단한다 — 하네스는
도메인 규칙을 모른다. 턴당 한 번. 주장한 파일이 없으면 아무것도 하지 않는다.

비용: 파일을 만든 턴에 모델 왕복 1회. 판정은 벤치(설계용 → 홀드아웃)로.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import posixpath
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Tuple

from xgen_agent_runtime.core.state import PipelineState

logger = logging.getLogger(__name__)

__all__ = [
    "CompletionReviewer",
    "DeliverableReviewer",
    "REVIEW_KEY",
    "build_digest",
    "claimed_paths",
    "describe_file",
]

#: ``state.shared`` 키 — 턴 단위(연속 슬라이스에 이어지고 새 턴에서 비운다).
REVIEW_KEY = "loop.completion_review"

#: 요약에 넣을 파일 수 상한과 파일당 읽는 바이트 상한.
MAX_FILES = 12
MAX_READ_BYTES = 2_000_000

#: 답변 본문에서 경로로 볼 토큰 — 확장자가 있는 상대/절대 경로. 문장 부호·따옴표·괄호
#: 로 둘러싸인 것을 벗긴다. URL 은 제외.
_PATH_RE = re.compile(
    r"(?<![\w/:@.])"  # 앞이 단어·경로 문자가 아닐 것 (URL 의 :// 배제)
    r"((?:\.{0,2}/)?(?:[\w.\-]+/)*[\w.\-]+\.[A-Za-z][A-Za-z0-9]{0,7})"  # 확장자는 글자로 시작 (v1.2.3 배제)
    r"(?![\w/])"
)
_PATH_STOP_EXT = {"com", "org", "net", "io", "kr", "co", "ai", "dev", "md5"}  # 도메인 오탐
_WRITE_TOOLS = {"Write", "Edit", "NotebookEdit", "MultiEdit"}


class CompletionReviewer(Protocol):
    """완료 직전에 한 번 끼어드는 검토자. 돌려준 문자열은 모델에게 보내고 한 바퀴 더 돈다."""

    name: str

    async def review(self, state: PipelineState) -> Optional[str]: ...


# ── 주장한 경로 수집 ───────────────────────────────────────────────────


def _turn_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """마지막 사용자 발화(도구 결과가 아닌 것)부터 끝까지 — 이 턴의 메시지."""
    start = 0
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        ):
            continue
        if isinstance(content, str) and content.startswith(_REVIEW_HEADER):
            continue
        start = i
        break
    return messages[start:]


def _tool_use_blocks(message: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    content = message.get("content")
    if not isinstance(content, list):
        return ()
    return (b for b in content if isinstance(b, dict) and b.get("type") == "tool_use")


def _paths_from_tool_input(name: str, tool_input: Any) -> List[str]:
    if not isinstance(tool_input, dict):
        return []
    if name in _WRITE_TOOLS:
        p = tool_input.get("file_path") or tool_input.get("path")
        return [str(p)] if p else []
    if name == "ToolBatch" and tool_input.get("tool") in _WRITE_TOOLS:
        out: List[str] = []
        for item in tool_input.get("inputs") or []:
            if isinstance(item, dict):
                p = item.get("file_path") or item.get("path")
                if p:
                    out.append(str(p))
        return out
    return []


def _paths_from_text(text: str) -> List[str]:
    out: List[str] = []
    for m in _PATH_RE.finditer(text or ""):
        cand = m.group(1)
        ext = cand.rsplit(".", 1)[-1].lower()
        if ext in _PATH_STOP_EXT:
            continue
        # 버전 번호(1.2.3)·소수·파일명 없는 확장자만인 토큰 제외
        if re.fullmatch(r"[\d.]+", cand):
            continue
        out.append(cand)
    return out


def claimed_paths(messages: List[Dict[str, Any]], final_text: str) -> List[str]:
    """이 턴에서 모델이 만들었다고 볼 수 있는 경로 — 쓴 것 + 마지막 답변에 언급한 것.

    쓴 경로가 먼저(확실), 언급 경로가 뒤(주장). 순서 유지·중복 제거.
    """
    seen: Dict[str, None] = {}
    for m in _turn_messages(messages):
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        for b in _tool_use_blocks(m):
            for p in _paths_from_tool_input(str(b.get("name") or ""), b.get("input")):
                seen.setdefault(p, None)
    for p in _paths_from_text(final_text):
        seen.setdefault(p, None)
    return list(seen)


# ── 파일 요약 ──────────────────────────────────────────────────────────


def _is_binary(data: bytes) -> bool:
    return b"\x00" in data[:8192]


def _decode(data: bytes) -> Optional[str]:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return data.decode("latin-1")
        except Exception:  # noqa: BLE001
            return None


def _clip(s: str, n: int) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _describe_csv(text: str, delimiter: str) -> str:
    rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return "empty"
    header = rows[0]
    widths = {len(r) for r in rows[1:]}
    ragged = bool(widths - {len(header)})
    return (
        f'header "{_clip(delimiter.join(header), 160)}" ({len(header)} cols), '
        f"{len(rows) - 1} data rows" + (", ragged rows" if ragged else "")
    )


def _describe_json(text: str) -> str:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        return f"INVALID JSON ({exc.msg} at line {exc.lineno})"
    if isinstance(obj, dict):
        keys = list(obj.keys())
        shown = ", ".join(str(k) for k in keys[:12]) + (", …" if len(keys) > 12 else "")
        return f"valid JSON object, {len(keys)} keys: {shown}"
    if isinstance(obj, list):
        kinds = {type(x).__name__ for x in obj[:50]}
        return f"valid JSON array of {len(obj)} ({'/'.join(sorted(kinds)) or 'empty'})"
    return f"valid JSON {type(obj).__name__}"


def _describe_jsonl(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    bad = 0
    first_bad = None
    for i, ln in enumerate(lines, 1):
        try:
            json.loads(ln)
        except json.JSONDecodeError:
            bad += 1
            first_bad = first_bad or i
    if bad:
        return f"{len(lines)} lines, {bad} INVALID JSON (first at line {first_bad})"
    return f"{len(lines)} JSON lines, all valid"


def describe_file(path: str, data: Optional[bytes]) -> str:
    """파일 하나의 한 줄 요약. ``data`` 가 None 이면 없는 파일."""
    if data is None:
        return "MISSING"
    if len(data) == 0:
        return "EMPTY (0 bytes)"
    if _is_binary(data):
        return f"binary, {len(data):,} bytes"
    text = _decode(data)
    if text is None:
        return f"undecodable, {len(data):,} bytes"
    ext = posixpath.splitext(path)[1].lower()
    n_lines = text.count("\n") + (0 if text.endswith("\n") else 1)
    if ext in (".csv", ".tsv"):
        return f"{n_lines} lines: " + _describe_csv(text, "\t" if ext == ".tsv" else ",")
    if ext == ".json":
        return _describe_json(text)
    if ext in (".jsonl", ".ndjson"):
        return _describe_jsonl(text)
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    return f'{n_lines} lines, {len(data):,} bytes, starts "{_clip(first, 100)}"'


# ── 메시지 ────────────────────────────────────────────────────────────

_REVIEW_HEADER = "[Deliverable check — automatic, before finishing]"

_REVIEW_TAIL = (
    "Compare each file against the explicit requirements in the request "
    "(file names and locations, format, exact headers or fields, row or item "
    "counts, ordering, required and forbidden content). If anything does not "
    "match, fix it now and then finish. If everything matches, finish — do not "
    "redo work that is already correct."
)


def build_digest(entries: List[Tuple[str, str]]) -> str:
    lines = [
        _REVIEW_HEADER,
        "These are the files this turn claims to have produced, as they exist right now:",
    ]
    lines += [f"- {path} — {desc}" for path, desc in entries]
    lines.append(_REVIEW_TAIL)
    return "\n".join(lines)


# ── 검토자 ────────────────────────────────────────────────────────────


class DeliverableReviewer:
    """주장한 산출물을 읽어 요약하고, 요청 조건과 대조하라고 한 번 돌려보낸다.

    ``context_getter`` 는 살아 있는 Stage-10 ``ToolContext`` 를 돌려준다 (세션이
    나중에 붙어도 보이도록 매번 읽는다). ``sandbox`` 가 있으면 그 안에서, 없으면
    ``working_dir``/``allowed_paths`` 기준 로컬 파일을 읽는다 — Read 도구와 같은 경로.
    """

    name = "deliverable"

    def __init__(
        self,
        context_getter: Callable[[], Any],
        *,
        max_files: int = MAX_FILES,
        max_read_bytes: int = MAX_READ_BYTES,
    ) -> None:
        self._ctx = context_getter
        self._max_files = int(max_files)
        self._max_read_bytes = int(max_read_bytes)

    async def _read(self, ctx: Any, path: str) -> Optional[bytes]:
        """없으면 None. 허용 밖 경로·읽기 오류는 None 이 아니라 예외 → 호출자가 건너뜀."""
        sandbox = getattr(ctx, "sandbox", None)
        wd = str(getattr(ctx, "working_dir", "") or "")
        if sandbox is not None:
            from xgen_agent_runtime.tools._xgeny_sandbox import sb_read_bytes

            try:
                data = await sb_read_bytes(sandbox, path, workdir=wd or "/workspace")
            except FileNotFoundError:
                return None
            return data[: self._max_read_bytes]
        from xgen_agent_runtime.tools.built_in._path_guard import resolve_and_validate

        resolved = resolve_and_validate(path, wd or ".", getattr(ctx, "allowed_paths", None))
        if not resolved.is_file():
            return None
        with open(resolved, "rb") as fh:
            return fh.read(self._max_read_bytes)

    async def review(self, state: PipelineState) -> Optional[str]:
        marker = state.shared.get(REVIEW_KEY)
        if isinstance(marker, dict) and marker.get("done"):
            return None
        ctx = self._ctx()
        if ctx is None:
            return None
        paths = claimed_paths(state.messages, state.final_text or "")[: self._max_files]
        if not paths:
            return None

        entries: List[Tuple[str, str]] = []
        missing = 0
        for p in paths:
            try:
                data = await self._read(ctx, p)
            except Exception as exc:  # noqa: BLE001 — 허용 밖·권한·세션 오류는 요약에서 뺀다
                logger.debug("deliverable review: skip %r (%s)", p, exc)
                continue
            desc = describe_file(p, data)
            missing += data is None
            entries.append((p, desc))
        if not entries:
            return None

        state.shared[REVIEW_KEY] = {"done": True, "files": len(entries), "missing": missing}
        state.add_event(
            "loop.completion_review",
            {
                "reviewer": self.name,
                "files": len(entries),
                "missing": missing,
                "paths": [p for p, _ in entries],
            },
        )
        return build_digest(entries)
