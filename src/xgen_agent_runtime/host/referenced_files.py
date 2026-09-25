"""요청에 이름이 나온 작업 폴더 파일을 **첫 턴에 붙여 준다** — 읽기 왕복을 없앤다.

왜
--
로컬 Harness-Bench 호출 해부(2026-09-24, qwen3.8-27b, 설계 과제 38개): 과제당 도구 호출 10.8회 중
**입력 읽기가 4.2회(39%)** 였다. 과제당 입력 파일 수(평균 5.1)와 거의 같다 — 헛읽기가 아니라 요청에
적힌 파일을 **하나씩 따로** 읽는 왕복이다(같은 대상 재읽기는 160회 중 1회). 프롬프트는 이미 "독립적인
조회는 한 응답에 묶어라" 를 말하지만 모델이 따르지 않는다(ToolBatch 38과제 중 1회) — 부탁 대신 구조로
없앤다. 사용자가 이름을 적은 파일을 대화에 붙이는 것은 널리 쓰는 방식이다(Claude Code ``@파일``,
Cursor·Aider 의 파일 추가).

규칙 (도메인 규칙 없음 — 경로 모양과 크기만 본다)
------------------------------------------------
* 요청 텍스트에서 경로처럼 생긴 토큰(절대 경로, ``a/b.ext``, ``이름.ext``)을 찾는다.
* 작업 폴더 경로 가드(``tool_fs().resolve``)를 통과하고 **실제로 있는 텍스트 파일**만 붙인다.
  없으면·폴더 밖이면 조용히 건너뛴다(추측한 이름일 수 있다).
* 비밀처럼 보이는 파일(.env·키·인증서 등)은 붙이지 않는다 — 요청이 이름만 적었을 수 있다.
* 파일당 ``PER_FILE`` 바이트, 합계 ``TOTAL`` 바이트까지. 넘치는 파일은 앞부분만 싣고 잘렸다고
  적는다. 바이너리(문서·이미지 등)는 이름·크기만 알려 주고 알맞은 도구로 열라고 한다.
* 끝까지 실은 파일은 "읽은 파일" 장부에 올린다 — 바로 Edit/Write 할 수 있다.
* 사용자 PC(커넥터 로컬 세션)는 건너뛴다 — 경로 규약이 다르고 매 파일이 원격 왕복이다.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

PER_FILE = 16_000
TOTAL = 48_000
MAX_FILES = 12

_TEXT_EXT = {
    "txt",
    "md",
    "markdown",
    "csv",
    "tsv",
    "json",
    "jsonl",
    "ndjson",
    "yaml",
    "yml",
    "xml",
    "html",
    "htm",
    "py",
    "js",
    "ts",
    "tsx",
    "jsx",
    "sql",
    "log",
    "ini",
    "toml",
    "cfg",
    "conf",
    "sh",
    "rst",
    "tex",
    "java",
    "go",
    "rs",
    "rb",
    "php",
    "c",
    "h",
    "cpp",
    "hpp",
    "cs",
    "kt",
    "swift",
    "css",
    "scss",
    "vue",
    "srt",
    "vtt",
    "eml",
    "ics",
    "properties",
    "gradle",
    "dockerfile",
    "mk",
    "graphql",
    "proto",
    "r",
    "lua",
    "pl",
}
_SECRETISH = re.compile(
    r"(^|/)(\.env(\..*)?|.*\.(pem|key|p12|pfx|jks|keystore|crt|cer)|id_(rsa|dsa|ecdsa|ed25519)(\.pub)?|"
    r".*credentials?.*|.*secrets?.*|\.netrc|\.npmrc|\.pypirc)$",
    re.I,
)
# 절대 경로 · 상대 경로(슬래시 포함) · 확장자가 붙은 파일명. 한글 등 유니코드 이름도 \w 로 잡는다.
_CANDIDATE = re.compile(
    r"(?<![\w$/.-])((?:/|\.{1,2}/)?(?:[\w.-]+/)*[\w.-]+\.[A-Za-z0-9]{1,10})(?![\w/])"
)
_ABS = re.compile(r"(?<![\w$])(/[^\s`'\"<>|*?(){}\[\],;]+)")
_TRAIL = ".,:;!?)\"'`*"
_URL = re.compile(r"[A-Za-z][\w+.-]*://\S+")


@dataclass
class Prefetch:
    block: str = ""
    witnessed: List[str] = field(default_factory=list)
    attached: List[str] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)


def candidates(text: str) -> List[str]:
    """요청 텍스트에서 경로 후보(등장 순서, 중복 제거)."""
    seen: List[str] = []
    text = _URL.sub(" ", text or "")  # URL 의 경로 부분을 파일로 오인하지 않는다
    for rx in (_ABS, _CANDIDATE):
        for m in rx.finditer(text):
            c = m.group(1).rstrip(_TRAIL)
            if c and c not in seen and not c.startswith(("http:", "https:")) and "://" not in c:
                seen.append(c)
    return seen


def _ext(path: str) -> str:
    base = os.path.basename(path).lower()
    return base.rsplit(".", 1)[-1] if "." in base else base


async def _head(fs: Any, resolved: str, cap: int) -> Optional[Tuple[bytes, int]]:
    """(앞부분 ≤cap+1 바이트, 전체 크기). 파일이 아니면 None."""
    kind = getattr(fs, "kind", "")
    if kind == "runner":
        sb = fs.sandbox
        if getattr(sb, "is_connector_local", False):
            return None
        size = await sb.exec(["stat", "-c", "%s", "--", resolved], timeout_s=10)
        if not size.ok or not size.stdout.strip().isdigit():
            return None
        head = await sb.exec(["head", "-c", str(cap + 1), "--", resolved], timeout_s=10)
        if not head.ok:
            return None
        return head.stdout, int(size.stdout.strip())

    def _local() -> Optional[Tuple[bytes, int]]:
        if not os.path.isfile(resolved):
            return None
        with open(resolved, "rb") as f:
            return f.read(cap + 1), os.path.getsize(resolved)

    return await asyncio.to_thread(_local)


async def collect(text: str, fs: Any, *, per_file: int = PER_FILE, total: int = TOTAL) -> Prefetch:
    out = Prefetch()
    budget = total
    parts: List[str] = []
    for cand in candidates(text)[: MAX_FILES * 3]:
        if len(out.attached) + len(out.skipped) >= MAX_FILES:
            break
        try:
            resolved = fs.resolve(cand)
        except Exception:  # noqa: BLE001 — 폴더 밖·빈 경로: 붙이지 않는다
            continue
        if resolved in out.attached or any(resolved == s for s, _ in out.skipped):
            continue
        if _SECRETISH.search(resolved):
            continue
        try:
            got = await _head(fs, resolved, per_file)
        except Exception:  # noqa: BLE001 — 읽기 실패는 그 파일만 포기
            got = None
        if got is None:
            continue
        data, size = got
        if _ext(resolved) not in _TEXT_EXT or b"\x00" in data[:4096]:
            out.skipped.append(
                (
                    resolved,
                    f"binary .{_ext(resolved)}, {size:,} bytes — open it with the matching tool",
                )
            )
            continue
        if budget <= 0:
            out.skipped.append(
                (resolved, f"{size:,} bytes — not attached (attachment budget used up); Read it")
            )
            continue
        cap = min(per_file, budget)
        body = data[:cap].decode("utf-8", "replace")
        full = size <= cap
        budget -= min(size, cap)
        note = (
            ""
            if full
            else f"\n…[truncated: showing the first {cap:,} of {size:,} bytes — Read with offset for the rest]"
        )
        parts.append(f'<file path="{resolved}" bytes="{size}">\n{body}{note}\n</file>')
        out.attached.append(resolved)
        if full:
            out.witnessed.extend(dict.fromkeys([cand, resolved]))
    if parts or out.skipped:
        lines = []
        if parts:
            lines.append(
                "[Files named in the request — already read for you from the working folder. "
                "Use this content directly; do not Read these files again.]"
            )
            lines.extend(parts)
        if out.skipped:
            lines.append("[Also named in the request but not attached:]")
            lines.extend(f"- {p}: {why}" for p, why in out.skipped)
        out.block = "\n".join(lines)
    return out
