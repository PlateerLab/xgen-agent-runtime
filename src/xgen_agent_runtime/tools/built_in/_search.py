"""Glob·Grep 의 실제 검색 — **표준 라이브러리만** 쓴다.

왜 이 파일이 따로 있고 stdlib 만 쓰는가
--------------------------------------
Glob·Grep 은 러너와 로컬 두 곳에서 돈다. 예전엔 두 곳이 **서로 다른 프로그램**이었다:

* 러너 — 셸 ``for f in {pattern}`` / ``grep -E``. 로컬 — ``Path.glob`` / Python ``re``.
* 결과: 같은 질문에 다른 답. ``Grep("order_id = \\d+")`` 는 로컬에선 찾고 러너에선
  **"No matches"** 였다 (``grep -E`` 에는 ``\\d`` 가 없다). 경로 표기(상대/절대)·출력
  모양·순서도 전부 갈렸다.
* 더 나쁜 것: 러너 Glob 은 모델이 준 패턴을 **따옴표 없이 셸에 끼워 넣었다.**
  ``$(…)`` 가 그대로 실행됐다. Glob 은 read_only 로 선언된 도구라 PLAN 모드에서도
  확인 없이 돈다 — 확인 없는 명령 실행기였다.

이제 **같은 코드가 두 곳에서 돈다.** 로컬은 이 모듈을 import 해서 부르고, 러너는
이 파일의 소스를 ``python3 -c <소스> <JSON 인자>`` 로 실행한다(러너는 workflow 와 같은
Python 3.14 라 ``re``·``glob`` 의미가 같다). 인자는 argv 로만 가므로 셸 해석이 없다.
출력 텍스트까지 여기서 만들어서 포맷도 구조적으로 같다.

⚠ 이 파일에 서드파티나 패키지 내부 import 를 넣지 말 것 — 러너에는 이 패키지가 없다.
"""

import json
import os
import re
import sys
from pathlib import Path

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox"}
GLOB_MAX_RESULTS = 500
GREP_MAX_FILES = 200
GREP_MAX_MATCHES = 300
GREP_MAX_FILE_SIZE = 2 * 1024 * 1024
#: 러너 exec 의 stdout 상한(200KB)보다 작게 — 잘리면 JSON 이 깨진다.
TEXT_CAP = 150_000


def _inside(path, roots):
    if not roots:
        return True
    try:
        real = os.path.realpath(path)
    except OSError:
        return False
    for root in roots:
        r = os.path.realpath(root)
        if real == r or real.startswith(r.rstrip("/") + "/"):
            return True
    return False


def _cap(text):
    if len(text) <= TEXT_CAP:
        return text
    return text[:TEXT_CAP] + "\n\n... (output truncated)"


def glob_files(req):
    base = Path(req["base"])
    pattern = req["pattern"]
    roots = req.get("roots") or []
    if not base.is_dir():
        return {"ok": False, "text": "Directory not found: %s" % base}
    try:
        matches = list(base.glob(pattern))
    except Exception as e:  # 잘못된 패턴 등
        return {"ok": False, "text": "Glob error: %s" % e}
    files = []
    for m in matches:
        try:
            if m.is_file() and _inside(m, roots):
                files.append((m.stat().st_mtime, str(m)))
        except OSError:
            continue
    if not files:
        return {"ok": True, "text": "No files matching '%s' in %s" % (pattern, base)}
    files.sort(reverse=True)  # 최근 수정 순
    shown = [p for _, p in files[:GLOB_MAX_RESULTS]]
    text = "\n".join(shown)
    if len(files) > GLOB_MAX_RESULTS:
        text += "\n\n... (showing %d of %d matches)" % (GLOB_MAX_RESULTS, len(files))
    return {"ok": True, "text": _cap(text)}


def grep_files(req):
    base = Path(req["base"])
    pattern = req["pattern"]
    mode = req.get("output_mode") or "files"
    ctx = int(req.get("context") or 0)
    file_glob = req.get("glob")
    roots = req.get("roots") or []
    try:
        regex = re.compile(pattern, re.IGNORECASE if req.get("case_insensitive") else 0)
    except re.error as e:
        return {"ok": False, "text": "Invalid regex: %s" % e}

    if base.is_file():
        targets = [base]
    elif base.is_dir():
        found = sorted(base.rglob(file_glob or "*"))
        targets = []
        for t in found:
            try:
                if not t.is_file():
                    continue
            except OSError:
                continue
            if set(t.relative_to(base).parts[:-1]) & SKIP_DIRS:
                continue
            if _inside(t, roots):
                targets.append(t)
        targets = targets[:GREP_MAX_FILES]
    else:
        return {"ok": False, "text": "Path not found: %s" % base}

    match_files, match_lines, total = [], [], 0
    for fpath in targets:
        try:
            if fpath.stat().st_size > GREP_MAX_FILE_SIZE:
                continue
            text = fpath.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lines = text.splitlines()
        hits = [(i, ln) for i, ln in enumerate(lines) if regex.search(ln)]
        if not hits:
            continue
        match_files.append(str(fpath))
        total += len(hits)
        if mode == "content":
            for n, _ in hits:
                if total > GREP_MAX_MATCHES:
                    break
                for ci in range(max(0, n - ctx), min(len(lines), n + ctx + 1)):
                    mark = ">" if ci == n else " "
                    match_lines.append("%s:%d:%s %s" % (fpath, ci + 1, mark, lines[ci]))
                if ctx > 0:
                    match_lines.append("--")

    if mode == "count":
        return {"ok": True, "text": "%d matches in %d files" % (total, len(match_files))}
    if mode == "files":
        if not match_files:
            return {"ok": True, "text": "No matches for '%s'" % pattern}
        text = "\n".join(match_files)
        if len(match_files) >= GREP_MAX_FILES:
            text += "\n\n... (limited to %d files)" % GREP_MAX_FILES
        return {"ok": True, "text": _cap(text)}
    if not match_lines:
        return {"ok": True, "text": "No matches for '%s'" % pattern}
    text = "\n".join(match_lines[: GREP_MAX_MATCHES * 3])
    if total > GREP_MAX_MATCHES:
        text += "\n\n... (%d total matches, showing first %d)" % (total, GREP_MAX_MATCHES)
    return {"ok": True, "text": _cap(text)}


def run(req):
    op = req.get("op")
    if op == "glob":
        return glob_files(req)
    if op == "grep":
        return grep_files(req)
    return {"ok": False, "text": "unknown search op: %r" % (op,)}


if __name__ == "__main__":
    sys.stdout.write(json.dumps(run(json.loads(sys.argv[1])), ensure_ascii=False))
