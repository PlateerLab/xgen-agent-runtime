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
#: Glob 이 한 번에 보여 주는 파일 수. 나머지는 offset 으로 이어 본다(개수는 늘 전부 센다).
GLOB_MAX_RESULTS = 500
#: Grep content 모드가 한 번에 보여 주는 매치 수. 나머지는 offset 으로 이어 본다 — 개수는 늘 전부 센다.
GREP_PAGE_MATCHES = 300
#: Grep files 모드가 한 번에 보여 주는 파일 수.
GREP_PAGE_FILES = 1000
#: 검색할 파일 수의 안전 상한. 넘으면 **넘었다고 말한다**(조용히 건너뛰지 않는다).
GREP_SCAN_MAX_FILES = 200_000
#: 검색 시간 상한(초). 러너 exec 제한(60초)보다 짧게 — 넘으면 어디까지 찾았는지 말하고 멈춘다.
GREP_TIME_BUDGET_S = 40.0
#: 바이너리 판정에 읽는 앞부분. NUL 이 있으면 글이 아니다.
_BINARY_PROBE = 8192
#: 한 줄을 보여 줄 때의 상한(문자). 넘으면 줄 끝에 남은 길이를 적는다(파일은 Read 로 본다).
LINE_SHOW_MAX = 2000
#: 출력 본문 상한 — JSON 으로 인코딩한 **UTF-8 바이트**. 러너 exec 의 stdout 상한(200KB)보다 작게:
#: 예전 상한(150,000 '자')은 한국어면 450KB 라 JSON 이 잘려 "Search failed" 가 됐다. 안내 줄은 이 상한
#: **밖에** 붙는다(예전에는 안내까지 잘려 몇 개를 못 봤는지 알 수 없었다).
TEXT_CAP_BYTES = 150_000
#: snapshot·read_texts 의 응답 상한(바이트, 같은 이유).
TEXT_CAP = TEXT_CAP_BYTES


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


def _json_bytes(text):
    """``text`` 가 결과 JSON 안에서 차지하는 바이트(이스케이프 포함)."""
    return len(json.dumps(text, ensure_ascii=False).encode("utf-8")) - 2


def _fit_lines(lines, budget=TEXT_CAP_BYTES):
    """앞에서부터 ``budget`` 바이트 안에 드는 줄 수(최소 1)."""
    used = 0
    for i, line in enumerate(lines):
        used += _json_bytes(line) + 2  # 줄바꿈(\n 이스케이프)
        if used > budget:
            return max(1, i)
    return len(lines)


def _show_line(line):
    if len(line) <= LINE_SHOW_MAX:
        return line
    return line[:LINE_SHOW_MAX] + " …[+%d chars on this line]" % (len(line) - LINE_SHOW_MAX)


def _offset(req):
    try:
        return max(0, int(req.get("offset") or 0))
    except (TypeError, ValueError):
        return 0


def _expand_braces(pattern):
    """``*.{ts,tsx}`` → ``['*.ts', '*.tsx']`` (중첩 없는 중괄호만). pathlib 은 중괄호를 몰라서 예전에는
    "No matches" 였다 — 도구 설명이 예로 든 바로 그 패턴이다."""
    m = re.search(r"\{([^{}]*)\}", pattern or "")
    if not m:
        return [pattern]
    out = []
    for alt in m.group(1).split(","):
        out.extend(_expand_braces(pattern[: m.start()] + alt + pattern[m.end() :]))
    return out


def _glob_match(rel, name, patterns):
    import fnmatch

    for pat in patterns:
        if "/" in pat:
            if fnmatch.fnmatch(rel, pat) or (
                pat.startswith("**/") and fnmatch.fnmatch(rel, pat[3:])
            ):
                return True
        elif fnmatch.fnmatch(name, pat):
            return True
    return False


def _walk_files(base, file_glob, roots):
    """``base`` 아래 파일(경로순)과 건너뛴 폴더. SKIP_DIRS 는 들어가지 않되 **건너뛰었다고 적는다**."""
    patterns = _expand_braces(file_glob) if file_glob else None
    files, skipped_dirs = [], {}
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        keep = []
        for d in sorted(dirnames):
            if d in SKIP_DIRS:
                skipped_dirs[d] = skipped_dirs.get(d, 0) + 1
            else:
                keep.append(d)
        dirnames[:] = keep
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, base).replace(os.sep, "/")
            if patterns and not _glob_match(rel, name, patterns):
                continue
            if not os.path.isfile(full) or not _inside(full, roots):
                continue
            files.append(Path(full))
    return files, skipped_dirs


def _is_binary(path):
    try:
        with open(path, "rb") as f:
            return b"\0" in f.read(_BINARY_PROBE)
    except OSError:
        return False


def glob_files(req):
    base = Path(req["base"])
    pattern = req["pattern"]
    roots = req.get("roots") or []
    if not base.is_dir():
        return {"ok": False, "text": "Directory not found: %s" % base}
    files, seen = [], set()
    try:
        for pat in _expand_braces(pattern):
            for m in base.glob(pat):
                key = str(m)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    if m.is_file() and _inside(m, roots):
                        stat = m.stat()
                        files.append((stat.st_mtime, str(m), stat.st_size))
                except OSError:
                    continue
    except Exception as e:  # 잘못된 패턴 등
        return {"ok": False, "text": "Glob error: %s" % e}
    if not files:
        return {"ok": True, "text": "No files matching '%s' in %s" % (pattern, base)}
    files.sort(reverse=True)  # 최근 수정 순
    offset = _offset(req)
    page = files[offset : offset + GLOB_MAX_RESULTS]
    if not page:
        return {
            "ok": True,
            "text": "No files at offset %d (%d files match '%s')" % (offset, len(files), pattern),
        }
    shown = [p for _, p, _ in page]
    fit = _fit_lines(shown)
    shown = shown[:fit]
    text = "\n".join(shown)
    end = offset + len(shown)
    if offset > 0 or end < len(files):
        text += "\n\n... (files %d-%d of %d, most recently modified first%s)" % (
            offset + 1,
            end,
            len(files),
            "; call Glob again with offset=%d for the next page" % end if end < len(files) else "",
        )
    result = {"ok": True, "text": text}
    if req.get("include_metadata") is True:
        result["files"] = [{"path": p, "size_bytes": size} for _, p, size in page[:fit]]
    return result


def grep_files(req):
    """정규식 검색. **모든 파일을 끝까지 세고**, 보여 주는 만큼만 보여 주며, 못 보여 준 것은 몇 개인지와
    다음 쪽을 보는 방법(offset)을 적는다.

    예전 결함(2026-10-23 감사, 재현): 앞 200개 파일만 찾고 2MiB 넘는 파일을 건너뛰면서 아무 말이 없었고,
    content 모드에서 첫 파일의 매치가 300개를 넘으면 **"No matches"** 를 돌려줬다(count 모드는 400개).
    모델은 "없다" 고 믿고 지우거나 고친다 — 결과를 사실과 반대로 만드는 결함이었다.
    """
    import time

    base = Path(req["base"])
    pattern = req["pattern"]
    mode = req.get("output_mode") or "files"
    ctx = int(req.get("context") or 0)
    file_glob = req.get("glob")
    roots = req.get("roots") or []
    offset = _offset(req)
    try:
        regex = re.compile(pattern, re.IGNORECASE if req.get("case_insensitive") else 0)
    except re.error as e:
        return {"ok": False, "text": "Invalid regex: %s" % e}

    skipped_dirs = {}
    if base.is_file():
        targets = [base]
    elif base.is_dir():
        targets, skipped_dirs = _walk_files(base, file_glob, roots)
    else:
        return {"ok": False, "text": "Path not found: %s" % base}

    notes = []
    if len(targets) > GREP_SCAN_MAX_FILES:
        notes.append(
            "searched the first %d of %d files (by path); narrow 'path' or 'glob' to search the rest"
            % (GREP_SCAN_MAX_FILES, len(targets))
        )
        targets = targets[:GREP_SCAN_MAX_FILES]

    started = time.monotonic()
    total = 0
    file_counts = []  # (path, hits)
    blocks = []  # content 모드에서 보여 줄 (줄, 매치 줄인가) — 구분선 포함
    page_end = offset + GREP_PAGE_MATCHES
    binary_skipped = unreadable = 0
    scanned = 0
    stopped_inside = None  # (file, line) where the time limit stopped a file mid-way
    for fpath in targets:
        if time.monotonic() - started > GREP_TIME_BUDGET_S:
            notes.append(
                "stopped at the time limit after %d of %d files (by path); the counts cover only "
                "those files — narrow 'path' or 'glob' to search the rest" % (scanned, len(targets))
            )
            break
        scanned += 1
        if _is_binary(fpath):
            binary_skipped += 1
            continue
        hits = 0
        before = []  # 앞 문맥(최근 ctx 줄)
        after_left = 0
        last_shown = -1
        try:
            with open(fpath, encoding="utf-8", errors="ignore") as f:
                for n, raw in enumerate(f):
                    # One huge file used to run past the runner's own limit, and the whole search failed
                    # with nothing to show. Stop inside the file and say where.
                    if n and n % 5000 == 0 and time.monotonic() - started > GREP_TIME_BUDGET_S:
                        stopped_inside = (fpath, n)
                        break
                    line = raw.rstrip("\n").rstrip("\r")
                    matched = regex.search(line) is not None
                    if matched:
                        idx = total + hits  # 전체에서 이 매치의 순번(0부터)
                        hits += 1
                        if mode == "content" and offset <= idx < page_end:
                            first = n - len(before)
                            if blocks and last_shown >= 0 and first > last_shown + 1 and ctx > 0:
                                blocks.append(("--", False))
                            for k, prev in enumerate(before):
                                ln = first + k
                                if ln > last_shown:
                                    blocks.append(
                                        ("%s:%d:  %s" % (fpath, ln + 1, _show_line(prev)), False)
                                    )
                                    last_shown = ln
                            blocks.append(("%s:%d:> %s" % (fpath, n + 1, _show_line(line)), True))
                            last_shown = n
                            after_left = ctx
                            before = []
                            continue
                    elif after_left > 0 and n > last_shown:
                        blocks.append(("%s:%d:  %s" % (fpath, n + 1, _show_line(line)), False))
                        last_shown = n
                        after_left -= 1
                        continue
                    if ctx > 0:
                        before.append(line)
                        if len(before) > ctx:
                            before.pop(0)
        except OSError:
            unreadable += 1
            continue
        if hits:
            file_counts.append((str(fpath), hits))
            total += hits
        if stopped_inside is not None:
            notes.append(
                "stopped at the time limit at line %d of %s (file %d of %d by path); the counts cover "
                "only what was searched; narrow 'path' or 'glob' to search the rest"
                % (stopped_inside[1], stopped_inside[0], scanned, len(targets))
            )
            break

    nfiles = len(file_counts)
    if binary_skipped:
        notes.append("%d binary file(s) not searched" % binary_skipped)
    if unreadable:
        notes.append("%d file(s) could not be read" % unreadable)
    if skipped_dirs:
        notes.append(
            "not searched: %s (pass a path inside one to search there)"
            % ", ".join("%s/ (%d)" % (d, c) for d, c in sorted(skipped_dirs.items()))
        )

    def _done(body_lines, page_note):
        all_notes = ([page_note] if page_note else []) + notes
        text = "\n".join(body_lines)
        if all_notes:
            text += ("\n\n" if text else "") + "\n".join("... (%s)" % n for n in all_notes)
        return {"ok": True, "text": text}

    if mode == "count":
        lines = ["%d matches in %d files" % (total, nfiles)]
        return _done(lines, "")
    if total == 0:
        return _done(["No matches for '%s'" % pattern], "")
    if mode == "files":
        page = [p for p, _ in file_counts[offset : offset + GREP_PAGE_FILES]]
        if not page:
            return _done([], "no files at offset %d (%d files match)" % (offset, nfiles))
        page = page[: _fit_lines(page)]
        end = offset + len(page)
        note = ""
        if offset > 0 or end < nfiles:
            note = "files %d-%d of %d matching files (%d matches)%s" % (
                offset + 1,
                end,
                nfiles,
                total,
                "; call Grep again with offset=%d for the next page" % end if end < nfiles else "",
            )
        return _done(page, note)
    # content
    if not blocks:
        return _done(
            [], "no matches at offset %d (%d matches in %d files)" % (offset, total, nfiles)
        )
    fit = _fit_lines([text for text, _ in blocks])
    shown_lines = [text for text, _ in blocks[:fit]]
    shown_matches = sum(1 for _, is_match in blocks[:fit] if is_match)
    end = offset + shown_matches
    note = ""
    if offset > 0 or end < total:
        note = "matches %d-%d of %d in %d files%s" % (
            offset + 1,
            end,
            total,
            nfiles,
            "; call Grep again with offset=%d for the next page" % end if end < total else "",
        )
    return _done(shown_lines, note)


def snapshot_files(req):
    """Return a complete, bounded UTF-8 snapshot of a small workspace.

    This is intentionally implemented beside Glob/Grep so local and runner
    filesystems execute the exact same traversal.  Size and file-count gates
    are checked *before* file contents are returned; a large workspace cannot
    become an unbounded host-to-runner transfer.
    """
    base = Path(req["base"])
    roots = req.get("roots") or []
    max_files = max(0, int(req.get("max_files") or 0))
    max_bytes = max(0, int(req.get("max_bytes") or 0))
    max_dirs = max(0, int(req.get("max_dirs") or 0))
    if not base.is_dir():
        return {
            "ok": False,
            "eligible": False,
            "reason": "workspace_unavailable",
            "files": [],
        }
    if not _inside(base, roots):
        return {
            "ok": False,
            "eligible": False,
            "reason": "workspace_outside_allowed_roots",
            "files": [],
        }

    paths = []
    total_bytes = 0
    directory_count = 0
    try:
        # Do not omit directories by name: an infrastructure-looking folder
        # can still be the user's actual input. Bound traversal structurally
        # instead, so an eligible result really is a complete snapshot.
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            dirnames[:] = sorted(dirnames)
            directory_count += len(dirnames)
            if directory_count > max_dirs:
                return {
                    "ok": True,
                    "eligible": False,
                    "reason": "too_many_directories",
                    "file_count": len(paths),
                    "total_bytes": total_bytes,
                    "files": [],
                }
            for dirname in dirnames:
                if (Path(dirpath) / dirname).is_symlink():
                    return {
                        "ok": True,
                        "eligible": False,
                        "reason": "non_regular_input",
                        "file_count": len(paths),
                        "total_bytes": total_bytes,
                        "files": [],
                    }
            for filename in sorted(filenames):
                path = Path(dirpath) / filename
                rel = path.relative_to(base)
                if path.is_symlink():
                    return {
                        "ok": True,
                        "eligible": False,
                        "reason": "non_regular_input",
                        "file_count": len(paths),
                        "total_bytes": total_bytes,
                        "files": [],
                    }
                if not path.is_file() or not _inside(path, roots):
                    continue
                size = path.stat().st_size
                paths.append((path, rel.as_posix(), size))
                total_bytes += size
                if len(paths) > max_files:
                    return {
                        "ok": True,
                        "eligible": False,
                        "reason": "too_many_files",
                        "file_count": len(paths),
                        "total_bytes": total_bytes,
                        "files": [],
                    }
                if total_bytes > max_bytes:
                    return {
                        "ok": True,
                        "eligible": False,
                        "reason": "workspace_over_budget",
                        "file_count": len(paths),
                        "total_bytes": total_bytes,
                        "files": [],
                    }
    except OSError:
        return {
            "ok": True,
            "eligible": False,
            "reason": "workspace_unreadable",
            "file_count": len(paths),
            "total_bytes": total_bytes,
            "files": [],
        }

    if not paths:
        return {
            "ok": True,
            "eligible": False,
            "reason": "empty_workspace",
            "file_count": 0,
            "total_bytes": 0,
            "files": [],
        }

    paths.sort(key=lambda item: item[1])

    entries = []
    actual_total = 0
    for path, rel, _size in paths:
        try:
            raw = path.read_bytes()
        except OSError:
            return {
                "ok": True,
                "eligible": False,
                "reason": "workspace_unreadable",
                "file_count": len(paths),
                "total_bytes": actual_total,
                "files": [],
            }
        actual_total += len(raw)
        if actual_total > max_bytes:
            return {
                "ok": True,
                "eligible": False,
                "reason": "workspace_over_budget",
                "file_count": len(paths),
                "total_bytes": actual_total,
                "files": [],
            }
        if b"\x00" in raw:
            return {
                "ok": True,
                "eligible": False,
                "reason": "non_text_input",
                "file_count": len(paths),
                "total_bytes": actual_total,
                "files": [],
            }
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            return {
                "ok": True,
                "eligible": False,
                "reason": "non_text_input",
                "file_count": len(paths),
                "total_bytes": actual_total,
                "files": [],
            }
        entries.append({"path": rel, "bytes": len(raw), "content": content})

    result = {
        "ok": True,
        "eligible": True,
        "reason": "eligible",
        "file_count": len(entries),
        "total_bytes": actual_total,
        "files": entries,
    }
    # The runner transports this dict as JSON on stdout. UTF-8 byte limits do
    # not bound JSON expansion (control characters and long paths are escaped),
    # so guard the actual wire representation before returning any contents.
    wire_bytes = len(json.dumps(result, ensure_ascii=False).encode("utf-8"))
    if wire_bytes > TEXT_CAP:
        return {
            "ok": True,
            "eligible": False,
            "reason": "snapshot_encoding_over_budget",
            "file_count": len(entries),
            "total_bytes": actual_total,
            "files": [],
        }
    return result


def _looks_binary(target):
    """앞 8KiB 로 텍스트가 아닌 파일인지 본다 — NUL 이 있거나 UTF-8 로 풀리지 않으면 바이너리."""
    import codecs

    try:
        with open(target, "rb") as stream:
            head = stream.read(8192)
    except OSError:
        return False
    if b"\x00" in head:
        return True
    try:
        codecs.getincrementaldecoder("utf-8")().decode(head, final=False)
    except UnicodeDecodeError:
        return True
    return False


def read_texts(req):
    """Read only explicitly named UTF-8 files under one bounded workspace."""
    base = Path(req["base"])
    roots = req.get("roots") or []
    requested = req.get("paths") if isinstance(req.get("paths"), list) else []
    max_files = max(0, int(req.get("max_files") or 0))
    max_bytes = max(0, int(req.get("max_bytes") or 0))
    if not base.is_dir() or not _inside(base, roots):
        return {
            "ok": False,
            "eligible": False,
            "reason": "workspace_unavailable",
            "files": [],
        }

    paths = list(dict.fromkeys(str(path) for path in requested if isinstance(path, str)))
    if not paths:
        return {"ok": True, "eligible": True, "reason": "eligible", "files": []}
    if len(paths) > max_files:
        return {
            "ok": True,
            "eligible": False,
            "reason": "too_many_files",
            "file_count": len(paths),
            "files": [],
        }

    entries = []
    skipped = []
    total_bytes = 0
    for relative in paths:
        target = base / relative
        try:
            lexical = target.relative_to(base)
        except ValueError:
            lexical = None
        if (
            lexical is None
            or ".." in lexical.parts
            or target.is_symlink()
            or not _inside(target, [str(base)])
        ):
            return {
                "ok": True,
                "eligible": False,
                "reason": "path_outside_workspace",
                "path": relative,
                "files": [],
            }
        try:
            if not target.is_file():
                return {
                    "ok": True,
                    "eligible": False,
                    "reason": "file_missing",
                    "path": relative,
                    "files": [],
                }
            size = target.stat().st_size
        except OSError:
            return {
                "ok": True,
                "eligible": False,
                "reason": "file_unreadable",
                "path": relative,
                "files": [],
            }
        # 텍스트가 아닌 파일(xlsx·docx·이미지 등)은 읽기 전체를 실패시키지 않고 그 파일만 건너뛴다 —
        # 호출자(산출물 계약)가 "검사 대상 아님" 으로 알린다. 예전에는 xlsx 하나 때문에 같이 적은
        # CSV 까지 검사되지 않고 실패로 돌아왔다(2026-09-26 dev 실사용 점검). 앞부분만 보고 먼저
        # 판정하므로 큰 바이너리도 용량 예산을 쓰지 않는다.
        if _looks_binary(target):
            skipped.append({"path": lexical.as_posix(), "reason": "non_text_input"})
            continue
        total_bytes += size
        if total_bytes > max_bytes:
            return {
                "ok": True,
                "eligible": False,
                "reason": "files_over_budget",
                "path": relative,
                "total_bytes": total_bytes,
                "files": [],
            }
        try:
            raw = target.read_bytes()
        except OSError:
            return {
                "ok": True,
                "eligible": False,
                "reason": "file_unreadable",
                "path": relative,
                "files": [],
            }
        actual_total = sum(entry["bytes"] for entry in entries) + len(raw)
        if actual_total > max_bytes:
            return {
                "ok": True,
                "eligible": False,
                "reason": "files_over_budget",
                "path": relative,
                "total_bytes": actual_total,
                "files": [],
            }
        try:
            content = raw.decode("utf-8") if b"\x00" not in raw else None
        except UnicodeDecodeError:
            content = None
        if content is None:
            skipped.append({"path": lexical.as_posix(), "reason": "non_text_input"})
            total_bytes -= size
            continue
        entries.append({"path": lexical.as_posix(), "bytes": len(raw), "content": content})

    result = {
        "ok": True,
        "eligible": True,
        "reason": "eligible",
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "files": entries,
        "skipped": skipped,
    }
    if len(json.dumps(result, ensure_ascii=False).encode("utf-8")) > TEXT_CAP:
        return {
            "ok": True,
            "eligible": False,
            "reason": "snapshot_encoding_over_budget",
            "file_count": len(entries),
            "total_bytes": result["total_bytes"],
            "files": [],
        }
    return result


def run(req):
    op = req.get("op")
    if op == "glob":
        return glob_files(req)
    if op == "grep":
        return grep_files(req)
    if op == "snapshot":
        return snapshot_files(req)
    if op == "read_texts":
        return read_texts(req)
    return {"ok": False, "text": "unknown search op: %r" % (op,)}


if __name__ == "__main__":
    sys.stdout.write(json.dumps(run(json.loads(sys.argv[1])), ensure_ascii=False))
