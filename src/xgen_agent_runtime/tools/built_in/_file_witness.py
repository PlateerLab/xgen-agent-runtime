"""에이전트가 **본 적 없는 파일을 말없이 덮어쓰는 것**을 막는 장부.

참고 하네스의 기본 계약이다 — 파일을 바꾸려면 먼저 읽는다. 우리에겐 그 계약이
없어서 ``Write`` 가 이미 있는 파일을 내용도 모른 채 잘라내고 새로 쓸 수 있었다.
사용자가 올린 문서·이전 세션의 산출물이 그렇게 사라져도 **로그에는 아무 흔적이
남지 않는다**(성공한 쓰기다). 그래서 이 구멍은 기록으로 셀 수 없고 계약으로만 막는다.

장부는 ``state.shared`` 의 키 하나다. 이 dict 는 턴이 아니라 **세션** 단위로 살아 있어
(``PipelineState.begin_turn`` 이 건드리지 않는다) 앞 턴에서 읽은 파일을 이번 턴에
고쳐도 마찰이 없다.

**경로만 적는다.** 읽은 뒤 밖에서 바뀐 파일까지 잡으려면 쓸 때마다 현재 내용을 다시
확인해야 하는데, 파일 도구는 로컬과 Geny 샌드박스 두 경로로 갈리고 샌드박스 쪽은
그 확인이 매번 원격 왕복이다. 막으려는 것은 "내용을 모른 채 지우는 일" 이고 그건
경로만으로 잡힌다 — 두 경로에 **같은 규칙**이 걸리는 쪽을 택했다.
"""

from __future__ import annotations

from typing import Any, Dict, List

from xgen_agent_runtime.core.shared_keys import SharedKeys

#: state.shared 키 — 이 세션에서 내용을 확인한 파일 경로들.
#: ⚠ ``executor.`` 접두어가 **필수**다. Stage 10 은 허용된 이름공간(executor.·memory.·geny.·plugin.)의
#: 키만 state.shared 에 반영하고 나머지는 경고만 남기고 버린다(stages/s10_tool/state_mutation.py).
#: 4.51.0~4.59.0 은 ``file.witnessed`` 여서 장부가 **한 번도 기록되지 않았다** — 이미 있는 파일의
#: Write 는 먼저 읽었든 자기가 방금 썼든 전부 거절됐다(2026-09-24 로컬 벤치 qwen: 실행당 11~19회,
#: 과제의 약 10%. Read 뒤 Write 도 거절돼 모델은 Bash 로 우회했다). 단위 테스트가 장부를 손으로
#: 반영해서 못 잡았다 — 이제 테스트는 실제 반영 함수를 거친다.
WITNESSED_KEY = SharedKeys.FILE_WITNESSED
RANGES_KEY = SharedKeys.FILE_READ_RANGES
# 4.51.0 shipped this unnamespaced spelling. Read it during the transition so
# an in-flight session does not forget what it inspected, but never write it
# again: Stage 10 correctly rejects unknown namespaces.
_LEGACY_WITNESSED_KEY = "file.witnessed"

#: 장부 상한. 넘으면 오래된 것부터 잊는다 — 잊은 파일은 "안 읽은 것" 이 되어
#: 한 번 더 읽으면 그만이다(데이터를 잃지는 않는다).
MAX_ENTRIES = 500


def _book(state_view: Any) -> List[str]:
    shared = getattr(state_view, "shared", None)
    if not isinstance(shared, dict):
        return []
    seen = shared.get(WITNESSED_KEY)
    if not isinstance(seen, list):
        seen = shared.get(_LEGACY_WITNESSED_KEY)
    return list(seen) if isinstance(seen, list) else []


def witnessed_mutation(state_view: Any, *paths: str) -> Dict[str, Any]:
    """읽은 파일을 장부에 올리는 ``state_mutations`` 조각.

    같은 파일이라도 모델이 준 표기와 해석된 절대 경로가 다를 수 있어 **둘 다** 적는다.
    """
    wanted = [p for p in paths if p]
    book = [p for p in _book(state_view) if p not in wanted]
    book.extend(dict.fromkeys(wanted))
    if len(book) > MAX_ENTRIES:
        book = book[-MAX_ENTRIES:]
    return {WITNESSED_KEY: book}


def is_witnessed(state_view: Any, path: str) -> bool:
    return path in _book(state_view)


#: Partly read files kept in the range ledger at most.
MAX_PARTIAL = 200


def _ranges_book(state_view: Any) -> Dict[str, Any]:
    shared = getattr(state_view, "shared", None)
    book = shared.get(RANGES_KEY) if isinstance(shared, dict) else None
    return dict(book) if isinstance(book, dict) else {}


def _merge(ranges: List[List[int]]) -> List[List[int]]:
    out: List[List[int]] = []
    for a, b in sorted([int(r[0]), int(r[1])] for r in ranges if len(r) == 2):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def read_mutation(
    state_view: Any, paths: List[str], *, start: int, end: int, total: int
) -> Dict[str, Any]:
    """What a Read of lines ``[start, end)`` of a ``total``-line file adds to the ledgers.

    The file counts as read (Write may replace it) only once the reads cover all of it. A preview
    of the first 2,000 lines used to count, so a Write could replace a 5,000-line file whose last
    3,000 lines the agent had never seen (2026-10-23 audit). Reading it in parts counts once the
    parts cover it.
    """
    wanted = [p for p in paths if p]
    key = wanted[-1] if wanted else ""
    book = _ranges_book(state_view)
    entry = book.get(key) if isinstance(book.get(key), dict) else None
    ranges = (
        list(entry.get("ranges") or []) if entry and int(entry.get("total") or -1) == total else []
    )
    ranges = _merge(ranges + [[max(0, start), min(total, end)]])
    covered = total <= 0 or (len(ranges) == 1 and ranges[0][0] <= 0 and ranges[0][1] >= total)
    if covered:
        for p in wanted:
            book.pop(p, None)
        out = witnessed_mutation(state_view, *wanted)
        if book != _ranges_book(state_view):
            out[RANGES_KEY] = book
        return out
    book[key] = {"total": total, "ranges": ranges}
    if len(book) > MAX_PARTIAL:
        for old in list(book)[: len(book) - MAX_PARTIAL]:
            book.pop(old, None)
    return {RANGES_KEY: book}


def read_ranges(state_view: Any, path: str) -> Dict[str, Any]:
    entry = _ranges_book(state_view).get(path)
    return entry if isinstance(entry, dict) else {}


def refusal(path: str, partial: Dict[str, Any] = None) -> str:
    """거절할 때 모델에게 주는 말 — 다음 한 수가 분명해야 한다."""
    if partial and partial.get("ranges"):
        seen = ", ".join(f"{a + 1}-{b}" for a, b in partial["ranges"])
        return (
            f"{path} already exists and you have read only lines {seen} of its "
            f"{partial.get('total')}. Nothing was written: a Write replaces the whole file, "
            f"including the lines you have not seen. Read the rest first, or use Edit to change "
            f"the part you know."
        )
    return (
        f"{path} already exists and you have not read it in this session. Nothing was "
        f"written: a blind Write would discard whatever is in it. Read it first, then use "
        f"Edit to change part of it, or Write again to replace it knowingly."
    )
