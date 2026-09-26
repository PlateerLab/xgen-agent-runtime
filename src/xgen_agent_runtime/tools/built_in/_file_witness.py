"""에이전트가 **본 적 없는 파일을 말없이 덮어쓰는 것**을 막는 장부.

참고 하네스의 기본 계약이다 — 파일을 바꾸려면 먼저 읽는다. 우리에겐 그 계약이
없어서 ``Write`` 가 이미 있는 파일을 내용도 모른 채 잘라내고 새로 쓸 수 있었다.
사용자가 올린 문서·이전 세션의 산출물이 그렇게 사라져도 **로그에는 아무 흔적이
남지 않는다**(성공한 쓰기다). 그래서 이 구멍은 기록으로 셀 수 없고 계약으로만 막는다.

장부는 ``state.shared`` 의 키 하나다. 이 dict 는 턴이 아니라 **세션** 단위로 살아 있어
(``PipelineState.begin_turn`` 이 건드리지 않는다) 앞 턴에서 읽은 파일을 이번 턴에
고쳐도 마찰이 없다.

**경로만 적는다.** 읽은 뒤 밖에서 바뀐 파일까지 잡으려면 쓸 때마다 현재 내용을 다시
확인해야 하는데, 파일 도구는 로컬과 XGeny 샌드박스 두 경로로 갈리고 샌드박스 쪽은
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


def refusal(path: str) -> str:
    """거절할 때 모델에게 주는 말 — 다음 한 수가 분명해야 한다."""
    return (
        f"{path} already exists and you have not read it in this session. Nothing was "
        f"written — a blind Write would discard whatever is in it. Read it first, then use "
        f"Edit to change part of it, or Write again to replace it knowingly."
    )
