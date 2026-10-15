"""숨김 목록 — 첫 화면에 없는 도구가 **있다는 사실**을 모델에게 알리는 글.

스키마는 보이는 도구만 나간다. 그래서 무엇이 더 있는지는 이 글로만 안다(모르는 도구는 검색도 못 한다).
SDK 경로는 Stage 3 이 시스템 프롬프트에 붙이고, CLI 경로(claude_code·codex)는 턴 조립이 같은 글을
시스템 프롬프트에 붙인다 — **같은 함수**가 만드므로 두 경로가 같은 글을 본다.

모양
----
* 문(Guide) 뒤의 도구는 **문 이름 한 줄**로 묶는다: ``- JobGuide opens: JobSchedule, JobList, JobCancel``.
  문은 첫 화면에 서 있고 자기 설명이 방을 말하므로, 도구마다 한 줄씩 적는 것은 같은 말을 두 번 하는 것이다.
* 문이 없는(또는 숨은 문의) 도구는 한 줄 설명: ``- ParseDocument — Extract the TEXT of a document file …``.
* 전체가 :data:`CATALOG_MAX_CHARS` 를 넘으면 한 줄 설명을 이름으로 줄인다(문 줄은 그대로).

예전 모양(도구마다 한 줄, 넘치면 **전부** 이름만)은 조건에 따라 뒤집혔다: 기본 웹 턴은 한 줄 설명형이었다가
SSH 를 켜거나 PC 브라우저를 켜면 숨긴 도구가 45개를 넘어 이름만 남았다(2026-09-30 실측). 같은 에이전트가
설정 하나로 문서 도구의 설명을 잃었다.

활성화 상태가 아니라 등록 시 core 표시로 만든다 — ToolSearch 로 도구를 열어도 글이 바뀌지 않아 프롬프트
캐시 접두가 유지된다.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

#: 한 줄 설명의 길이.
CATALOG_ONE_LINER_CHARS = 72
#: 글 전체의 상한.
CATALOG_MAX_CHARS = 4_000

HEADER = (
    "## Additional tools (hidden — not in your tool list)\n"
    "{count} more tools exist. To use one, call "
    'ToolSearch("<keyword or exact name>") — its schema arrives on '
    "your next step. ToolSearch with no query browses this catalog. "
    "Tools listed under a guide open when you call that guide."
)


def _one_liner(tool: Any) -> str:
    desc = str(getattr(tool, "description", "") or "").strip()
    line = desc.splitlines()[0] if desc else ""
    if len(line) > CATALOG_ONE_LINER_CHARS:
        line = line[: CATALOG_ONE_LINER_CHARS - 1] + "…"
    return line


def deferred_catalog_text(registry: Any) -> str:
    """이 레지스트리의 숨김 목록 글. 숨긴 도구가 없으면 빈 문자열."""
    if registry is None:
        return ""
    is_core = getattr(registry, "is_core", None)
    get = getattr(registry, "get", None)
    list_names = getattr(registry, "list_names", None)
    if not (callable(is_core) and callable(get) and callable(list_names)):
        return ""
    from xgen_agent_runtime.tools.gates import owner_gate

    names = sorted(list_names())
    grouped: Dict[str, List[str]] = {}
    single: List[Tuple[str, str]] = []
    count = 0
    for name in names:
        try:
            if is_core(name):
                continue
            count += 1
            gate = owner_gate(name, names)
            if gate is not None:
                grouped.setdefault(gate, []).append(name)
                continue
            single.append((name, _one_liner(get(name))))
        except Exception:  # noqa: BLE001 — 도구 하나가 프롬프트를 깨지 않는다
            continue
    if not count:
        return ""
    header = HEADER.format(count=count)
    gate_lines = [f"- {g} opens: {', '.join(members)}" for g, members in sorted(grouped.items())]
    rich = [f"- {n} — {d}" if d else f"- {n}" for n, d in single]
    body = "\n".join(gate_lines + rich)
    if len(header) + len(body) > CATALOG_MAX_CHARS:
        # 넘치면 한 줄 설명만 이름으로 줄인다 — 문 줄은 짧고, 그 도구들의 설명은 문이 들고 있다.
        # 그래도 넘치면 **이름 단위로** 끊고 남은 수를 말한다 — 글자 수로 자르면 이름 중간이 잘려
        # 모델이 없는 이름을 부른다.
        room = CATALOG_MAX_CHARS - len(header) - len("\n".join(gate_lines)) - 80
        shown: List[str] = []
        used = 0
        for n, _ in single:
            if used + len(n) + 2 > room:
                break
            shown.append(n)
            used += len(n) + 2
        rest = len(single) - len(shown)
        line = ", ".join(shown)
        if rest:
            line = (
                f"{line}, … and {rest} more (ToolSearch with no query lists every one)"
                if line
                else (f"{rest} more (ToolSearch with no query lists every one)")
            )
        body = "\n".join(gate_lines + ([f"- {line}"] if line else []))
    return header + "\n" + body


__all__ = ["CATALOG_MAX_CHARS", "CATALOG_ONE_LINER_CHARS", "deferred_catalog_text"]
