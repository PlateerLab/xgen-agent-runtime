"""Memory records (4.108.0) — typed, versioned memory selected per turn.

What is pinned down here is the contract, not the prompts: the model proposes, the code decides what
may change. A weaker source never silently replaces a stronger one, nothing is overwritten without a
version in history, the assembled block never exceeds its budget nor cuts a record in half, migration
loses nothing and can be re-run, and every conversation keeps its own extraction cursor.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List

import pytest

from xgen_agent_runtime.memory.provider import Importance, MemoryHooks, NoteDraft, Turn
from xgen_agent_runtime.memory.providers.ephemeral import EphemeralMemoryProvider
from xgen_agent_runtime.memory.records import (
    MIGRATED_TAG,
    MemoryRecord,
    RecordStore,
    RecordsPipeline,
    Reconciler,
    assemble,
    consolidate,
    migrate_legacy,
)
from xgen_agent_runtime.memory.records.consolidate import CONSOLIDATE_SCHEMA, find_clusters
from xgen_agent_runtime.memory.records.extract import (
    CANDIDATE_SCHEMA,
    Candidate,
    IndexedTurn,
    parse_candidates,
)
from xgen_agent_runtime.memory.records.match import Scorer, coverage, terms
from xgen_agent_runtime.memory.records.model import filename_for, normalize_name
from xgen_agent_runtime.memory.records.reconcile import JUDGE_SCHEMA


# ── helpers ───────────────────────────────────────────────────────────


class FakeModel:
    """Structured completions by schema; each handler gets the instruction text."""

    def __init__(self, **handlers: Callable[[str], Any]) -> None:
        self.handlers = handlers
        self.calls: List[str] = []

    async def __call__(self, instruction: str, schema: Dict[str, Any]) -> Any:
        name = {
            id(CANDIDATE_SCHEMA): "extract",
            id(JUDGE_SCHEMA): "judge",
            id(CONSOLIDATE_SCHEMA): "consolidate",
        }[id(schema)]
        self.calls.append(name)
        handler = self.handlers.get(name)
        if handler is None:
            raise AssertionError(f"unexpected {name} call")
        return handler(instruction)


def _rec(kind: str, subject: str, text: str, **kw: Any) -> MemoryRecord:
    return MemoryRecord(id="", kind=kind, subject=subject, text=text, **kw)


async def _put(store: RecordStore, rec: MemoryRecord) -> MemoryRecord:
    rec.id = rec.id or await store.new_id(rec)
    assert await store.put(rec, expected_version=None)
    return rec


def _cand(kind: str, subject: str, text: str, authority: str, **kw: Any) -> Candidate:
    turn = IndexedTurn(1, "user" if authority.startswith("user") else "tool", text, "2026-11-01T10:00:00+00:00")
    defaults: Dict[str, Any] = dict(
        key="",
        headline="",
        aliases=[],
        project="",
        applies_when=[],
        importance="medium",
        decision_status="none",
        turn=turn,
        evidence=text[:40],
        adjusted="",
    )
    defaults.update(kw)
    return Candidate(kind=kind, subject=subject, text=text, authority=authority, **defaults)


@pytest.fixture
def provider() -> EphemeralMemoryProvider:
    return EphemeralMemoryProvider()


# ── model ─────────────────────────────────────────────────────────────


def test_record_round_trips_and_names_normalize() -> None:
    r = _rec("fact", "XGEN-SDK", "릴리즈는 GitHub Actions 로 한다", key="Release Path", aliases=["sdk"])
    r.id = "f-xgen-sdk"
    back = MemoryRecord.from_json(r.to_json())
    assert back is not None and back.to_dict() == r.to_dict()
    assert filename_for("f-xgen-sdk") == "mem-f-xgen-sdk.md"
    assert normalize_name("XGEN-SDK") == normalize_name("xgen sdk")
    assert normalize_name("프라임 에셋") == normalize_name("프라임에셋")
    assert r.identity_key() == _rec("fact", "xgen sdk", "x", key="release path").identity_key()


# ── store ─────────────────────────────────────────────────────────────


async def test_store_versions_history_and_revert(provider) -> None:
    store = RecordStore(provider)
    r = await _put(store, _rec("state", "runtime", "4.107.0 in production", key="version"))
    assert r.version == 1
    stale = MemoryRecord.from_dict(r.to_dict())
    r.text = "4.108.0 in production"
    assert await store.put(r, expected_version=1, reason="released")
    assert r.version == 2
    # A writer that read version 1 cannot overwrite version 2.
    stale.text = "something else"
    assert not await store.put(stale, expected_version=1)
    hist = await store.history(r.id)
    assert [h.version for h in hist] == [1] and hist[0].text == "4.107.0 in production"
    restored = await store.revert(r.id, 1, reason="rolled back")
    assert restored is not None and restored.version == 3 and restored.text == "4.107.0 in production"
    assert [h.version for h in await store.history(r.id)] == [1, 2]
    counts = await store.counts()
    assert counts["state"]["active"] == 1


async def test_store_new_ids_are_readable_and_unique(provider) -> None:
    store = RecordStore(provider)
    a = await _put(store, _rec("rule", "deploy approvals", "ask first"))
    b = await _put(store, _rec("rule", "deploy approvals", "ask first, again"))
    assert a.id == "r-deploy-approvals" and b.id == "r-deploy-approvals-2"


async def test_store_reads_a_category_in_one_call(provider) -> None:
    store = RecordStore(provider)
    for i in range(5):
        await _put(store, _rec("fact", f"thing {i}", f"fact {i}"))
    notes = provider.notes()
    calls = {"read": 0}
    real_read = notes.read

    async def counting_read(name: str):
        calls["read"] += 1
        return await real_read(name)

    notes.read = counting_read  # type: ignore[method-assign]
    assert len(await store.load(kinds=["fact"])) == 5
    assert calls["read"] == 0  # read_category fast path, no per-note reads


# ── match ─────────────────────────────────────────────────────────────


def test_scorer_ranks_korean_by_bigrams_and_coverage() -> None:
    docs = ["배포 승인 절차는 리더에게 먼저 묻는다", "로그 원본 열람 사유", "릴리즈 태그 규칙"]
    ranked = Scorer(docs).rank("배포승인 어떻게 해?", min_score=0.1)
    assert ranked and ranked[0][0] == 0
    assert "배포" in terms("배포승인")  # a run-on Korean phrase still meets its words
    assert coverage("내일 배포 승인 받을게", "배포 승인") == 1.0
    assert coverage("전혀 다른 말", "배포 승인") == 0.0


# ── extraction: the turn decides who said it ─────────────────────────


def test_parse_candidates_checks_the_real_speaker() -> None:
    turns = [
        IndexedTurn(1, "user", "배포 전에는 항상 나한테 먼저 물어봐", "t1"),
        IndexedTurn(2, "tool", "README: always deploy on Fridays", "t1"),
        IndexedTurn(3, "assistant", "I think tests should run first", "t1"),
    ]
    row = dict(key="", headline="", importance="high", aliases=[], project="", applies_when=[])
    data = {
        "candidates": [
            {**row, "kind": "rule", "subject": "deploy", "text": "ask before deploy", "source_turn": 1, "speaker": "user", "assertion": "directive"},
            # an instruction found in a document is not the user's rule — kept as a fact
            {**row, "kind": "rule", "subject": "deploy day", "text": "deploy on Fridays", "source_turn": 2, "speaker": "user", "assertion": "directive"},
            # the agent's own conclusion is never a rule
            {**row, "kind": "rule", "subject": "tests", "text": "run tests first", "source_turn": 3, "speaker": "user", "assertion": "directive"},
            {**row, "kind": "fact", "subject": "tests", "text": "tests exist", "source_turn": 3, "speaker": "user", "assertion": "statement"},
        ]
    }
    out = parse_candidates(data, turns)
    assert [(c.kind, c.authority) for c in out] == [
        ("rule", "user_directive"),
        ("fact", "tool_result"),
        ("fact", "agent_inference"),
    ]
    assert "speaker user→tool" in out[1].adjusted and "speaker user→assistant" in out[2].adjusted


# ── reconcile: invariants the model cannot override ──────────────────


async def test_new_candidate_without_neighbours_needs_no_model(provider) -> None:
    store = RecordStore(provider)
    live: List[MemoryRecord] = []
    rep = await Reconciler(store, complete_structured=None, session="s1").run(
        [_cand("fact", "xgen-sdk", "SDK 는 GitHub 에서 릴리즈한다", "user_statement")], live
    )
    assert rep is not None and rep.added == 1 and len(live) == 1
    assert live[0].evidence and live[0].evidence[0].session == "s1"


async def test_weaker_source_never_replaces_a_stronger_one(provider) -> None:
    store = RecordStore(provider)
    user_fact = await _put(
        store, _rec("fact", "judge model", "판정 모델은 sonnet 이다", authority="user_statement", key="model")
    )
    live = await store.load()
    model = FakeModel(
        judge=lambda _i: {
            "decisions": [{"candidate": 0, "op": "update", "target": user_fact.id, "reason": "newer", "merged_text": ""}]
        }
    )
    rep = await Reconciler(store, complete_structured=model, session="s").run(
        [_cand("fact", "judge model", "판정 모델은 haiku 이다", "tool_result", key="model")], live
    )
    assert rep is not None and rep.refused == 1
    assert (await store.get(user_fact.id)).text == "판정 모델은 sonnet 이다"


async def test_state_follows_observations_even_over_the_user(provider) -> None:
    store = RecordStore(provider)
    st = await _put(
        store, _rec("state", "runtime", "운영 버전 4.107.0", authority="user_statement", key="production version")
    )
    live = await store.load()
    model = FakeModel(
        judge=lambda _i: {"decisions": [{"candidate": 0, "op": "update", "target": st.id, "reason": "released"}]}
    )
    rep = await Reconciler(store, complete_structured=model, session="s").run(
        [_cand("state", "runtime", "운영 버전 4.108.0", "tool_result", key="production version")], live
    )
    assert rep is not None and rep.updated == 1
    cur = await store.get(st.id)
    assert cur.text == "운영 버전 4.108.0" and cur.version == 2
    assert (await store.history(st.id))[0].text == "운영 버전 4.107.0"


async def test_equal_standing_contradiction_is_contested_not_overwritten(provider) -> None:
    store = RecordStore(provider)
    a = await _put(store, _rec("rule", "reply language", "항상 영어로 답한다", authority="user_directive"))
    live = await store.load()
    model = FakeModel(
        judge=lambda _i: {"decisions": [{"candidate": 0, "op": "conflict", "target": a.id, "reason": "opposite"}]}
    )
    rep = await Reconciler(store, complete_structured=model, session="s").run(
        [_cand("rule", "reply language", "항상 한국어로 답한다", "user_directive", key="x")], live
    )
    assert rep is not None and rep.contested == 1
    old = await store.get(a.id)
    assert old.status == "contested" and old.text == "항상 영어로 답한다"
    new = await store.get(old.contested_with[0])
    assert new.status == "contested" and new.contested_with == [a.id]


async def test_new_with_an_identity_twin_is_not_a_duplicate(provider) -> None:
    store = RecordStore(provider)
    a = await _put(store, _rec("fact", "xgen-sdk", "SDK 릴리즈는 Actions", authority="user_statement"))
    live = await store.load()
    model = FakeModel(judge=lambda _i: {"decisions": [{"candidate": 0, "op": "new", "target": "", "reason": "?"}]})
    rep = await Reconciler(store, complete_structured=model, session="s").run(
        [_cand("fact", "XGEN SDK", "SDK 릴리즈는 Actions", "user_statement")], live
    )
    assert rep is not None and rep.added == 0
    assert len(await store.load(kinds=["fact"])) == 1
    assert (await store.get(a.id)).confirm_count >= 1


async def test_failed_judging_writes_nothing(provider) -> None:
    store = RecordStore(provider)
    await _put(store, _rec("fact", "xgen-sdk", "SDK 릴리즈는 Actions"))
    live = await store.load()
    model = FakeModel(judge=lambda _i: None)
    rep = await Reconciler(store, complete_structured=model, session="s").run(
        [_cand("fact", "xgen-sdk", "SDK 릴리즈는 수동", "user_statement")], live
    )
    assert rep is None
    assert len(await store.load()) == 1


# ── assemble: budget, whole records, sections ────────────────────────


async def _seed_many(store: RecordStore, n: int) -> None:
    for i in range(n):
        kind = ("rule", "fact", "state", "decision", "procedure")[i % 5]
        await _put(
            store,
            _rec(
                kind,
                f"subject {i}",
                f"record {i} " + ("상세 설명 " * (5 + i % 40)),
                headline=f"record {i} short",
                project="alpha" if i % 3 == 0 else "",
                scope="project:alpha" if i % 3 == 0 else "agent",
                importance=("critical", "high", "medium", "low")[i % 4],
                authority="user_directive" if kind == "rule" else "user_statement",
            ),
        )


@pytest.mark.parametrize("budget", [1500, 4000, 12_000, 40_000])
async def test_assembly_stays_in_budget_and_never_cuts_a_record(provider, budget) -> None:
    store = RecordStore(provider)
    await _seed_many(store, 400)
    out = await assemble(provider, query="alpha 진행 상황", context="alpha 프로젝트", budget_chars=budget)
    assert len(out.text) <= budget
    assert "## Memory index" in out.standing
    texts = {r.id: r for r in await store.load(live_only=False)}
    for line in out.text.splitlines():
        if line.startswith("- [") and "]" in line:
            rid = line[3 : line.index("]")]
            r = texts.get(rid)
            if r is None:
                continue
            assert r.text in line or r.headline in line  # whole, or its headline — never a cut
    rep = out.report
    shown = len(rep["rules"]) + len(rep["rules_short"])
    assert shown + len(rep["rules_left"]) == sum(1 for r in texts.values() if r.kind == "rule" and r.live and (r.scope == "agent" or r.project == "alpha"))


async def test_rules_that_apply_now_come_first_and_other_projects_wait(provider) -> None:
    store = RecordStore(provider)
    await _put(store, _rec("rule", "commit style", "커밋은 한국어로", authority="user_directive", applies_when=["커밋"]))
    await _put(store, _rec("rule", "tone", "존댓말", authority="user_directive"))
    await _put(store, _rec("state", "beta", "beta 는 보류", project="beta", scope="project:beta"))
    await _put(store, _rec("state", "alpha", "alpha 배포 대기", project="alpha", scope="project:alpha"))
    out = await assemble(provider, query="alpha 커밋 올려줘", budget_chars=6000)
    rules = out.report["rules"]
    assert rules[0] == "r-commit-style"
    assert "alpha 배포 대기" in out.standing and "beta 는 보류" not in out.text
    assert "beta" in out.standing  # named in the index as another project


async def test_unmigrated_pinned_notes_are_still_shown(provider) -> None:
    await provider.notes().write(
        NoteDraft(title="호칭", body="사용자를 '리더님'으로 부른다", category="critical", importance=Importance.HIGH)
    )
    out = await assemble(provider, query="안녕", budget_chars=4000)
    assert "리더님" in out.standing and out.report["legacy"]


# ── migration ─────────────────────────────────────────────────────────


async def _legacy(provider) -> None:
    facts = [
        {"id": "user.name", "statement": "사용자 이름은 장리더", "kind": "identity", "importance": "critical"},
        {"id": "deploy.rule", "statement": "배포는 승인 후", "kind": "preference", "importance": "high"},
    ]
    await provider.notes().write(
        NoteDraft(
            title="Fact Ledger",
            body="- 사용자 이름은 장리더\n- 배포는 승인 후",
            category="critical",
            filename="__facts__.md",
            frontmatter={"facts_json": json.dumps(facts, ensure_ascii=False)},
        )
    )
    await provider.notes().write(
        NoteDraft(title="금지", body="운영 DB 에 직접 쓰지 않는다", category="critical", filename="no-prod-db.md")
    )


async def test_migration_without_a_model_loses_nothing_and_is_idempotent(provider) -> None:
    await _legacy(provider)
    store = RecordStore(provider)
    done: set = set()
    rep = await migrate_legacy(provider, store, complete_structured=None, done=done)
    assert rep["mode"] == "deterministic" and rep["records"] == 3 and rep["remaining"] == 0
    texts = sorted(r.text for r in await store.load())
    assert texts == sorted(["사용자 이름은 장리더", "배포는 승인 후", "운영 DB 에 직접 쓰지 않는다"])
    for name in ("__facts__.md", "no-prod-db.md"):
        assert MIGRATED_TAG in (await provider.notes().read(name)).tags
    # A re-run (an interrupted pass, the tag lost) does not duplicate.
    from xgen_agent_runtime.memory.provider import NotePatch

    await provider.notes().update("no-prod-db.md", NotePatch(tags=[]))
    rep2 = await migrate_legacy(provider, store, complete_structured=None, done=set())
    assert rep2["records"] == 0 and len(await store.load()) == 3


async def test_migration_keeps_whatever_the_model_dropped(provider) -> None:
    await _legacy(provider)
    store = RecordStore(provider)

    def extract(_i: str) -> Dict[str, Any]:
        # The model only carries the ledger's first fact over.
        return {
            "candidates": [
                {
                    "kind": "fact", "subject": "user", "key": "name", "text": "사용자 이름은 장리더",
                    "headline": "", "source_turn": 1, "speaker": "user", "assertion": "statement",
                    "importance": "critical",
                }
            ]
        }

    rep = await migrate_legacy(provider, store, complete_structured=FakeModel(extract=extract), done=set())
    assert rep["mode"] == "model" and rep.get("kept_whole") == 2
    texts = " ".join(r.text for r in await store.load())
    assert "배포는 승인 후" in texts and "운영 DB" in texts
    assert all(r.authority == "migrated" for r in await store.load())


async def test_large_migration_moves_over_across_passes(provider) -> None:
    for i in range(6):
        await provider.notes().write(
            NoteDraft(title=f"note {i}", body=f"규칙 {i} " + "내용 " * 9000, category="critical", filename=f"n{i}.md")
        )
    store = RecordStore(provider)

    def extract(instruction: str) -> Dict[str, Any]:
        import re

        return {
            "candidates": [
                {
                    "kind": "rule", "subject": f"note {m.group(2)}", "key": "", "text": f"규칙 {m.group(2)} " + "내용 " * 9000,
                    "headline": "", "source_turn": int(m.group(1)), "speaker": "user", "assertion": "directive",
                    "importance": "high",
                }
                for m in re.finditer(r"\[#(\d+) memory[^\]]*\] \(note “note (\d+)”\)", instruction)
            ]
        }

    model = FakeModel(extract=extract, judge=lambda _i: {"decisions": []})
    done: set = set()
    first = await migrate_legacy(provider, store, complete_structured=model, done=done, max_batches=2)
    assert 0 < first["remaining"] < 6 and first["finished_notes"] == 6 - first["remaining"]
    second = await migrate_legacy(provider, store, complete_structured=model, done=done, max_batches=10)
    assert second["remaining"] == 0
    for i in range(6):
        assert MIGRATED_TAG in (await provider.notes().read(f"n{i}.md")).tags
    assert not done  # finished notes leave the progress set


# ── consolidation ─────────────────────────────────────────────────────


async def test_consolidation_keeps_the_strongest_and_never_retires_the_users(provider) -> None:
    store = RecordStore(provider)
    weak = await _put(store, _rec("fact", "sdk", "SDK 는 Actions 로 배포", authority="agent_inference"))
    strong = await _put(store, _rec("fact", "xgen sdk", "SDK 는 Actions 로 배포한다", authority="user_statement", aliases=["sdk"]))
    rule = await _put(store, _rec("rule", "tone", "존댓말", authority="user_directive"))
    live = await store.load()
    clusters = find_clusters(live)
    ci = next(i for i, c in enumerate(clusters) if {r.id for r in c} >= {weak.id, strong.id})

    def handler(_i: str) -> Dict[str, Any]:
        return {
            "actions": [
                {"cluster": ci, "op": "merge", "ids": [weak.id, strong.id], "canonical": weak.id, "reason": "dup"},
                {"cluster": ci, "op": "retire", "ids": [rule.id], "reason": "old"},
            ],
            "headlines": [],
        }

    rep = await consolidate(store, complete_structured=FakeModel(consolidate=handler), records=live)
    assert rep is not None and rep.merged == 1
    assert (await store.get(strong.id)).status == "active"  # the user's record remains
    assert (await store.get(weak.id)).superseded_by == strong.id
    assert (await store.get(rule.id)).status == "active"  # not in that cluster, and the user's anyway


# ── pipeline: per-conversation cursors, failure pass-over ────────────


def _extract_from_turns(instruction: str) -> Dict[str, Any]:
    import re

    rows = []
    for m in re.finditer(r"\[#(\d+) user [^\]]*\] (.+)", instruction):
        rows.append(
            {
                "kind": "fact", "subject": m.group(2)[:20], "key": "", "text": m.group(2), "headline": "",
                "source_turn": int(m.group(1)), "speaker": "user", "assertion": "statement", "importance": "medium",
            }
        )
    return {"candidates": rows}


async def test_each_conversation_keeps_its_own_cursor(provider) -> None:
    t0 = datetime(2026, 11, 1, 9, 0, tzinfo=timezone.utc)
    model = FakeModel(extract=_extract_from_turns, judge=lambda _i: {"decisions": []})
    await provider.stm().append(Turn(role="user", content="A 대화의 사실: 사과는 빨갛다", timestamp=t0 + timedelta(hours=2)))
    rep_a = await RecordsPipeline(provider, complete_structured=model, session_id="A").run()
    assert rep_a.reconcile["added"] == 1

    # Conversation B's turn is OLDER than A's newest — one shared cursor skipped it.
    other = EphemeralMemoryProvider()
    other._notes = provider._notes  # same vault, B's own short-term store
    await other.stm().append(Turn(role="user", content="B 대화의 사실: 하늘은 파랗다", timestamp=t0))
    rep_b = await RecordsPipeline(other, complete_structured=model, session_id="B").run()
    assert rep_b.reconcile["added"] == 1
    state = await RecordStore(provider).load_state()
    assert set(state["cursors"]) == {"A", "B"}

    again = await RecordsPipeline(provider, complete_structured=model, session_id="A").run()
    assert again.skipped_reason == "no_new_turns"


async def test_turns_the_model_keeps_failing_on_are_passed_over(provider) -> None:
    await provider.stm().append(Turn(role="user", content="이건 저장해 둬"))
    model = FakeModel(extract=lambda _i: None)
    for _ in range(3):
        await RecordsPipeline(provider, complete_structured=model, session_id="S").run()
    state = await RecordStore(provider).load_state()
    assert state["cursors"].get("S")  # moved past after the third failure
    assert "S" not in state.get("failures", {})


# ── retriever: standing part with the pinned tier, related per turn ───


async def test_retriever_injects_records_and_falls_back_when_assembly_fails(provider, monkeypatch) -> None:
    from xgen_agent_runtime.core.state import PipelineState
    from xgen_agent_runtime.memory.retriever import MemoryAwareRetriever

    store = RecordStore(provider)
    await _put(store, _rec("rule", "tone", "항상 존댓말로 답한다", authority="user_directive"))
    await _put(store, _rec("fact", "배포 승인", "배포 승인은 리더가 한다", authority="user_statement"))
    hooks = MemoryHooks(records_mode="on", max_inject_chars=20_000)
    retriever = MemoryAwareRetriever(provider, hooks=hooks)
    chunks = await retriever.retrieve("배포 승인 누가 해?", PipelineState())
    standing = [c for c in chunks if c.key == "memory_records"]
    related = [c for c in chunks if c.key == "memory_records_related"]
    assert standing and standing[0].metadata["layer"] == "pinned" and "존댓말" in standing[0].content
    assert related and "리더가 한다" in related[0].content
    # Records never come back a second time through the search layers.
    assert not any(c.key.startswith("mem-") for c in chunks)

    import importlib

    asm = importlib.import_module("xgen_agent_runtime.memory.records.assemble")

    async def broken(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(asm, "assemble", broken)
    await provider.notes().write(NoteDraft(title="호칭", body="리더님", category="critical", importance=Importance.HIGH))
    chunks = await MemoryAwareRetriever(provider, hooks=hooks).retrieve("안녕", PipelineState())
    assert any((c.metadata or {}).get("layer") == "pinned" and "리더님" in c.content for c in chunks)


# ── tools ─────────────────────────────────────────────────────────────


class _Ctx:
    def __init__(self, user_text: str) -> None:
        self.session_id = "conv-1"
        self.state_view = type("S", (), {"messages": [{"role": "user", "content": user_text}], "session_id": "conv-1"})()


async def _tool(provider, name: str):
    from xgen_agent_runtime.host.memory_tools import build_memory_tools

    tools = {t.name: t for t in build_memory_tools(provider, records=True)}
    return tools[name]


async def test_memory_pin_writes_a_record_whose_owner_the_conversation_decides(provider) -> None:
    pin = await _tool(provider, "memory_pin")
    r = await pin.execute({"title": "보고 형식", "content": "보고는 표로 정리한다"}, _Ctx("앞으로 보고는 표로 정리해줘"))
    out = json.loads(r.content)
    assert not r.is_error and out["status"] == "pinned"
    assert out["record"]["authority"] == "user_directive" and out["record"]["kind"] == "rule"

    # The agent's own idea cannot overwrite the user's rule.
    r2 = await pin.execute({"title": "보고 형식", "content": "보고는 글로 쓴다"}, _Ctx("고마워"))
    assert json.loads(r2.content)["status"] == "kept"
    # Pinning the same thing again in the user's words updates the same record.
    r3 = await pin.execute({"title": "보고 형식", "content": "보고는 표와 요약으로"}, _Ctx("보고는 표와 요약으로 해"))
    out3 = json.loads(r3.content)
    assert out3["status"] == "updated" and out3["id"] == out["id"]
    assert len(await RecordStore(provider).load(kinds=["rule"])) == 1


async def test_memory_record_guards_the_users_records(provider) -> None:
    store = RecordStore(provider)
    rule = await _put(store, _rec("rule", "tone", "존댓말", authority="user_directive"))
    tool = await _tool(provider, "memory_record")
    denied = await tool.execute({"action": "retire", "id": rule.id, "reason": "looks old"}, _Ctx("hi"))
    assert denied.is_error and "user" in json.loads(denied.content)["error"]
    ok = await tool.execute(
        {"action": "correct", "id": rule.id, "text": "반말", "reason": "user changed it", "user_asked": True}, _Ctx("이제 반말로 해")
    )
    assert not ok.is_error and (await store.get(rule.id)).text == "반말"
    hist = json.loads((await tool.execute({"action": "history", "id": rule.id}, _Ctx("x"))).content)
    assert [v["text"] for v in hist["versions"]] == ["존댓말", "반말"]
    listed = json.loads((await tool.execute({"action": "list", "query": "반말"}, _Ctx("x"))).content)
    assert listed["records"][0]["id"] == rule.id
    read = await (await _tool(provider, "memory_read")).execute({"filename": rule.filename}, _Ctx("x"))
    assert json.loads(read.content)["record"]["text"] == "반말"


async def test_memory_search_hides_record_internals(provider) -> None:
    store = RecordStore(provider)
    r = await _put(store, _rec("fact", "배포", "배포 승인은 리더"))
    r.text = "배포 승인은 리더가 한다"
    await store.put(r, expected_version=1)
    search = await _tool(provider, "memory_search")
    out = json.loads((await search.execute({"query": "배포 승인"}, _Ctx("x"))).content)
    names = [x["filename"] for x in out["results"]]
    assert r.filename in names and not any(n.startswith("memh-") for n in names)


# ── distillation: records replace the ledger and the evergreen ───────


class _DistillLLM:
    def __init__(self) -> None:
        self.purposes: List[str] = []

    async def complete(self, instruction: str, *, purpose: str = "") -> str:
        self.purposes.append(purpose)
        return "digest"

    async def complete_structured(self, instruction: str, schema: Dict[str, Any], *, purpose: str = "") -> Any:
        self.purposes.append(purpose)
        if schema is CANDIDATE_SCHEMA:
            return _extract_from_turns(instruction)
        if schema is JUDGE_SCHEMA:
            return {"decisions": []}
        return None


async def test_distillation_runs_the_records_pass_instead_of_the_ledger(provider, tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    from xgen_agent_runtime.host import distill
    from xgen_agent_runtime.memory import facts

    monkeypatch.setenv("XGEN_MEMORY_RECORDS", "on")

    def _no_ledger(*_a, **_k):
        raise AssertionError("the fact ledger must not run in records mode")

    monkeypatch.setattr(facts.FactExtraction, "run", _no_ledger)
    await provider.stm().append(Turn(role="user", content="회의는 매주 화요일 10시다"))
    host = SimpleNamespace(agent_vault_root=lambda wf: str(tmp_path))
    spec = SimpleNamespace(workflow_id="wf", interaction_id="conv-9", host=host)
    llm = _DistillLLM()
    report = await distill._distill_async(spec, provider, llm)
    assert report["mode"] == "records" and report["records"]["added"] == 1
    assert report["evergreen"] is False and "memory.records" in llm.purposes
    rec = (await RecordStore(provider).load(kinds=["fact"]))[0]
    assert rec.evidence[0].session == "conv-9" and rec.authority == "user_statement"


# ── the owner (memory browser) ───────────────────────────────────────


async def test_owner_edits_are_versions_with_the_users_authority(provider) -> None:
    from xgen_agent_runtime.memory.records import OwnerActionError, owner_action, owner_create, records_overview

    store = RecordStore(provider)
    guess = await _put(store, _rec("fact", "deploy day", "배포는 금요일", authority="agent_inference"))
    out = await owner_action(provider, guess.id, "correct", text="배포는 화요일", expected_version=1, reason="틀림")
    assert out["record"]["authority"] == "user_statement" and out["record"]["version"] == 2
    with pytest.raises(OwnerActionError):
        await owner_action(provider, guess.id, "retire", expected_version=1)  # the browser showed v1
    await owner_action(provider, guess.id, "retire", expected_version=2)
    assert (await store.get(guess.id)).status == "retired"
    await owner_action(provider, guess.id, "restore")
    await owner_action(provider, guess.id, "revert", version=1)
    assert (await store.get(guess.id)).text == "배포는 금요일"

    a = await _put(store, _rec("rule", "lang", "영어로", authority="user_directive", status="contested"))
    b = await _put(store, _rec("rule", "lang", "한국어로", authority="user_directive", status="contested", key="2"))
    for x, y in ((a, b), (b, a)):
        x.contested_with = [y.id]
        await store.put(x, expected_version=x.version)
    over = await records_overview(provider)
    assert {c["id"] for c in over["contested"]} == {a.id, b.id}
    settled = await owner_action(provider, b.id, "settle", reason="한국어가 맞다")
    assert settled["superseded"] == [a.id] and (await store.get(a.id)).superseded_by == b.id
    created = await owner_create(provider, kind="rule", subject="보고", text="보고는 표로")
    assert created["record"]["authority"] == "user_directive"


async def test_memory_write_keeps_plain_notes_out_of_record_categories(provider) -> None:
    write = await _tool(provider, "memory_write")
    out = json.loads((await write.execute({"title": "메모", "content": "그냥 메모", "category": "facts"}, _Ctx("x"))).content)
    assert out["category"] == "topics"


async def test_small_budget_shows_every_rule_as_a_line_before_any_in_full(provider) -> None:
    store = RecordStore(provider)
    for i in range(30):
        await _put(
            store,
            _rec("rule", f"rule {i}", f"규칙 {i} 상세 " + "설명 " * 150, headline=f"규칙 {i} 요약", authority="user_directive"),
        )
    out = await assemble(provider, query="", budget_chars=4000)
    rep = out.report
    assert not rep["rules_left"] and len(rep["rules_short"]) + len(rep["rules"]) == 30
    big = await assemble(provider, query="", budget_chars=200_000)
    assert len(big.report["rules"]) == 30  # room for all: every rule whole
