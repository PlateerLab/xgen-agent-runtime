"""Memory records at scale — a synthetic bench on the real code (no model, no network).

    python scripts/bench_memory_records.py [--sizes 300,1000,3000,10000] [--store file|ephemeral]

What it measures, per memory size:

* the per-turn block (``assemble``) for 128k / 200k / 1M windows: time, size against its budget, how
  many rules / state lines were shown whole, as a headline, or only counted in the index;
* the write path's bounded inputs: catalog size of an extraction prompt, neighbour lookup per candidate;
* a consolidation scan (``find_clusters``);
* migrating an old memory shaped like a large production agent (ledger of 200 facts, a long evergreen,
  dozens of pinned notes) without a model, and what the old pinned layer showed of it at the same budget.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

from xgen_agent_runtime.host.context_budget import memory_inject_cap_chars
from xgen_agent_runtime.memory.provider import (
    DEFAULT_MAX_INJECT_CHARS,
    Importance,
    MemoryHooks,
    NoteDraft,
)
from xgen_agent_runtime.memory.records import (
    MemoryRecord,
    RecordStore,
    assemble,
    find_clusters,
    migrate_legacy,
)
from xgen_agent_runtime.memory.records.extract import Candidate, IndexedTurn, build_catalog
from xgen_agent_runtime.memory.records.reconcile import Reconciler

WINDOWS = {"128k": 128_000, "200k": 200_000, "1M": 1_000_000}
_WORDS = (
    "배포 승인 릴리즈 태그 로그 원본 열람 사유 다운로드 검색 에이전트 메모리 규칙 상태 결정 절차 "
    "runtime sdk workflow frontend gitlab github jenkins docker kubernetes approval release table "
    "보고서 형식 표 요약 회의 일정 고객 계약 정산 자산 포트폴리오 수익률 리스크 한도 승인권자"
).split()


_SYL = "가나다라마바사아자차카타파하고노도로모보소오조초코토포호구누두루무부수우주추쿠투푸후기니디리미비시이지치키티피히"


def _vocab(n: int = 4000) -> list:
    """Common work words plus many rarer ones (names, codes, terms) — memory text is not a 60-word language."""
    rng = random.Random(0)
    words = list(_WORDS)
    while len(words) < n:
        if rng.random() < 0.5:
            words.append("".join(rng.choice(_SYL) for _ in range(rng.randint(2, 4))))
        else:
            words.append(
                "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3, 9)))
            )
    return words


_VOCAB = _vocab()
_ZIPF = [1.0 / (i + 1) ** 1.05 for i in range(len(_VOCAB))]


def _sentence(rng: random.Random, n: int) -> str:
    return " ".join(rng.choices(_VOCAB, weights=_ZIPF, k=n))


def _provider(kind: str, root: Path) -> Any:
    if kind == "ephemeral":
        from xgen_agent_runtime.memory.providers.ephemeral import EphemeralMemoryProvider

        return EphemeralMemoryProvider()
    from xgen_agent_runtime.memory.provider import Scope
    from xgen_agent_runtime.memory.providers.file import FileMemoryProvider

    return FileMemoryProvider(root=root, scope=Scope.SESSION, timezone_name="UTC")


async def _seed(store: RecordStore, n: int, rng: random.Random) -> None:
    projects = [f"project-{i}" for i in range(12)]
    kinds = ("rule",) * 2 + ("fact",) * 5 + ("state",) * 2 + ("decision",) + ("procedure",)
    for i in range(n):
        kind = rng.choice(kinds)
        project = rng.choice(projects) if rng.random() < 0.6 and kind != "rule" else ""
        long = rng.random() < 0.15
        rec = MemoryRecord(
            id="",
            kind=kind,
            subject=f"{_sentence(rng, 2)} {i}",
            key=_sentence(rng, 1) if kind in ("fact", "state") else "",
            text=_sentence(rng, rng.randint(60, 220) if long else rng.randint(8, 40)),
            headline=_sentence(rng, 8) if long else "",
            project=project,
            scope=f"project:{project}" if project else "agent",
            applies_when=[_sentence(rng, 2)] if kind == "rule" and rng.random() < 0.5 else [],
            authority=rng.choice(
                ("user_directive", "user_statement", "tool_result", "agent_inference", "migrated")
            ),
            importance=rng.choice(("critical", "high", "medium", "medium", "low")),
        )
        rec.id = f"{kind[0]}-{i}"
        await store.put(rec, expected_version=None)


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 1)


async def bench_size(n: int, store_kind: str) -> Dict[str, Any]:
    rng = random.Random(n)
    with tempfile.TemporaryDirectory() as td:
        provider = _provider(store_kind, Path(td))
        if hasattr(provider, "initialize"):
            await provider.initialize()
        store = RecordStore(provider)
        t = time.perf_counter()
        await _seed(store, n, rng)
        seed_s = time.perf_counter() - t
        # What the background pass leaves for the per-turn index.
        state = await store.load_state()
        state["counts"] = await store.counts()
        await store.save_state(state)

        t = time.perf_counter()
        live = await store.load()
        load_ms = _ms(time.perf_counter() - t)
        out: Dict[str, Any] = {
            "records": n,
            "seed_s": round(seed_s, 2),
            "load_all_ms": load_ms,
            "windows": {},
        }

        hooks = MemoryHooks()
        for label, window in WINDOWS.items():
            inject = memory_inject_cap_chars(window, DEFAULT_MAX_INJECT_CHARS)
            budget = int(inject * hooks.layer_budget_ratio["records"])
            times: List[float] = []
            res = None
            for q in range(5):
                query = f"{_sentence(rng, 6)} project-{q}"
                t = time.perf_counter()
                res = await assemble(
                    provider, query=query, context=f"project-{q} 진행 상황", budget_chars=budget
                )
                times.append(time.perf_counter() - t)
            assert res is not None
            r = res.report
            out["windows"][label] = {
                "budget": budget,
                "chars": r["chars"],
                "within": r["chars"] <= budget,
                "assemble_ms_median": _ms(statistics.median(times)),
                "rules_whole/headline/indexed": [
                    len(r["rules"]),
                    len(r["rules_short"]),
                    len(r["rules_left"]),
                ],
                "state_whole/headline/indexed": [
                    len(r["state"]),
                    len(r["state_short"]),
                    len(r["state_left"]),
                ],
                "related_shown": len(r["related"]) + len(r["related_short"]),
            }

        text = _sentence(rng, 400)
        t = time.perf_counter()
        catalog = build_catalog(live, text)
        out["extraction_catalog_chars"] = len(catalog)
        out["catalog_ms"] = _ms(time.perf_counter() - t)

        rec = Reconciler(store, complete_structured=None, session="bench")
        cands = [
            Candidate(
                kind=rng.choice(("fact", "state", "rule")),
                subject=_sentence(rng, 2),
                key="",
                text=_sentence(rng, 20),
                headline="",
                aliases=[],
                project="",
                applies_when=[],
                importance="medium",
                authority="user_statement",
                decision_status="none",
                turn=IndexedTurn(1, "user", "x", ""),
                evidence="",
                adjusted="",
            )
            for _ in range(20)
        ]
        t = time.perf_counter()
        for c in cands:
            rec.neighbours(c, live)
        out["neighbours_ms_per_candidate"] = _ms((time.perf_counter() - t) / len(cands))

        t = time.perf_counter()
        clusters = find_clusters(live)
        out["find_clusters_ms"] = _ms(time.perf_counter() - t)
        out["clusters"] = len(clusters)

        # One background pass without the model's time: 6 new turns → 12 candidates → judged → written.
        out["pass_overhead_s"] = round(await _pass_overhead(provider, rng), 2)
        await provider.close()
    return out


async def _pass_overhead(provider: Any, rng: random.Random) -> float:
    from datetime import datetime, timedelta, timezone

    from xgen_agent_runtime.memory.provider import Turn
    from xgen_agent_runtime.memory.records import RecordsPipeline
    from xgen_agent_runtime.memory.records.extract import CANDIDATE_SCHEMA

    t0 = datetime.now(timezone.utc)
    for i in range(6):
        await provider.stm().append(
            Turn(role="user", content=_sentence(rng, 30), timestamp=t0 + timedelta(seconds=i))
        )

    async def model(instruction: str, schema: Dict[str, Any]) -> Any:
        if schema is CANDIDATE_SCHEMA:
            return {
                "candidates": [
                    {
                        "kind": rng.choice(("fact", "state", "rule")),
                        "subject": _sentence(rng, 2),
                        "key": "",
                        "text": _sentence(rng, 20),
                        "headline": "",
                        "source_turn": 1 + j % 6,
                        "speaker": "user",
                        "assertion": "directive",
                        "importance": "medium",
                    }
                    for j in range(12)
                ]
            }
        if "decisions" in schema.get("properties", {}):
            return {"decisions": []}
        return {"actions": [], "headlines": []}

    t = time.perf_counter()
    await RecordsPipeline(provider, complete_structured=model, session_id="bench").run()
    return time.perf_counter() - t


async def bench_migration(store_kind: str) -> Dict[str, Any]:
    """An old memory shaped like a large production agent, moved over without a model."""
    rng = random.Random(7)
    with tempfile.TemporaryDirectory() as td:
        provider = _provider(store_kind, Path(td))
        if hasattr(provider, "initialize"):
            await provider.initialize()
        notes = provider.notes()
        facts = [
            {
                "id": f"{_sentence(rng, 1)}.{i}",
                "statement": _sentence(rng, rng.randint(10, 60)),
                "kind": rng.choice(
                    ("identity", "preference", "relationship", "commitment", "context", "knowledge")
                ),
                "importance": rng.choice(("critical", "high", "medium")),
            }
            for i in range(200)
        ]
        await notes.write(
            NoteDraft(
                title="Fact Ledger",
                body="\n".join(f"- {f['statement']}" for f in facts),
                category="critical",
                filename="__facts__.md",
                importance=Importance.CRITICAL,
                frontmatter={"facts_json": json.dumps(facts, ensure_ascii=False)},
            )
        )
        await notes.write(
            NoteDraft(
                title="Evergreen",
                body="\n\n".join(_sentence(rng, 120) for _ in range(40)),
                category="critical",
                filename="__evergreen__.md",
                importance=Importance.CRITICAL,
            )
        )
        for i in range(40):
            await notes.write(
                NoteDraft(
                    title=f"pinned {i}",
                    body=_sentence(rng, rng.randint(20, 400)),
                    category="critical",
                    filename=f"pinned-{i}.md",
                    importance=Importance.HIGH,
                )
            )
        legacy_chars = 0
        for m in await notes.list(category="critical"):
            legacy_chars += len((await notes.read(m.ref.filename)).body)

        old: Dict[str, Any] = {}
        for label, window in WINDOWS.items():
            cap = int(memory_inject_cap_chars(window, DEFAULT_MAX_INJECT_CHARS) * 0.30)
            shown = len(await notes.load_pinned(category="critical", max_chars=cap) or "")
            old[label] = {
                "budget": cap,
                "shown_chars": shown,
                "of_legacy": round(shown / legacy_chars, 3),
            }

        store = RecordStore(provider)
        t = time.perf_counter()
        rep = await migrate_legacy(provider, store, complete_structured=None, done=set())
        mig_s = time.perf_counter() - t
        new: Dict[str, Any] = {}
        hooks = MemoryHooks()
        for label, window in WINDOWS.items():
            budget = int(
                memory_inject_cap_chars(window, DEFAULT_MAX_INJECT_CHARS)
                * hooks.layer_budget_ratio["records"]
            )
            res = await assemble(provider, query=_sentence(rng, 6), budget_chars=budget)
            r = res.report
            new[label] = {
                "budget": budget,
                "chars": r["chars"],
                "rules_whole/headline/indexed": [
                    len(r["rules"]),
                    len(r["rules_short"]),
                    len(r["rules_left"]),
                ],
                "legacy_still_shown": len(r["legacy"]),
            }
        await provider.close()
    return {
        "legacy_notes": 42,
        "legacy_chars": legacy_chars,
        "old_pinned_layer": old,
        "migration": {"mode": rep["mode"], "records": rep["records"], "seconds": round(mig_s, 2)},
        "records_block_after": new,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="300,1000,3000,10000")
    ap.add_argument("--store", default="file", choices=("file", "ephemeral"))
    args = ap.parse_args()
    results = {"store": args.store, "sizes": [], "migration": await bench_migration(args.store)}
    for n in (int(x) for x in args.sizes.split(",") if x.strip()):
        results["sizes"].append(await bench_size(n, args.store))
        print(json.dumps(results["sizes"][-1], ensure_ascii=False), flush=True)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
