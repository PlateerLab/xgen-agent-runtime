"""턴-종료 메모리 증류 — Geny 의 compact_now 케이던스를 XGEN 턴 모델로 이식.

Geny 는 상주 세션의 유휴 틱(60s)마다 ``compact_now()`` 를 돌려 ① Fact
Ledger(원자 사실 원장) ② L1 롤링 다이제스트 ③ L2 데일리 ④ L3 에버그린을
갱신한다. XGEN agent_geny 는 턴 단위 실행이라 유휴 틱이 없으므로 **턴 종료
직후 백그라운드 데몬 스레드**에서 같은 파이프라인을 돌린다:

  - 응답 스트림이 이미 끝난 뒤라 사용자 체감 지연 0.
  - 스레드는 자기 이벤트 루프에서 **새 provider 를 만들어 쓰고 닫는다**
    (같은-루프 수명 계약 준수; 파일 스토어는 원자적 쓰기라 다음 턴과 동시
    접근에도 안전 — Geny 도 유휴 컴팩션이 채팅 턴과 상시 병행).
  - 워크플로우당 in-flight 1개 (연타 턴은 스킵 — Geny 틱의 coalesce 동일).

LLM 콜 예산: 턴당 2콜 (facts 1 + 다이제스트 1). daily 는 다이제스트를 코드
렌더로 재사용(0콜), evergreen 은 pass 카운터 임계(기본 5 pass)마다 +1콜.

계약(executor facts.py/rollup.py): "LLM 이 판단하고, 스키마가 구속하고,
코드가 저장한다" — 스키마 위반/실패는 이전 상태를 절대 훼손하지 않는다.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger("editor.geny_bridge.distill")

#: 에버그린(L3) 병합 케이던스 — 증류 pass 횟수 기준 (Geny 의 "느린 케이던스").
EVERGREEN_EVERY_PASSES = 5
#: 증류 상태 스탬프 (pass 카운터 + 최근 실행 기록) — vault 루트에 보관.
DISTILL_STATE_FILENAME = "_distill_state.json"
#: 스테이지별 LLM 타임아웃(s) — 행이 걸리면 in-flight 가드가 영구 잠기는 것 방지.
FACTS_TIMEOUT_S = 90.0
ROLLUP_TIMEOUT_S = 180.0

#: 워크플로우당 in-flight 증류 1개 (연타 coalesce).
_inflight: set = set()
_inflight_lock = threading.Lock()


@dataclass(frozen=True)
class DistillSpec:
    """턴이 넘겨주는 증류 입력 — 자격증명은 그 턴의 LLM 그대로."""

    workflow_id: str
    interaction_id: str
    provider: str
    model: str
    api_key: str
    base_url: Optional[str] = None
    # claude_code 전용 — 노드(_build_cli_runtime 규약)가 해석한 인증 채널.
    # 구독(setup_token) 모드도 증류가 돌아야 한다(Geny 프로드 동작 미러).
    # 본 spec 은 메모리 상주 전용 — _distill_state.json 에 절대 저장 금지.
    cli_auth_mode: str = ""
    cli_oauth_token: str = ""
    cli_binary_path: str = ""
    #: bedrock/vertex 등 다중 필드 자격증명 (runner.build_client credentials
    #: 계약) — 메모리 상주 전용, _distill_state.json 저장 금지는 동일.
    credentials: Optional[Dict[str, Any]] = None
    #: HostServices — vault 경로/증류 LLM/메모리 provider 를 host 로 얻는다
    #: (서버=editor, 커넥터=degrade). 메모리 상주 전용, 상태파일 저장 금지.
    host: Any = None


# ── pass 카운터 (에버그린 케이던스) ────────────────────────────────


def _state_path(workflow_id: str, host: Any) -> str:
    # vault 루트는 host 로 얻는다(서버=editor vault, 커넥터=로컬 .memory).
    return os.path.join(host.agent_vault_root(workflow_id), DISTILL_STATE_FILENAME)


def _load_state(workflow_id: str, host: Any) -> Dict[str, Any]:
    try:
        with open(_state_path(workflow_id, host), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(workflow_id: str, state: Dict[str, Any], host: Any) -> None:
    try:
        path = _state_path(workflow_id, host)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except OSError:
        logger.debug("distill state save failed", exc_info=True)


# ── 증류 본체 ─────────────────────────────────────────────────────


async def _distill_async(spec: DistillSpec, provider: Any, llm: Any) -> Dict[str, Any]:
    """facts → rollup(L1) → daily(L2, 0콜) → evergreen(L3, 임계) 순서.

    Geny compact_now 와 동일한 순서 — 방금 진술된 내구 사실이 내러티브 압축
    전에 원장에 먼저 박히도록 facts 가 선행한다.
    """
    from xgen_agent_runtime.memory import FactExtraction, MemoryRollup

    report: Dict[str, Any] = {
        "facts_changes": 0,
        "digest": False,
        "daily": False,
        "evergreen": False,
    }

    # ① Fact Ledger — 스키마 구속 추출 (critical/__facts__.md, 매 턴 주입 대상)
    try:
        import asyncio as _asyncio

        fact_report = await _asyncio.wait_for(
            FactExtraction(provider, complete_structured=llm.complete_structured).run(),
            timeout=FACTS_TIMEOUT_S,
        )
        if fact_report.ran:
            report["facts_changes"] = fact_report.changes
            if fact_report.changes:
                logger.info(
                    "distill: fact ledger updated (workflow=%s, %d change(s), %d active)",
                    spec.workflow_id,
                    fact_report.changes,
                    fact_report.active_facts,
                )
        elif fact_report.skipped_reason:
            logger.debug("distill: facts skipped (%s)", fact_report.skipped_reason)
    except Exception:  # noqa: BLE001 — facts 는 best-effort (Geny 동일)
        logger.warning("distill: fact extraction failed", exc_info=True)

    # ② L1 롤링 다이제스트 + ③ 데일리(코드 렌더) + ④ 에버그린(임계)
    state = _load_state(spec.workflow_id, spec.host)
    passes = int(state.get("passes", 0)) + 1
    last_evergreen = int(state.get("last_evergreen_pass", 0))
    run_evergreen = (passes - last_evergreen) >= EVERGREEN_EVERY_PASSES

    async def _summarize(instruction: str) -> str:
        return await llm.complete(instruction, purpose="memory.rollup")

    async def _summarize_structured(instruction: str, schema: Dict[str, Any]):
        return await llm.complete_structured(instruction, schema, purpose="memory.rollup")

    try:
        rollup = MemoryRollup(
            provider,
            summarize=_summarize,
            complete_structured=_summarize_structured,
        )
        rollup_report = await _asyncio.wait_for(
            rollup.run(
                evergreen=run_evergreen,
                daily_key=datetime.now().strftime("%Y-%m-%d"),
            ),
            timeout=ROLLUP_TIMEOUT_S,
        )
        report["digest"] = bool(getattr(rollup_report, "segment_written", False))
        report["daily"] = bool(getattr(rollup_report, "daily_written", False))
        report["evergreen"] = bool(getattr(rollup_report, "evergreen_written", False))
    except Exception:  # noqa: BLE001
        logger.warning("distill: rollup failed", exc_info=True)

    # ③.5 벡터 백필 — 자동 색인 도입 전/임베딩 모델 전환 후 미색인 노트를
    # 점진 색인 (호출당 100개, 실패 무해). 시맨틱 검색이 기존 기억도 커버.
    try:
        vector = provider.vector()
        if vector is not None and hasattr(vector, "index_missing"):
            backfilled = await _asyncio.wait_for(
                vector.index_missing(provider.notes()),
                timeout=60.0,
            )
            if backfilled:
                report["vector_backfill"] = backfilled
                logger.info("distill: 벡터 백필 %d개 (workflow=%s)", backfilled, spec.workflow_id)
    except Exception:  # noqa: BLE001
        logger.debug("distill: vector backfill failed", exc_info=True)

    state["passes"] = passes
    if report["evergreen"]:
        state["last_evergreen_pass"] = passes
    _save_state(spec.workflow_id, state, spec.host)
    return report


def run_distillation(spec: DistillSpec, llm: Any = None) -> Optional[Dict[str, Any]]:
    """동기 증류 1 pass — 호출 스레드에 자기 루프를 만들어 돌리고 정리한다.

    ``llm`` 주입은 테스트용(스크립트된 MemoryLLM). None 이면 스펙 자격증명으로
    구성하고, 구성 불가(예: claude_code 구독 모드)면 조용히 스킵한다.
    """
    import asyncio

    if llm is None:
        llm = spec.host.build_turn_memory_llm(
            spec.provider,
            spec.model,
            spec.api_key,
            spec.base_url,
            cli_auth_mode=spec.cli_auth_mode,
            cli_oauth_token=spec.cli_oauth_token,
            cli_binary_path=spec.cli_binary_path,
            credentials=spec.credentials,
        )
    if llm is None:
        return None

    provider = spec.host.build_memory_provider(spec.workflow_id, spec.interaction_id)
    if provider is None:
        return None

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_distill_async(spec, provider, llm))
    finally:
        try:
            loop.run_until_complete(provider.close())
        except Exception:  # noqa: BLE001
            pass
        try:
            aclose = getattr(llm.client, "aclose", None)
            if callable(aclose):
                loop.run_until_complete(aclose())
        except Exception:  # noqa: BLE001
            pass
        try:
            if callable(getattr(llm, "cleanup", None)):
                llm.cleanup()
        except Exception:  # noqa: BLE001
            pass
        loop.close()


def launch_distillation(spec: DistillSpec) -> bool:
    """백그라운드 데몬 스레드로 증류 발사 (fire-and-forget).

    같은 워크플로우의 증류가 이미 도는 중이면 스킵(coalesce) — 다음 턴이
    어차피 그 사이 STM 을 커서 기준으로 이어서 증류한다. 반환값은 발사 여부.
    """
    wf = spec.workflow_id
    with _inflight_lock:
        if wf in _inflight:
            logger.debug("distill: already in flight (workflow=%s) — coalesced", wf)
            return False
        _inflight.add(wf)

    def _worker() -> None:
        state = _load_state(wf, spec.host)
        state["last_launch"] = datetime.now().isoformat(timespec="seconds")
        state["last_status"] = "running"
        _save_state(wf, state, spec.host)
        try:
            report = run_distillation(spec)
            state = _load_state(wf, spec.host)
            if report:
                state["last_status"] = "ok"
                state["last_error"] = None
                state["last_report"] = report
                logger.info(
                    "distill: pass done (workflow=%s, facts=%s, digest=%s, daily=%s, evergreen=%s)",
                    wf,
                    report["facts_changes"],
                    report["digest"],
                    report["daily"],
                    report["evergreen"],
                )
            else:
                # LLM 구성 불가(claude_code 구독 등) / provider 실패 — 스킵 기록
                state["last_status"] = "skipped"
                state["last_error"] = "llm_or_provider_unavailable"
            _save_state(wf, state, spec.host)
        except Exception as exc:  # noqa: BLE001 — 백그라운드; 절대 전파 금지
            logger.warning("distill: background pass failed (workflow=%s)", wf, exc_info=True)
            state = _load_state(wf, spec.host)
            state["last_status"] = "error"
            state["last_error"] = f"{type(exc).__name__}: {exc}"[:500]
            _save_state(wf, state, spec.host)
        finally:
            with _inflight_lock:
                _inflight.discard(wf)

    threading.Thread(target=_worker, name=f"geny-distill-{wf[:8]}", daemon=True).start()
    return True
