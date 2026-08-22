"""대화 자동 아카이브 — Geny conversation_archiver 형식의 세션 rollup.

Geny 의 counterpart-aware rollup 형식을 XGEN(user_chat 버킷 단일)으로 이식:
세션당 파일 하나(``<sid>__user.md``), 발화마다 ``## turn-<eid8>`` H2 앵커 +
``<!--meta ... -->`` 메타 주석 + 원문, frontmatter 에 session_id ·
date_first/date_last · turn_count · kinds · importance_max 누적. 에이전트와
나눈 대화가 **아무 도구 호출 없이도** 메모리 브라우저에 구조화되어 쌓인다.

왜 호스트에서 하나: executor 파일 provider 의 ``reflect()`` 는 의도적
no-op(LLM 없음)이고 STM(transcripts)은 세션 플레인이라 브라우저(vault)에
안 보인다 — Geny 도 conversation rollup 을 호스트(SessionMemoryManager)에서
쓴다. 여기서는 MemoryStage 의 strategy 슬롯에 들어가는
:class:`ConversationArchivingStrategy` 가 STM 기록(부모) 직후 같은 메시지를
vault 노트로도 흘린다.

중복 방지: 부모(ProviderDrivenStrategy)와 동일한 state.metadata 워터마크
방식(``_ARCHIVED_KEY``) — 새 메시지만 append 한다. 메모리 포트(DB Memory)로
preload 된 과거 대화는 agent_geny 가 두 워터마크(부모 ``_RECORDED_KEY`` 포함)를
preload 길이로 초기화해 STM/rollup 어느 쪽에도 재기록되지 않는다.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime
from typing import Any, List, Tuple

from xgen_agent_runtime.memory.provider import Importance, NoteDraft, NotePatch
from xgen_agent_runtime.memory.strategy import ProviderDrivenStrategy

from xgen_agent_runtime.host.ids import _safe_id

logger = logging.getLogger("editor.geny_bridge.conversation_archive")

#: 브라우저의 대화 카테고리 (Geny Opsidian 과 동일).
CONVERSATIONS_CATEGORY = "conversations"
#: 발화 하나당 노트에 남길 최대 길이 (rollup 은 원문 전체가 아니라 열람용).
_MAX_UTTERANCE_CHARS = 2000
#: state.metadata 워터마크 키 — 부모 전략의 _RECORDED_KEY 와 같은 패턴.
_ARCHIVED_KEY = "geny_bridge.conversation_archived_idx"
#: 부모(ProviderDrivenStrategy)의 STM 워터마크 키 (preload 스킵 초기화용).
STM_RECORDED_KEY = "memory.provider_strategy_recorded_idx"


def _block_text(content: Any) -> str:
    """Anthropic-형 message content(str | block list)에서 사람 텍스트만 추출.

    tool_use/tool_result 블록은 대화 rollup 에서 제외한다 — 도구 왕복은
    transcripts(STM)에 원본이 남고, rollup 은 사람이 읽는 대화 기록이다.
    """
    if isinstance(content, str):
        return content.strip()
    parts: List[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = str(block.get("text") or "").strip()
                if text:
                    parts.append(text)
    return "\n".join(parts).strip()


def _clip(text: str) -> str:
    if len(text) <= _MAX_UTTERANCE_CHARS:
        return text
    return text[:_MAX_UTTERANCE_CHARS].rstrip() + " …(생략)"


class ConversationArchivingStrategy(ProviderDrivenStrategy):
    """STM 기록(부모) + vault ``conversations/`` rollup 노트 append.

    노트 파일명은 ``session-<session_id>.md`` — vault 가 에이전트당 하나이므로
    세션당 정확히 한 노트가 쌓인다. 아카이브 실패는 절대 턴을 깨지 않는다.
    """

    @property
    def name(self) -> str:  # type: ignore[override]
        return "conversation_archiving"

    @property
    def description(self) -> str:  # type: ignore[override]
        return "provider_driven + vault conversations/ rollup append"

    # ── helpers ──────────────────────────────────────────────────
    @staticmethod
    def _title_slug(text: str, *, max_len: int = 28) -> str:
        """Geny title_slug 미러 — 첫 사용자 발화에서 파일명 조각 파생.

        파일명은 최초 아카이브 시 고정되고 이후 절대 안 바뀐다(위키링크
        안정성 — Geny invariant #2).
        """
        words = re.findall(r"[0-9A-Za-z가-힣]+", (text or "").lower())
        slug = "-".join(words)[:max_len].strip("-")
        return slug or "chat"

    @staticmethod
    def _session_prefix(state: Any) -> str:
        sid = _safe_id(str(getattr(state, "session_id", "") or "default"))
        return f"{sid}__user"

    async def _resolve_filename(
        self,
        provider: Any,
        state: Any,
        first_user: str,
    ) -> Tuple[str, Any]:
        """세션의 대화 노트 파일명 확정 + 기존 노트 반환.

        우선순위: ① 레거시 정확명(`<sid>__user.md`, 기존 프로드 데이터)
        ② 슬러그형 기존 노트(`<sid>__user__*` prefix 탐색) ③ 신규 생성
        (`<sid>__user__<title_slug>.md`).
        """
        notes = provider.notes()
        prefix = self._session_prefix(state)
        legacy = f"{prefix}.md"
        existing = await notes.read(legacy)
        if existing is not None:
            return legacy, existing
        try:
            # ⚠ index().list_notes 의 메타는 .filename 직접 보유(브라우즈 DTO),
            # notes.list 는 NoteMeta(.ref.filename) — 양 백엔드 공통 계약인
            # 후자를 쓴다 (전자를 쓰면 파일 백엔드에서 빈손 → 턴마다 새 노트).
            metas = await notes.list(category=CONVERSATIONS_CATEGORY)
            for m in metas:
                fname = getattr(getattr(m, "ref", None), "filename", "") or ""
                bare = fname.rsplit("/", 1)[-1]
                if bare.startswith(prefix + "__"):
                    return bare, await notes.read(bare)
        except Exception:  # noqa: BLE001 — 탐색 실패 시 신규 생성으로 진행
            logger.debug("conversation filename lookup failed", exc_info=True)
        return f"{prefix}__{self._title_slug(first_user)}.md", None

    @staticmethod
    def _new_utterances(state: Any) -> Tuple[List[Tuple[str, str]], int]:
        """워터마크 이후의 (role, text) 목록과 새 워터마크."""
        messages = list(getattr(state, "messages", []) or [])
        archived_upto = int(state.metadata.get(_ARCHIVED_KEY, 0))
        fresh: List[Tuple[str, str]] = []
        for msg in messages[archived_upto:]:
            role = str(msg.get("role", "")) if isinstance(msg, dict) else ""
            if role not in ("user", "assistant"):
                continue
            text = _block_text(msg.get("content") if isinstance(msg, dict) else None)
            if text:
                fresh.append((role, text))
        return fresh, len(messages)

    async def _archive(self, state: Any) -> None:
        provider = self._provider
        if provider is None:
            return
        fresh, watermark = self._new_utterances(state)
        if not fresh:
            state.metadata[_ARCHIVED_KEY] = watermark
            return

        now = datetime.now()
        stamp_iso = now.isoformat(timespec="seconds")
        day = now.strftime("%Y-%m-%d")
        sid = str(getattr(state, "session_id", "") or "default")

        # Geny rollup 형식: 발화마다 H2 앵커 + meta 주석 + 원문.
        blocks: List[str] = []
        kinds_new = set()
        event_ids_new: List[str] = []
        for role, text in fresh:
            eid8 = uuid.uuid4().hex[:8]
            event_ids_new.append(eid8)
            kind = "user_chat" if role == "user" else "assistant_chat"
            kinds_new.add(kind)
            blocks.append(
                f"## turn-{eid8}\n\n"
                f"<!--meta\n"
                f"event_id: {eid8}\n"
                f"ts: {stamp_iso}\n"
                f"kind: {kind}\n"
                f"role: {role}\n"
                f"content_chars: {len(text)}\n"
                f"-->\n\n"
                f"{_clip(text)}\n\n---"
            )
        body_block = "\n\n".join(blocks)

        first_user = next((t for r, t in fresh if r == "user"), fresh[0][1])
        filename, existing = await self._resolve_filename(provider, state, first_user)
        notes = provider.notes()
        if existing is None:
            title = f"대화: {first_user[:48]}" + ("…" if len(first_user) > 48 else "")
            await notes.write(
                NoteDraft(
                    title=title,
                    body=body_block,
                    category=CONVERSATIONS_CATEGORY,
                    tags=["conversation", "auto"],
                    importance=Importance.LOW,
                    filename=filename,
                    frontmatter={
                        "session_id": sid,
                        "date_first": day,
                        "date_last": day,
                        "turn_count": len(fresh),
                        "kinds": sorted(kinds_new),
                        "counterparts": ["user"],
                        "event_ids": event_ids_new[:200],
                    },
                )
            )
        else:
            # frontmatter 는 patch 시 교체 — 기존 값을 읽어 누적 병합.
            fm = dict(getattr(existing, "frontmatter", {}) or {})
            fm["session_id"] = fm.get("session_id") or sid
            fm.setdefault("date_first", day)
            fm["date_last"] = day
            fm["turn_count"] = int(fm.get("turn_count") or 0) + len(fresh)
            fm["kinds"] = sorted(set(fm.get("kinds") or []) | kinds_new)
            fm.setdefault("counterparts", ["user"])
            fm["event_ids"] = (list(fm.get("event_ids") or []) + event_ids_new)[:200]
            await notes.update(
                filename,
                NotePatch(
                    append_body="\n\n" + body_block,
                    frontmatter=fm,
                ),
            )
        state.metadata[_ARCHIVED_KEY] = watermark

    # ── strategy hook ────────────────────────────────────────────
    #: MemoryStage 경로 전체 상한(s) — 저장 백엔드가 어떤 상태든(테이블 미생성,
    #: DB 지연, 재시도 폭주) 턴 꼬리를 이 이상 잡을 수 없다.
    UPDATE_TIMEOUT_S = 20.0

    async def _update_inner(self, state: Any) -> None:
        await super().update(state)  # STM 기록 — 기존 동작 유지
        try:
            await self._archive(state)
        except Exception:  # noqa: BLE001 — 아카이브는 턴을 깨지 않는다
            logger.debug("conversation archive failed (turn unaffected)", exc_info=True)

    async def update(self, state: Any) -> None:  # type: ignore[override]
        import asyncio

        try:
            await asyncio.wait_for(self._update_inner(state), timeout=self.UPDATE_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.warning(
                "memory update timed out (>%ss) — 이번 턴 기록 스킵 (턴 무영향)",
                self.UPDATE_TIMEOUT_S,
            )
        except Exception:  # noqa: BLE001
            logger.debug("memory update failed (turn unaffected)", exc_info=True)
