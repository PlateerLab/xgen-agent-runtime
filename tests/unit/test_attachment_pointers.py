"""첨부 포인터 — 지난 턴·새 대화에서도 첨부 파일의 자리를 잃지 않는다.

2026-09-29 dev(gpt-5.4): 첫 턴에 첨부한 PDF 두 개는 작업 폴더에 그대로 있었는데,
① 지난 턴 첨부가 ``[Current-turn attachment: …]`` 로 재생되자 모델이 "지금 대화에서는 원문을 다시
열 수 없다" 고 했고, ② 긴 턴 뒤 첫 지시가 단기 기억 창에서 대화로 강등되며 file 블록이 버려져
경로가 사라졌고, ③ 새 대화는 대화 노트에 경로가 없어 원문 자리를 몰랐다.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, List

from xgen_agent_runtime.core.file_blocks import POINTER_PREFIX, file_block_pointer
from xgen_agent_runtime.host.conversation_archive import ConversationArchivingStrategy
from xgen_agent_runtime.llm_client.translators._canonical import (
    _user_content_to_openai_parts,
    canonical_messages_to_anthropic,
)
from xgen_agent_runtime.memory.providers.ephemeral import EphemeralMemoryProvider
from xgen_agent_runtime.memory.short_term_window import WindowConfig, build_window

ABS = "/xgeny/workspace/workflow/wf_x/workspace/uploads/users_1/chat_a/0a85-report.pdf"
REL = "uploads/users_1/chat_a/0a85-report.pdf"
FILE = {
    "type": "file",
    "name": "report.pdf",
    "mime_type": "application/pdf",
    "path": ABS,
    "workspace_path": REL,
}


def _t(role: str, content: Any) -> SimpleNamespace:
    return SimpleNamespace(role=role, content=content)


def test_pointer_names_the_saved_location_not_the_turn():
    line = file_block_pointer(FILE)
    assert line.startswith(POINTER_PREFIX)
    assert ABS in line and "working folder" in line
    assert "Current-turn" not in line
    rel = file_block_pointer({**FILE, "path": None})
    assert REL in rel and "relative to your working folder" in rel


def test_every_backend_translation_uses_the_same_pointer():
    openai = _user_content_to_openai_parts([FILE, {"type": "text", "text": "summarize"}])
    assert openai[0] == {"type": "text", "text": file_block_pointer(FILE)}
    anthropic = canonical_messages_to_anthropic([{"role": "user", "content": [FILE]}])
    assert anthropic[0]["content"][0]["text"] == file_block_pointer(FILE)


def _attached_turn() -> List[Any]:
    return [
        _t("user", [FILE, {"type": "text", "text": "read the attachment and summarize"}]),
        _t("assistant", [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": f"cat {ABS}"}}]),
        _t("user", [{"type": "tool_result", "tool_use_id": "t1", "content": "text of the pdf"}]),
        _t("assistant", [{"type": "text", "text": "summary"}]),
    ]


def _plain_turn(n: int, *, big: int = 0) -> List[Any]:
    tid = f"t{n}"
    return [
        _t("user", f"request {n}"),
        _t("assistant", [{"type": "tool_use", "id": tid, "name": "WebFetch", "input": {"url": "u"}}]),
        _t("user", [{"type": "tool_result", "tool_use_id": tid, "content": "x" * big or "ok"}]),
        _t("assistant", [{"type": "text", "text": f"answer {n}"}]),
    ]


def test_dialogue_turn_keeps_the_attachment_path():
    # 첨부 턴이 먼 턴(대화만)이 된다 — 도구 블록은 빠져도 경로 한 줄은 남는다.
    rows = _attached_turn() + _plain_turn(2) + _plain_turn(3)
    msgs, report = build_window(rows, WindowConfig())
    assert report.dialogue == 1
    first = msgs[0]
    assert first["role"] == "user" and isinstance(first["content"], str)
    assert ABS in first["content"] and "read the attachment" in first["content"]


def test_budget_demotion_keeps_the_attachment_path():
    # 실측 모양: 첨부 턴 뒤에 결과가 큰 턴 — 예산 때문에 첨부 턴이 대화로 강등된다.
    rows = _attached_turn() + _plain_turn(2, big=60_000) + _plain_turn(3, big=60_000)
    msgs, report = build_window(rows, WindowConfig())
    assert "demote_oldest_full" in report.degraded or report.dialogue >= 1
    assert any(ABS in (m["content"] if isinstance(m["content"], str) else "") for m in msgs)


def test_conversation_note_keeps_path_but_title_stays_human():
    provider = EphemeralMemoryProvider()
    strategy = ConversationArchivingStrategy(provider)
    state = SimpleNamespace(
        session_id="chat_a",
        metadata={},
        messages=[
            {"role": "user", "content": [FILE, {"type": "text", "text": "read the attachment and summarize"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "summary"}]},
        ],
    )
    asyncio.run(strategy._archive(state))
    notes = asyncio.run(provider.notes().list(category="conversations"))
    assert len(notes) == 1
    note = asyncio.run(provider.notes().read(notes[0].ref.filename))
    assert ABS in note.body
    assert POINTER_PREFIX not in (note.title or "")
    assert "attached-file" not in notes[0].ref.filename.lower()


def test_turn_out_of_window_still_leaves_its_attachment_path():
    # 첨부 턴이 5턴 창 밖으로 밀려도 경로(내용 아님)는 창 첫 사용자 메시지에 넘어온다 — 한 번만.
    rows = _attached_turn() + [r for n in range(2, 8) for r in _plain_turn(n)]
    msgs, report = build_window(rows, WindowConfig())
    assert report.turns == 5
    first_user = next(m for m in msgs if m["role"] == "user")
    text = first_user["content"] if isinstance(first_user["content"], str) else str(first_user["content"])
    assert "[Earlier in this conversation]" in text and ABS in text
    assert sum(str(m["content"]).count(ABS) for m in msgs) == 1
    assert "text of the pdf" not in str(msgs)
