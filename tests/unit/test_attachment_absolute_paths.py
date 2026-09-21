"""첨부 경로는 한 가지 방식으로만 말한다 — 세션 절대 경로.

2026-09-21 실측: 첨부는 워크스페이스 상대(``uploads/…``)로 들어오는데 파일 도구는 절대
경로를 요구했다. 기준을 모델이 정하게 두었더니 작업 폴더(``…/<wf>/workspace``)의 마지막
조각을 건너뛴 ``…/<wf>/uploads/…`` 를 만들어 샌드박스 가드에 막혔다. 파일은 제자리에
있었는데 열지 못했다.
"""

from __future__ import annotations

from xgen_agent_runtime.host.attachment_paths import (
    absolutize_attachments,
    absolutize_input,
    session_path,
)
from xgen_agent_runtime.llm_client.translators._canonical import (
    _file_block_to_text_fallback,
)
from xgen_agent_runtime.stages.s01_input.artifact.default.normalizers import (
    MultimodalNormalizer,
)
from xgen_agent_runtime.stages.s01_input.types import NormalizedInput

WORKDIR = "/xgeny/workspace/workflow/wf_1/workspace"
REL = "uploads/users_1/chat_9/c88e042416f0-deck.pptx"
ABS = f"{WORKDIR}/{REL}"


class TestSessionPath:
    def test_joins_against_the_working_folder(self):
        assert session_path(WORKDIR, REL) == ABS

    def test_keeps_the_workspace_segment(self):
        # 이 한 조각을 잃은 것이 사고의 전부였다.
        assert "/workspace/uploads/" in session_path(WORKDIR, REL)

    def test_an_absolute_reference_is_left_alone(self):
        assert session_path(WORKDIR, "/etc/hosts") == "/etc/hosts"

    def test_no_base_means_no_absolute_claim(self):
        assert session_path("", REL) == ""
        assert session_path("relative/base", REL) == ""

    def test_dot_slash_and_empty(self):
        assert session_path(WORKDIR, "./a/b.txt") == f"{WORKDIR}/a/b.txt"
        assert session_path(WORKDIR, "") == ""


class TestAbsolutizeAttachments:
    def test_fills_path_and_keeps_provenance(self):
        out = absolutize_attachments(
            [{"kind": "file", "name": "deck.pptx", "workspace_path": REL}], WORKDIR
        )
        assert out[0]["path"] == ABS
        assert out[0]["workspace_path"] == REL, "출처 기록(기억·원장)은 상대 경로 그대로"

    def test_the_original_descriptor_is_not_mutated(self):
        item = {"kind": "file", "workspace_path": REL}
        absolutize_attachments([item], WORKDIR)
        assert "path" not in item

    def test_without_a_base_nothing_is_invented(self):
        item = {"kind": "file", "workspace_path": REL}
        assert absolutize_attachments([item], "")[0] == item

    def test_non_dict_items_pass_through(self):
        assert absolutize_attachments(["x", None], WORKDIR) == ["x", None]

    def test_input_shape_covers_every_bucket(self):
        raw = {
            "input_str": "hi",
            "attachments": [{"kind": "file", "workspace_path": REL}],
            "images": [{"kind": "image", "workspace_path": "uploads/a.png"}],
        }
        out = absolutize_input(raw, WORKDIR)
        assert out["attachments"][0]["path"] == ABS
        assert out["images"][0]["path"] == f"{WORKDIR}/uploads/a.png"
        assert out["input_str"] == "hi"


class TestAttachmentIsAnnouncedByAbsolutePath:
    def test_absolute_path_is_stated_plainly(self):
        text = _file_block_to_text_fallback(
            {"type": "file", "name": "deck.pptx", "mime_type": "application/x-pptx", "path": ABS}
        )
        assert ABS in text
        assert "relative" not in text

    def test_a_relative_path_always_says_what_it_is_relative_to(self):
        text = _file_block_to_text_fallback(
            {"type": "file", "name": "deck.pptx", "workspace_path": REL}
        )
        assert REL in text
        assert "relative to your working folder" in text

    def test_a_bare_attachment_makes_no_path_claim(self):
        text = _file_block_to_text_fallback({"type": "file", "name": "deck.pptx"})
        assert "Read it at" not in text


class TestTheWholeWayThrough:
    def test_an_uploaded_file_reaches_the_prompt_as_an_absolute_path(self):
        raw = {
            "input_str": "이 파일 요약해 줘",
            "attachments": absolutize_attachments(
                [{"kind": "file", "name": "deck.pptx", "workspace_path": REL}], WORKDIR
            ),
        }
        normalized: NormalizedInput = MultimodalNormalizer().normalize(raw)
        blocks = normalized.to_message_content()
        rendered = " ".join(
            _file_block_to_text_fallback(b) for b in blocks if b.get("type") == "file"
        )
        assert ABS in rendered

    def test_an_image_attachment_names_the_same_absolute_path(self):
        raw = {
            "input_str": "",
            "attachments": absolutize_attachments(
                [
                    {
                        "kind": "image",
                        "name": "shot.png",
                        "mime_type": "image/png",
                        "workspace_path": "uploads/users_1/chat_9/shot.png",
                        "url": "file:///whatever.png",
                    }
                ],
                WORKDIR,
            ),
        }
        normalized = MultimodalNormalizer().normalize(raw)
        texts = [b.get("text", "") for b in normalized.to_message_content()]
        assert any(f"{WORKDIR}/uploads/users_1/chat_9/shot.png" in t for t in texts)


class TestOversizedResultNeverPointsOutside:
    """도달할 수 없는 경로를 "여기 저장했다" 고 말하면 모델이 그 뿌리를 학습한다."""

    def _run(self, *, storage: str, workdir: str, sandbox=None):
        from xgen_agent_runtime.stages.s10_tool.persistence import (
            maybe_persist_large_result,
        )
        from xgen_agent_runtime.tools.base import (
            ToolCapabilities,
            ToolContext,
            ToolResult,
        )

        context = ToolContext(
            session_id="s1",
            working_dir=workdir,
            storage_path=storage,
            allowed_paths=[workdir],
            sandbox=sandbox,
        )
        out = maybe_persist_large_result(
            ToolResult(content="x" * 5_000),
            tool_use_id="tu_1",
            tool_name="Bash",
            capabilities=ToolCapabilities(max_result_chars=1_000),
            context=context,
        )
        return out.display_text or ""

    def test_a_sibling_storage_dir_is_not_advertised_as_a_path(self, tmp_path):
        storage = tmp_path / "executor"
        workdir = tmp_path / "workspace"
        workdir.mkdir()
        text = self._run(storage=str(storage), workdir=str(workdir))
        assert "cannot open it" in text
        assert str(storage) not in text

    def test_a_reachable_location_is_named(self, tmp_path):
        workdir = tmp_path / "workspace"
        workdir.mkdir()
        storage = workdir / "inner"
        text = self._run(storage=str(storage), workdir=str(workdir))
        assert "Full body saved at:" in text
        assert str(storage) in text
