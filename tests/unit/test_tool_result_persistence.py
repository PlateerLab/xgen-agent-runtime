"""Phase 2 Week 4 Checkpoint 2 — tool result persistence tests."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import pytest

from xgen_agent_runtime.stages.s10_tool.persistence import (
    TOOL_RESULTS_DIRNAME,
    maybe_persist_large_result,
)
from xgen_agent_runtime.tools.base import ToolCapabilities, ToolContext, ToolResult


def _ctx(storage_path: str | None) -> ToolContext:
    return ToolContext(
        session_id="s1",
        working_dir=storage_path or "",
        storage_path=storage_path,
    )


class TestSmallPayloadPassthrough:
    def test_string_content_under_limit_unchanged(self, tmp_path):
        result = ToolResult(content="hello")
        out = maybe_persist_large_result(
            result,
            tool_use_id="t_1",
            tool_name="echo",
            capabilities=ToolCapabilities(max_result_chars=100_000),
            context=_ctx(str(tmp_path)),
        )
        assert out is result
        assert out.display_text is None
        assert out.persist_full is None
        # No tool-results dir created when nothing to persist
        assert not (tmp_path / TOOL_RESULTS_DIRNAME).exists()

    def test_dict_content_under_limit_unchanged(self, tmp_path):
        result = ToolResult(content={"ok": True, "items": [1, 2, 3]})
        out = maybe_persist_large_result(
            result,
            tool_use_id="t_2",
            tool_name="api",
            capabilities=ToolCapabilities(max_result_chars=10_000),
            context=_ctx(str(tmp_path)),
        )
        assert out is result


class TestImageBlocksNotCountedAsText:
    """Image base64 is not text — a normal-size render must reach the model (2026-10-11)."""

    @staticmethod
    def _image_result(text: str, b64_len: int) -> ToolResult:
        return ToolResult(
            content=[
                {"type": "text", "text": text},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "A" * b64_len}},
            ]
        )

    def test_large_image_with_short_text_passes_through(self, tmp_path):
        result = self._image_result("[Image file: a.png] The image is attached below.", 300_000)
        out = maybe_persist_large_result(
            result,
            tool_use_id="t_img",
            tool_name="Read",
            capabilities=ToolCapabilities(max_result_chars=100_000),
            context=_ctx(str(tmp_path)),
        )
        assert out is result
        assert not (tmp_path / TOOL_RESULTS_DIRNAME).exists()

    def test_oversized_text_next_to_image_still_persisted_without_base64(self, tmp_path):
        result = self._image_result("y" * 5000, 300_000)
        out = maybe_persist_large_result(
            result,
            tool_use_id="t_img_big",
            tool_name="Read",
            capabilities=ToolCapabilities(max_result_chars=1_000),
            context=_ctx(str(tmp_path)),
        )
        assert out is not result
        assert out.persist_full
        assert "AAAA" not in out.display_text
        # 잘렸다고 말하고, 원본을 읽는 길(id)을 준다 — 480자 미리보기만 주던 것은 2026-10-23 에 걷었다.
        assert "[tool result clipped:" in out.display_text
        assert 'ToolResultRead(id="t_img_big"' in out.display_text

    def test_list_without_image_measured_as_before(self, tmp_path):
        content = [{"type": "text", "text": "z" * 900}]
        result = ToolResult(content=content)
        out = maybe_persist_large_result(
            result,
            tool_use_id="t_txt",
            tool_name="x",
            capabilities=ToolCapabilities(max_result_chars=920),
            context=_ctx(str(tmp_path)),
        )
        # JSON rendering adds the block wrapper, which pushes 900 chars over 920 — unchanged rule.
        assert out is not result


class TestLargePayloadPersisted:
    def test_oversized_string_written_to_disk(self, tmp_path):
        big = "x" * 5000
        result = ToolResult(content=big, metadata={"rows": 1000})
        out = maybe_persist_large_result(
            result,
            tool_use_id="t_large",
            tool_name="grep",
            capabilities=ToolCapabilities(max_result_chars=1000),
            context=_ctx(str(tmp_path)),
        )
        # A new ToolResult is returned (dataclass replace)
        assert out is not result
        # persist_full points to the correct file
        expected = tmp_path / TOOL_RESULTS_DIRNAME / "t_large.json"
        assert out.persist_full == str(expected)
        assert expected.is_file()
        # Full body written as JSON envelope
        envelope: Dict[str, Any] = json.loads(expected.read_text(encoding="utf-8"))
        assert envelope["tool_use_id"] == "t_large"
        assert envelope["tool_name"] == "grep"
        assert envelope["content"] == big
        assert envelope["metadata"] == {"rows": 1000}
        # Display text says it was clipped, how big the whole is, and how to read the rest
        assert out.display_text is not None
        assert str(expected) in out.display_text
        assert "5000 chars" in out.display_text
        assert 'ToolResultRead(id="t_large"' in out.display_text
        # head and tail are both kept (the end of a Bash result holds the exit code and STDERR)
        assert out.display_text.rstrip().endswith("x")

    def test_oversized_dict_rendered_then_persisted(self, tmp_path):
        payload = {"rows": [{"i": i} for i in range(2000)]}  # >> 100 chars rendered
        result = ToolResult(content=payload)
        out = maybe_persist_large_result(
            result,
            tool_use_id="t_dict",
            tool_name="search",
            capabilities=ToolCapabilities(max_result_chars=100),
            context=_ctx(str(tmp_path)),
        )
        path = tmp_path / TOOL_RESULTS_DIRNAME / "t_dict.json"
        assert path.is_file()
        envelope = json.loads(path.read_text(encoding="utf-8"))
        assert envelope["content"] == payload
        assert out.persist_full == str(path)

    def test_original_display_text_preserved_on_persist(self, tmp_path):
        big = "y" * 5000
        result = ToolResult(content=big, display_text="12 matches across 3 files")
        out = maybe_persist_large_result(
            result,
            tool_use_id="t_disp",
            tool_name="grep",
            capabilities=ToolCapabilities(max_result_chars=1000),
            context=_ctx(str(tmp_path)),
        )
        # When the tool already has a hand-crafted display_text, we keep it
        # (the tool knows best what the LLM should see) but still persist.
        assert out.display_text == "12 matches across 3 files"
        assert out.persist_full is not None
        assert os.path.isfile(out.persist_full)


class TestCapOffAndSentinels:
    def test_zero_max_chars_disables_cap(self, tmp_path):
        result = ToolResult(content="z" * 50_000)
        out = maybe_persist_large_result(
            result,
            tool_use_id="t_zero",
            tool_name="cat",
            capabilities=ToolCapabilities(max_result_chars=0),
            context=_ctx(str(tmp_path)),
        )
        assert out is result
        assert not (tmp_path / TOOL_RESULTS_DIRNAME).exists()

    def test_negative_max_chars_treated_as_disabled(self, tmp_path):
        result = ToolResult(content="z" * 5000)
        out = maybe_persist_large_result(
            result,
            tool_use_id="t_neg",
            tool_name="cat",
            capabilities=ToolCapabilities(max_result_chars=-1),
            context=_ctx(str(tmp_path)),
        )
        assert out is result

    def test_already_persisted_is_no_op(self, tmp_path):
        prior = tmp_path / "somewhere.json"
        prior.write_text("stub")
        result = ToolResult(
            content="x" * 5000,
            persist_full=str(prior),
            display_text="preset",
        )
        out = maybe_persist_large_result(
            result,
            tool_use_id="t_dup",
            tool_name="x",
            capabilities=ToolCapabilities(max_result_chars=100),
            context=_ctx(str(tmp_path)),
        )
        assert out is result


class TestFallbacks:
    def test_no_storage_path_inlines_original(self, tmp_path, caplog):
        big = "x" * 5000
        result = ToolResult(content=big)
        caplog.set_level("WARNING")
        out = maybe_persist_large_result(
            result,
            tool_use_id="t_nopath",
            tool_name="dump",
            capabilities=ToolCapabilities(max_result_chars=1000),
            context=_ctx(None),
        )
        assert out is result
        assert any("no storage_path" in r.message for r in caplog.records)

    def test_os_error_falls_back_to_original(self, tmp_path, monkeypatch):
        big = "x" * 5000
        result = ToolResult(content=big)

        def _boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(
            "xgen_agent_runtime.stages.s10_tool.persistence.os.makedirs",
            _boom,
        )
        out = maybe_persist_large_result(
            result,
            tool_use_id="t_boom",
            tool_name="dump",
            capabilities=ToolCapabilities(max_result_chars=1000),
            context=_ctx(str(tmp_path)),
        )
        assert out is result

    def test_tool_use_id_with_unsafe_chars_sanitized(self, tmp_path):
        big = "x" * 5000
        result = ToolResult(content=big)
        out = maybe_persist_large_result(
            result,
            tool_use_id="../evil/../id",
            tool_name="dump",
            capabilities=ToolCapabilities(max_result_chars=1000),
            context=_ctx(str(tmp_path)),
        )
        # Traversal characters replaced with underscores — file must sit
        # inside the sink directory, never outside.
        assert out.persist_full is not None
        assert out.persist_full.startswith(
            str(tmp_path / TOOL_RESULTS_DIRNAME) + os.sep
        )
        # And the file should exist where we said
        assert os.path.isfile(out.persist_full)


# ─────────────────────────────────────────────────────────────────
# Integration — through SequentialExecutor / PartitionExecutor
# ─────────────────────────────────────────────────────────────────


from xgen_agent_runtime.stages.s10_tool import (  # noqa: E402
    ParallelExecutor,
    PartitionExecutor,
    SequentialExecutor,
    StreamingToolExecutor,
)
from xgen_agent_runtime.stages.s10_tool.artifact.default.routers import (  # noqa: E402
    RegistryRouter,
)
from xgen_agent_runtime.tools.base import Tool  # noqa: E402
from xgen_agent_runtime.tools.registry import ToolRegistry  # noqa: E402


class _FixedOutputTool(Tool):
    def __init__(self, name: str, payload: Any, max_chars: int = 1000):
        self._name = name
        self._payload = payload
        self._max = max_chars

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return "fixture"

    @property
    def input_schema(self):
        return {"type": "object"}

    def capabilities(self, input):
        return ToolCapabilities(
            concurrency_safe=True,
            max_result_chars=self._max,
        )

    async def execute(self, input, context):
        return ToolResult(content=self._payload)


def _make_registry(tools):
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


class TestExecutorIntegration:
    @pytest.mark.asyncio
    async def test_sequential_persists_large_payload(self, tmp_path):
        tool = _FixedOutputTool("big", "x" * 5000, max_chars=500)
        reg = _make_registry([tool])
        router = RegistryRouter(reg)
        execu = SequentialExecutor()

        ctx = _ctx(str(tmp_path))
        results = await execu.execute_all(
            [{"tool_use_id": "u1", "tool_name": "big", "tool_input": {}}],
            router,
            ctx,
        )
        assert len(results) == 1
        content = results[0]["content"]
        assert "clipped" in content and 'ToolResultRead(id="u1"' in content
        assert str(tmp_path / TOOL_RESULTS_DIRNAME / "u1.json") in content
        assert (tmp_path / TOOL_RESULTS_DIRNAME / "u1.json").is_file()

    @pytest.mark.asyncio
    async def test_parallel_persists_large_payload(self, tmp_path):
        tool = _FixedOutputTool("big", "x" * 5000, max_chars=500)
        reg = _make_registry([tool])
        router = RegistryRouter(reg)
        execu = ParallelExecutor(max_concurrency=2)

        ctx = _ctx(str(tmp_path))
        results = await execu.execute_all(
            [
                {"tool_use_id": "p1", "tool_name": "big", "tool_input": {}},
                {"tool_use_id": "p2", "tool_name": "big", "tool_input": {}},
            ],
            router,
            ctx,
        )
        assert len(results) == 2
        for r, tuid in zip(results, ["p1", "p2"]):
            assert (tmp_path / TOOL_RESULTS_DIRNAME / f"{tuid}.json").is_file()
            assert "clipped" in r["content"]

    @pytest.mark.asyncio
    async def test_partition_persists_large_payload(self, tmp_path):
        tool = _FixedOutputTool("big", "x" * 5000, max_chars=500)
        reg = _make_registry([tool])
        router = RegistryRouter(reg)
        execu = PartitionExecutor(registry=reg, max_concurrency=2)

        ctx = _ctx(str(tmp_path))
        await execu.execute_all(
            [
                {"tool_use_id": "a1", "tool_name": "big", "tool_input": {}},
                {"tool_use_id": "a2", "tool_name": "big", "tool_input": {}},
            ],
            router,
            ctx,
        )
        for tuid in ("a1", "a2"):
            assert (tmp_path / TOOL_RESULTS_DIRNAME / f"{tuid}.json").is_file()

    @pytest.mark.asyncio
    async def test_streaming_persists_large_payload(self, tmp_path):
        tool = _FixedOutputTool("big", "x" * 5000, max_chars=500)
        reg = _make_registry([tool])
        router = RegistryRouter(reg)
        execu = StreamingToolExecutor(registry=reg, router=router)

        ctx = _ctx(str(tmp_path))
        await execu.add(
            {"tool_use_id": "s1", "tool_name": "big", "tool_input": {}}, ctx
        )
        await execu.add(
            {"tool_use_id": "s2", "tool_name": "big", "tool_input": {}}, ctx
        )
        results = await execu.drain(ctx)
        assert len(results) == 2
        for tuid in ("s1", "s2"):
            assert (tmp_path / TOOL_RESULTS_DIRNAME / f"{tuid}.json").is_file()

    @pytest.mark.asyncio
    async def test_no_storage_path_still_returns_full_content(self, tmp_path):
        tool = _FixedOutputTool("big", "HELLO", max_chars=500)
        reg = _make_registry([tool])
        router = RegistryRouter(reg)
        execu = SequentialExecutor()

        ctx = _ctx(None)
        results = await execu.execute_all(
            [{"tool_use_id": "n1", "tool_name": "big", "tool_input": {}}],
            router,
            ctx,
        )
        # Small payload → unchanged
        assert results[0]["content"] == "HELLO"
