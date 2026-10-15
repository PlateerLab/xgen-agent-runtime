"""Read 가 이미지를 **보는 모델에게는 그림으로**, 못 보는 모델에게는 못 본다는 사실로 돌려준다.

예전엔 어느 모델에게나 ``[Image file: …]`` 한 줄뿐이었다. dev 30일(2026-10-11 감사) Qwen 대화 24개가
그림을 Read 하고 이름만 받은 뒤 "프레임 31장을 확인했다" 고 답하거나 OCR·픽셀 분석 코드로 우회했다 —
광고 영상 프레임, 도면 표시 그림, 영수증 사진(PC) 세 갈래. 그 모델(qwen3.8-27b)은 그림을 본다.
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
from pathlib import Path

import pytest
from PIL import Image

from xgen_agent_runtime.host import token_budget
from xgen_agent_runtime.host.token_budget import resolve_supports_vision
from xgen_agent_runtime.llm_client.translators._canonical import canonical_messages_to_openai
from xgen_agent_runtime.stages.s10_tool.artifact.default.executors import _result_text
from xgen_agent_runtime.tools._geny_sandbox import ExecResult
from xgen_agent_runtime.tools.base import ToolContext
from xgen_agent_runtime.tools.built_in.read_tool import MODEL_VISION_KEY, ReadTool


class _Session:
    """러너 백엔드 — 같은 파일을 러너 길로도 읽어 두 길이 같은 답을 하는지 본다."""

    def __init__(self, root: Path) -> None:
        self.workdir = str(root)
        self.extra_roots: list = []
        self.readonly_roots: list = []

    async def ensure(self) -> None:
        return None

    async def exec(self, argv, *, cwd=None, stdin=None, env=None, timeout_s=120.0):
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=cwd or self.workdir,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env={"PATH": os.environ.get("PATH", "")},
        )
        out, err = await proc.communicate()
        return ExecResult(rc=proc.returncode or 0, stdout=out, stderr=err)

    async def read_bytes(self, path: str) -> bytes:
        return Path(path).read_bytes()


def _png(size=(64, 32), color="red") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def _ctx(root: Path, backend: str, vision: bool) -> ToolContext:
    meta = {MODEL_VISION_KEY: True} if vision else {}
    if backend == "local":
        return ToolContext(session_id="t", working_dir=str(root), allowed_paths=[str(root)], metadata=meta)
    return ToolContext(session_id="t", working_dir=str(root), sandbox=_Session(root), metadata=meta)


def _read(ctx: ToolContext, path: str):
    return asyncio.run(ReadTool().execute({"file_path": path}, ctx))


@pytest.fixture(params=["local", "runner"])
def backend(request) -> str:
    return request.param


class TestReadImage:
    def test_a_seeing_model_gets_the_picture(self, tmp_path, backend):
        (tmp_path / "frame.png").write_bytes(_png())
        r = _read(_ctx(tmp_path, backend, vision=True), "frame.png")
        assert not r.is_error
        text, image = r.content
        assert text["type"] == "text" and text["text"].startswith("[Image file: frame.png")
        assert image["type"] == "image" and image["source"]["media_type"] == "image/png"
        with Image.open(io.BytesIO(base64.b64decode(image["source"]["data"]))) as im:
            assert im.size == (64, 32)

    def test_a_large_picture_is_shrunk_for_the_model_only(self, tmp_path, backend):
        original = _png(size=(3200, 800))
        (tmp_path / "sheet.png").write_bytes(original)
        r = _read(_ctx(tmp_path, backend, vision=True), "sheet.png")
        image = r.content[1]
        assert image["source"]["media_type"] == "image/jpeg"
        with Image.open(io.BytesIO(base64.b64decode(image["source"]["data"]))) as im:
            assert max(im.size) == 1568
        assert (tmp_path / "sheet.png").read_bytes() == original  # 원본은 그대로

    def test_a_blind_model_is_told_it_cannot_see(self, tmp_path, backend):
        (tmp_path / "receipt.jpg").write_bytes(_png())
        r = _read(_ctx(tmp_path, backend, vision=False), "receipt.jpg")
        assert isinstance(r.content, str)
        assert r.content.startswith("[Image file: receipt.jpg")
        assert "cannot view images" in r.content and "Do not describe or guess" in r.content

    def test_a_broken_picture_is_not_sent(self, tmp_path, backend):
        (tmp_path / "broken.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
        r = _read(_ctx(tmp_path, backend, vision=True), "broken.png")
        assert isinstance(r.content, str) and "could not be decoded" in r.content

    def test_svg_is_read_as_text(self, tmp_path, backend):
        (tmp_path / "icon.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>\n')
        r = _read(_ctx(tmp_path, backend, vision=True), "icon.svg")
        assert isinstance(r.content, str) and "<rect/>" in r.content


class TestResolveVision:
    @pytest.fixture(autouse=True)
    def _no_catalog(self, monkeypatch):
        monkeypatch.setattr(token_budget, "_catalog_capabilities", lambda p, m: None)

    @pytest.mark.parametrize(
        "provider, model, expected",
        [
            ("anthropic", "claude-sonnet-5-5", True),
            ("bedrock", "apac.anthropic.claude-sonnet-4-5-20250929-v1:0", True),
            ("anthropic", "claude-3-5-haiku-20241022", False),
            ("openai", "gpt-5.4", True),
            ("openai", "gpt-4.1-mini-2025-04-14", True),
            ("openai", "o3-mini", False),
            ("google", "gemini-2.5-pro", True),
            ("claude_code", "sonnet", True),
            ("vllm", "Qwen/Qwen3-VL-8B-Instruct", True),
            # 자체 서빙은 같은 이름도 체크포인트에 따라 다르다 — 카탈로그가 말해야 켠다.
            ("vllm", "qwen3.8-27b", False),
            ("vllm", "x2bee/Polar-14B", False),
            ("deepseek", "deepseek-v4-flash", False),
            # 짧은 접두가 이름 속에서 걸리지 않는다.
            ("vllm", "qwen3-32b-o3x", False),
            ("vllm", "", False),
        ],
    )
    def test_public_families_and_unknown(self, provider, model, expected):
        assert resolve_supports_vision(provider, model) is expected

    def test_catalog_decides_before_the_table(self, monkeypatch):
        caps = {
            ("vllm", "qwen3.8-27b"): {"supports_vision": True},
            ("vllm", "Qwen3.6-27B-FP8"): {"vision": True},
            ("vllm", "gemma-4-31b-it-qat"): {"supports_vision": False},
            ("openai", "gpt-4.1"): {"supports_vision": False},
        }
        monkeypatch.setattr(token_budget, "_catalog_capabilities", lambda p, m: caps.get((p, m)))
        assert resolve_supports_vision("vllm", "qwen3.8-27b") is True
        assert resolve_supports_vision("vllm", "Qwen3.6-27B-FP8") is True
        assert resolve_supports_vision("vllm", "gemma-4-31b-it-qat") is False
        assert resolve_supports_vision("openai", "gpt-4.1") is False

    def test_node_override_wins(self, monkeypatch):
        monkeypatch.setattr(token_budget, "_catalog_capabilities", lambda p, m: {"supports_vision": True})
        assert resolve_supports_vision("vllm", "qwen3.8-27b", False) is False
        assert resolve_supports_vision("vllm", "polar-14b", True) is True


def test_the_picture_reaches_an_openai_compatible_server(tmp_path):
    """OpenAI 형식의 tool 메시지는 글만 받는다 — 그림은 바로 뒤 user 메시지로 옮겨 간다."""
    (tmp_path / "frame.png").write_bytes(_png())
    r = _read(_ctx(tmp_path, "local", vision=True), "frame.png")
    messages = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]},
        {"role": "user", "content": [r.to_api_format("t1")]},
    ]
    out = canonical_messages_to_openai(messages, system_role="system")
    tool_msg, image_msg = out[-2], out[-1]
    assert tool_msg["role"] == "tool" and tool_msg["content"].startswith("[Image file: frame.png")
    assert image_msg["role"] == "user"
    assert image_msg["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_progress_text_names_the_image_without_base64(tmp_path):
    """작업 과정·기록에 쓰는 결과 글 — 파이썬 표기·Base64 대신 ``[image]``."""
    (tmp_path / "frame.png").write_bytes(_png())
    r = _read(_ctx(tmp_path, "local", vision=True), "frame.png")
    text = _result_text({"content": r.content})
    assert text.startswith("[Image file: frame.png") and text.endswith("[image]")
    assert "base64" not in text and "{" not in text
