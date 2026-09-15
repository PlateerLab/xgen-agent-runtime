"""Multimodal pipeline tests \u2014 NormalizedInput, translators, dehydration."""

import sys
import os
import base64
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


from xgen_agent_runtime.stages.s01_input.artifact.default.normalizers import (
    DefaultNormalizer,
    MultimodalNormalizer,
)
from xgen_agent_runtime.stages.s06_api._translate import (
    canonical_messages_to_anthropic,
    canonical_messages_to_openai,
    canonical_messages_to_google,
)
from xgen_agent_runtime.stages.s18_memory._dehydrate import (
    dehydrate_message,
    dehydrate_messages,
)
from xgen_agent_runtime.stages.s01_input.artifact.default.validators import DefaultValidator


SAMPLE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="  # 1x1 PNG


# \u2501\u2501 NormalizedInput \u2501\u2501


class TestMultimodalNormalizer:
    def test_text_only(self):
        n = MultimodalNormalizer().normalize("hello")
        assert n.text == "hello"
        assert n.images == []
        assert n.files == []
        assert n.to_message_content() == "hello"

    def test_dict_with_inline_image(self):
        n = MultimodalNormalizer().normalize(
            {
                "text": "what is this?",
                "images": [
                    {"mime_type": "image/png", "data": SAMPLE_B64, "name": "x.png"},
                ],
            }
        )
        assert n.text == "what is this?"
        assert len(n.images) == 1
        block = n.images[0]
        assert block["type"] == "image"
        assert block["source"]["type"] == "base64"
        assert block["source"]["media_type"] == "image/png"
        assert block["source"]["data"] == SAMPLE_B64
        assert block["_meta"]["name"] == "x.png"

    def test_dict_with_url_image(self):
        n = MultimodalNormalizer().normalize(
            {"text": "describe", "images": [{"mime_type": "image/jpeg", "url": "https://e.x/y.jpg"}]}
        )
        block = n.images[0]
        assert block["source"] == {"type": "url", "url": "https://e.x/y.jpg"}

    def test_attachments_routes_by_kind(self):
        n = MultimodalNormalizer().normalize(
            {
                "text": "see",
                "attachments": [
                    {"kind": "image", "mime_type": "image/png", "data": SAMPLE_B64},
                    {"kind": "file", "name": "report.pdf", "mime_type": "application/pdf",
                     "url": "https://e.x/r.pdf"},
                ],
            }
        )
        assert len(n.images) == 1
        assert len(n.files) == 1
        assert n.files[0]["mime_type"] == "application/pdf"

    def test_to_message_content_with_image(self):
        n = MultimodalNormalizer().normalize(
            {"text": "hi", "images": [{"mime_type": "image/png", "data": SAMPLE_B64}]}
        )
        content = n.to_message_content()
        assert isinstance(content, list)
        assert content[0]["type"] == "image"
        assert content[-1] == {"type": "text", "text": "hi"}

    def test_canonical_block_passthrough(self):
        canonical = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": SAMPLE_B64},
        }
        n = MultimodalNormalizer().normalize({"text": "x", "images": [canonical]})
        assert n.images[0] is canonical or n.images[0] == canonical

    def test_workspace_image_path_stays_lazy(self, tmp_path):
        image = tmp_path / "a.png"
        image.write_bytes(base64.b64decode(SAMPLE_B64))
        n = MultimodalNormalizer().normalize({
            "text": "see",
            "attachments": [{"kind": "image", "mime_type": "image/png", "local_path": str(image)}],
        })
        assert n.images[0]["source"] == {"type": "path", "path": str(image), "media_type": "image/png"}


def test_validator_counts_text_instead_of_base64_payload():
    payload = {"text": "describe", "attachments": [{"kind": "image", "data": "A" * 2_000_000}]}
    assert DefaultValidator().validate(payload) is None


def test_validator_still_rejects_oversized_user_text():
    assert DefaultValidator(max_length=10).validate({"text": "x" * 11, "attachments": [{}]}) == (
        "Input too long (max 10 chars)"
    )


class TestDefaultNormalizerAutoDelegates:
    def test_attachments_dict_uses_multimodal_path(self):
        n = DefaultNormalizer().normalize(
            {"text": "  hello  ", "images": [{"mime_type": "image/png", "data": SAMPLE_B64}]}
        )
        assert n.text == "hello"
        assert len(n.images) == 1

    def test_text_only_unchanged(self):
        n = DefaultNormalizer().normalize("  hi  ")
        assert n.text == "hi"
        assert n.images == []


# \u2501\u2501 Translators \u2501\u2501


def _user_msg_with_image():
    return {
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": SAMPLE_B64},
             "_meta": {"name": "x.png"}},
            {"type": "text", "text": "what is this?"},
        ],
    }


class TestAnthropicSanitization:
    def test_strips_internal_meta_key(self):
        out = canonical_messages_to_anthropic([_user_msg_with_image()])
        block = out[0]["content"][0]
        assert "_meta" not in block
        assert block["source"]["data"] == SAMPLE_B64

    def test_file_block_falls_back_to_text(self):
        msg = {
            "role": "user",
            "content": [
                {"type": "file", "name": "x.pdf", "mime_type": "application/pdf"},
                {"type": "text", "text": "summarise"},
            ],
        }
        out = canonical_messages_to_anthropic([msg])
        types = [b["type"] for b in out[0]["content"]]
        assert types == ["text", "text"]
        assert "x.pdf" in out[0]["content"][0]["text"]

    def test_file_fallback_includes_workspace_path(self):
        msg = {"role": "user", "content": [{
            "type": "file", "name": "a.json", "mime_type": "application/json",
            "workspace_path": "uploads/a.json",
        }]}
        out = canonical_messages_to_anthropic([msg])
        assert "uploads/a.json" in out[0]["content"][0]["text"]

    def test_local_image_materializes_only_in_translated_copy(self, tmp_path):
        path = tmp_path / "a.png"
        path.write_bytes(base64.b64decode(SAMPLE_B64))
        source = {"type": "path", "path": str(path), "media_type": "image/png"}
        msg = {"role": "user", "content": [{"type": "image", "source": source}]}
        out = canonical_messages_to_anthropic([msg])
        assert out[0]["content"][0]["source"]["type"] == "base64"
        assert msg["content"][0]["source"] == source


class TestOpenAIMultimodal:
    def test_user_image_becomes_image_url_part(self):
        out = canonical_messages_to_openai([_user_msg_with_image()])
        assert len(out) == 1
        msg = out[0]
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        kinds = [p["type"] for p in msg["content"]]
        assert "image_url" in kinds
        assert "text" in kinds
        img_part = next(p for p in msg["content"] if p["type"] == "image_url")
        assert img_part["image_url"]["url"].startswith("data:image/png;base64,")

    def test_url_image_passes_through(self):
        msg = {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "url", "url": "https://e.x/a.jpg"}},
                {"type": "text", "text": "?"},
            ],
        }
        out = canonical_messages_to_openai([msg])
        img_part = next(p for p in out[0]["content"] if p["type"] == "image_url")
        assert img_part["image_url"]["url"] == "https://e.x/a.jpg"


class TestGoogleMultimodal:
    def test_user_image_becomes_inline_data_part(self):
        out = canonical_messages_to_google([_user_msg_with_image()])
        parts = out[0]["parts"]
        kinds = [list(p.keys())[0] for p in parts]
        assert "inlineData" in kinds
        assert "text" in kinds
        inline = next(p for p in parts if "inlineData" in p)["inlineData"]
        assert inline["mimeType"] == "image/png"
        assert inline["data"] == SAMPLE_B64

    def test_url_image_becomes_file_data(self):
        msg = {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "url", "url": "https://e.x/a.jpg",
                                              "media_type": "image/jpeg"}},
            ],
        }
        out = canonical_messages_to_google([msg])
        parts = out[0]["parts"]
        assert "fileData" in parts[0]


# \u2501\u2501 Dehydration \u2501\u2501


class TestDehydration:
    def test_image_base64_data_is_stripped(self):
        msg = _user_msg_with_image()
        out = dehydrate_message(msg)
        block = out["content"][0]
        assert block["source"]["data"] is None
        assert block["source"]["_dehydrated"] is True
        # Original is not mutated
        assert msg["content"][0]["source"]["data"] == SAMPLE_B64

    def test_url_image_unchanged(self):
        msg = {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "url", "url": "https://e.x/a.jpg"}},
            ],
        }
        out = dehydrate_message(msg)
        assert out["content"][0]["source"] == {"type": "url", "url": "https://e.x/a.jpg"}

    def test_string_content_unchanged(self):
        out = dehydrate_messages([{"role": "user", "content": "hi"}])
        assert out[0]["content"] == "hi"


def test_workspace_files_reach_cli_single_and_multi_turn():
    import json
    from xgen_agent_runtime.llm_client.translators._cli import build_stream_json_stdin, flatten_messages_to_prompt
    attachment = {"type": "file", "name": "config.json", "mime_type": "application/json", "workspace_path": "uploads/config.json"}
    current = {"role": "user", "content": [attachment]}
    wire = json.loads(build_stream_json_stdin([current]))
    assert wire["message"]["content"][0]["type"] == "text"
    assert "uploads/config.json" in wire["message"]["content"][0]["text"]
    history = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}, current]
    assert "uploads/config.json" in flatten_messages_to_prompt(history)
    assert "uploads/config.json" in build_stream_json_stdin(history).decode()


def test_provider_image_copy_is_resized_without_changing_original(tmp_path):
    import io
    from PIL import Image
    from xgen_agent_runtime.llm_client.translators._canonical import materialize_local_image_block
    path = tmp_path / "large.png"
    Image.new("RGB", (4096, 1024), "white").save(path)
    original = path.read_bytes()
    block = {"type": "image", "source": {"type": "path", "path": str(path)}}
    wire = materialize_local_image_block(block)
    with Image.open(io.BytesIO(base64.b64decode(wire["source"]["data"]))) as resized:
        assert resized.size == (2048, 512)
    assert path.read_bytes() == original
    assert block["source"]["type"] == "path"


def test_truncated_provider_image_is_rejected(tmp_path):
    from xgen_agent_runtime.llm_client.translators._canonical import materialize_local_image_block
    path = tmp_path / "broken.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\ninvalid")
    with pytest.raises(ValueError, match="invalid image"):
        materialize_local_image_block({"type": "image", "source": {"type": "path", "path": str(path)}})


def test_input_str_and_original_image_path_are_preserved(tmp_path):
    path = tmp_path / "picture.png"
    path.write_bytes(base64.b64decode(SAMPLE_B64))
    normalized = DefaultNormalizer().normalize({"input_str": "describe this", "attachments": [
        {"kind": "image", "local_path": str(path), "workspace_path": "uploads/picture.png", "name": "picture.png"}
    ]})
    assert normalized.text == "describe this"
    assert any("uploads/picture.png" in block.get("text", "") for block in normalized.to_message_content())
