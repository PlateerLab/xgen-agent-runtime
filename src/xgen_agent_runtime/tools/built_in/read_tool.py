"""ReadTool — read file contents with line numbers."""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import PurePosixPath
from typing import Any, Dict

from xgen_agent_runtime.core.images import bound_image
from xgen_agent_runtime.tools.built_in._file_witness import read_mutation
from xgen_agent_runtime.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult
from xgen_agent_runtime.tools.fs import tool_fs

_DEFAULT_LIMIT = 2000


#: 문서 형식 — Read 로는 내용을 못 본다. ParseDocument 가 글을 뽑는다.
_DOCUMENT_SUFFIXES = frozenset(
    {".docx", ".xlsx", ".pptx", ".pdf", ".hwp", ".hwpx", ".doc", ".xls", ".ppt", ".rtf"}
)


def _binary_note(name: str, size: int) -> str:
    """바이너리 파일 결과. 문서면 읽는 도구를 알려 준다.

    예전엔 ``[Binary file: …]`` 한 줄뿐이었다 — 모델은 문서를 읽을 방법이 없다고 결론짓거나 Bash 로
    zip 을 풀었다(2026-09-30 감사).
    """
    note = f"[Binary file: {name}, {size} bytes]"
    suffix = os.path.splitext(str(name))[1].lower()
    if suffix in _DOCUMENT_SUFFIXES:
        note += (
            " This is a document — extract its text with ParseDocument (text-centric: text, "
            "tables as text, page markers; not the layout). If it is not in your tool list, "
            'open it with ToolSearch("ParseDocument").'
        )
    return note


#: 모델에게 건네는 그림의 긴 변 상한 — 공개 비전 모델들도 이보다 큰 그림은 줄여서 본다. 세부가
#: 필요하면 모델이 잘라서(Bash) 다시 읽는다.
_IMAGE_MAX_EDGE = 1568

#: 호스트가 ``ToolContext.metadata`` 에 싣는 "이 턴의 모델이 그림을 보는가" (host.turn_executor).
MODEL_VISION_KEY = "model_vision"

_NO_VISION = (
    "The current model cannot view images, so the content of this image is NOT available to you. "
    "Do not describe or guess what it shows. If you need text from it, use an OCR tool if one is "
    "available; otherwise tell the user this model cannot look at images."
)


def _image_result(name: str, raw: bytes, mime: str, context: ToolContext) -> ToolResult:
    """이미지 파일 — 그림을 보는 모델에게는 그림을, 못 보는 모델에게는 못 본다는 사실을.

    예전엔 어느 모델에게나 ``[Image file: …]`` 한 줄만 돌려줬다. 그림을 보는 모델(Qwen 3.8·Claude·GPT)도
    파일 이름만 받고 "프레임 31장을 확인했다" 고 답하거나 OCR·픽셀 분석 코드로 우회했다(dev 30일
    Qwen 24대화 — Read 36회, OCR 38회·픽셀 71회, 2026-10-11 감사).
    """
    head = f"[Image file: {name}, {len(raw)} bytes, type={mime}]"
    if not (context.metadata or {}).get(MODEL_VISION_KEY):
        return ToolResult(content=f"{head} {_NO_VISION}")
    try:
        data, sent_mime = bound_image(raw, max_edge=_IMAGE_MAX_EDGE)
    except ValueError:
        return ToolResult(
            content=f"{head} The image could not be decoded (unsupported format or corrupt)."
        )
    return ToolResult(
        content=[
            {"type": "text", "text": f"{head} The image is attached below."},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": sent_mime,
                    "data": base64.b64encode(data).decode("ascii"),
                },
            },
        ]
    )


class ReadTool(Tool):
    """Read a file from the local filesystem.

    Returns content with line numbers (cat -n format).
    Detects binary files and refuses to read them. Images go to the model as images when it can
    see them (``MODEL_VISION_KEY``), otherwise a note says it cannot.
    """

    @property
    def name(self) -> str:
        return "Read"

    @property
    def description(self) -> str:
        return (
            "Read a file in your sandbox. Returns content with line numbers. "
            "Use offset and limit to read specific portions of large files. "
            "Image files (PNG, JPEG, WebP, GIF) are shown to you as images if your model can see them."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": (
                        "Absolute path to the file to read (starts with `/`). "
                        "A relative path is resolved against your working folder."
                    ),
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (0-based). Default: 0.",
                    "minimum": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": f"Max lines to read. Default: {_DEFAULT_LIMIT}.",
                    "exclusiveMinimum": 0,
                },
            },
            "required": ["file_path"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        # Pure filesystem read — no side effects, safe to parallelise.
        return ToolCapabilities(
            concurrency_safe=True,
            read_only=True,
            idempotent=True,
        )

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        file_path = input.get("file_path", "")
        offset = input.get("offset", 0)
        limit = input.get("limit", _DEFAULT_LIMIT)

        # 러너든 로컬이든 한 길로 읽는다 — 백엔드 선택은 tool_fs 한 곳에서만 일어난다.
        # 예전엔 분기마다 따로 구현해서 같은 파일에 두 곳이 다르게 답했다(이미지는
        # 로컬만 "Image file", 러너는 바이너리 검사에 걸려 "Binary file").
        fs = tool_fs(context)
        try:
            resolved = fs.resolve(file_path)
        except (PermissionError, ValueError) as e:
            return ToolResult(content=str(e), is_error=True)
        name = PurePosixPath(resolved).name

        try:
            raw = await fs.read_bytes(file_path)
        except FileNotFoundError:
            return ToolResult(content=f"File not found: {resolved}", is_error=True)
        except IsADirectoryError:
            return ToolResult(
                content=f"Cannot read directory: {resolved}. Use Bash with 'ls' instead.",
                is_error=True,
            )
        except PermissionError as e:
            return ToolResult(content=str(e), is_error=True)
        except Exception as e:  # noqa: BLE001
            return ToolResult(content=f"Read error: {e}", is_error=True)

        mime, _ = mimetypes.guess_type(name)
        # SVG 는 글(XML)이다 — 그림 길로 보내지 않고 아래에서 글로 읽는다.
        if mime and mime.startswith("image/") and mime != "image/svg+xml":
            return _image_result(name, raw, mime, context)

        if b"\x00" in raw[:8192]:
            return ToolResult(content=_binary_note(name, len(raw)))

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw.decode("latin-1")
            except Exception:
                return ToolResult(content=_binary_note(name, len(raw)))

        lines = text.splitlines(keepends=True)
        total = len(lines)
        selected = lines[offset : offset + limit]
        if not selected and total > 0:
            return ToolResult(
                content=f"Offset {offset} is beyond file end ({total} lines).", is_error=True
            )

        # 줄 번호는 1부터 (cat -n 과 같은 모양)
        output = "\n".join(
            f"{i}\t{line.rstrip()}" for i, line in enumerate(selected, start=offset + 1)
        )
        if offset + limit < total:
            output += f"\n\n... ({total - offset - limit} more lines, {total} total)"

        return ToolResult(
            content=output,
            # 이 세션에서 내용을 본 파일로 기록한다(_file_witness). 모델이 준 표기와
            # 해석된 절대 경로 둘 다 — Write 가 어느 쪽으로 와도 알아본다. 예전엔 러너
            # 분기만 모델 표기 하나를 적어서, 상대 경로로 읽고 절대 경로로 쓰면 러너에서만
            # 거절됐다.
            # 파일 전체를 본 읽기만 "읽었다" 로 친다(나눠 읽으면 다 덮었을 때) — _file_witness.read_mutation.
            state_mutations=read_mutation(
                context.state_view,
                [file_path, resolved],
                start=offset,
                end=min(total, offset + limit),
                total=total,
            ),
        )
