"""ParseDocument — 문서 파일에서 **글 위주의 요소**를 뽑는다(doc2chunk 추출 단계).

문서 도구는 이것 하나다. 예전의 편집 도구 묶음(DocGuide·DocEdit·DocBuild … edit2docs 기반)은
2026-09-30 에 걷어 냈다 — 에이전트는 문서를 **읽기만** 한다.

무엇을 주고 무엇을 주지 않는가
------------------------------
xgen-doc2chunk 의 ``DocumentProcessor.extract_text`` 를 부른다(청크로 자르는 단계는 쓰지 않는다).
결과는 **글이 중심인 표현**이다:

- 본문 글, 페이지·슬라이드·시트 구분 표식(``[Page Number: N]`` · ``[Slide Number: N]`` · ``[Sheet: …]``)
- 표는 글로 옮긴 모양(HTML ``<table>`` 이나 마크다운), 차트는 ``[chart]…[/chart]`` 글
- 이미지는 ``[Image]`` 자리표시뿐(그림 내용은 없다, OCR 도 하지 않는다)
- 문서 메타데이터(제목·작성자·날짜 등, 있으면)

**원본의 전체 구조가 아니다** — 글꼴·서식·위치·레이아웃·병합 규칙·도형·주석 등은 담지 않는다.
그래서 이 결과로 원본을 다시 만들거나 고칠 수 없다. 모델에게도 그렇게 말한다(설명).

파일은 런타임의 파일 경로 하나(:func:`xgen_agent_runtime.tools.fs.tool_fs`)로 읽는다 — 샌드박스가
붙어 있으면 러너 세션에서, 아니면 이 파드의 작업 폴더에서. 추출은 CPU 를 오래 쓸 수 있어(PDF·HWP)
스레드에서 돈다.
"""

from __future__ import annotations

import asyncio
import re
import tempfile
import threading
from pathlib import PurePosixPath
from typing import Any, Dict, Optional, Tuple

from xgen_agent_runtime.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult

#: 한 번에 돌려주는 글자 수. 넘으면 offset 으로 다음 부분을 읽게 한다.
DEFAULT_LIMIT = 40_000
MAX_LIMIT = 200_000

#: doc2chunk 가 이미지 자리에 남기는 표식 — 경로는 이 호출의 임시 폴더라 호출이 끝나면 없다.
_IMAGE_TAG = re.compile(r"\[Image:[^\]]*\]")

#: 문서 형식이 아닌데 이 도구로 오면 Read 가 낫다고 알려 줄 확장자.
_PLAIN_TEXT = frozenset({"txt", "md", "markdown", "log"})


#: 추출은 프로세스 전체에서 한 번에 하나만 돈다. PDF 를 여는 pypdfium2(PDFium)는 스레드 안전하지
#: 않다 — dev 10-02: 모델이 ToolBatch 로 PDF 두 개를 동시에 읽자 글자가 깨졌고(``æîô WHÅçL…``),
#: 그 뒤 같은 프로세스에서는 하나씩 읽어도 "broken document" 가 났다. 같은 파일 둘을 스레드 넷으로
#: 동시에 추출하면 프로세스가 Segmentation fault 로 죽는다(워크플로 서버 전체가 내려간다).
#: 추출은 몇 초짜리라 줄 세워도 비용이 작고, 다른 형식의 파서도 같은 처지일 수 있어 형식과
#: 상관없이 잠근다. 다른 세션의 추출과도 겹치면 안 되므로 도구 인스턴스가 아니라 모듈에 둔다.
_EXTRACT_LOCK = threading.Lock()


def _load_processor(image_directory: str) -> Any:
    from xgen_doc2chunk import DocumentProcessor

    return DocumentProcessor(image_directory=image_directory)


def _extract(local_path: str, extension: str) -> str:
    """doc2chunk 추출 — 청크 없이 글만. 지원하지 않는 형식이면 ValueError."""
    with _EXTRACT_LOCK, tempfile.TemporaryDirectory(prefix="xgen-parse-img-") as image_dir:
        processor = _load_processor(image_dir)
        if not processor.is_supported(extension):
            raise ValueError(f"unsupported file type: .{extension}")
        return processor.extract_text(
            local_path,
            file_extension=extension,
            extract_metadata=True,
            ocr_processing=False,
        )


class ParseDocumentTool(Tool):
    """문서 파일의 글과 단순 요소를 뽑는다 — 편집·재구성용이 아니다."""

    @property
    def name(self) -> str:
        return "ParseDocument"

    @property
    def description(self) -> str:
        return (
            "Extract the TEXT of a document file — PDF, Word (docx/doc), PowerPoint (pptx/ppt), "
            "Excel (xlsx/xls), HWP/HWPX, RTF, CSV/TSV, HTML and similar. Returns text-centric "
            "content only: the body text with page/slide/sheet markers, tables rendered as text "
            "(HTML or markdown), chart data as text, [Image] placeholders and document metadata. "
            "It is NOT the full document structure: fonts, formatting, positions, layout, shapes "
            "and images are not returned, so the result cannot be used to rebuild or edit the file. "
            "Long results come in parts: pass offset to continue. For plain text or code files use Read."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path of the document file."},
                "offset": {
                    "type": "integer",
                    "description": "Character offset to start from (0 = beginning). Use the value the "
                    "previous call reported to read the next part.",
                },
                "limit": {
                    "type": "integer",
                    "description": f"Maximum characters to return (default {DEFAULT_LIMIT}).",
                },
            },
            "required": ["file_path"],
        }

    @property
    def input_aliases(self) -> Dict[str, Tuple[str, ...]]:
        """모델이 ``path`` 로 불러도 받는다(다른 파일 도구의 옛 이름). 스키마에는 싣지 않는다."""
        return {"file_path": ("path",)}

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:  # noqa: A002
        # 결과를 이 도구가 나눠 준다(offset·limit) — 큰 결과 저장 한도는 한 번에 줄 수 있는 최대로.
        return ToolCapabilities(
            concurrency_safe=True, read_only=True, max_result_chars=MAX_LIMIT + 2_000
        )

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:  # noqa: A002
        from xgen_agent_runtime.tools.fs import FsAccessError, tool_fs

        raw = str((input or {}).get("file_path") or "").strip()
        if not raw:
            return ToolResult(content="file_path is required.", is_error=True)
        offset = _int(input.get("offset"), 0, low=0)
        limit = _int(input.get("limit"), DEFAULT_LIMIT, low=1, high=MAX_LIMIT)
        fs = tool_fs(context)
        try:
            resolved = fs.resolve(raw)
        except (FsAccessError, PermissionError) as exc:
            return ToolResult(content=f"Access denied: {exc}", is_error=True)
        except ValueError as exc:
            return ToolResult(content=f"File not found: {exc}", is_error=True)
        name = PurePosixPath(resolved).name
        extension = PurePosixPath(name).suffix.lstrip(".").lower()
        if not extension:
            return ToolResult(
                content=f"{name} has no file extension, so its format is unknown.", is_error=True
            )
        try:
            async with fs.materialize(raw) as local:
                text = await asyncio.to_thread(_extract, str(local), extension)
        except FileNotFoundError:
            return ToolResult(content=f"File not found: {resolved}", is_error=True)
        except IsADirectoryError:
            return ToolResult(content=f"{resolved} is a directory, not a file.", is_error=True)
        except ValueError as exc:
            return ToolResult(content=f"Cannot parse {name}: {exc}", is_error=True)
        except ImportError:
            return ToolResult(
                content="The document parser is not installed on this server.", is_error=True
            )
        except Exception as exc:  # noqa: BLE001 — 깨진 파일은 모델이 읽을 사유로 돌려준다
            return ToolResult(
                content=f"Could not parse {name}: {type(exc).__name__}: {exc}", is_error=True
            )
        text = _IMAGE_TAG.sub("[Image]", text or "")
        return ToolResult(
            content=_page(name, text, offset, limit, extension),
            metadata={
                "file": resolved,
                "chars": len(text),
                "offset": offset,
            },
        )


def _int(value: Any, default: int, *, low: int, high: Optional[int] = None) -> int:
    try:
        number = int(value) if value is not None else default
    except (TypeError, ValueError):
        number = default
    number = max(low, number)
    return min(high, number) if high is not None else number


def _page(name: str, text: str, offset: int, limit: int, extension: str) -> str:
    """머리 한 줄(무엇이 담겼는지) + 요청한 부분 + 이어 읽기 안내."""
    total = len(text)
    head = (
        f"[ParseDocument: {name} — text-centric extraction (text, page/slide/sheet markers, tables as "
        f"text, [Image] placeholders); not the full document structure or formatting]"
    )
    if not text.strip():
        return (
            head
            + "\n(No text was found. A scanned or image-only document has no extractable text.)"
        )
    part = text[offset : offset + limit]
    lines = [head]
    if offset or offset + limit < total:
        end = min(total, offset + limit)
        lines.append(f"[chars {offset}-{end} of {total}]")
    lines.append(part)
    if offset + limit < total:
        lines.append(
            f"[{total - (offset + limit)} more characters: call ParseDocument again with offset={offset + limit}]"
        )
    if extension in _PLAIN_TEXT:
        lines.append("[This is a plain text file — Read shows it with line numbers.]")
    return "\n".join(lines)


__all__ = ["ParseDocumentTool", "DEFAULT_LIMIT", "MAX_LIMIT"]
