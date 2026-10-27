"""ParseDocument 추출 직렬화 — doc2chunk 없이도 돈다(가짜 처리기)."""

from __future__ import annotations


# ── 추출은 프로세스에서 한 번에 하나 (4.81.1) ────────────────────────────────
#
# dev 10-02: 모델이 ToolBatch 로 PDF 두 개를 동시에 읽자 글자가 깨졌고, 그 뒤엔 하나씩 읽어도
# "broken document". pypdfium2(PDFium)는 스레드 안전하지 않다 — 같은 PDF 둘을 스레드 넷으로
# 동시에 추출하면 프로세스가 Segmentation fault 로 죽었다(워크플로 서버 전체).


def test_concurrent_extractions_never_overlap(monkeypatch):
    import threading
    import time

    from xgen_agent_runtime.tools.built_in import parse_document_tool as m

    state = {"inside": 0, "peak": 0}
    guard = threading.Lock()

    class _Processor:
        def is_supported(self, extension):
            return True

        def extract_text(self, path, **kwargs):
            with guard:
                state["inside"] += 1
                state["peak"] = max(state["peak"], state["inside"])
            time.sleep(0.02)
            with guard:
                state["inside"] -= 1
            return f"text of {path}"

    monkeypatch.setattr(m, "_load_processor", lambda image_directory: _Processor())
    out: list = []
    threads = [
        threading.Thread(target=lambda i=i: out.append(m._extract(f"/tmp/{i}.pdf", "pdf")))
        for i in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(out) == sorted(f"text of /tmp/{i}.pdf" for i in range(6))
    assert state["peak"] == 1


# ── 그림 읽기 함수(4.104.0) ──────────────────────────────────────────────────
#
# 호스트가 ``extras["image_reader"]`` 를 주면 그림 자리에 그림에서 읽은 글이 들어간다. XGEN 은 파일
# 저장소 그림 처리와 같은 VLM · 같은 결과를 준다. 없으면 예전처럼 ``[Image]``.


def _fake_document(monkeypatch, tmp_path, images):
    from xgen_agent_runtime.tools.built_in import parse_document_tool as m

    class _Processor:
        def __init__(self, image_directory):
            self.dir = image_directory

        def is_supported(self, extension):
            return True

        def extract_text(self, path, **kwargs):
            parts = ["[Page Number: 1]", "본문"]
            for n, data in enumerate(images):
                where = f"{self.dir}/img{n}.png"
                with open(where, "wb") as handle:
                    handle.write(data)
                parts.append(f"[Image:{where}]")
            return "\n".join(parts)

    monkeypatch.setattr(m, "_load_processor", lambda image_directory: _Processor(image_directory))
    (tmp_path / "doc.pdf").write_bytes(b"%PDF")
    return m


def _run(tmp_path, extras):
    import asyncio

    from xgen_agent_runtime.tools.base import ToolContext
    from xgen_agent_runtime.tools.built_in.parse_document_tool import ParseDocumentTool
    from xgen_agent_runtime.tools.fs import LocalFS

    ctx = ToolContext(session_id="s", working_dir=str(tmp_path), fs=LocalFS(str(tmp_path), [str(tmp_path)]),
                      extras=extras)
    return asyncio.run(ParseDocumentTool().execute({"file_path": "doc.pdf"}, ctx))


def test_with_an_image_reader_each_image_is_read_once_and_numbered(monkeypatch, tmp_path):
    _fake_document(monkeypatch, tmp_path, [b"chart", b"logo", b"chart"])
    calls = []

    def reader(data, name):
        calls.append(data)
        if data == b"logo":
            return None  # 읽지 못한 그림은 [Image] 로 남는다
        return "[그림 1 · 차트]\n받아 적은 글: Q3 12.5"

    out = _run(tmp_path, {"image_reader": reader})
    assert not out.is_error, out.content
    assert sorted(calls) == [b"chart", b"logo"]  # 같은 그림은 한 번만
    assert out.content.count("받아 적은 글: Q3 12.5") == 2 and "[Image]" in out.content
    assert "/img" not in out.content  # 임시 경로는 남기지 않는다


def test_without_an_image_reader_images_stay_placeholders(monkeypatch, tmp_path):
    _fake_document(monkeypatch, tmp_path, [b"chart"])
    out = _run(tmp_path, {})
    assert "[Image]" in out.content and "받아 적은" not in out.content
