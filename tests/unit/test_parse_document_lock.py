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
