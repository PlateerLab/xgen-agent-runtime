"""모델에게 보낼 이미지 한 장의 크기 상한 — 첨부(provider 요청 직전)와 Read 가 같은 규칙을 쓴다.

원본은 작업 공간에 그대로 두고, 모델에게 가는 사본만 줄인다(Base64 는 다시 3분의 1이 붙는다).
형식은 클라이언트가 붙인 MIME 이 아니라 파일의 매직 바이트로 정한다.
"""

from __future__ import annotations

import io
import warnings
from typing import Tuple

from PIL import Image, ImageOps

#: 모델이 받는 형식 — 이 밖은 거절한다.
MODEL_IMAGE_MIMES = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})


def bound_image(
    data: bytes, *, max_edge: int = 2048, max_bytes: int = 4 * 1024 * 1024
) -> Tuple[bytes, str]:
    """``(보낼 바이트, MIME)``. 긴 변이 ``max_edge`` 보다 크거나 ``max_bytes`` 를 넘으면 JPEG 로 줄인다.

    지원하지 않는 형식·깨진 파일·압축 폭탄은 ``ValueError``.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as original:
                mime = Image.MIME.get(original.format or "", "")
                if mime not in MODEL_IMAGE_MIMES:
                    raise ValueError("unsupported image format")
                original.load()
                if max(original.size) > max_edge or len(data) > max_bytes:
                    resized = ImageOps.exif_transpose(original).convert("RGB")
                    resized.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
                    output = io.BytesIO()
                    resized.save(output, format="JPEG", quality=85, optimize=True)
                    data, mime = output.getvalue(), "image/jpeg"
    except (
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError("unsupported or invalid image") from exc
    return data, mime
