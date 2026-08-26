"""Canonical turn input at the host/runtime boundary.

The workflow and connector transports may carry plain text, OpenAI-style
content arrays, or the XGen attachment envelope.  Runtime stages already know
how to normalize images; this module only preserves that structure while the
host executor performs text-only work such as RAG and context budgeting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from xgen_agent_runtime.host._constants import _coerce_text


_DATA_IMAGE_RE = re.compile(
    r"^data:(image/[a-z0-9.+-]+);base64,([A-Za-z0-9+/=\r\n]+)$",
    re.IGNORECASE,
)


def _image_from_url(item: Dict[str, Any]) -> Dict[str, Any] | None:
    image_url = item.get("image_url")
    url = image_url.get("url") if isinstance(image_url, dict) else image_url
    if not isinstance(url, str) or not url:
        return None
    match = _DATA_IMAGE_RE.match(url)
    if match:
        return {
            "kind": "image",
            "mime_type": match.group(1).lower(),
            "data": "".join(match.group(2).split()),
        }
    return {"kind": "image", "url": url}


@dataclass(frozen=True)
class TurnInput:
    """Provider-neutral input retained until Stage 1 normalization."""

    text: str = ""
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Any) -> "TurnInput":
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, str) or raw is None:
            return cls(text=_coerce_text(raw))

        if isinstance(raw, list):
            text_parts: List[str] = []
            attachments: List[Dict[str, Any]] = []
            for item in raw:
                if not isinstance(item, dict):
                    text_parts.append(_coerce_text(item))
                    continue
                kind = str(item.get("type") or item.get("kind") or "").lower()
                if kind in ("text", "input_text"):
                    value = item.get("text", item.get("content", ""))
                    if value:
                        text_parts.append(_coerce_text(value))
                elif kind in ("image_url", "input_image"):
                    image = _image_from_url(item)
                    if image is not None:
                        attachments.append(image)
                elif kind in ("image", "img"):
                    attachments.append(dict(item))
                else:
                    text_parts.append(_coerce_text(item))
            return cls(
                text="\n".join(part for part in text_parts if part).strip(),
                attachments=attachments,
            )

        if isinstance(raw, dict):
            body = raw.get("text")
            if body is None:
                body = raw.get("input_str", raw.get("input", raw.get("content", "")))
            nested = cls.from_raw(body) if isinstance(body, (list, dict)) else None
            attachments: List[Dict[str, Any]] = list(nested.attachments) if nested else []
            for key in ("attachments", "images", "files"):
                values = raw.get(key) or []
                if isinstance(values, dict):
                    values = [values]
                for item in values:
                    if isinstance(item, dict):
                        copied = dict(item)
                        if key == "images" and not copied.get("kind"):
                            copied["kind"] = "image"
                        attachments.append(copied)
            metadata = raw.get("metadata")
            return cls(
                text=nested.text if nested else _coerce_text(body),
                attachments=attachments,
                metadata=dict(metadata) if isinstance(metadata, dict) else {},
            )

        return cls(text=_coerce_text(raw))

    def with_text(self, text: str) -> "TurnInput":
        return TurnInput(text=text, attachments=self.attachments, metadata=self.metadata)

    def as_pipeline_input(self) -> Any:
        """Keep the historical string shape for text-only turns."""
        if not self.attachments and not self.metadata:
            return self.text
        return {
            "text": self.text,
            "attachments": list(self.attachments),
            "metadata": dict(self.metadata),
        }
