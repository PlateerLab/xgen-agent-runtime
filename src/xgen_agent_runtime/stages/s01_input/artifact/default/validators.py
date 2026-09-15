"""Default artifact validators for Stage 1: Input."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from xgen_agent_runtime.stages.s01_input.interface import InputValidator


def _text_and_attachment_state(raw_input: Any) -> tuple[str, bool]:
    """Return user text without serializing binary attachment payloads."""

    if isinstance(raw_input, dict):
        has_attachments = any(
            bool(raw_input.get(key)) for key in ("attachments", "images", "files")
        )
        if not has_attachments and not any(
            key in raw_input for key in ("text", "content", "input_str")
        ):
            return str(raw_input).strip(), False
        value = raw_input.get("text", raw_input.get("content", raw_input.get("input_str", "")))
        if isinstance(value, list):
            text = "\n".join(
                str(item.get("text") or "")
                for item in value
                if isinstance(item, dict) and item.get("type") in ("text", "input_text")
            )
        else:
            text = str(value or "")
        return text.strip(), has_attachments
    if isinstance(raw_input, list):
        text = "\n".join(
            str(item.get("text") or "")
            for item in raw_input
            if isinstance(item, dict) and item.get("type") in ("text", "input_text")
        )
        has_attachments = any(
            isinstance(item, dict)
            and item.get("type") in ("image", "file", "document", "image_url")
            for item in raw_input
        )
        if not text and not has_attachments:
            return str(raw_input).strip(), False
        return text.strip(), has_attachments
    return str(raw_input).strip(), False


class DefaultValidator(InputValidator):
    """Standard validator — length and type checks."""

    def __init__(self, max_length: int = 1_000_000, min_length: int = 1):
        self._max_length = max_length
        self._min_length = min_length

    @property
    def name(self) -> str:
        return "default"

    @property
    def description(self) -> str:
        return "Standard length and type validation"

    def validate(self, raw_input: Any) -> Optional[str]:
        if raw_input is None:
            return "Input cannot be None"

        text, has_attachments = _text_and_attachment_state(raw_input)
        if len(text) < self._min_length and not has_attachments:
            return f"Input too short (min {self._min_length} chars)"
        if len(text) > self._max_length:
            return f"Input too long (max {self._max_length} chars)"
        return None


class PassthroughValidator(InputValidator):
    """No validation — pass everything through."""

    @property
    def name(self) -> str:
        return "passthrough"

    @property
    def description(self) -> str:
        return "No validation, accepts all input"

    def validate(self, raw_input: Any) -> Optional[str]:
        return None


class StrictValidator(InputValidator):
    """Strict validator — additional pattern checks."""

    def __init__(
        self,
        max_length: int = 100_000,
        blocked_patterns: Optional[List[str]] = None,
    ):
        self._max_length = max_length
        self._blocked_patterns = blocked_patterns or []

    @property
    def name(self) -> str:
        return "strict"

    @property
    def description(self) -> str:
        return "Strict validation with pattern blocking"

    def configure(self, config: Dict[str, Any]) -> None:
        # Manifest-restore feeds strategy_configs.validator into here. The
        # ctor wires the same fields, but without configure() the snapshot
        # restore path silently drops them (mutation.py catches AttributeError).
        patterns = config.get("blocked_patterns")
        if isinstance(patterns, list):
            self._blocked_patterns = [str(p) for p in patterns]
        max_length = config.get("max_length")
        if isinstance(max_length, int) and max_length > 0:
            self._max_length = max_length

    def get_config(self) -> Dict[str, Any]:
        return {
            "blocked_patterns": list(self._blocked_patterns),
            "max_length": self._max_length,
        }

    def validate(self, raw_input: Any) -> Optional[str]:
        if raw_input is None:
            return "Input cannot be None"

        text, has_attachments = _text_and_attachment_state(raw_input)
        if not text and not has_attachments:
            return "Input cannot be empty"
        if len(text) > self._max_length:
            return f"Input too long (max {self._max_length} chars)"

        text_lower = text.lower()
        for pattern in self._blocked_patterns:
            if pattern.lower() in text_lower:
                return "Input contains blocked pattern"

        return None


class SchemaValidator(InputValidator):
    """JSON Schema based validation for structured input."""

    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        self._schema = schema or {}

    @property
    def name(self) -> str:
        return "schema"

    @property
    def description(self) -> str:
        return "JSON Schema based validation"

    def configure(self, config: Dict[str, Any]) -> None:
        schema = config.get("schema")
        if isinstance(schema, dict):
            self._schema = schema

    def get_config(self) -> Dict[str, Any]:
        return {"schema": self._schema}

    def validate(self, raw_input: Any) -> Optional[str]:
        if not isinstance(raw_input, dict):
            return "Input must be a dictionary for schema validation"
        # Basic type checking without jsonschema dependency
        required = self._schema.get("required", [])
        for key in required:
            if key not in raw_input:
                return f"Missing required field: {key}"
        return None
