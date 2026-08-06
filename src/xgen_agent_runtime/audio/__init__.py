"""Audio capability layer — model-gated media bridges.

The model's content vocabulary is text/image/document only: there is no
audio block. This package therefore holds the bridges that turn
workspace audio into something the model can consume — starting with
STT (:mod:`xgen_agent_runtime.audio.stt`). The tool surface lives in
``tools/built_in/audio_tools.py`` and is hidden behind
``feature:stt_enabled`` until the host wires a provider.
"""

from xgen_agent_runtime.audio.stt import (  # noqa: F401
    STTError,
    STTProvider,
    STTResult,
    STTSegment,
    create_stt_client,
    register_stt_provider,
    unregister_stt_provider,
)

__all__ = [
    "STTError",
    "STTProvider",
    "STTResult",
    "STTSegment",
    "create_stt_client",
    "register_stt_provider",
    "unregister_stt_provider",
]
