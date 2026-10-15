"""span 속성 키 — OpenTelemetry GenAI semantic conventions + ``xgen.*`` 확장.

호스트(xgen-workflow ``service/observability/attributes.py``)와 **같은 키**를 쓴다. 대시보드·알림이
키 이름으로 읽으므로, 여기서 이름을 바꾸면 호스트 쪽 조회가 조용히 비게 된다.
"""

from __future__ import annotations

# ── GenAI semantic conventions ─────────────────────────────────────────────
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_RESPONSE_ID = "gen_ai.response.id"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read.input_tokens"
GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS = "gen_ai.usage.cache_creation.input_tokens"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_TYPE = "gen_ai.tool.type"
ERROR_TYPE = "error.type"

OP_CHAT = "chat"
OP_EXECUTE_TOOL = "execute_tool"
TOOL_TYPE_FUNCTION = "function"

# ── xgen 확장 ──────────────────────────────────────────────────────────────
XGEN_LLM_STREAM = "xgen.llm.stream"
XGEN_LLM_MESSAGE_COUNT = "xgen.llm.message_count"
XGEN_LLM_HAS_TOOLS = "xgen.llm.has_tools"
XGEN_LLM_OUTPUT_LENGTH = "xgen.llm.output_length"
XGEN_LLM_TOOL_CALLS = "xgen.llm.tool_calls"
XGEN_TOOL_IS_ERROR = "xgen.tool.is_error"
XGEN_TOOL_RESULT_SIZE = "xgen.tool.result_size"
