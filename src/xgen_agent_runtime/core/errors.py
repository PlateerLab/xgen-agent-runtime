"""Error classification and exception hierarchy.

Two parallel classification dimensions
--------------------------------------

* :class:`ErrorCategory` (existing) — coarse, retry-aware buckets used by
  the executor's own retry / backoff machinery. Stable since 2.0.x; one
  value per *behavioural* class (recoverable vs fatal).

* :class:`ExecutorErrorCode` (new in 2.1.0) — fine-grained, stable string
  identifier the host (Geny, CI runners, downstream consumers) uses for
  logging, i18n, telemetry grouping, and alerting. Each code is a stable
  string in the form ``exec.<component>.<reason>``. Once published, a
  code's value MUST NOT change — see ``docs/error_codes.md``.

The two coexist on every :class:`GenyExecutorError`:

  - ``e.code`` answers *what specifically went wrong* (``exec.cli.auth_failed``)
  - ``e.category`` (on :class:`APIError`) answers *should we retry?*
    (``ErrorCategory.CLI_AUTH_FAILED.is_fatal == True``)

Backward compatibility
----------------------
Existing call sites of the form
``raise APIError("...", category=ErrorCategory.CLI_AUTH_FAILED)``
keep working unchanged — the base class now derives a reasonable default
``code`` from the ``category`` argument via :data:`_CATEGORY_TO_CODE_DEFAULT`.
Sites that want to be specific can pass ``code=ExecutorErrorCode.EXEC_CLI_AUTH_FAILED``
explicitly; the explicit code wins. The ``code`` attribute is always set
(no ``None`` checks needed downstream).
"""

import re
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────────────── ExecutorErrorCode ─


class ExecutorErrorCode(str, Enum):
    """Stable, fine-grained error identifiers (2.1.0+).

    Format: ``exec.<component>.<reason>``. Lowercase, dot-separated, ≤4
    segments deep. Designed for:

      - Host logging / Sentry grouping
      - Frontend i18n key lookup (Geny maps ``exec.cli.auth_failed`` →
        ``t("executor.exec.cli.auth_failed")``)
      - Telemetry alerting (route ``exec.api.*`` rate to a different
        channel than ``exec.cli.*``)
      - Operator dashboards

    Stability contract:
      - **NEVER renumber** — once shipped, a code's string value is API surface.
      - **NEVER repurpose** — if the meaning of a code drifts, deprecate
        and add a new code rather than mutating the old one.
      - Adding new codes is non-breaking.

    The full list with descriptions, recoverability, and example
    scenarios lives in ``docs/error_codes.md`` — keep it in sync.
    """

    # ── exec.api.* — vendor API surface (Anthropic/OpenAI/Google/vLLM SDK) ──
    EXEC_API_AUTH_INVALID_KEY = "exec.api.auth.invalid_key"
    EXEC_API_AUTH_EXPIRED = "exec.api.auth.expired"
    EXEC_API_RATE_LIMITED = "exec.api.rate_limited"
    EXEC_API_TIMEOUT = "exec.api.timeout"
    EXEC_API_NETWORK = "exec.api.network"
    EXEC_API_TOKEN_LIMIT = "exec.api.token_limit"
    EXEC_API_BAD_REQUEST = "exec.api.bad_request"
    EXEC_API_SERVER_ERROR = "exec.api.server_error"
    EXEC_API_TERMINAL = "exec.api.terminal"
    EXEC_API_UNKNOWN = "exec.api.unknown"
    EXEC_API_NO_CLIENT = "exec.api.no_client"
    EXEC_API_STREAM_INCOMPLETE = "exec.api.stream_incomplete"
    EXEC_API_RETRY_EXHAUSTED = "exec.api.retry_exhausted"

    # ── exec.cli.* — CLI-driven backends (claude_code_cli) ──
    EXEC_CLI_BINARY_NOT_FOUND = "exec.cli.binary_not_found"
    EXEC_CLI_AUTH_FAILED = "exec.cli.auth_failed"
    EXEC_CLI_TIMEOUT = "exec.cli.timeout"
    EXEC_CLI_PROTOCOL_ERROR = "exec.cli.protocol_error"
    EXEC_CLI_PERMISSION_DENIED = "exec.cli.permission_denied"
    EXEC_CLI_EXITED = "exec.cli.exited"

    # ── exec.pipeline.* / exec.stage.* — pipeline / stage orchestration ──
    EXEC_PIPELINE_NOT_INITIALIZED = "exec.pipeline.not_initialized"
    EXEC_PIPELINE_INVALID_MANIFEST = "exec.pipeline.invalid_manifest"
    EXEC_STAGE_FAILED = "exec.stage.failed"
    EXEC_STAGE_GUARD_REJECTED = "exec.stage.guard_rejected"

    # ── exec.tool.* — Stage 10 tool dispatch ──
    EXEC_TOOL_UNKNOWN = "exec.tool.unknown"
    EXEC_TOOL_INVALID_INPUT = "exec.tool.invalid_input"
    EXEC_TOOL_ACCESS_DENIED = "exec.tool.access_denied"
    EXEC_TOOL_CRASHED = "exec.tool.crashed"
    EXEC_TOOL_TRANSPORT = "exec.tool.transport"

    # ── exec.mutation.* — runtime config mutation ──
    EXEC_MUTATION_INVALID = "exec.mutation.invalid"
    EXEC_MUTATION_LOCKED = "exec.mutation.locked"

    # ── exec.mcp.* — MCP server lifecycle ──
    EXEC_MCP_CONNECT_FAILED = "exec.mcp.connect_failed"
    EXEC_MCP_INITIALIZE_FAILED = "exec.mcp.initialize_failed"
    EXEC_MCP_LIST_TOOLS_FAILED = "exec.mcp.list_tools_failed"
    EXEC_MCP_SDK_MISSING = "exec.mcp.sdk_missing"

    # ── exec.unknown — fallback when no code is attached ──
    EXEC_UNKNOWN = "exec.unknown"

    @classmethod
    def from_category(cls, category: "ErrorCategory") -> "ExecutorErrorCode":
        """Map a legacy ``ErrorCategory`` to a default code.

        Used by :class:`APIError` to keep existing call sites
        (``raise APIError("...", category=ErrorCategory.CLI_AUTH_FAILED)``)
        working unchanged — the resulting exception still gets a
        meaningful ``code`` attribute without the caller having to
        thread it explicitly.
        """
        return _CATEGORY_TO_CODE_DEFAULT.get(category, cls.EXEC_UNKNOWN)


# ─────────────────────────────────────────────────────── ErrorCategory ─


class ErrorCategory(str, Enum):
    """API error classification for retry decisions."""

    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    NETWORK = "network"
    TOKEN_LIMIT = "token_limit"
    AUTH = "auth"
    BAD_REQUEST = "bad_request"
    SERVER_ERROR = "server_error"
    TERMINAL = "terminal"
    UNKNOWN = "unknown"

    # --- CLI-backend specific categories ---
    CLI_NOT_FOUND = "cli_not_found"
    CLI_AUTH_FAILED = "cli_auth_failed"
    CLI_TIMEOUT = "cli_timeout"
    CLI_PROTOCOL_ERROR = "cli_protocol_error"
    CLI_PERMISSION_DENIED = "cli_permission_denied"
    #: 구독 한도·지출 한도 소진. 같은 계정으로 다시 보내도 초기화 전에는 풀리지
    #: 않으므로 재시도 대상이 아니다(is_recoverable False). 호스트가 다른 계정으로
    #: 넘길지 정한다. APIError.details 에 reset_at·rate_limits 가 실린다.
    CLI_RATE_LIMITED = "cli_rate_limited"

    @property
    def is_recoverable(self) -> bool:
        return self in {
            ErrorCategory.RATE_LIMITED,
            ErrorCategory.TIMEOUT,
            ErrorCategory.NETWORK,
            ErrorCategory.SERVER_ERROR,
            ErrorCategory.CLI_TIMEOUT,
            ErrorCategory.CLI_PROTOCOL_ERROR,
        }

    @property
    def is_fatal(self) -> bool:
        """Categories where retry has no chance of recovery."""
        return self in {
            ErrorCategory.AUTH,
            ErrorCategory.BAD_REQUEST,
            ErrorCategory.CLI_NOT_FOUND,
            ErrorCategory.CLI_AUTH_FAILED,
            ErrorCategory.CLI_PERMISSION_DENIED,
        }


# ─────────────────────────────────────────────────────── default mapping ─


# Category → default error code. Kept private — call sites should rely on
# the public ``ExecutorErrorCode.from_category(...)`` accessor so the
# mapping can evolve without churn on consumers.
_CATEGORY_TO_CODE_DEFAULT: dict = {
    ErrorCategory.RATE_LIMITED: ExecutorErrorCode.EXEC_API_RATE_LIMITED,
    ErrorCategory.TIMEOUT: ExecutorErrorCode.EXEC_API_TIMEOUT,
    ErrorCategory.NETWORK: ExecutorErrorCode.EXEC_API_NETWORK,
    ErrorCategory.TOKEN_LIMIT: ExecutorErrorCode.EXEC_API_TOKEN_LIMIT,
    ErrorCategory.AUTH: ExecutorErrorCode.EXEC_API_AUTH_INVALID_KEY,
    ErrorCategory.BAD_REQUEST: ExecutorErrorCode.EXEC_API_BAD_REQUEST,
    ErrorCategory.SERVER_ERROR: ExecutorErrorCode.EXEC_API_SERVER_ERROR,
    ErrorCategory.TERMINAL: ExecutorErrorCode.EXEC_API_TERMINAL,
    ErrorCategory.UNKNOWN: ExecutorErrorCode.EXEC_API_UNKNOWN,
    ErrorCategory.CLI_NOT_FOUND: ExecutorErrorCode.EXEC_CLI_BINARY_NOT_FOUND,
    ErrorCategory.CLI_AUTH_FAILED: ExecutorErrorCode.EXEC_CLI_AUTH_FAILED,
    ErrorCategory.CLI_TIMEOUT: ExecutorErrorCode.EXEC_CLI_TIMEOUT,
    ErrorCategory.CLI_PROTOCOL_ERROR: ExecutorErrorCode.EXEC_CLI_PROTOCOL_ERROR,
    ErrorCategory.CLI_PERMISSION_DENIED: ExecutorErrorCode.EXEC_CLI_PERMISSION_DENIED,
    ErrorCategory.CLI_RATE_LIMITED: ExecutorErrorCode.EXEC_API_RATE_LIMITED,
}


# ─────────────────────────────────────────────────────── exceptions ─


class GenyExecutorError(Exception):
    """Base exception for xgen-agent-runtime.

    Every executor exception carries a stable :class:`ExecutorErrorCode`
    accessible as ``e.code``. Subclasses set a class-level default in
    :attr:`_DEFAULT_CODE` so call sites that pre-date the 2.1.0 code
    field still get a meaningful code on the resulting exception.
    """

    _DEFAULT_CODE: ExecutorErrorCode = ExecutorErrorCode.EXEC_UNKNOWN

    def __init__(
        self,
        message: str,
        *,
        code: Optional[ExecutorErrorCode] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.code: ExecutorErrorCode = code or self._DEFAULT_CODE
        self.cause = cause


class PipelineError(GenyExecutorError):
    """Pipeline-level error."""

    _DEFAULT_CODE = ExecutorErrorCode.EXEC_PIPELINE_NOT_INITIALIZED


class StageError(GenyExecutorError):
    """Stage execution error."""

    _DEFAULT_CODE = ExecutorErrorCode.EXEC_STAGE_FAILED

    def __init__(
        self,
        message: str,
        *,
        stage_name: str = "",
        stage_order: int = 0,
        code: Optional[ExecutorErrorCode] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, code=code, cause=cause)
        self.stage_name = stage_name
        self.stage_order = stage_order


class GuardRejectError(StageError):
    """Guard rejected execution."""

    _DEFAULT_CODE = ExecutorErrorCode.EXEC_STAGE_GUARD_REJECTED

    def __init__(self, message: str, *, guard_name: str = "", **kwargs):
        super().__init__(message, stage_name="guard", stage_order=4, **kwargs)
        self.guard_name = guard_name


class APIError(GenyExecutorError):
    """API call error with classification.

    Carries both a coarse :class:`ErrorCategory` (used by the executor's
    retry machinery) and a fine-grained :class:`ExecutorErrorCode` (used
    by hosts for logging / i18n / telemetry). When ``code`` is omitted,
    it is derived from ``category`` via
    :meth:`ExecutorErrorCode.from_category` so legacy call sites keep
    their existing semantics.
    """

    def __init__(
        self,
        message: str,
        *,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        status_code: Optional[int] = None,
        code: Optional[ExecutorErrorCode] = None,
        cause: Optional[Exception] = None,
        details: Optional[dict] = None,
    ):
        # Resolve code: explicit > derived-from-category > UNKNOWN.
        resolved_code = code or ExecutorErrorCode.from_category(category)
        super().__init__(message, code=resolved_code, cause=cause)
        self.category = category
        self.status_code = status_code
        #: 분류에 딸린 구조화 값(없으면 빈 dict). CLI_RATE_LIMITED 는
        #: ``reset_at``(유닉스 초 또는 None)·``rate_limits``·``kind`` 를 싣는다.
        self.details: dict = dict(details or {})


class ToolExecutionError(GenyExecutorError):
    """Tool execution failed."""

    _DEFAULT_CODE = ExecutorErrorCode.EXEC_TOOL_CRASHED

    def __init__(
        self,
        message: str,
        *,
        tool_name: str = "",
        code: Optional[ExecutorErrorCode] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, code=code, cause=cause)
        self.tool_name = tool_name


class MutationError(GenyExecutorError):
    """Invalid mutation request (bad stage/slot/impl)."""

    _DEFAULT_CODE = ExecutorErrorCode.EXEC_MUTATION_INVALID

    def __init__(
        self,
        message: str,
        *,
        stage_order: int = 0,
        slot_name: str = "",
        code: Optional[ExecutorErrorCode] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, code=code, cause=cause)
        self.stage_order = stage_order
        self.slot_name = slot_name


class MutationLocked(GenyExecutorError):
    """Mutation blocked because the target stage is currently executing."""

    _DEFAULT_CODE = ExecutorErrorCode.EXEC_MUTATION_LOCKED

    def __init__(
        self,
        message: str,
        *,
        stage_order: int = 0,
        code: Optional[ExecutorErrorCode] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, code=code, cause=cause)
        self.stage_order = stage_order


# ─────────────────────────────────────────────── context overflow detection ─

#: Phrases providers use when the PROMPT is larger than the model accepts (lower-case). One list for
#: every client: before 4.89.0 only the Anthropic client recognised an overflow, so OpenAI / vLLM /
#: Azure / Gemini overflows were plain BAD_REQUEST and nothing tried to recover.
CONTEXT_OVERFLOW_PHRASES: tuple = (
    "maximum context length",  # OpenAI chat, vLLM, most OpenAI-compatible servers
    "context_length_exceeded",  # OpenAI error code (Chat Completions / Responses)
    "reduce the length of the messages",  # OpenAI
    "prompt is too long",  # Anthropic
    "exceed context limit",  # Anthropic: "input length and `max_tokens` exceed context limit"
    "input length exceeds",
    "input is too long",  # Bedrock
    "exceeds the maximum number of tokens",  # Gemini (AI Studio)
    "model only supports up to",  # Gemini on Vertex: "input token count is N but model only supports up to M"
    "more than the max tokens limit",  # Gemini on Vertex: "Input request contains (N) tokens, which is more than ..."
    "exceeds the context window",
    "context window exceeded",
    "longer than the model's context length",
)


def is_context_overflow_message(message: str) -> bool:
    """True when an error message says the prompt does not fit the model's context window."""
    text = str(message or "").lower()
    return any(phrase in text for phrase in CONTEXT_OVERFLOW_PHRASES)


_OVERFLOW_COUNT_PATTERNS = (
    # Anthropic: "prompt is too long: 210000 tokens > 200000 maximum"
    re.compile(r"prompt is too long:\s*([\d,]+)\s*tokens", re.I),
    # Anthropic: "input length and `max_tokens` exceed context limit: 190000 + 21333 > 200000"
    re.compile(r"exceed context limit:\s*([\d,]+)\s*\+", re.I),
    # OpenAI: "However, your messages resulted in 140000 tokens"
    re.compile(r"messages resulted in\s*([\d,]+)\s*tokens", re.I),
    # vLLM / OpenAI: "you requested 40000 tokens (32000 in the messages, 8000 in the completion)"
    re.compile(r"\(([\d,]+)\s*in the messages", re.I),
    # vLLM >= 0.6: "your request has 40000 input tokens"
    re.compile(r"request has\s*([\d,]+)\s*input tokens", re.I),
    # Gemini: "The input token count (1234567) exceeds the maximum number of tokens allowed"
    # Vertex: "... because the input token count is 1234567 but model only supports up to 1048576"
    re.compile(r"input token count\s*(?:is\s*)?\(?\s*([\d,]+)", re.I),
    # Vertex: "Input request contains (1234567) tokens, which is more than the max tokens limit"
    re.compile(r"contains\s*\(\s*([\d,]+)\s*\)\s*tokens", re.I),
)


_OVERFLOW_LIMIT_PATTERNS = (
    # Anthropic: "prompt is too long: 210000 tokens > 200000 maximum"
    re.compile(r"tokens\s*>\s*([\d,]+)\s*maximum", re.I),
    # Anthropic: "input length and `max_tokens` exceed context limit: 190000 + 21333 > 200000"
    re.compile(r"exceed context limit:\s*[\d,]+\s*\+\s*[\d,]+\s*>\s*([\d,]+)", re.I),
    # OpenAI / vLLM: "This model's maximum context length is 128000 tokens."
    re.compile(r"maximum context length is\s*([\d,]+)", re.I),
    # Gemini: "... exceeds the maximum number of tokens allowed (1048576)."
    re.compile(r"maximum number of tokens allowed\s*\(?\s*([\d,]+)", re.I),
    # Vertex: "... but model only supports up to 1048576."
    re.compile(r"only supports up to\s*([\d,]+)", re.I),
    # Vertex: "... more than the max tokens limit allowed (1048576)."
    re.compile(r"max tokens limit(?:\s*allowed)?\s*\(?\s*([\d,]+)", re.I),
)


def _first_int(patterns: tuple, message: str) -> int:
    text = str(message or "")
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            try:
                return int(match.group(1).replace(",", ""))
            except ValueError:
                continue
    return 0


def parse_overflow_tokens(message: str) -> int:
    """The prompt size a provider reported in its overflow error, or 0 when it said none."""
    return _first_int(_OVERFLOW_COUNT_PATTERNS, message)


def parse_overflow_limit(message: str) -> int:
    """The context window a provider named in its overflow error, or 0 when it named none."""
    return _first_int(_OVERFLOW_LIMIT_PATTERNS, message)
