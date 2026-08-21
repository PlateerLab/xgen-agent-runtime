"""OpenAI Codex CLI backend.

Wraps OpenAI's ``codex`` command-line agent as a :class:`BaseClient` —
the second subprocess backend after Claude Code, speaking the same
canonical APIRequest/APIResponse contract. Process plumbing (spawn, env
scrubbing, kill ladder, line streaming) reuses the vendor-neutral
:class:`CLIProcessRunner`; the Codex-specific wire lives in
``translators/_codex.py``.

Authentication
--------------
``codex`` reads credentials from one of:
  - ``OPENAI_API_KEY`` env var (passed by this client when ``api_key=``
    is set and ``auth_mode`` resolves to the API-key channel)
  - ChatGPT subscription login stored under ``$CODEX_HOME/auth.json``
    (``codex login`` on the host)

``auth_mode`` (``'api_key' | 'oauth' | 'auto'``) declares the channel;
``'auto'`` resolves from ``api_key`` presence. In subscription mode the
API key is deliberately **not** exported — leaking it flips the CLI's
billing channel silently.

Tool execution
--------------
Like Claude Code, the CLI executes its own tools (shell, file edits,
MCP) inside the subprocess: ``is_subprocess && supports_tools &&
requires_workspace`` makes the host tool stage skip dispatch. MCP
servers travel as ``-c mcp_servers.*`` config overrides so the user's
``$CODEX_HOME`` (and the login stored there) stays untouched.
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import aclosing
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Sequence

from xgen_agent_runtime.core.config import ModelConfig
from xgen_agent_runtime.core.errors import APIError, ErrorCategory
from xgen_agent_runtime.llm_client._cli_runtime import (
    CLIAuthFailed,
    CLIBinaryNotFound,
    CLIProcessRunner,
    CLIProtocolError,
    CLIResult,
    CLITimeout,
    aiter_bytes,
    detect_binary,
    parse_stream_json_line,
)
from xgen_agent_runtime.llm_client.base import BaseClient, ClientCapabilities
from xgen_agent_runtime.llm_client.translators._codex import (
    CodexEventAccumulator,
    codex_argv,
    flatten_messages_to_prompt,
    parse_codex_output_to_response,
)
from xgen_agent_runtime.llm_client.types import APIRequest, APIResponse

logger = logging.getLogger(__name__)

__all__ = ["CodexCLIClient"]


#: Anchored stderr/wire phrases that indicate an authentication failure.
#: Same posture as the Claude backend: only phrases the CLI itself emits
#: on credential problems — never generic 'auth'/'fail' substrings.
_AUTH_FAILURE_PHRASES = (
    "not logged in",
    "please run `codex login`",
    "please run codex login",
    "invalid api key",
    "unauthorized",
    "authentication failed",
)


def _classify_cli_result(result: CLIResult, *, cli_version: str = "") -> APIError:
    stderr = result.stderr.decode("utf-8", errors="replace").lower()
    suffix = f" [cli_version={cli_version}]" if cli_version else ""
    if any(phrase in stderr for phrase in _AUTH_FAILURE_PHRASES):
        return APIError(
            f"Codex CLI auth failed (exit {result.returncode}): {stderr[:300]}{suffix}",
            category=ErrorCategory.CLI_AUTH_FAILED,
        )
    if "permission" in stderr and ("denied" in stderr or "deny" in stderr or "blocked" in stderr):
        return APIError(
            f"Codex CLI permission denied: {stderr[:300]}{suffix}",
            category=ErrorCategory.CLI_PERMISSION_DENIED,
        )
    return APIError(
        f"Codex CLI exited with code {result.returncode}: {stderr[:300]}{suffix}",
        category=ErrorCategory.CLI_PROTOCOL_ERROR,
    )


class CodexCLIClient(BaseClient):
    """Subprocess-backed OpenAI Codex client."""

    provider = "codex_cli"
    capabilities = ClientCapabilities(
        supports_thinking=True,
        supports_tools=True,
        supports_streaming=True,
        supports_tool_choice=False,
        supports_stop_sequences=False,
        supports_top_k=False,
        supports_system_prompt=True,
        supports_structured_output=True,
        supports_session_continuity=True,
        supports_mcp_passthrough=True,
        supports_budget_limit=False,
        supports_token_usage=True,
        supports_cost_usage=False,
        is_subprocess=True,
        requires_workspace=True,
        # Codex emits completed items (message granularity), not
        # per-token deltas — declare honestly so hosts don't wait for
        # token cadence that never comes.
        streaming_granularity="message",
        drops=(
            "tool_choice",
            "stop_sequences",
            "top_k",
            "temperature",
            "top_p",
            "max_tokens",
        ),
    )

    _VERSION_PROBE_TIMEOUT_S = 10.0

    def __init__(
        self,
        *,
        binary_path: Optional[str] = None,
        workspace_dir: Optional[str] = None,
        api_key: str = "",
        auth_mode: str = "auto",
        sandbox_mode: str = "workspace-write",
        bypass_sandbox: bool = False,
        mcp_config: Any = None,
        extra_args: Sequence[str] = (),
        timeout_s: float = 300.0,
        env_extras: Optional[Dict[str, str]] = None,
        event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        strict_wire: bool = False,
        runner_factory: Optional[Callable[..., CLIProcessRunner]] = None,
        session_hint: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=None,
            default_headers=None,
            event_sink=event_sink,
        )
        if binary_path:
            self._binary = detect_binary("codex", binary_path) or ""
        else:
            env_override = os.environ.get("CODEX_BINARY", "")
            self._binary = (
                (detect_binary("codex", env_override) if env_override else None)
                or detect_binary("codex", None)
                or ""
            )
        self._workspace_dir = workspace_dir
        self._auth_mode = auth_mode
        self._sandbox_mode = sandbox_mode
        self._bypass_sandbox = bool(bypass_sandbox)
        self._mcp_config = mcp_config
        self._extra_args = tuple(extra_args)
        self._timeout_s = timeout_s
        self._extra_env: Dict[str, str] = dict(env_extras) if env_extras else {}
        self._strict_wire = strict_wire
        self._runner_factory = runner_factory
        self._session_hint: Optional[Dict[str, Any]] = dict(session_hint) if session_hint else None
        self._cli_version_value: Optional[str] = None

    # ─────────────────────────────────────────────────────── helpers ─

    def _env_extras(self) -> Dict[str, str]:
        extras: Dict[str, str] = dict(self._extra_env)
        mode = self._auth_mode
        if mode == "auto":
            mode = "api_key" if self._api_key else "oauth"
        # Subscription (ChatGPT login) mode must NOT see the API key —
        # its presence silently flips the CLI onto API billing (the same
        # trap the Claude backend documents for --bare).
        if mode == "api_key" and self._api_key:
            extras["OPENAI_API_KEY"] = self._api_key
        return extras

    def _make_runner(self, *, timeout_s: Optional[float] = None) -> CLIProcessRunner:
        effective_timeout = self._timeout_s if timeout_s is None else timeout_s
        if self._runner_factory is not None:
            return self._runner_factory(
                binary=self._binary,
                cwd=self._workspace_dir,
                env_extras=self._env_extras(),
                timeout_s=effective_timeout,
            )
        if not self._binary:
            raise CLIBinaryNotFound(
                "codex binary not found. Set binary_path=, CODEX_BINARY env var, "
                "or ensure 'codex' is on PATH."
            )
        return CLIProcessRunner(
            binary=self._binary,
            cwd=self._workspace_dir,
            env_extras=self._env_extras(),
            timeout_s=effective_timeout,
        )

    def _schema_tempfile(self, request: APIRequest) -> str:
        """Write a json_schema response_format to a temp file for
        ``--output-schema``; empty string when not applicable."""
        rf = request.response_format or {}
        if rf.get("type") != "json_schema":
            return ""
        schema = rf.get("json_schema") or {}
        # OpenAI-style nesting ({"json_schema": {"schema": {...}}}) and a
        # bare schema dict are both accepted from callers.
        payload = (
            schema.get("schema") if isinstance(schema, dict) and "schema" in schema else schema
        )
        if not payload:
            return ""
        try:
            fd, path = tempfile.mkstemp(prefix="codex-schema-", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as fp:
                import json as _json

                _json.dump(payload, fp, ensure_ascii=False)
            return path
        except OSError:
            logger.debug("codex: schema tempfile write failed", exc_info=True)
            return ""

    def _build_argv(self, request: APIRequest, *, output_schema_path: str = "") -> List[str]:
        return codex_argv(
            request,
            sandbox_mode=self._sandbox_mode,
            bypass_sandbox=self._bypass_sandbox,
            mcp_config=self._mcp_config,
            output_schema_path=output_schema_path,
            extra_args=self._extra_args,
        )

    def _build_stdin(self, request: APIRequest) -> bytes:
        parts: List[str] = []
        system = request.system
        if isinstance(system, str) and system.strip():
            parts.append(system.strip())
        elif isinstance(system, list):
            texts = [str(b.get("text", "")) for b in system if isinstance(b, dict)]
            joined = "\n".join(t for t in texts if t).strip()
            if joined:
                parts.append(joined)
        prompt = flatten_messages_to_prompt(request.messages)
        if prompt:
            parts.append(prompt)
        return ("\n\n".join(parts) + "\n").encode("utf-8")

    def _build_request(
        self,
        *,
        model_config: ModelConfig,
        messages: List[Dict[str, Any]],
        system: Any,
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Dict[str, Any]],
        stream: bool,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> APIRequest:
        request = super()._build_request(
            model_config=model_config,
            messages=messages,
            system=system,
            tools=tools,
            tool_choice=tool_choice,
            stream=stream,
            response_format=response_format,
        )
        if self._session_hint and request.session_hint is None:
            request.session_hint = dict(self._session_hint)
        return request

    # ─────────────────────────────────────── version handshake ─

    async def _ensure_cli_version(self) -> str:
        if self._cli_version_value is not None:
            return self._cli_version_value
        version = "unknown"
        try:
            runner = self._make_runner(
                timeout_s=min(self._VERSION_PROBE_TIMEOUT_S, self._timeout_s)
            )
            result = await runner.run_oneshot(["--version"])
            text = result.stdout.decode("utf-8", errors="replace").strip()
            if result.returncode == 0 and text:
                version = text.splitlines()[0].strip()
        except Exception:  # noqa: BLE001 — the handshake is telemetry
            version = "unknown"
        self._cli_version_value = version
        logger.info("Codex CLI version handshake: %s (binary=%s)", version, self._binary)
        return version

    async def warmup(self, *, timeout_s: float = 8.0) -> bool:
        import asyncio

        try:
            await asyncio.wait_for(self._ensure_cli_version(), timeout=timeout_s)
            return self._cli_version_value not in (None, "unknown")
        except Exception:  # noqa: BLE001 — warmup is best-effort by contract
            return False

    def _with_version(self, message: str) -> str:
        if self._cli_version_value:
            return f"{message} [cli_version={self._cli_version_value}]"
        return message

    def _attach_cli_version(self, response: APIResponse) -> APIResponse:
        if isinstance(response.raw, dict) and self._cli_version_value:
            response.raw.setdefault("cli_version", self._cli_version_value)
        return response

    def _capture_session(self, response: APIResponse) -> None:
        """Remember the thread id so the next turn can ``exec resume``."""
        raw = response.raw if isinstance(response.raw, dict) else {}
        sid = str(raw.get("session_id") or "")
        if sid:
            self._session_hint = {"session_id": sid, "resume": True}

    # ─────────────────────────────────────── wire-shape telemetry ─

    def _report_unknown_wire(self, accum: CodexEventAccumulator) -> None:
        total = accum.unknown_line_count + accum.malformed_line_count
        if not total:
            return
        if self._event_sink is not None:
            self._event_sink(
                {
                    "type": "llm_client.unknown_wire_shape",
                    "provider": self.provider,
                    "unknown_type": accum.first_unknown_type,
                    "count": total,
                    "unknown_line_count": accum.unknown_line_count,
                    "malformed_line_count": accum.malformed_line_count,
                    "cli_version": self._cli_version_value or "unknown",
                }
            )
        if self._strict_wire:
            raise APIError(
                self._with_version(
                    f"Codex CLI emitted {total} unknown/malformed JSONL line(s) "
                    f"(first unknown type: {accum.first_unknown_type!r}) and this "
                    "client was constructed with strict_wire=True"
                ),
                category=ErrorCategory.CLI_PROTOCOL_ERROR,
            )

    def _raise_on_wire_error(self, line_obj: Dict[str, Any]) -> None:
        """Surface CLI-side error frames as APIError before accumulation."""
        etype = str(line_obj.get("type") or "")
        message = ""
        if etype == "error":
            message = str(line_obj.get("message") or line_obj)
        elif etype == "turn.failed":
            err = line_obj.get("error")
            message = str((err or {}).get("message") if isinstance(err, dict) else err or line_obj)
        elif etype == "item.completed":
            item = line_obj.get("item")
            if (
                isinstance(item, dict)
                and str(item.get("item_type") or item.get("type") or "") == "error"
            ):
                message = str(item.get("text") or item.get("message") or item)
        else:
            msg = line_obj.get("msg")
            if isinstance(msg, dict) and str(msg.get("type") or "") == "error":
                message = str(msg.get("message") or msg)
        if not message:
            return
        lowered = message.lower()
        if any(p in lowered for p in _AUTH_FAILURE_PHRASES):
            raise APIError(
                self._with_version(
                    "Codex CLI is not authenticated: run `codex login` on the "
                    f"host or configure an OpenAI API key. ({message[:200]})"
                ),
                category=ErrorCategory.CLI_AUTH_FAILED,
            )
        raise APIError(
            self._with_version(f"Codex CLI reported error: {message[:300]}"),
            category=ErrorCategory.CLI_PROTOCOL_ERROR,
        )

    # ─────────────────────────────────────────────────────── _send ─

    async def _send(self, request: APIRequest, *, purpose: str = "") -> APIResponse:
        try:
            runner = self._make_runner()
        except CLIBinaryNotFound as e:
            raise APIError(str(e), category=ErrorCategory.CLI_NOT_FOUND) from e

        cli_version = await self._ensure_cli_version()
        schema_path = self._schema_tempfile(request)
        argv = self._build_argv(request, output_schema_path=schema_path)
        stdin = self._build_stdin(request)

        try:
            result = await runner.run_oneshot(argv, stdin=stdin)
            if result.returncode != 0:
                raise _classify_cli_result(result, cli_version=cli_version)
            response = parse_codex_output_to_response(
                result.stdout, model=request.model, cli_version=cli_version
            )
            self._capture_session(response)
            # One-shot wire drift shares the streaming telemetry channel.
            accum_counts = response.raw if isinstance(response.raw, dict) else {}
            if accum_counts.get("unknown_line_count") or accum_counts.get("malformed_line_count"):
                shim = CodexEventAccumulator(model=request.model)
                shim.unknown_line_count = int(accum_counts.get("unknown_line_count", 0) or 0)
                shim.malformed_line_count = int(accum_counts.get("malformed_line_count", 0) or 0)
                shim.first_unknown_type = accum_counts.get("first_unknown_type")
                self._report_unknown_wire(shim)
            return self._attach_cli_version(response)
        except CLIBinaryNotFound as e:
            raise APIError(self._with_version(str(e)), category=ErrorCategory.CLI_NOT_FOUND) from e
        except CLITimeout as e:
            raise APIError(self._with_version(str(e)), category=ErrorCategory.CLI_TIMEOUT) from e
        except CLIAuthFailed as e:
            raise APIError(
                self._with_version(str(e)), category=ErrorCategory.CLI_AUTH_FAILED
            ) from e
        except CLIProtocolError as e:
            raise APIError(
                self._with_version(str(e)), category=ErrorCategory.CLI_PROTOCOL_ERROR
            ) from e
        finally:
            if schema_path:
                try:
                    os.unlink(schema_path)
                except OSError:
                    pass

    # ───────────────────────────────────────────────── streaming API ─

    async def create_message_stream(
        self,
        *,
        model_config: ModelConfig,
        messages: List[Dict[str, Any]],
        system: Any = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        purpose: str = "",
    ) -> AsyncIterator[Dict[str, Any]]:
        """Yield canonical events as the CLI streams JSONL items.

        Granularity is per-item (``streaming_granularity="message"``);
        the terminal ``{"type": "message_complete", "response": ...}``
        envelope matches every other client's streaming contract.
        """
        request = self._build_request(
            model_config=model_config,
            messages=messages,
            system=system,
            tools=tools,
            tool_choice=tool_choice,
            stream=True,
        )

        try:
            runner = self._make_runner()
        except CLIBinaryNotFound as e:
            raise APIError(str(e), category=ErrorCategory.CLI_NOT_FOUND) from e

        cli_version = await self._ensure_cli_version()
        schema_path = self._schema_tempfile(request)
        argv = self._build_argv(request, output_schema_path=schema_path)
        stdin = self._build_stdin(request)
        accum = CodexEventAccumulator(model=model_config.model, cli_version=cli_version)

        try:
            async with aclosing(runner.stream(argv, stdin_iter=aiter_bytes(stdin))) as lines:
                async for raw in lines:
                    line_obj = parse_stream_json_line(raw)
                    if line_obj is None:
                        continue
                    self._raise_on_wire_error(line_obj)
                    for event in accum.feed(line_obj):
                        yield event

            self._report_unknown_wire(accum)
            response = self._attach_cli_version(accum.finalize())
            self._capture_session(response)
            yield {"type": "message_complete", "response": response}
        except CLIBinaryNotFound as e:
            raise APIError(self._with_version(str(e)), category=ErrorCategory.CLI_NOT_FOUND) from e
        except CLITimeout as e:
            raise APIError(self._with_version(str(e)), category=ErrorCategory.CLI_TIMEOUT) from e
        except CLIProtocolError as e:
            raise APIError(
                self._with_version(str(e)), category=ErrorCategory.CLI_PROTOCOL_ERROR
            ) from e
        finally:
            if schema_path:
                try:
                    os.unlink(schema_path)
                except OSError:
                    pass
