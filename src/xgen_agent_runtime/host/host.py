"""HostServices — the extraction boundary for a single agent turn.

``AgentTurnExecutor`` (the lifted body of ``agent_geny.AgentGenyNode.execute``)
is host-agnostic: it assembles a turn (tools, sandbox, memory, prompt, provider
client), runs the ``xgen_agent_runtime`` pipeline, and tears down. Everything it
cannot do in pure Python — read admin settings, resolve credentials, reach the
sandbox runner, hydrate/publish a workspace, mount the user's cloud, build the
server-owned tool families — it obtains through this ``HostServices`` protocol.

Two implementations exist, and they MUST produce byte-identical turn behaviour
(the whole point of the extraction — the connector must never diverge from web):

* ``ServerHostServices`` (in xgen-workflow): admin config DB, MinIO+DB workspace
  store, xgen-workflow-sandbox HTTP runner, cloud DB, connector reverse-WS
  bridge. Web sessions and the current server-routed connector path.
* ``LocalHostServices`` (in the desktop connector's Python sidecar): env/local
  config, the local synced folder as the workspace, direct host execution
  (codex/claude_code/SDK all spawn locally), and thin RPC back to the server for
  the capabilities that are inherently multi-tenant (④ below).

Method groups map 1:1 to the dependency categories established in the extraction
survey (see memory ``xgeny-shared-host-extraction``):

  B/C/E-abstract → injected here (③ "needs abstraction").
  D/E-server, workspace store, sandbox runner → ④ "server-resident": the server
    impl does the real thing; the connector impl calls the server over RPC, or
    returns a graceful no-op when the capability is genuinely unavailable
    locally (e.g. self-evolution edits a server-owned graph).

Nothing here imports server symbols; the protocol is defined against
``xgen_agent_runtime`` types + plain data only, so this package stays a pure,
bundle-able dependency of BOTH hosts.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

if TYPE_CHECKING:
    # Import only for typing — keeps import-time deps to xgen_agent_runtime.
    from xgen_agent_runtime.tools import ToolRegistry
    from xgen_agent_runtime.tools._xgeny_sandbox import XgenySandbox

#: Cloud/shared mount tuple as produced by cloud_mount: (index, local_path[, mode]).
CloudMount = Tuple[Any, ...]
#: Result of a built provider LLM client + its per-run cleanup callback.
CliRuntime = Tuple[Any, Optional[Any]]


@runtime_checkable
class HostServices(Protocol):
    """Everything a turn needs from its host. See module docstring for the two
    implementations and the non-divergence contract they must honour."""

    # ── A. settings & credentials ────────────────────────────────────────
    # Server: admin config-composer DB → env → default. Connector: env/local
    # config → default (no DB). ``_cli_setting`` already carries the env
    # fallback, so this seam is a drop-in.
    def setting(self, name: str, default: str = "") -> str: ...
    def setting_truthy(self, name: str) -> bool: ...
    def resolve_model(self, provider: str, params: Mapping[str, Any]) -> str: ...
    def resolve_api_key(self, provider: str, params: Mapping[str, Any]) -> str: ...
    def resolve_base_url(self, provider: str, params: Mapping[str, Any]) -> Optional[str]: ...
    def resolve_credentials(
        self, provider: str, params: Mapping[str, Any]
    ) -> Optional[Dict[str, Any]]: ...

    # ── B. execution host: sandbox + workspace ───────────────────────────
    # The one seam the runtime already models (ToolContext.sandbox / XgenySandbox).
    # Server: SandboxSession (HTTP runner) or, for a connector-surface turn,
    # ConnectorLocalSandbox (reverse-WS to the user PC). Connector sidecar: a
    # direct-host sandbox rooted at the local synced folder — this is what makes
    # codex & every provider operate on local files natively.
    def probe_connector_workspace(
        self,
        user_id: Any,
        workflow_id: str,
        workflow_name: str,
        client_surface: Any,
        provider: str,
    ) -> Optional[dict]: ...
    def make_sandbox(
        self,
        workflow_id: str,
        user_id: Any,
        connector_ws: Optional[dict],
    ) -> Optional["XgenySandbox"]: ...
    def agent_workspace_dir(self, workflow_id: str, *, create: bool = True) -> str: ...
    def workspace_storage_root(self, workflow_id: str) -> str: ...
    #: Restore the persistent workspace from the source of truth into ``run_dir``
    #: BEFORE the turn. Returns True iff hydration succeeded — the executor only
    #: publishes afterwards when it did (empty-cache-deletes-everything guard).
    def hydrate_workspace(self, workflow_id: str, run_dir: str) -> bool: ...
    #: Reflect a turn's workspace changes back to the source of truth. Connector
    #: local mode is a no-op here (the connector's own sync engine owns it).
    def publish_workspace(
        self, workflow_id: str, run_dir: str, *, origin: str = "agent"
    ) -> None: ...

    #: 실행 환경 안내 프롬프트 — 도구가 **어디서** 도는지 에이전트에게 알린다.
    #: 서버: 러너 sandbox / 커넥터 로컬(ConnectorLocalSandbox) 설명. 커넥터
    #: 사이드카: 이 PC 로컬 환경 설명(OS 포함). 없으면 "".
    def environment_prompt(
        self, sandbox: Any, connector_ws: Optional[dict], provider: str
    ) -> str: ...

    # ── C. memory ────────────────────────────────────────────────────────
    # Server: xgen-db provider. Connector: file/sqlite vault OR server RPC so
    # web↔connector share the same memory (confirmed decision: state is shared).
    def build_memory_provider(self, workflow_id: str, interaction_id: str) -> Optional[Any]: ...

    # ── D. user cloud (④ server-resident) ────────────────────────────────
    def prepare_cloud(
        self, user_id: Any, workflow_id: str, *, pod_local: bool
    ) -> Optional[CloudMount]: ...
    def cloud_inventory(self, user_id: Any, path: str) -> str: ...
    def cloud_not_mounted_note(self, user_id: Any, workflow_id: str) -> str: ...
    def open_shared(self, sandbox: Any, user_id: Any, *, workflow_id: str) -> List[CloudMount]: ...
    def build_cloud_skill(
        self, index: Any, path: str, session: Any, user_id: Any
    ) -> Optional[Any]: ...
    #: Server-resident feature **prompt blocks** — tell the agent what cloud /
    #: jobs / shared folders are available. Server returns the product's prompt
    #: text; connector returns "" (feature absent locally) until wired via RPC.
    def cloud_prompt_block(self, path: str) -> str: ...
    def jobs_prompt_block(self) -> str: ...
    def shared_prompt_block(self, mounts: Sequence[Any]) -> str: ...
    #: The cloud byte-plane tool (registered into the SDK ToolRegistry) for a
    #: built cloud skill. Server returns the tool instance; connector → None.
    def build_cloud_file_tool(self, cloud_skill: Any) -> Optional[Any]: ...

    # ── E. server-owned tool families ────────────────────────────────────
    # Injected as tools into the runtime ToolRegistry (SDK path) or advertised
    # via the connector MCP bridge (CLI path). Each returns tools/None or
    # registers into the passed registry. Connector impl routes to server RPC
    # for the DB-backed ones (jobs/self-evolution/delegation) or no-ops when
    # unavailable locally, without ever changing the executor's call shape.
    def build_connector_mcp_tools(self, user_id: Any, client_surface: Any) -> List[Any]: ...
    def build_job_tools(
        self,
        workflow_id: str,
        workflow_name: str,
        user_id: Any,
        *,
        in_scheduled_run: bool,
        interaction_id: str,
    ) -> List[Any]: ...
    def register_workflow_self_tools(
        self,
        registry: "ToolRegistry",
        *,
        workflow_id: str,
        user_id: Any,
        workflow_name: str,
    ) -> None: ...
    #: Build the per-turn delegation extras (sub-pipeline factory + report sink).
    #: ``spec_fields`` is a plain dict of the turn's LLM/run fields — the server
    #: impl constructs its SubPipelineSpec from it (the spec type is server-owned,
    #: so it never appears in the shared executor). Connector → {} (no delegation).
    def build_turn_delegation(
        self,
        *,
        workflow_id: str,
        interaction_id: str,
        user_id: Any,
        spec_fields: Mapping[str, Any],
    ) -> Dict[str, Any]: ...
    #: True iff this turn's text is a delegation report (recursion guard). Server
    #: consults its delegation module; connector → False.
    def is_report_turn(self, text: str) -> bool: ...
    #: Server-owned delegation tool classes (name → class) to register into the
    #: SDK registry. Connector → {}.
    def delegation_extra_tool_classes(self) -> Dict[str, type]: ...
    #: Filesystem root for a workflow's delegation run when no sandbox is present.
    #: Server → local pod path; connector → the local synced folder.
    def delegation_workspace(self, workflow_id: str) -> str: ...
    #: Claim & format any unreported delegation completions to inject into this
    #: turn (alarm/pod-restart fallback). Server → the report block; connector → "".
    def drain_pending_reports(self, workflow_id: str, interaction_id: str) -> str: ...
    def make_sub_cli_client_factory(
        self, params: Mapping[str, Any], workflow_id: str
    ) -> Optional[Any]: ...
    def register_forged_tools(
        self,
        registry: "ToolRegistry",
        *,
        workflow_id: str,
        workspace_dir: str,
        core: bool,
        sandboxed: bool,
    ) -> None: ...
    def register_builtin_tools(
        self,
        registry: "ToolRegistry",
        *,
        core: bool,
        user_id: Any,
        anthropic_api_key: str,
        ssh_servers: Sequence[Any],
    ) -> Dict[str, Any]: ...
    def build_run_tool_context(self, **kwargs: Any) -> Any: ...
    def load_ssh_servers(self) -> List[Any]: ...

    # ── H. product helpers injected into the pure ② modules ──────────────
    # rag / token_budget / distill are otherwise-pure orchestration helpers;
    # these are the last xgen-workflow-specific calls they need, injected by the
    # host so the modules stay import-clean. Server delegates to editor; the
    # connector returns a graceful default (None/""/skip).
    #: Build one RAG context block from a retrieval port item (or None to skip).
    def rag_context_builder(self, text: str, item: Any) -> Optional[str]: ...
    #: Live vLLM ``max_model_len`` probe for an OpenAI-compatible base_url (or None).
    def fetch_vllm_max_model_len(self, base_url: str, model: Optional[str]) -> Optional[int]: ...
    #: Filesystem root of a workflow's memory vault (for distill pass-state).
    def agent_vault_root(self, workflow_id: str) -> str: ...
    #: Build the memory-distillation LLM client for a turn (or None if unavailable,
    #: e.g. claude_code subscription mode / codex).
    def build_turn_memory_llm(
        self,
        provider: str,
        model: str,
        api_key: str,
        base_url: Optional[str],
        *,
        cli_auth_mode: str = "",
        cli_oauth_token: str = "",
        cli_binary_path: str = "",
        credentials: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Any]: ...

    # ── G. turn teardown: reflect the turn's file changes ────────────────
    # The 3-way publish decision (connector-local flush / runner publish / pod
    # publish) + cloud/shared publish. Server does the real reflection; the
    # connector sidecar flushes through its own sync engine. Local resource
    # cleanups (cli/run-dir) stay in the executor — they are not host state.
    def finalize_turn(
        self,
        *,
        sandbox: Any,
        workflow_id: str,
        user_id: Any,
        hydrated_wf: str,
        hydrated_ws: Optional[str],
        shared_mounts: Sequence[Any],
        cloud_mount: Optional[CloudMount],
        connector_ws: Optional[dict],
    ) -> None: ...

    # ── F. CLI provider runtime (process spawn + connector MCP bridge) ────
    # Builds the claude_code / codex subprocess client. On the connector sidecar
    # this is exactly where "codex runs locally" lives: same runtime CLI client,
    # cwd = local workspace, so its native file/shell hit the user's machine.
    def build_cli_runtime(
        self,
        provider: str,
        params: Mapping[str, Any],
        *,
        cloud_workspace: str = "",
        shared_workspaces: Optional[Sequence[str]] = None,
    ) -> CliRuntime: ...
