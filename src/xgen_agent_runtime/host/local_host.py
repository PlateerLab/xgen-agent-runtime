"""LocalHostServices — 데스크톱 커넥터(사이드카) 의 HostServices 구현.

서버의 ``ServerHostServices`` 와 **같은 AgentTurnExecutor.run** 을 돌리되,
**로컬과 웹의 유일한 차이는 실행 환경뿐**이다. 계정 상태(에이전트 설정·자격증명·
설정·메모리·이력)는 **전부 서버가 진실의 원본**이고, 로컬 실행은 그것을 그대로
쓴다(무발산·완전 공유):

* **상태 = 서버**: 자격증명/설정/base_url/credentials 는 서버가 로그인 계정으로
  해석해 넘긴 ``context`` 에서 읽는다(로컬 env 가 아니라). 메모리는 서버 브릿지
  (``server_bridge``)를 통해 **웹과 같은 저장소**를 읽고 쓴다 — 로컬 턴이 쌓은
  기억도 웹에서 보인다. 브릿지가 없으면(오프라인 등) 발산을 막기 위해 메모리는
  비활성(로컬 전용 vault 를 만들지 않는다).
* **실행 = 이 PC**: ``make_sandbox`` 가 ``None`` → 런타임 Bash/Read/Write 가
  ``ToolContext.sandbox`` 없이 **로컬 호스트(사용자 PC)** 에서 직접 실행된다.
  codex/claude_code 는 로컬 프로세스로 스폰돼 네이티브 도구가 곧 로컬 파일을
  만진다 — 서버-codex 처럼 tool 을 되돌려 라우팅할 필요가 없다.
* **워크스페이스**: 로컬 동기화 폴더(커넥터 동기화 엔진이 서버 인덱스와 sync).
  hydrate/publish 는 그 엔진이 하므로 여기선 무동작.
* **④(cloud/jobs/delegation/self-evolution)**: 서버 자산이라 로컬은 서버 브릿지
  RPC 또는 v1 degrade.

순수 Python(xgen_agent_runtime + xgen_agent_runtime.host 만 의존) — 커넥터 사이드카가
그대로 임포트한다.
"""

from __future__ import annotations

import os
import platform
from typing import Any, Dict, List, Mapping, Optional, Sequence


class LocalHostServices:
    """커넥터 로컬 실행용 HostServices — 상태는 서버(context/bridge), 실행은 로컬."""

    def __init__(
        self,
        workspace_dir: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
        server_bridge: Optional[Any] = None,
        builtin_features: Sequence[str] = ("filesystem", "shell", "web", "meta"),
    ) -> None:
        #: 에이전트가 파일/셸을 조작하는 로컬 폴더(커넥터 동기화 대상=서버와 sync).
        self._workspace = os.path.abspath(workspace_dir)
        os.makedirs(self._workspace, exist_ok=True)
        #: 서버가 로그인 계정으로 해석해 넘긴 상태(진실의 원본은 서버):
        #: {api_keys:{provider:key}, base_urls:{provider:url}, credentials:{provider:{...}},
        #:  settings:{name:value}}. 로컬 env 가 아니라 이걸 쓴다.
        self._ctx = dict(context or {})
        #: 서버 브릿지(인증 RPC) — 메모리 등 라이브 공유 상태. None=미연결.
        self._bridge = server_bridge
        self._builtin_features = tuple(builtin_features)

    # ── A. settings & credentials (서버 계정 상태) ───────────────────────
    def setting(self, name: str, default: str = "") -> str:
        # 관리자 설정은 서버가 진실 — context.settings. (미연결 dev 는 env 폴백.)
        val = (self._ctx.get("settings") or {}).get(name)
        if val:
            return str(val)
        return os.getenv(name, "") or default

    def setting_truthy(self, name: str) -> bool:
        return self.setting(name).strip().lower() in ("1", "true", "yes", "on")

    def resolve_model(self, provider: str, params: Mapping[str, Any]) -> str:
        # 에이전트 설정(provider/model)은 서버 저장 에이전트에서 온다 = run() kwargs.
        return str(params.get("model") or "").strip()

    def resolve_api_key(self, provider: str, params: Mapping[str, Any]) -> str:
        explicit = str(params.get("api_key") or "").strip()
        if explicit:
            return explicit
        # 로그인 계정의 키 — 서버가 해석해 넘긴 것(로컬 env 아님).
        key = (self._ctx.get("api_keys") or {}).get(provider)
        if key:
            return str(key)
        return os.getenv(_ENV_KEY.get(provider, ""), "") or ""

    def resolve_base_url(self, provider: str, params: Mapping[str, Any]) -> Optional[str]:
        explicit = str(params.get("base_url") or "").strip()
        if explicit:
            return explicit
        return (self._ctx.get("base_urls") or {}).get(provider) or None

    def resolve_credentials(
        self, provider: str, params: Mapping[str, Any]
    ) -> Optional[Dict[str, Any]]:
        creds = params.get("credentials") or (self._ctx.get("credentials") or {}).get(provider)
        return dict(creds) if isinstance(creds, dict) and creds else None

    # ── B. execution host: sandbox + workspace ───────────────────────────
    def probe_connector_workspace(
        self,
        user_id: Any,
        workflow_id: str,
        workflow_name: str,
        client_surface: Any,
        provider: str,
    ) -> Optional[dict]:
        # 이미 로컬이다 — 되돌아볼 커넥터가 없다.
        return None

    def make_sandbox(
        self, workflow_id: str, user_id: Any, connector_ws: Optional[dict]
    ) -> Optional[Any]:
        # None → 런타임 도구가 이 PC(사이드카 호스트)에서 직접 실행된다. 그게 로컬.
        return None

    def agent_workspace_dir(self, workflow_id: str, *, create: bool = True) -> str:
        return self._workspace

    def workspace_storage_root(self, workflow_id: str) -> str:
        # 워크스페이스 밖 형제 경로(도구 산출물·상태가 사용자 파일과 안 섞이게).
        return os.path.join(os.path.dirname(self._workspace), ".xgen-agent-storage")

    def hydrate_workspace(self, workflow_id: str, run_dir: str) -> bool:
        # 이미 로컬·동기화됨 — 복원 불필요. markers 를 세워 두면 안 되므로 False.
        return False

    def publish_workspace(self, workflow_id: str, run_dir: str, *, origin: str = "agent") -> None:
        # 커넥터 동기화 엔진이 인덱스에 반영한다 — 여기선 무동작.
        return None

    def environment_prompt(self, sandbox: Any, connector_ws: Optional[dict], provider: str) -> str:
        osname = {"Windows": "Windows", "Darwin": "macOS", "Linux": "Linux"}.get(
            platform.system(), platform.system()
        )
        return (
            "## 실행 환경 — 사용자 PC (데스크톱 커넥터, 로컬)\n"
            f"- **OS**: {osname}. 셸/파일 도구는 **이 컴퓨터에서 직접** 실행됩니다.\n"
            f"- **작업 폴더**: `{self._workspace}` — Bash·Read·Write 가 여기서 돕니다. "
            "산출물은 이 폴더 안에 저장하세요.\n"
        )

    # ── C. memory (서버 공유 — 웹과 같은 저장소) ─────────────────────────
    def build_memory_provider(self, workflow_id: str, interaction_id: str) -> Optional[Any]:
        # 메모리는 계정 자산 — 서버가 진실. 브릿지로 **웹과 같은 저장소**를 읽고
        # 쓴다(로컬 턴의 기억도 웹에서 보인다). 브릿지 없으면(미연결) 로컬 전용
        # vault 를 만들지 않는다 — 발산 방지(기억이 갈라지느니 이번 턴은 무기억).
        if self._bridge is None:
            return None
        try:
            return self._bridge.build_memory_provider(
                str(workflow_id or ""), str(interaction_id or "")
            )
        except Exception:  # noqa: BLE001 — 메모리 실패가 턴을 죽이면 안 된다
            return None

    # ── D. user cloud (v1: 로컬 미제공) ──────────────────────────────────
    def prepare_cloud(self, user_id: Any, workflow_id: str, *, pod_local: bool) -> Optional[Any]:
        return None

    def cloud_inventory(self, user_id: Any, path: str) -> str:
        return ""

    def cloud_not_mounted_note(self, user_id: Any, workflow_id: str) -> str:
        return ""

    def open_shared(self, sandbox: Any, user_id: Any, *, workflow_id: str) -> List[Any]:
        return []

    def build_cloud_skill(self, index: Any, path: str, session: Any, user_id: Any) -> Optional[Any]:
        return None

    def cloud_prompt_block(self, path: str) -> str:
        return ""  # ④: 클라우드 마운트 — v1 로컬 미제공.

    def jobs_prompt_block(self) -> str:
        return ""  # ④: 영구 작업 — v1 로컬 미제공.

    def shared_prompt_block(self, mounts: Sequence[Any]) -> str:
        return ""  # ④: 공유 폴더 — v1 로컬 미제공.

    def build_cloud_file_tool(self, cloud_skill: Any) -> Optional[Any]:
        return None

    # ── E. tool families ─────────────────────────────────────────────────
    def build_connector_mcp_tools(self, user_id: Any, client_surface: Any) -> List[Any]:
        return []  # 커넥터-of-커넥터 없음.

    def build_job_tools(
        self, workflow_id, workflow_name, user_id, *, in_scheduled_run, interaction_id
    ) -> List[Any]:
        return []  # ④: 서버 스케줄러 — v1 로컬 미제공.

    def register_workflow_self_tools(
        self, registry, *, workflow_id, user_id, workflow_name
    ) -> None:
        return None  # ④: 그래프는 서버 자산 — 로컬 편집 미제공(v1).

    def build_turn_delegation(
        self, *, workflow_id, interaction_id, user_id, spec_fields
    ) -> Dict[str, Any]:
        return {}  # ④: 위임 — v1 로컬 미제공.

    def is_report_turn(self, text: str) -> bool:
        return False  # ④: 위임 보고 턴 개념 없음(로컬).

    def delegation_extra_tool_classes(self) -> Dict[str, type]:
        return {}

    def delegation_workspace(self, workflow_id: str) -> str:
        return self._workspace  # 로컬 동기화 폴더.

    def drain_pending_reports(self, workflow_id: str, interaction_id: str) -> str:
        return ""

    def make_sub_cli_client_factory(
        self, params: Mapping[str, Any], workflow_id: str
    ) -> Optional[Any]:
        return None

    def register_forged_tools(
        self, registry, *, workflow_id, workspace_dir, core, sandboxed
    ) -> None:
        return None  # v1: 자가제작 도구 복원 미제공.

    def register_builtin_tools(
        self,
        registry,
        *,
        core: bool,
        user_id: Any,
        anthropic_api_key: str,
        ssh_servers: Sequence[Any],
    ) -> Dict[str, Any]:
        from xgen_agent_runtime.tools.built_in import get_builtin_tools

        tools = get_builtin_tools(features=list(self._builtin_features))
        names: List[str] = []
        for name, tool_cls in tools.items():
            if registry.get(name) is None:
                registry.register(tool_cls(), core=core)
                names.append(name)
        return {"tools": names, "extras": {}}

    def build_run_tool_context(self, **kwargs: Any) -> Any:
        from xgen_agent_runtime.tools.base import ToolContext

        return ToolContext(
            session_id=str(kwargs.get("interaction_id") or ""),
            working_dir=str(kwargs.get("run_dir") or self._workspace),
            storage_path=kwargs.get("storage_dir"),
            allowed_paths=list(kwargs.get("extra_allowed") or []) or None,
            metadata=dict(kwargs.get("extras") or {}),
            sandbox=kwargs.get("sandbox"),  # None → 로컬 호스트 실행
        )

    def load_ssh_servers(self) -> List[Any]:
        return []

    # ── H. product helpers injected into rag / token_budget / distill ────
    def rag_context_builder(self, text: str, item: Any) -> Optional[str]:
        return None  # RAG 포트 컨텍스트 빌더는 서버 노드 헬퍼 — 로컬 스킵.

    def fetch_vllm_max_model_len(self, base_url: str, model: Optional[str]) -> Optional[int]:
        return None  # 라이브 프로브 없음 — token_budget 이 카탈로그로 폴백.

    def agent_vault_root(self, workflow_id: str) -> str:
        import os

        root = os.path.join(self._workspace, ".memory")
        os.makedirs(root, exist_ok=True)
        return root

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
    ) -> Optional[Any]:
        # v1: 로컬 턴-종료 증류는 스킵(None) — 자동 계층(주입/STM 기록)은
        # RemoteMemoryProvider 로 서버 vault 에 그대로 반영된다.
        return None

    # ── F. CLI provider runtime (로컬 프로세스 스폰) ──────────────────────
    def build_cli_runtime(
        self,
        provider: str,
        params: Mapping[str, Any],
        *,
        cloud_workspace: str = "",
        shared_workspaces: Optional[Sequence[str]] = None,
    ) -> Any:
        # 커넥터에서는 codex/claude_code 가 **로컬 프로세스**로 뜨고 cwd 가 로컬
        # 워크스페이스라, 네이티브 도구가 곧 로컬 파일이다. 서버로 되돌리는 MCP
        # 브릿지(mcp_config)는 없다 — 로컬엔 필요 없다.
        api_key = self.resolve_api_key(
            "anthropic" if provider == "claude_code" else "openai", params
        )
        if provider == "codex":
            from xgen_agent_runtime.host.runner import build_codex_cli_client

            client = build_codex_cli_client(
                auth_mode=self.setting("CODEX_AUTH_MODE", "api_key") or "api_key",
                api_key=api_key,
                binary_path=self.setting("CODEX_BINARY_PATH"),
                workspace_dir=self._workspace,
                mcp_config=None,
            )
            return client, None
        from xgen_agent_runtime.host.runner import build_cli_client

        client = build_cli_client(
            auth_mode=self.setting("CLAUDE_CODE_AUTH_MODE", "api_key") or "api_key",
            api_key=api_key,
            oauth_token=self.setting("CLAUDE_CODE_OAUTH_TOKEN"),
            binary_path=self.setting("CLAUDE_CODE_BINARY_PATH"),
            workspace_dir=self._workspace,
            mcp_config=None,
        )
        return client, None

    # ── G. teardown ──────────────────────────────────────────────────────
    def finalize_turn(self, **kwargs: Any) -> None:
        # 커넥터 동기화 엔진이 로컬 변경을 인덱스로 민다 — 여기선 무동작.
        return None


#: provider → 사용자 키 env var (로컬 폴백).
_ENV_KEY = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY",
    "vertex": "VERTEX_API_KEY",
    "claude_code": "ANTHROPIC_API_KEY",
    "codex": "OPENAI_API_KEY",
}
