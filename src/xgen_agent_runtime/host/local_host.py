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
  만진다 — 서버-codex 처럼 tool 을 되돌려 라우팅할 필요가 없다 (커넥터의
  ``mcp_local_*`` 브릿지 도구도 필요 없다 — 이미 로컬이다).
* **CLI 홈 격리**: 커넥터는 ``XGEN_LOCAL_CODEX_HOME`` / ``XGEN_LOCAL_CLAUDE_CONFIG_DIR``
  설정(설치 폴더 아래)을 넘긴다. 그러면 codex/claude 의 설정·로그인이 사용자의
  개인 ``~/.codex`` / ``~/.claude`` 와 섞이지 않고, 서버 중앙 자격증명
  (``CODEX_CREDENTIALS_JSON`` / ``CLAUDE_CODE_OAUTH_TOKEN``)이 그 격리 홈에
  물질화된다 — 서버 파드의 materialize 와 동형.
* **워크스페이스**: 로컬 동기화 폴더(커넥터 동기화 엔진이 서버 인덱스와 sync).
  hydrate/publish 는 그 엔진이 하므로 여기선 무동작.
* **④(cloud/jobs/delegation/self-evolution)**: 서버 자산이라 로컬은 서버 브릿지
  RPC 또는 v1 degrade.

순수 Python(xgen_agent_runtime + xgen_agent_runtime.host 만 의존) — 커넥터 사이드카가
그대로 임포트한다.
"""

from __future__ import annotations

import json
import logging
import os
import platform
from typing import Any, Dict, List, Mapping, Optional, Sequence

logger = logging.getLogger("xgen_agent_runtime.host.local_host")

#: 서버(editor/geny_bridge/builtin_tools._EXPOSED_FAMILIES)와 **같은** 기본 노출 패밀리
#: (meta/Plan 은 서버도 미노출 — HITL 배선 없음; ToolSearch 는 build_pipeline 이 필요 시
#: 자동 등록). 관리자 kill-switch(GENY_TOOLS_<FAMILY>_ENABLED)는
#: context.settings 로 전달돼 여기서도 같은 규칙(명시적으로 꺼야 비활성)이 적용된다.
_DEFAULT_FAMILIES = ("web", "documents", "browser", "workflow", "filesystem", "shell")
_FAMILY_FLAGS = {
    "web": "GENY_TOOLS_WEB_ENABLED",
    "documents": "GENY_TOOLS_DOCUMENTS_ENABLED",
    "browser": "GENY_TOOLS_BROWSER_ENABLED",
    "workflow": "GENY_TOOLS_WORKFLOW_ENABLED",
    "filesystem": "GENY_TOOLS_FILESYSTEM_ENABLED",
    "shell": "GENY_TOOLS_SHELL_ENABLED",
}
#: 커넥터가 넘기는 로컬 CLI 홈 격리 설정 이름.
SETTING_LOCAL_CODEX_HOME = "XGEN_LOCAL_CODEX_HOME"
SETTING_LOCAL_CLAUDE_CONFIG_DIR = "XGEN_LOCAL_CLAUDE_CONFIG_DIR"
#: Claude Code **네이티브** 도구 사전 허용 표면(로컬 턴). ``--print`` 비대화 모드는
#: 허용되지 않은 도구 호출을 프롬프트 없이 **자동 거부**하므로, 서버가 mcp__connector
#: 를 settings+allowedTools 로 통째로 사전 허용하는 것과 같은 방식으로 네이티브 도구를
#: 미리 연다(settings permissions.allow + --allowedTools, permission_mode 는 default 유지
#: — bypassPermissions 는 root 차단·dontAsk 는 거부 모드라 쓰지 않는다).
CLAUDE_LOCAL_ALLOW_TOOLS = (
    "Bash",
    "Read",
    "Write",
    "Edit",
    "MultiEdit",
    "Glob",
    "Grep",
    "LS",
    "WebFetch",
    "WebSearch",
    "NotebookEdit",
    "TodoWrite",
)
#: 격리 CLAUDE_CONFIG_DIR 안에 쓰는 settings 파일 이름(--settings <path>).
CLAUDE_LOCAL_SETTINGS_FILENAME = "xgen-local-settings.json"


class LocalHostServices:
    """커넥터 로컬 실행용 HostServices — 상태는 서버(context/bridge), 실행은 로컬."""

    #: 턴이 **출력을 하나도 내기 전에** 실패하면 메모리 vault 실행 기록
    #: (record_turn_execution)을 남기지 않는다. 커넥터는 그런 실패를 서버 실행으로
    #: 폴백하므로, 여기서 실패 카드를 남기면 서버 턴의 기록과 **중복**된다.
    #: (runner.stream_turn 이 getattr(host, "record_failed_starts", True) 로 읽는다.)
    record_failed_starts: bool = False

    def __init__(
        self,
        workspace_dir: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
        server_bridge: Optional[Any] = None,
        builtin_features: Optional[Sequence[str]] = None,
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
        self._builtin_features: Optional[tuple] = (
            tuple(builtin_features) if builtin_features is not None else None
        )
        #: 메모리가 **기대됐으나**(서버 브릿지 구성/첫 RPC) 실패해 이번 턴이 무기억으로
        #: degrade 됐는지. 사이드카가 이 신호를 읽어 진단 notice 이벤트를 낸다.
        #: (서버 url 이 아예 없는 의도적 오프라인은 실패가 아니므로 세우지 않는다.)
        self._memory_offline = False

    # ── A. settings & credentials (서버 계정 상태) ───────────────────────
    def setting(self, name: str, default: str = "") -> str:
        # 관리자 설정은 서버가 진실 — context.settings. (미연결 dev 는 env 폴백.)
        val = (self._ctx.get("settings") or {}).get(name)
        if val is not None and str(val) != "":
            return str(val)
        return os.getenv(name, "") or default

    def setting_truthy(self, name: str) -> bool:
        return self.setting(name).strip().lower() in ("1", "true", "yes", "on")

    def _family_enabled(self, family: str) -> bool:
        flag = _FAMILY_FLAGS.get(family, "")
        if not flag:
            return True
        raw = self.setting(flag).strip().lower()
        return raw not in ("0", "false", "no", "off")

    def cli_bridge_available(self, provider: str) -> bool:
        """CLI(claude_code/codex) 턴에 ``mcp__connector__*`` 브릿지가 붙는가 — 로컬은 **항상 False**.

        로컬 CLI 턴은 서버로 되돌아가는 MCP 브릿지가 없다(네이티브 도구가 곧 로컬
        파일). 실행기는 이 값으로 CLI 전용 프롬프트 안내(mcp__connector__memory_*
        메모리 도구·DelegateTask 위임·SELF_EVOLUTION 블록)를 **붙이지 않는다** —
        없는 도구를 안내하면 유령 도구가 된다. 기억은 RemoteMemoryProvider 단계
        (주입/STM 기록)가 자동으로 처리하므로 도구 없이도 웹과 같은 기억을 쓴다.
        """
        return False

    def resolve_model(self, provider: str, params: Mapping[str, Any]) -> str:
        # 에이전트 설정(provider/model)은 서버 저장 에이전트에서 온다 = run() kwargs.
        model = str(params.get("model") or "").strip()
        if model:
            return model
        # 서버 해석기와 같은 폴백(관리자 기본 모델) — context.settings 로 전달된다.
        if provider == "claude_code":
            return self.setting("CLAUDE_CODE_MODEL_DEFAULT", "")
        if provider == "codex":
            return self.setting("CODEX_MODEL_DEFAULT", "")
        return ""

    def _llm_proxy_for(self, provider: str) -> Optional[Dict[str, str]]:
        """이 provider 의 LLM 호출을 **서버로 프록시**하라는 마커(context.llm_proxy) 해석.

        내부 서빙(vLLM 등) base_url 은 커넥터 PC 에서 도달 불가라, 서버가 그 URL·모델
        키를 싣는 대신 이 마커만 싣는다. 런타임은 **서버 브릿지**(=서버에 닿는 경로,
        base_url·토큰 보유)로 LLM 호출을 프록시한다:

            connector 런타임 → xgen-server /llm-proxy → 내부 provider

        서버 프록시는 OpenAI 호환 패스스루라 런타임 OpenAI 클라이언트는 그냥
        ``{base}{path}/chat/completions`` 를 부른다. 브릿지가 없으면(오프라인 dev)
        프록시 불가 → None(그때는 종전대로 base_urls 를 본다)."""
        if self._bridge is None:
            return None
        marker = self._ctx.get("llm_proxy")
        if not isinstance(marker, Mapping) or str(marker.get("provider") or "") != provider:
            return None
        path = str(marker.get("path") or "").strip()
        base = str(getattr(self._bridge, "base_url", "") or "").rstrip("/")
        token = str(getattr(self._bridge, "token", "") or "")
        if not path or not base or not token:
            return None
        return {"base_url": base + path, "token": token}

    def resolve_api_key(self, provider: str, params: Mapping[str, Any]) -> str:
        explicit = str(params.get("api_key") or "").strip()
        if explicit:
            return explicit
        # 서버 LLM 프록시 대상이면 인증은 **브릿지 토큰**(사용자 세션) — 프록시가
        # 업스트림(내부 모델)의 실키를 서버에서 주입한다(실키는 PC 로 안 나간다).
        proxy = self._llm_proxy_for(provider)
        if proxy:
            return proxy["token"]
        # 로그인 계정의 키 — 서버가 해석해 넘긴 것(로컬 env 아님).
        key = (self._ctx.get("api_keys") or {}).get(provider)
        if key:
            return str(key)
        return os.getenv(_ENV_KEY.get(provider, ""), "") or ""

    def resolve_base_url(self, provider: str, params: Mapping[str, Any]) -> Optional[str]:
        explicit = str(params.get("base_url") or "").strip()
        if explicit:
            return explicit
        # 내부 서빙 provider → 서버 LLM 프록시 URL 로 재작성(PC 도달 가능한 서버 경유).
        proxy = self._llm_proxy_for(provider)
        if proxy:
            return proxy["base_url"]
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

    def hydrate_workspace(self, workflow_id: str, run_dir: str):  # -> Optional[bool]
        # 이미 로컬·동기화됨 — 복원 개념이 없다(None = 해당 없음). False 를 돌려주면
        # 실행기가 "복원 실패" 경고를 매 턴 찍는다.
        return None

    def publish_workspace(self, workflow_id: str, run_dir: str, *, origin: str = "agent") -> None:
        # 커넥터 동기화 엔진이 인덱스에 반영한다 — 여기선 무동작.
        return None

    def environment_prompt(self, sandbox: Any, connector_ws: Optional[dict], provider: str) -> str:
        # 주입 프롬프트는 영문이 규약이다 — 환경 사실(어디서 도는가/무엇이 남는가)만
        # 담고 개별 도구 사용법은 가르치지 않는다(도구 description 의 몫).
        osname = {"Windows": "Windows", "Darwin": "macOS", "Linux": "Linux"}.get(
            platform.system(), platform.system()
        )
        shell_hint = (
            "commands run under **PowerShell** — use PowerShell syntax"
            " (`Get-ChildItem`, `$env:VAR`, join commands with `;`). bash-only syntax"
            " (`&&`, `$(...)`, `ls -la`) may fail on this machine."
            if osname == "Windows"
            else "commands run under a POSIX shell (bash/sh)."
        )
        return (
            "## Execution environment — the user's PC (desktop connector, local run)\n"
            f"- **OS**: {osname} ({platform.machine()}). Shell/file tools run **directly on"
            " this computer** — no remote bridge tools (mcp_local_* etc.) are needed.\n"
            f"- **Shell**: {shell_hint}\n"
            f"- **Working folder**: `{self._workspace}` — file and shell work happens here."
            " Save deliverables inside this folder (it syncs automatically with the server"
            " workspace, so they are also visible on the web).\n"
            "- **Memory/history**: shared with the server — you read and write the same"
            " memory as web conversations.\n"
        )

    # ── 메모리 오프라인 진단 신호(사이드카가 읽는다) ─────────────────────
    def note_memory_offline(self) -> None:
        """메모리 브릿지 구성 실패 또는 첫 RPC 실패로 이번 턴이 무기억으로 degrade
        됐음을 기록한다. 사이드카가 첫 청크 이전에 notice 이벤트로 승격한다.
        (사이드카가 브릿지 구성 실패를 감지했을 때도 이 헬퍼로 신호를 세운다.)"""
        self._memory_offline = True

    def memory_offline(self) -> bool:
        """이번 host 수명에서 메모리 오프라인 degrade 가 관측됐는가."""
        return self._memory_offline

    # ── C. memory (서버 공유 — 웹과 같은 저장소) ─────────────────────────
    def build_memory_provider(self, workflow_id: str, interaction_id: str) -> Optional[Any]:
        # 메모리는 계정 자산 — 서버가 진실. 브릿지로 **웹과 같은 저장소**를 읽고
        # 쓴다(로컬 턴의 기억도 웹에서 보인다). 브릿지 없으면(미연결) 로컬 전용
        # vault 를 만들지 않는다 — 발산 방지(기억이 갈라지느니 이번 턴은 무기억).
        if self._bridge is None:
            # 브릿지 미연결 — 서버 url 이 없어 의도적 오프라인이거나, 구성 실패라
            # 사이드카가 이미 note_memory_offline() 로 신호를 세웠다(여긴 판단 안 함).
            return None
        try:
            provider = self._bridge.build_memory_provider(
                str(workflow_id or ""), str(interaction_id or "")
            )
        except Exception:  # noqa: BLE001 — 메모리 실패가 턴을 죽이면 안 된다
            # 브릿지는 있으나(메모리 기대됨) 첫 RPC/구성이 던졌다 → 무기억 degrade 신호.
            self.note_memory_offline()
            return None
        if provider is None:
            # 브릿지는 있으나 provider 를 못 열었다(엔드포인트 부재 등) → 같은 신호.
            self.note_memory_offline()
        return provider

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
        # 로컬 실행 = 이미 로컬. 서버 reverse-WS 브릿지 도구(mcp_local_* — 셸/파일 프록시)는
        # 런타임 자체 도구와 중복이라 노출하지 않는다. 다만 사용자가 커넥터에 등록한 **외부 MCP
        # 서버**(Atlassian 등)는 로컬 실행 에이전트도 써야 하므로, 런타임 MCP 매니저로 직접
        # 연결해 노출한다(서버 경로가 브릿지로 프록시하는 것과 같은 결과, 커넥터 로컬에서 직접).
        # 설정은 커넥터가 context.connector_mcp_servers 로 실어 보낸다(resolved). 실패는 무 MCP.
        servers = self._ctx.get("connector_mcp_servers")
        if not servers:
            return []
        try:
            from xgen_agent_runtime.host.connector_mcp_local import connector_mcp_tools

            return connector_mcp_tools(servers)
        except Exception as exc:  # noqa: BLE001 — MCP 실패가 턴을 깨면 안 된다
            logger.warning("local_host: 외부 MCP 도구 빌드 실패(무시): %s", exc)
            return []

    def build_job_tools(
        self, workflow_id, workflow_name, user_id, *, in_scheduled_run, interaction_id
    ) -> List[Any]:
        return []  # ④: 서버 스케줄러 — v1 로컬 미제공.

    def register_workflow_self_tools(
        self, registry, *, workflow_id, user_id, workflow_name
    ) -> None:
        """자기진화(WorkflowSelf) — 그래프는 서버 자산이지만 편집은 **서버 RPC** 로
        로컬에서도 웹과 완전 동일하게 동작한다(메모리·RAG·LLM 프록시와 같은
        '서버 호출' 원칙). 프록시 도구는 컨텍스트 메타가 실어 준 서버 실물의
        description/input_schema 를 그대로 광고한다(정적 사본=드리프트 금지).

        메타 없음(구서버)/비활성(관리자 kill-switch)/브릿지 없음이면 조용히
        미등록 — turn_executor 가 registry.get("WorkflowSelf") 게이트로 프롬프트
        블록도 함께 생략한다(유령 안내 방지).
        """
        meta = self._ctx.get("workflow_self")
        if not isinstance(meta, dict) or not meta.get("enabled"):
            return None
        if self._bridge is None or registry is None:
            return None
        path = str(meta.get("path") or "").strip()
        description = str(meta.get("description") or "").strip()
        schema = meta.get("input_schema")
        if not path or not description or not isinstance(schema, dict):
            return None

        from xgen_agent_runtime.tools.base import (
            Tool,
            ToolCapabilities,
            ToolContext,
            ToolResult,
        )

        bridge = self._bridge

        class _WorkflowSelfProxy(Tool):
            """서버 WorkflowSelfTool 프록시 — 입력을 그대로 위임, 결과를 그대로 반환."""

            @property
            def name(self) -> str:
                return "WorkflowSelf"

            @property
            def description(self) -> str:
                return description

            @property
            def input_schema(self) -> Dict[str, Any]:
                return schema

            def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:  # noqa: A002
                return ToolCapabilities(concurrency_safe=False, network_egress=True)

            async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:  # noqa: A002
                import asyncio as _aio

                data = await _aio.to_thread(bridge.workflow_self, path, dict(input or {}))
                if not isinstance(data, dict):
                    return ToolResult(
                        content="WorkflowSelf server call failed — the server is "
                                "unreachable or refused the request. The graph was "
                                "not changed; try again shortly.",
                        is_error=True,
                    )
                if not data.get("ok"):
                    return ToolResult(
                        content=str(data.get("error") or "WorkflowSelf failed."),
                        is_error=True,
                    )
                return ToolResult(
                    content=str(data.get("content") or ""),
                    is_error=bool(data.get("is_error")),
                    metadata=data.get("metadata") if isinstance(data.get("metadata"), dict) else None,
                )

        try:
            registry.register(_WorkflowSelfProxy(), core=True)
            logger.info("local-host: WorkflowSelf(자기진화) 서버-RPC 프록시 등록 (wf=%s)", workflow_id)
        except Exception:  # noqa: BLE001 — 등록 실패가 턴을 죽이면 안 된다
            logger.warning("local-host: WorkflowSelf 프록시 등록 실패 (스킵)", exc_info=True)
        return None

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

    def builtin_families(self) -> List[str]:
        """이 턴에 노출할 built-in 패밀리 — 서버 `_EXPOSED_FAMILIES` + meta 를
        같은 kill-switch 규칙으로 거른 것(명시적 주입이 있으면 그것)."""
        if self._builtin_features is not None:
            return list(self._builtin_features)
        return [f for f in _DEFAULT_FAMILIES if self._family_enabled(f)]

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

        # 서버 register_builtin_tools 와 같은 게이트: required_config_keys 가
        # 충족되지 않는 도구는 광고하지 않는다(죽은 도구 미광고 원칙).
        satisfied: set = set()
        extras: Dict[str, Any] = {}
        if anthropic_api_key:
            satisfied.add("feature:docs_llm")
            extras["docs"] = {"api_key": anthropic_api_key}

        names: List[str] = []
        families = self.builtin_families()
        for family in families:
            try:
                tools = get_builtin_tools(features=[family])
            except Exception as exc:  # noqa: BLE001 — 옵셔널 미설치 등은 해당 패밀리만 스킵
                logger.warning("built-in 패밀리 %s 로드 실패 (스킵): %s", family, exc)
                continue
            for name, tool_cls in tools.items():
                if not name or registry.get(name) is not None:
                    continue
                try:
                    tool = tool_cls()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("built-in 도구 %s 생성 실패 (스킵): %s", name, exc)
                    continue
                try:
                    required = list(tool.required_config_keys() or [])
                except Exception:  # noqa: BLE001
                    required = []
                if any(tok not in satisfied for tok in required):
                    continue
                registry.register(tool, core=core)
                names.append(name)
        return {"families": families, "tools": names, "extras": extras}

    def build_run_tool_context(self, **kwargs: Any) -> Any:
        from xgen_agent_runtime.tools.base import ToolContext

        run_dir = str(kwargs.get("run_dir") or self._workspace)
        # 서버 builtin_tools.build_run_tool_context 와 같은 규약: 허용 트리는
        # **항상** [run_dir(=동기화 폴더), *명시적 추가분]. None/[] 로 두면 path guard
        # 가 "제한 없음"이라 Read/Write/Edit 가 사용자 PC 전체를 만진다 — 로컬이
        # 샌드박스 없이(sandbox=None) 호스트에서 도는 만큼 이 격리가 유일한 울타리다.
        allowed = [run_dir]
        for extra in kwargs.get("extra_allowed") or []:
            if extra and str(extra) not in allowed:
                allowed.append(str(extra))
        return ToolContext(
            session_id=str(kwargs.get("interaction_id") or ""),
            working_dir=run_dir,
            storage_path=kwargs.get("storage_dir"),
            allowed_paths=allowed,
            metadata=dict(kwargs.get("extras") or {}),
            sandbox=kwargs.get("sandbox"),  # None → 로컬 호스트 실행
        )

    def load_ssh_servers(self) -> List[Any]:
        return []

    # ── H. product helpers injected into rag / token_budget / distill ────
    def rag_context_builder(self, text: str, item: Any) -> Optional[str]:
        # RAG(컬렉션·rag_service)는 **서버 자산** — 메모리와 같은 '서버 호출' 원칙이다.
        # 로컬은 직렬화된 search_params 만 받고, 서버 RAG RPC 로 실제 검색을 위임해
        # [DOC_n] 블록을 받는다. 브릿지/파라미터/workflow_id 가 없거나 실패하면 None
        # (그 RAG 아이템은 컨텍스트 없이 진행 — 턴은 깨지지 않는다). rag_service 는
        # 로컬로 오지 않으므로(직렬화 불가) 여기서 무시하고 search_params 만 쓴다.
        if self._bridge is None:
            return None
        search_params = item.get("search_params") if isinstance(item, dict) else None
        if not isinstance(search_params, dict) or not search_params:
            return None
        workflow_id = str(self._ctx.get("workflow_id") or "").strip()
        if not workflow_id:
            return None
        try:
            return self._bridge.rag_search(workflow_id, text, search_params)
        except Exception as exc:  # noqa: BLE001 — RAG 실패가 턴을 깨면 안 된다
            logger.warning("local_host: RAG 서버 호출 실패(무시): %s", exc)
            return None

    def fetch_vllm_max_model_len(self, base_url: str, model: Optional[str]) -> Optional[int]:
        return None  # 라이브 프로브 없음 — token_budget 이 카탈로그로 폴백.

    def agent_vault_root(self, workflow_id: str) -> str:
        # 워크스페이스(동기화 폴더) **밖** — 사용자 파일 트리에 .memory 를 남기지 않는다.
        root = os.path.join(self.workspace_storage_root(workflow_id), "memory")
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
    def _isolated_home(self, setting_name: str) -> str:
        """커넥터가 넘긴 격리 홈(설치 폴더 아래). 없으면 ""(CLI 기본 홈)."""
        path = self.setting(setting_name).strip()
        if not path:
            return ""
        try:
            os.makedirs(path, mode=0o700, exist_ok=True)
        except OSError as exc:
            logger.warning("CLI 격리 홈 생성 실패(%s): %s — 기본 홈 사용", path, exc)
            return ""
        return path

    def _materialize_codex_credentials(self, codex_home: str) -> bool:
        """서버 중앙 자격증명(CODEX_CREDENTIALS_JSON)을 격리 CODEX_HOME/auth.json 에
        물질화 — 서버 파드의 codex_service.materialize_credentials 와 동형(멱등)."""
        cred = self.setting("CODEX_CREDENTIALS_JSON").strip()
        if not cred or not codex_home:
            return False
        try:
            json.loads(cred)
        except ValueError:
            logger.warning("CODEX_CREDENTIALS_JSON 이 JSON 이 아닙니다 — 물질화 생략")
            return False
        path = os.path.join(codex_home, "auth.json")
        try:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    if f.read() == cred:
                        return True
            tmp = f"{path}.tmp-{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(cred)
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
            os.replace(tmp, path)
            return True
        except OSError as exc:
            logger.warning("codex 자격증명 물질화 실패(%s): %s", path, exc)
            return False

    def _claude_local_settings(self, claude_home: str, allow_tools: Any = None) -> str:
        """네이티브 도구 사전 허용 settings — 격리 홈이 있으면 그 안의 파일 경로,
        없으면 인라인 JSON(서버 agent_geny 의 mcp__connector 사전 허용과 같은 형식).
        둘 다 ``--settings`` 로 전달된다. ``allow_tools`` 로 **유지된 네이티브만**
        사전 허용한다(에이전트별 도구 선택) — None 이면 전체 카탈로그."""
        allow = list(allow_tools) if allow_tools is not None else list(CLAUDE_LOCAL_ALLOW_TOOLS)
        payload = {"permissions": {"allow": allow}}
        body = json.dumps(payload, ensure_ascii=False)
        if not claude_home:
            return body
        path = os.path.join(claude_home, CLAUDE_LOCAL_SETTINGS_FILENAME)
        try:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    if f.read() == body:
                        return path
            tmp = f"{path}.tmp-{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(body)
            os.replace(tmp, path)
            return path
        except OSError as exc:
            logger.warning("claude settings 파일 쓰기 실패(%s): %s — 인라인 JSON 사용", path, exc)
            return body

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
        if provider == "codex":
            from xgen_agent_runtime.host.runner import build_codex_cli_client

            # 관리자 enable 게이트 — 서버(agent_geny._build_codex_cli_runtime)와 같은 문구.
            if not self.setting_truthy("CODEX_ENABLED"):
                raise ValueError(
                    "Codex 백엔드가 비활성화되어 있습니다. "
                    "관리자 설정(CODEX_ENABLED)에서 활성화 후 사용하세요."
                )
            auth_mode = (self.setting("CODEX_AUTH_MODE", "api_key") or "api_key").strip()
            api_key = self.resolve_api_key("openai", params) if auth_mode == "api_key" else ""
            env_extras: Dict[str, str] = {}
            codex_home = self._isolated_home(SETTING_LOCAL_CODEX_HOME)
            if codex_home:
                env_extras["CODEX_HOME"] = codex_home
            if auth_mode != "api_key":
                # 구독(ChatGPT) — 서버 중앙 자격증명을 격리 홈에 물질화. 중앙값이 없으면
                # 이 PC 의 기존 로그인(격리 홈/기본 홈)을 그대로 쓴다.
                self._materialize_codex_credentials(codex_home)
            timeout_s = float(self.setting("CODEX_TIMEOUT_S", "3600") or 3600.0)
            # 샌드박스/승인: CodexCLIClient 기본 ``--sandbox workspace-write`` + ``codex exec``
            # (헤드리스 — 승인 정책 never) 로 cwd(=동기화 폴더) 안 쓰기는 프롬프트 없이
            # 허용되고 밖은 거부된다. bypass(--dangerously-bypass…)는 쓰지 않는다.
            client = build_codex_cli_client(
                auth_mode=auth_mode,
                api_key=api_key,
                binary_path=self.setting("CODEX_BINARY_PATH"),
                workspace_dir=self._workspace,
                timeout_s=timeout_s,
                mcp_config=None,
                env_extras=env_extras or None,
            )
            return client, None

        from xgen_agent_runtime.host.runner import build_cli_client, resolve_native_tools

        if not self.setting_truthy("CLAUDE_CODE_ENABLED"):
            raise ValueError(
                "Claude Code 백엔드가 비활성화되어 있습니다. "
                "관리자 설정(CLAUDE_CODE_ENABLED)에서 활성화 후 사용하세요."
            )
        # 네이티브 도구 선택(에이전트별 tool-toggle-list) — 유지/제거 분리.
        # 커넥터 로컬엔 MCP 브릿지가 없어 네이티브가 파일/셸의 유일 경로이므로
        # 기본 정책(Bash 만 유지)이 그대로 로컬에도 적용된다. 제거된 도구는
        # --disallowedTools 로 차단(비어 있어도 무해). 서버와 같은 판정 함수를 쓴다.
        _kept_natives, _removed_natives = resolve_native_tools(
            params.get("cli_native_tools_disabled"), catalog=CLAUDE_LOCAL_ALLOW_TOOLS
        )
        auth_mode = (self.setting("CLAUDE_CODE_AUTH_MODE", "api_key") or "api_key").strip()
        api_key = self.resolve_api_key("anthropic", params) if auth_mode == "api_key" else ""
        oauth_token = self.setting("CLAUDE_CODE_OAUTH_TOKEN") if auth_mode == "setup_token" else ""
        timeout_s = float(self.setting("CLAUDE_CODE_TIMEOUT_S", "3600") or 3600.0)
        budget = float(params.get("cli_max_budget_usd") or 0.0) or float(
            self.setting("CLAUDE_CODE_MAX_BUDGET_USD", "0") or 0.0
        )
        extra_env: Dict[str, str] = {}
        claude_home = self._isolated_home(SETTING_LOCAL_CLAUDE_CONFIG_DIR)
        if claude_home:
            extra_env["CLAUDE_CONFIG_DIR"] = claude_home
        # --print(비대화) 는 허용 목록 밖 도구 호출을 프롬프트 없이 자동 거부한다 —
        # 네이티브 표면을 settings(permissions.allow) + --allowedTools 로 사전 허용.
        settings_path = self._claude_local_settings(claude_home, allow_tools=_kept_natives)
        client = build_cli_client(
            auth_mode=auth_mode,
            api_key=api_key,
            oauth_token=oauth_token,
            binary_path=self.setting("CLAUDE_CODE_BINARY_PATH"),
            workspace_dir=self._workspace,
            timeout_s=timeout_s,
            max_budget_usd=budget,
            # ⚠ 로컬: CLI 의 **네이티브** 도구(Read/Write/Edit/Bash/…)를 켠다.
            # 서버는 러너 sandbox 가 붙어 있어 끄고 mcp__connector__* 로 브릿지하지만,
            # 여기엔 브릿지가 없다 — 끄면 모델에게 파일/셸 도구가 하나도 남지 않는다.
            allow_local_tools=True,
            permission_mode="default",
            settings_path=settings_path,
            # 유지된 네이티브만 사전 허용하고, 제거된 것은 disallow 로 차단한다
            # (--disallowedTools 가 우선). 기본은 Bash 만 유지.
            allow_tools=tuple(_kept_natives),
            disallow_tools_extra=tuple(_removed_natives),
            mcp_config=None,
            extra_env=extra_env or None,
            # 이 호스트는 턴마다 파이프라인/CLI 클라이언트를 새로 만들고 턴 끝에 닫는다
            # (one-shot). hot-spare 프리웜은 '다음 턴'이 이 클라이언트를 재사용할 때만
            # 이득인데, 여기선 매 턴 새 클라이언트라 프리웜된 프로세스가 즉시 teardown
            # 에 회수될 뿐이다 → 매 CLI 턴 낭비 spawn. 끈다(runner.build_cli_client 지침).
            prewarm_spawn=False,
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
