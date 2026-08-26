"""AgentTurnExecutor — execute() 본체를 host-무관 실행부로 들어올린 것.

xgen-workflow 의 AgentGenyNode.execute() 가 이 run() 에 위임한다. 지금은 본체가
서버에서 resolve 되는 import(editor.geny_bridge shim/editor.nodes 상수)를 그대로
쓴다 — 서버 웹 경로는 무변경. 커넥터(Phase 2)에서는 이 import 들을
xgen_agent_runtime.host.* 로 repoint 하면 같은 본체가 로컬에서 돈다(무발산).

인프라는 전부 ``host``(HostServices)로, 노드 고유값은 ``node_name`` 으로 받는다.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

# 본체가 쓰는 모듈-수준 상수/헬퍼(항상 실행) — 서버에서 resolve. lazy 트리거라
# 순환 없음(execute 가 turn_executor 를 지연 import). Phase 2 에서 패키지로 이전.
from xgen_agent_runtime.host._constants import (  # noqa: E402
    _CLI_BACKENDS,
    _coerce_text,
    _delegation_wired,
    _self_evolution_policy,
    default_prompt,
    SELF_EVOLUTION_PROMPT_BLOCK,
)

logger = logging.getLogger("editor.nodes.xgen.agent.agent_geny")


def _coerce_schema(schema: Any) -> Optional[Dict[str, Any]]:
    """raw dict 또는 pydantic model class(Schema Provider 출력)를 스키마 dict 로."""
    if isinstance(schema, dict):
        return schema
    for attr in ("model_json_schema", "schema"):
        fn = getattr(schema, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:  # noqa: BLE001
                return None
    return None


class AgentTurnExecutor:
    """execute() 의 host-무관 판. 서버·커넥터가 같은 run() 을 돈다."""

    def run(self, host: Any, **kwargs):
        # node_name 은 워크플로 executor 가 kwargs 에 실어 준다(node_info nodeName)
        # = 노드의 self.nodeName. build_pipeline name= 에만 쓰인다.
        node_name = kwargs.get("node_name") or ""
        from xgen_agent_runtime.host.memory import history_messages
        from xgen_agent_runtime.host.rag import collect_rag
        from xgen_agent_runtime.host.runner import build_pipeline, run_turn, stream_turn
        from xgen_agent_runtime.host.tools import adapt_tools
        from xgen_agent_runtime import PipelineState

        text = _coerce_text(kwargs.get("text"))
        streaming = bool(kwargs.get("streaming", True))
        interaction_id = str(kwargs.get("interaction_id") or "")
        response_io_id = kwargs.get("response_io_id")

        try:
            # host(HostServices)는 run() 인자로 받는다 — 서버는 ServerHostServices,
            # 커넥터는 LocalHostServices. 본체는 인프라에 오직 host.* 로만 닿는다.
            provider = (kwargs.get("provider") or "openai").strip()

            # provider별 파라미터 사전 검증 — 허용 범위 밖이면 실행 전 정확한 메시지로 차단.
            # (temperature: OpenAI/vLLM/Google 0~2, Anthropic/Claude Code 0~1.)
            # provider 런타임 에러 매핑(geny-executor 내부)은 executor 측 과제 — 여기서는
            # 노드에서 아는 값만 실행 전에 막는다.
            from xgen_agent_runtime.host.param_validator import validate_agent_params

            param_error = validate_agent_params(
                provider, temperature=kwargs.get("temperature", 0.7)
            )
            if param_error:
                logger.error(
                    "agents/geny: 파라미터 검증 실패 (provider=%s, temperature=%r): %s",
                    provider,
                    kwargs.get("temperature"),
                    param_error,
                )
                return iter([param_error]) if streaming else param_error

            model = host.resolve_model(provider, kwargs)
            api_key = host.resolve_api_key(provider, kwargs)
            base_url = host.resolve_base_url(provider, kwargs)
            credentials = host.resolve_credentials(provider, kwargs)
            schema = (
                _coerce_schema(kwargs.get("output_schema"))
                if kwargs.get("output_schema") is not None
                else None
            )

            # Tools port + embedded tools carried on Context dicts (tool-search mode).
            # tool_exposure="search" → Tools 포트는 deferred 등록(2.42.0 노출 모델,
            # ToolSearch 가 런타임 발견·활성화). Context 포트의 내장 검색 도구는
            # 연결된 지식소스의 1차 도구이므로 항상 core 로 즉시 노출한다.
            exposure = (kwargs.get("tool_exposure") or "all").strip()
            result_sink: Dict[str, str] = {}
            # ── CLI 백엔드의 도구 표면 ───────────────────────────────────
            # CLI(claude_code/codex)는 자기 루프를 소유해 이 registry 를 직접 보지
            # 못한다. 도구를 건네는 표준 경로는 MCP 뿐이고, 서버 CLI 브릿지가 자기
            # 조립으로 그 표면을 만든다 — 그래서 여기서는 registry 를 조립하지 않는다.
            #
            # (예전엔 데스크톱에서 런타임을 직접 돌리는 경로가 있어, 호스트가 registry
            #  자체를 루프백 MCP 로 내주는 분기가 있었다. 로컬 실행은 폐기됐다 —
            #  에이전트는 언제나 서버 세션에서 돈다.)
            _sdk_tools = provider not in _CLI_BACKENDS
            #: CLI 에 도구가 광고되는 MCP 서버 이름 — 프롬프트 이름 규약 안내가 쓴다.
            _cli_mcp_server = "connector"

            registry = adapt_tools(
                kwargs.get("tools"), result_sink=result_sink, core=(exposure != "search")
            )
            rag_block, embedded_tools = collect_rag(
                text,
                kwargs.get("context"),
                context_builder=host.rag_context_builder,
            )
            if embedded_tools:
                registry = adapt_tools(
                    embedded_tools, result_sink=result_sink, registry=registry, core=True
                )
            # Connector-hosted Local MCP 도구 자동 주입 — 실행자(user_id)의 데스크톱 커넥터가
            # 로컬 MCP 서버 도구를 노출하고 있으면 registry 에 core 로 합산한다(그래프 노드
            # 없이 실행 시점 자동). 커넥터 미연결 시 빈 리스트 → no-op.
            # ⚠ CLI 백엔드는 이 registry 를 못 본다 — 서버 CLI 브릿지가 자기 조립으로
            # 커넥터 도구를 따로 광고한다.
            if _sdk_tools:
                try:
                    # client_surface 게이트(host 내부): 대화 출처가 데스크톱 커넥터일
                    # 때만 로컬 도구를 주입한다 (web 대화엔 커넥터가 연결돼 있어도 no-op).
                    connector_tools = host.build_connector_mcp_tools(
                        kwargs.get("user_id"), kwargs.get("client_surface")
                    )
                    if connector_tools:
                        registry = adapt_tools(
                            connector_tools, result_sink=result_sink, registry=registry, core=True
                        )
                        logger.info(
                            "agents/geny: Connector MCP 도구 %d개 자동 주입", len(connector_tools)
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("agents/geny: Connector MCP 도구 주입 실패 (무시): %s", exc)
            if registry:
                logger.info(
                    "agents/geny: %d tool(s) registered (%d deferred) from Tools/Context ports",
                    len(registry),
                    len(registry.list_deferred()),
                )

            # RAG results join the user turn (agent_xgen convention), so the
            # model sees [DOC_n] chunks next to the question they answer.
            # ⚠ 융합은 컨텍스트 예산 적용 **후** (build_pipeline 직전) — 예산은
            # text/rag 를 따로 잘라야 한다 (사용자 텍스트 최후 보존 원칙).

            state = PipelineState(session_id=interaction_id)
            history = history_messages(kwargs.get("memory"))
            if history:
                state.messages = history
                # preload 된 과거 대화는 내장 메모리의 STM 기록/대화 아카이브
                # 대상에서 제외 — 두 전략의 워터마크를 preload 길이로 초기화
                # (없으면 매 턴 과거 이력이 통째로 재기록되어 중복 폭증).
                from xgen_agent_runtime.host.conversation_archive import (
                    _ARCHIVED_KEY,
                    STM_RECORDED_KEY,
                )

                state.metadata[STM_RECORDED_KEY] = len(history)
                state.metadata[_ARCHIVED_KEY] = len(history)
                logger.info(
                    "agents/geny: preloaded %d history message(s) from Memory port", len(history)
                )

            # ── 내장 메모리 (에이전트당 하나) ──────────────────────
            # enable_memory 기본 True — 메모리 노드와 무관하게 geny-executor 파일
            # vault 를 attach 한다. provider 수명은 turn teardown 이 소유
            # (runner._close_memory_provider). 도구 등록은 executor-native Tool
            # 직접 register (LangChain 어댑터 불필요). claude_code 백엔드는 CLI 가
            # 도구 루프를 소유하므로 자가-조회 도구는 제외되지만, Stage 2 retriever
            # (Pinned Facts/Relevant Knowledge 주입)와 Stage 15 STM 기록은 동일하게
            # 동작한다.
            memory_provider = None
            # 명시적으로 비운 것("")과 아예 안 준 것(None/키 없음)을 구분한다 —
            # `or default_prompt` 는 둘을 똑같이 취급해, 사용자가 System Prompt 를
            # 의도적으로 비워도 조용히 기본 문구로 되돌아갔다(진짜 "시스템 프롬프트
            # 없음" 을 요청할 방법이 없었다). 키 자체가 없을 때만 기본값을 쓴다.
            system_prompt = kwargs.get("system_prompt")
            if system_prompt is None:
                system_prompt = default_prompt

            # ── 연결된 사용자 클라우드 ─────────────────────────────────
            #
            # **한 번만** 준비한다. 준비는 곧 복원(hydrate)이라, 백엔드마다 따로
            # 부르면 같은 트리를 두 번 내려받는다.
            #
            # 그리고 **반드시 알려 준다.** 경로만 열어 두면 에이전트는 그곳이
            # 있는 줄 모르고 자기 workspace 만 뒤진 뒤 "클라우드에는 이 파일
            # 하나뿐" 이라고 답한다 — 실제로 그랬다. 열어 주는 것과 알려 주는
            # 것은 다른 일이다.
            # ── 커넥터 로컬 워크스페이스 프로브 ──────────────────────
            #
            # 데스크톱 커넥터 대화이고 이 에이전트의 워크스페이스가 사용자 PC 로
            # 동기화되고 있으면, 파일/셸 도구는 sandbox 가 아니라 **사용자 PC**
            # 에서 돈다 (ConnectorLocalSandbox — 같은 XgenySandbox 프로토콜,
            # 도구 표면은 동일). 실패/미지원은 전부 None = 조용한 sandbox 폴백.
            #
            # claude_code CLI 도 로컬 실행한다 — 도구는 ToolContext.sandbox 하나로
            # 라우팅되므로 그 값이 ConnectorLocalSandbox 면 CLI 의
            # mcp__connector__Bash/Read/Write 도 사용자 PC 에서 돈다. 단 CLI
            # run-ctx 의 working_dir 을 가상 /ws 로 맞춰야 한다(_build_cli_runtime
            # 에서 처리 — 안 그러면 sandbox_path 가드가 전부 거절).
            #
            # codex 만 제외: 자체 OS 샌드박스라 파일/셸을 ToolContext.sandbox 로
            # 돌릴 표면이 없다.
            _cloud_mount: Optional[tuple] = None
            try:
                _cloud_mount = host.prepare_cloud(
                    kwargs.get("user_id"),
                    str(kwargs.get("workflow_id") or ""),
                    pod_local=False,
                )
            except Exception as _cexc:  # noqa: BLE001 — 클라우드가 실행을 막으면 안 된다
                logger.warning("agents/geny: 클라우드 연결 준비 실패 (스킵): %s", _cexc)
            if _cloud_mount:
                system_prompt = system_prompt + host.cloud_prompt_block(_cloud_mount[1])
                # 무엇이 있는지까지 말해 준다. 경로만 열어 주면 에이전트는
                # 뒤지지 않는다 — 두 번 겪었다. 최상위 목록이 프롬프트에 있으면
                # "클라우드에 뭐 있어?"에 ls 없이 정확히 답하고, 파일 요청은
                # 정확한 경로로 바로 연다.
                _inv = host.cloud_inventory(kwargs.get("user_id"), _cloud_mount[1])
                if _inv:
                    system_prompt = system_prompt + "\nCurrent top-level entries:\n" + _inv + "\n"
            else:
                # 클라우드를 못 열었다 — **절대 침묵하지 않는다.** 침묵하면(프롬프트도
                # 안내도 없으면) 에이전트는 '클라우드가 없다'고 보고 외부 서비스
                # (Drive/Dropbox/S3)로 오판한다 (실증). not_mounted_note 는 항상
                # 문자열을 돌려주므로, "XgenCloud 는 이 플랫폼 저장소이고 이번 턴엔
                # 이 사유로 접근 불가"를 못박아 연결된 에이전트가 무조건 클라우드의
                # 존재를 인식하게 한다. 진단 보조라 실패는 조용히 무시(실행 안 막음).
                try:
                    _note = host.cloud_not_mounted_note(
                        kwargs.get("user_id"), str(kwargs.get("workflow_id") or "")
                    )
                    if _note:
                        system_prompt = (
                            system_prompt + "\n## User cloud storage (XgenCloud)\n" + _note + "\n"
                        )
                except Exception:  # noqa: BLE001
                    pass

            # 파일 클라우드 위의 fs_* 스킬 — 문서/표/OCR 전문 도구 37종.
            # 캔버스 노드 없이 **기본 내장**이다: 클라우드가 마운트되면 항상.
            # 한 번만 세워 SDK·CLI 가 같은 객체를 쓴다 (표면이 갈리면 안 된다).
            _cloud_skill = None
            if _cloud_mount:
                try:
                    _cloud_skill = host.build_cloud_skill(
                        _cloud_mount[0],
                        _cloud_mount[1],
                        None,
                        kwargs.get("user_id"),
                    )
                except Exception as _cfexc:  # noqa: BLE001
                    logger.warning("agents/geny: FileCloud 스킬 실패 (스킵): %s", _cfexc)

            # ── 코드 실행 기반 (xgen-workflow-sandbox) ─────────────────
            #
            # 켜져 있으면 파일/셸 도구는 **이 파드가 아니라** 러너 세션에서
            # 돈다. 준비가 곧 복원이라 여기서 한 번만 붙인다 (클라우드와 같은
            # 규약). 실패하면 붙이지 않는다 — 반쯤 붙은 상태가 제일 나쁘다.
            # 실패해도 계속 진행하지 **않는다.** 세션 없이 가면 도구가 이 파드
            # 에서 돌고, 그건 이 기능이 없애려던 바로 그 상태다 — 게다가 조용히
            # 그렇게 되면 격리가 사라진 줄 아무도 모른 채 무거운 도구 하나가
            # 같은 파드의 다른 대화를 함께 느리게 만든다.
            #
            # 안 쓰기로 했다면 관리자 설정에서 끄면 된다 (그때는 None 이 온다).
            # 실행 환경: 커넥터 로컬(사용자 PC) 또는 러너 세션 — host 가 분기를
            # 캡슐화한다(같은 XgenySandbox 프로토콜). 커넥터 로컬은 attach/publish
            # 를 하지 않는다: 진실은 사용자 PC 폴더이고 인덱스 반영은 커넥터 동기화
            # 엔진이 한다(이중 기록 금지).
            _sandbox = host.make_sandbox(
                str(kwargs.get("workflow_id") or ""),
                kwargs.get("user_id"),
            )
            # 연결된 클라우드를 이 세션에서 다룰 수 있게 연다.
            #
            # 형제 트리다 — 에이전트 workspace 와 **다른 인덱스**를 갖는다.
            # 한 트리로 합치면 에이전트 산출물과 사용자 파일이 같은 인덱스에
            # 잡히고, 한쪽의 삭제 전파가 다른 쪽 파일을 지운다.
            #
            if _sandbox is not None and _cloud_mount:
                _sandbox.open_tree(_cloud_mount[0], _cloud_mount[1])
            # 다른 사람이 공유한 폴더도 같은 방식으로 연다 — 형제 트리다.
            # 소유자는 바뀌지 않고, 무엇을 할 수 있는지는 공유 레코드가 정한다
            # (읽기 공유는 읽기 전용으로 열린다).
            _shared_mounts = []
            if _sandbox is not None and kwargs.get("user_id"):
                try:
                    _shared_mounts = host.open_shared(
                        _sandbox,
                        kwargs.get("user_id"),
                        workflow_id=str(kwargs.get("workflow_id") or ""),
                    )
                except Exception as _shexc:  # noqa: BLE001
                    logger.warning("agents/geny: 공유 폴더 열기 실패 (스킵): %s", _shexc)
            # CLI 백엔드의 브릿지 run ctx 가 여기서 꺼내 쓴다 (같은 세션).
            kwargs["_sandbox_session"] = _sandbox
            kwargs["_cloud_skill"] = _cloud_skill
            # 영구 작업 도구 — 서버 스케줄러에 이 에이전트를 건다. CLI 의
            # 세션 한정 Cron* 은 runner 가 차단하므로, 이게 없으면 반복 요청을
            # 받을 길 자체가 없다.
            _job_tools = []
            # enable_builtin_tools 게이트에 함께 묶는다 — 아래 SDK registry/CLI
            # ctx 등록이 이 플래그 안에 있어서, 여기서만 만들면 도구 없이
            # 프롬프트만 "JobSchedule 로 걸어라"라고 약속하는 유령이 된다.
            if (
                kwargs.get("workflow_id")
                and kwargs.get("user_id")
                and bool(kwargs.get("enable_builtin_tools", True))
            ):
                try:
                    _job_tools = host.build_job_tools(
                        str(kwargs.get("workflow_id")),
                        str(kwargs.get("workflow_name") or ""),
                        kwargs.get("user_id"),
                        # 스케줄 발화 턴에는 JobSchedule 을 빼고 준다 — 작업이 매
                        # 실행마다 새 작업을 낳는 자기복제 방지.
                        in_scheduled_run=str(kwargs.get("interaction_id") or "").startswith(
                            "workflow_schedule_"
                        ),
                        # notify 옵션의 귀착지 — 이 작업을 요청한 바로 이 대화.
                        interaction_id=str(kwargs.get("interaction_id") or ""),
                    )
                except Exception as _jexc:  # noqa: BLE001
                    logger.warning("agents/geny: 영구 작업 도구 실패 (스킵): %s", _jexc)
            kwargs["_job_tools"] = _job_tools
            if _job_tools:
                system_prompt = system_prompt + "\n\n" + host.jobs_prompt_block()
            if _cloud_skill is not None and _sandbox is not None:
                # 어댑터의 바이트 경로를 러너로 돌린다 — 러너가 붙어 있으면
                # 이 파드의 클라우드 트리는 복원되지 않은 캐시라 읽으면 안 된다.
                # (커넥터 로컬 모드는 pod_local 복원 트리를 그대로 쓴다 — 로컬
                #  어댑터는 파드 실경로를 모른다.)
                try:
                    _cloud_skill._ctx.adapter._session = _sandbox
                except Exception:  # noqa: BLE001
                    pass
            # 실행 환경 안내 — 도구가 어디서 도는지 host 가 설명한다(서버: 러너/
            # 커넥터 로컬 sandbox, 커넥터 사이드카: 이 PC). 안 알려 주면 에이전트는
            # 자기 코드가 어디서 도는지 모른 채 /tmp 에 쓰고 다음 턴에 잃는다.
            _env_block = host.environment_prompt(_sandbox, provider)
            if _env_block:
                system_prompt = system_prompt + "\n\n" + _env_block
            if _shared_mounts:
                # 여는 것과 알려 주는 것은 다른 일이다 — 클라우드에서 배웠다.
                system_prompt = system_prompt + "\n\n" + host.shared_prompt_block(_shared_mounts)
            # ── 자기진화(self-evolution) 판정 — 배선보다 **먼저** ────────────
            # 여기서 정하는 이유: 호스트의 CLI 브릿지 가용성 판정이 이 결과를 본다
            # (내장 도구를 꺼도 WorkflowSelf 하나 때문에 run ctx 를 바인딩해야 한다).
            # 늦게 스태시하면 probe 가 항상 '미허용'을 보고, 내장 도구를 끈 에이전트는
            # 자기진화를 조용히 잃는다.
            #
            # ★ 보안: 배포(deploy_)·게스트(guest_) 실행에서는 절대 허용하지 않는다. 그
            # 실행은 워크플로 OWNER user_id 로 돌아 write-access 검사를 통과하므로,
            # 익명 사용자/문서 프롬프트 인젝션이 라이브 프로덕션 그래프를 영구 변조할
            # 수 있다(감사 CRITICAL). 판정은 SDK/CLI 공용이다.
            _se_allowed, _se_reason = _self_evolution_policy(kwargs, host.setting)
            kwargs["_self_evolution_allowed"] = _se_allowed
            if not _se_allowed:
                logger.info("agents/geny: self-evolution 미배선 — %s", _se_reason)

            # ── CLI 도구 브릿지 가용성 (claude_code/codex 전용) ───────────
            # CLI 백엔드는 registry 를 못 보고, 비네이티브 도구(memory_*/WorkflowSelf/
            # DelegateTask…)는 host 의 MCP 브릿지가 mcp__<서버>__* 로 광고할 때만
            # 존재한다. 브릿지가 없는 host 에서 그 도구를 프롬프트로 약속하면 유령
            # 호출이 된다(감사 #25). host.cli_bridge_available 은
            # OPTIONAL — 없으면 True(레거시 서버 동작). 예외도 True(판정 불가 = 레거시).
            _cli_bridge_ok = True
            _cli_bridge_reason = ""
            if provider in _CLI_BACKENDS:
                _probe = getattr(host, "cli_bridge_available", None)
                if callable(_probe):
                    try:
                        if not bool(_probe(provider)):
                            _cli_bridge_ok = False
                            _cli_bridge_reason = "host 가 CLI 도구 브릿지를 제공하지 않음"
                    except Exception as _bexc:  # noqa: BLE001
                        logger.warning(
                            "agents/geny: cli_bridge_available 판정 실패 (브릿지 있음으로 간주): %s",
                            _bexc,
                        )
            # 도구(run ctx) 표면 — WorkflowSelf/위임은 브릿지 run ctx 에 산다. 서버는
            # 내장 도구가 꺼져 **있어도** 자기진화가 허용되면 run ctx 를 바인딩한다
            # (WorkflowSelf 는 registry + workflow_id 만 필요하다). 둘 다 아니면
            # 바인딩이 없으므로 host 가 True 라 해도 여기서 '없음'으로 본다.
            # memory_* 는 run ctx 와 무관하게(memory eager) 광고되므로 _cli_bridge_ok 만 본다.
            _cli_tools_bridge_ok = _cli_bridge_ok and (
                bool(kwargs.get("enable_builtin_tools", True)) or _se_allowed
            )
            _cli_tools_bridge_reason = _cli_bridge_reason or (
                "enable_builtin_tools=off + 자기진화 미허용 (브릿지 run ctx 미바인딩)"
            )
            if bool(kwargs.get("enable_memory", True)):
                from xgen_agent_runtime.host._constants import (
                    MEMORY_AUTO_PROMPT_BLOCK,
                    MEMORY_PROMPT_BLOCK,
                )
                from xgen_agent_runtime.host.memory_tools import build_memory_tools

                memory_provider = host.build_memory_provider(
                    str(kwargs.get("workflow_id") or ""), interaction_id
                )
                if memory_provider is not None and provider in _CLI_BACKENDS and not _cli_bridge_ok:
                    # 브릿지 없는 CLI(데스크톱 사이드카): 도구는 없고 자동 계층(Stage 2
                    # 주입 + Stage 15 기록)만 돈다 — 그 사실만 알리고 도구는 광고 안 함.
                    system_prompt = system_prompt + MEMORY_AUTO_PROMPT_BLOCK
                    logger.info(
                        "agents/geny: CLI 메모리 도구 미광고 — %s (자동 계층만 동작)",
                        _cli_bridge_reason,
                    )
                if (
                    memory_provider is not None
                    and provider in _CLI_BACKENDS
                    and _cli_bridge_ok
                    and not _sdk_tools
                ):
                    # 서버 CLI 브릿지 경로: memory_* 는 그 브릿지가 자기 조립으로
                    # 광고한다(여기 registry 에는 넣지 않는다). 자동 계층(주입/기록)과
                    # 별개로 에이전트가 도구를 인지하도록 정책 블록 + 이름 규약 노트.
                    system_prompt = (
                        system_prompt
                        + MEMORY_PROMPT_BLOCK
                        + (
                            f"\n(Note: on this backend the memory tools appear as"
                            f" mcp__{_cli_mcp_server}__memory_write,"
                            f" mcp__{_cli_mcp_server}__memory_read, etc.)"
                            if provider == "claude_code"
                            else f"\n(Note: on this backend the memory tools are served by the"
                            f" '{_cli_mcp_server}' MCP server.)"
                        )
                    )
                if memory_provider is not None and _sdk_tools:
                    try:
                        from xgen_agent_runtime.tools import ToolRegistry

                        if registry is None:
                            registry = ToolRegistry()
                        for mem_tool in build_memory_tools(memory_provider):
                            registry.register(mem_tool, core=True)
                        system_prompt = system_prompt + MEMORY_PROMPT_BLOCK
                        if provider in _CLI_BACKENDS:
                            # 같은 registry 가 MCP 로 나가므로 도구 이름에 접두가 붙는다.
                            system_prompt += (
                                f"\n(Note: on this backend the memory tools appear as"
                                f" mcp__{_cli_mcp_server}__memory_write,"
                                f" mcp__{_cli_mcp_server}__memory_read, etc.)"
                                if provider == "claude_code"
                                else f"\n(Note: on this backend the memory tools are served"
                                f" by the '{_cli_mcp_server}' MCP server.)"
                            )
                        logger.info("agents/geny: 내장 메모리 활성 (self-serve 도구 6개 등록)")
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "agents/geny: 메모리 도구 등록 실패 (자동 계층만 동작): %s", exc
                        )

            # ── built-in 도구 패밀리 (web/documents/browser/ssh/workflow) ──
            # geny-executor 옵셔널 전부 채택 (Geny 동형). CLI 백엔드도 로컬 표면이
            # 있으면 **같은 조립**을 지난다 — 네이티브는 전면 차단이므로 파일/셸도
            # 여기서 나온 우리 도구가 유일한 경로다. 파일 도구는 workspace 에 격리되고
            # (path guard), 문서 산출물은 사용자 스토리지 '결과물' 폴더로 업로드되어
            # 다운로드 버튼으로 나타난다. 관리자 차단: GENY_TOOLS_*_ENABLED.
            run_tool_context = None
            run_dir_cleanup = None
            # 턴 종료 시 원본(MinIO+DB)에 반영할 workspace. hydrate 가 **성공한**
            # 경우에만 채운다 — 복원 실패한 빈 캐시로 삭제를 전파하면 원본이
            # 통째로 날아간다.
            _hydrated_ws: Optional[str] = None
            _hydrated_wf: str = ""
            if _sdk_tools and bool(kwargs.get("enable_builtin_tools", True)):
                try:
                    import shutil as _shutil
                    import tempfile as _tempfile

                    from xgen_agent_runtime.tools import ToolRegistry

                    if registry is None:
                        registry = ToolRegistry()
                    bt_summary = host.register_builtin_tools(
                        registry,
                        core=(exposure != "search"),
                        user_id=kwargs.get("user_id"),
                        anthropic_api_key=host.resolve_api_key("anthropic", kwargs),
                        ssh_servers=host.load_ssh_servers(),
                    )
                    if _cloud_skill is not None and registry.get("FileCloud") is None:
                        _cloud_tool = host.build_cloud_file_tool(_cloud_skill)
                        if _cloud_tool is not None:
                            registry.register(_cloud_tool, core=True)
                    for _jt in _job_tools:
                        if registry.get(_jt.name) is None:
                            registry.register(_jt, core=True)
                    if bt_summary["tools"]:
                        # 영속 workspace (Drive형 동기화의 전제): workflow(에이전트)
                        # 축의 안정 디렉터리 — 턴을 가로질러 파일이 살아남고,
                        # 데스크톱 커넥터 레플리카·workspace API 와 같은 트리를
                        # 공유한다 (Geny ONE-workspace 동형). workflow_id 가 없는
                        # 비정형 실행만 임시 디렉터리로 폴백(턴 종료 시 정리).
                        _wf_for_ws = str(kwargs.get("workflow_id") or "")
                        _storage_dir = None
                        if _wf_for_ws:
                            run_dir = host.agent_workspace_dir(_wf_for_ws)
                            # 파드 로컬 트리는 **캐시**다 — 재배포/다른 파드로
                            # 비어 있을 수 있으므로 원본(host)에서 복원한 뒤 턴을
                            # 시작한다. 이걸 빠뜨리면 에이전트가 빈 workspace 를
                            # 보고, 턴 끝의 publish 가 그 공백을 원본에 전파해
                            # 사용자 파일을 지운다.
                            if _sandbox is not None:
                                # 러너가 이미 복원했다. 여기서 또 hydrate 하면
                                # 같은 트리에 두 작성자가 생기고, 턴 끝에 양쪽이
                                # publish 해 서로의 삭제를 되살린다. run_dir 은
                                # 러너가 알려 준 경로를 그대로 쓴다 — 배포가 두
                                # 루트를 같은 문자열로 맞추므로 값은 같지만,
                                # 어긋났을 때 조용히 갈라지지 않게 명시한다.
                                run_dir = _sandbox.workdir
                            else:
                                # 원본(host)에서 복원한 뒤 턴을 시작한다. hydrate 가
                                # **성공한** 경우에만 markers 를 세운다 — 복원 실패한
                                # 빈 캐시로 삭제를 전파하면 원본이 통째로 날아간다.
                                _hyd = host.hydrate_workspace(_wf_for_ws, run_dir)
                                if _hyd:
                                    _hydrated_ws, _hydrated_wf = run_dir, _wf_for_ws
                                elif _hyd is None:
                                    # 호스트가 복원 개념이 없다(데스크톱: 동기화 폴더가 곧 원본)
                                    logger.debug(
                                        "agents/geny: workspace hydrate 해당 없음(host 관리 동기화)"
                                    )
                                else:
                                    logger.warning(
                                        "agents/geny: workspace 복원 실패 — 이번 턴은 "
                                        "원본 반영을 건너뛴다"
                                    )
                            # executor 내부 저장소는 workspace 밖 형제 경로로
                            # (tool-results/·ssh/ 가 사용자 파일 목록·동기화에
                            #  섞이지 않게).
                            _storage_dir = os.path.join(
                                host.workspace_storage_root(_wf_for_ws), "executor"
                            )
                        else:
                            run_dir = _tempfile.mkdtemp(prefix="xgen-geny-run-")

                            def run_dir_cleanup(_d: str = run_dir) -> None:
                                _shutil.rmtree(_d, ignore_errors=True)

                        # 연결된 에이전트는 사용자 클라우드도 만진다. 위에서
                        # 이미 준비했으므로 여기서는 경로만 연다 — 형제 트리다
                        # (하위로 끼우면 같은 파일이 두 인덱스에 잡혀, 한쪽의
                        # delete_missing 이 다른 쪽 파일을 지운다).
                        # 클라우드 + 공유받은 폴더 — executor 의 로컬 경로
                        # 가드가 보는 목록이다. 여기 없으면 샌드박스 가드가
                        # 허용해도 도구가 먼저 막는다.
                        _cloud_extra = ([_cloud_mount[1]] if _cloud_mount else []) + [
                            _p for _o, _p, _m in _shared_mounts
                        ]
                        # 클라우드 extras 는 서버 자산(editor.geny_bridge.cloud_mount) —
                        # 데스크톱 호스트(사이드카)에는 그 모듈이 없다. 마운트가 있을
                        # 때만, 그리고 import 가 가능할 때만 붙인다(호스트 비의존 유지).
                        _cloud_extras_kv: Dict[str, Any] = {}
                        if _cloud_mount:
                            try:
                                _cloud_extras_kv = __import__(
                                    "editor.geny_bridge.cloud_mount", fromlist=["describe"]
                                ).describe(_cloud_mount[1])
                            except Exception:  # noqa: BLE001 — 서버 전용 헬퍼 부재
                                _cloud_extras_kv = {}

                        run_tool_context = host.build_run_tool_context(
                            interaction_id=interaction_id,
                            run_dir=run_dir,
                            extras={**bt_summary["extras"], **_cloud_extras_kv},
                            storage_dir=_storage_dir,
                            extra_allowed=_cloud_extra,
                            sandbox=_sandbox,
                        )

                        # 자기확장: 이 에이전트가 만들어 저장한 도구를 복원하고
                        # 제작 도구(ForgeTool/List/Delete)를 배선한다. 스크립트는
                        # 영속 workspace 안에 살아 있으므로 세션이 바뀌어도
                        # 도구로 되살아난다.
                        if _wf_for_ws:
                            try:
                                host.register_forged_tools(
                                    registry,
                                    workflow_id=_wf_for_ws,
                                    workspace_dir=run_dir,
                                    core=(exposure != "search"),
                                    sandboxed=_sandbox is not None,
                                )
                            except Exception as _fexc:  # noqa: BLE001
                                logger.warning(
                                    "agents/geny: 저장된 도구 복원 실패 (스킵): %s", _fexc
                                )

                except Exception as exc:  # noqa: BLE001 — 내장 도구는 실행을 깨지 않는다
                    logger.warning("agents/geny: built-in 도구 등록 실패 (스킵): %s", exc)

            # ── 자기진화(self-evolution) 등록 — built-in tools 와 독립 ──────────
            # WorkflowSelf 는 registry + workflow_id 만 있으면 되고(편집은 DB, workspace
            # 불필요), built-in tools 설정과 무관해야 한다. 예전엔 enable_builtin_tools 와
            # `if bt_summary['tools']` 안에 중첩돼, 내장도구를 끄거나 모든 패밀리를 kill-switch
            # 로 비우면 self-evolution 이 조용히 죽었다(감사 HIGH).
            # (판정 _se_allowed 는 위에서 끝났다 — 브릿지 가용성 판정이 그 결과를 본다.)
            # 서버 CLI 경로는 여기서 registry 에 넣지 않고(그 registry 를 CLI 가 못 본다)
            # 커넥터 MCP 브릿지가 같은 판정으로 WorkflowSelf 를 광고한다
            # (build_cli_run_context 의 self_evolution 플래그 → cli_bridge_registry).
            if _sdk_tools and _se_allowed:
                try:
                    from xgen_agent_runtime.tools import ToolRegistry as _ToolRegistry

                    if registry is None:
                        registry = _ToolRegistry()
                    host.register_workflow_self_tools(
                        registry,
                        workflow_id=str(kwargs.get("workflow_id")),
                        user_id=kwargs.get("user_id"),
                        workflow_name=str(kwargs.get("workflow_name") or ""),
                    )
                    # 프롬프트 블록은 도구가 **실제로 등록된** 경우에만 — 호스트가 미제공
                    # (데스크톱 사이드카 v1: WorkflowSelf 없음)이면 "그래프를 영구 편집할 수
                    # 있다"고 말해 놓고 도구가 없는 유령 안내가 된다.
                    if registry.get("WorkflowSelf") is not None:
                        system_prompt = system_prompt + SELF_EVOLUTION_PROMPT_BLOCK
                        if provider == "claude_code":
                            # 하네스 자체 'Workflow'(서브에이전트 조율)와 이름이 비슷해
                            # 그래프 편집 요청을 그쪽으로 오인하는 회귀가 있었다(프로드 실증).
                            system_prompt += (
                                f"\n(Note: on this backend the graph-editing tool appears as"
                                f" mcp__{_cli_mcp_server}__WorkflowSelf. Your harness's own"
                                " 'Workflow' tool is subagent orchestration — NOT XGEN"
                                " graph editing.)"
                            )
                        elif provider == "codex":
                            system_prompt += (
                                f"\n(Note: on this backend the WorkflowSelf tool is served"
                                f" by the '{_cli_mcp_server}' MCP server.)"
                            )
                    else:
                        logger.info(
                            "agents/geny: self-evolution 미배선 — host 가 WorkflowSelf 를 제공하지 않음"
                        )
                except Exception as _sexc:  # noqa: BLE001
                    logger.warning("agents/geny: self-evolution 도구 등록 실패 (스킵): %s", _sexc)
            elif provider in _CLI_BACKENDS and _se_allowed and not _cli_tools_bridge_ok:
                # 브릿지 run ctx 가 없으면 WorkflowSelf 는 CLI 에 보이지 않는다 —
                # 블록을 붙이면 유령 안내(감사 #25). 같은 판정(_self_evolution_allowed)
                # 스태시는 그대로 두어 브릿지가 생기는 호스트 쪽 계약은 불변.
                logger.info(
                    "agents/geny: self-evolution 미배선 — CLI 브릿지 없음 (%s)",
                    _cli_tools_bridge_reason,
                )
            elif (
                provider == "claude_code"
                and _se_allowed
                and kwargs.get("user_id") not in (None, 0, "0", "")
            ):
                # CLI 표면에선 mcp__connector__WorkflowSelf 로 광고된다. 하네스가
                # 자체 "Workflow"(서브에이전트 조율) 도구를 갖고 있어 이름이 비슷할
                # 뿐 전혀 다른 물건이다 — 이 각주가 없으면 모델이 그래프 편집
                # 요청을 하네스 Workflow 로 오인하고 "안 된다" 고 답한다 (프로드 실증).
                system_prompt = (
                    system_prompt
                    + SELF_EVOLUTION_PROMPT_BLOCK
                    + (
                        "\n(Note: on this backend the graph-editing tool appears as"
                        " mcp__connector__WorkflowSelf. Your harness's own 'Workflow'"
                        " tool is subagent orchestration — NOT XGEN graph editing.)"
                    )
                )
            elif (
                provider == "codex"
                and _se_allowed
                and kwargs.get("user_id") not in (None, 0, "0", "")
            ):
                system_prompt = (
                    system_prompt
                    + SELF_EVOLUTION_PROMPT_BLOCK
                    + (
                        "\n(Note: on this backend the WorkflowSelf tool is served by"
                        " the 'connector' MCP server.)"
                    )
                )

            # ── 위임 (sub-worker/sub-agent/백그라운드 작업) — Geny 위임 스택 ──
            # Agent(one-shot)/SubAgent*(상주+inbox)/Task*(백그라운드) 도구를
            # 배선한다. lifecycle 은 geny_agent_tasks 미러로 남아 '작업' 뷰가
            # 읽는다. 위임 도구는 **항상 core** — ToolSearch 뒤에 숨기면 모델이
            # 위임을 서술만 하고 호출하지 않는 회귀가 Geny 에서 실증됐다.
            #
            # 완료 트리거 계약 (모델의 자발적 inbox 폴링에 의존하지 않는다):
            #   - 보고 턴([SUB_AGENT_RESULT] 로 시작 — alarm 반응 턴)은 위임
            #     도구를 배선하지 않는다 (재귀 위임 차단, Geny Stage-12 동형).
            #   - 일반 턴은 미보고 완료분을 DB 에서 클레임해 user 턴 앞에
            #     주입한다 (alarm 실패/파드 재시작 폴백 — drain_pending_reports).
            _deleg_report_turn = False
            try:
                _deleg_report_turn = host.is_report_turn(text)
            except Exception:  # noqa: BLE001
                pass
            if _deleg_report_turn:
                logger.info("agents/geny: 보고 턴 — 위임 도구 비활성 (재귀 차단)")
            if provider == "codex" and bool(kwargs.get("enable_delegation", True)):
                # Codex v1: 위임 미지원 — registry 도구는 CLI 에 보이지 않고,
                # claude 의 mcp__connector__* 위임 표면(_delegation_extras 브릿지
                # 배선)은 claude 전용 계약이다. 조용한 반쪽 배선보다 명시 스킵.
                logger.info("agents/geny: codex 백엔드 — 위임 도구 미배선 (v1 미지원)")
            if (
                provider != "codex"
                and not _deleg_report_turn
                and bool(kwargs.get("enable_delegation", True))
                and str(kwargs.get("workflow_id") or "")
            ):
                try:
                    wf_id = str(kwargs.get("workflow_id"))
                    delegation_extras: Dict[str, Any] = {}
                    _deleg_no_bridge = provider == "claude_code" and not _cli_tools_bridge_ok
                    if _deleg_no_bridge:
                        # 브릿지 run ctx 가 없으면 mcp__connector__DelegateTask 는 CLI 에
                        # 보이지 않는다 — 스태시(→ CLI 내장 Task/Agent 차단)·노트 모두
                        # 생략해 CLI 가 최소한 자기 위임 도구는 쓰게 둔다(감사 #25).
                        # host.build_turn_delegation 도 부르지 않는다(쓸 데 없는 백엔드 생성).
                        logger.info(
                            "agents/geny: 위임 미배선 — CLI 브릿지 없음 (%s)",
                            _cli_tools_bridge_reason,
                        )
                    else:
                        cli_factory = None
                        if provider == "claude_code":
                            cli_factory = host.make_sub_cli_client_factory(kwargs, wf_id)
                        # SubPipelineSpec 은 서버 소유 타입 — 여기선 plain dict 필드만
                        # 주고, host(서버 impl)가 그 타입을 구성한다(공유 executor 무결).
                        delegation_extras = host.build_turn_delegation(
                            workflow_id=wf_id,
                            interaction_id=interaction_id,
                            user_id=kwargs.get("user_id"),
                            spec_fields=dict(
                                provider=provider,
                                model=model,
                                api_key=api_key,
                                base_url=base_url,
                                temperature=float(kwargs.get("temperature", 0.7) or 0.7),
                                max_tokens=int(kwargs.get("max_tokens", 8192)),
                                user_id=kwargs.get("user_id"),
                                anthropic_api_key=host.resolve_api_key("anthropic", kwargs),
                                ssh_servers=host.load_ssh_servers(),
                                llm_client_factory=cli_factory,
                                sandbox=_sandbox,
                                credentials=credentials,
                            ),
                        )
                    if _deleg_no_bridge:
                        pass  # 위에서 판정·로그 끝 — 미보고 완료분 주입만 계속
                    elif not _delegation_wired(delegation_extras):
                        # host 가 위임 백엔드(subagent_manager/task_runner/task_registry)
                        # 를 주지 않았다(데스크톱 사이드카 v1: {}). 이때 SDK 패밀리
                        # (SubAgent*/Task*)를 등록하면 도구는 보이는데 extras 가 비어
                        # 매 호출이 NO_SUBAGENT_MANAGER 로 죽는 유령 도구가 된다 —
                        # WorkflowSelf 와 같은 원칙으로 등록·노트 모두 생략.
                        logger.info("agents/geny: 위임 미배선 — host 미제공")
                    elif provider == "claude_code":
                        # 서버 CLI 경로: 도구는 커넥터 MCP 브릿지가 광고/실행한다 —
                        # _build_connector_mcp_bridge 가 run ctx 로 가져가도록 스태시.
                        # (로컬 표면이면 아래 SDK 분기가 registry 에 직접 등록한다.)
                        kwargs["_delegation_extras"] = delegation_extras
                        # 도구 계약은 도구 설명이 담는다 (Geny 동형 — 위임 프롬프트
                        # 블록 없음). CLI 에만 이름 매핑/내장 Task 비활성 사실을 한 줄로.
                        system_prompt = system_prompt + (
                            "\n(Delegation/background-task tools appear as"
                            " mcp__connector__DelegateTask etc.; your built-in Task tool"
                            " is disabled here — delegate via mcp__connector__DelegateTask.)"
                        )
                    else:
                        from xgen_agent_runtime.tools import ToolRegistry as _TR
                        from xgen_agent_runtime.tools.built_in import get_builtin_tools as _gbt

                        if registry is None:
                            registry = _TR()
                        added: List[str] = []
                        # DelegateTask = 단일 위임 동사 (background 전용, Geny
                        # send_direct_message 동형). one-shot `agent` 패밀리는
                        # 의도적으로 미등록 — 턴 블로킹 위임 회귀 방지.
                        for name, tool_cls in host.delegation_extra_tool_classes().items():
                            if registry.get(name) is None:
                                registry.register(tool_cls(), core=True)
                                added.append(name)
                        for fam in ("subagent", "tasks"):
                            for name, tool_cls in _gbt(features=[fam]).items():
                                if registry.get(name) is None:
                                    registry.register(tool_cls(), core=True)
                                    added.append(name)
                        if run_tool_context is not None:
                            run_tool_context.extras.update(delegation_extras)
                        else:
                            # built-in 도구가 꺼져 있어도 위임 도구는 돈다.
                            # 이 폴백에도 실행 기반과 내부 저장소를 똑같이 준다 —
                            # 여기만 빠지면 "도구는 러너에서 도는데 위임만 이
                            # 파드에서 도는" 상태가 되고, 그건 어느 로그를 봐도
                            # 드러나지 않는다.
                            run_tool_context = host.build_run_tool_context(
                                interaction_id=interaction_id,
                                run_dir=(
                                    _sandbox.workdir
                                    if _sandbox is not None
                                    else host.delegation_workspace(wf_id)
                                ),
                                extras=delegation_extras,
                                storage_dir=(
                                    os.path.join(host.workspace_storage_root(wf_id), "executor")
                                    if wf_id
                                    else None
                                ),
                                sandbox=_sandbox,
                            )
                        logger.info(
                            "agents/geny: 위임 활성 — 도구 %d개 (sub-worker/sub-agent/tasks)",
                            len(added),
                        )
                    # 미보고 완료분 주입 — alarm 이 못 전한 결과(파드 재시작·
                    # interaction 부재·반응 턴 실패)를 이 턴에서 보고하게 한다.
                    pending_block = host.drain_pending_reports(wf_id, interaction_id)
                    if pending_block:
                        # user_text 융합 전이므로 text 에 붙인다 — 융합 순서상
                        # (pending + text) + rag 로 결과 동일.
                        text = f"{pending_block}\n\n---\n\n{text}"
                        logger.info("agents/geny: 미보고 위임 완료분 주입 (다음-턴 폴백)")
                except Exception as exc:  # noqa: BLE001 — 위임은 실행을 깨지 않는다
                    logger.warning("agents/geny: 위임 배선 실패 (스킵): %s", exc)

            # 턴-종료 증류 스펙 — 이 턴의 LLM 자격증명 그대로 (memory_distill 기본 ON).
            memory_distill_spec = None
            if (
                memory_provider is not None
                and provider == "codex"
                and bool(kwargs.get("memory_distill", True))
            ):
                # Codex v1: 증류용 MemoryLLM 경로(build_turn_memory_llm)가 codex
                # 클라이언트 구성을 모른다 — 자동 계층(주입/STM 기록)은 그대로
                # 동작하고 턴-종료 증류만 스킵한다.
                logger.info("agents/geny: codex 백엔드 — 메모리 증류 스킵 (v1 미지원)")
            if (
                memory_provider is not None
                and provider != "codex"
                and bool(kwargs.get("memory_distill", True))
            ):
                from xgen_agent_runtime.host.distill import DistillSpec

                # claude_code 는 인증 채널 해석을 노드가 소유(_build_cli_runtime
                # 규약) — 구독(setup_token) 모드도 증류가 돌도록 그대로 전달.
                cli_auth_mode = ""
                cli_oauth_token = ""
                cli_binary_path = ""
                distill_api_key = api_key
                if provider == "claude_code":
                    cli_auth_mode = (
                        host.setting("CLAUDE_CODE_AUTH_MODE", "api_key") or "api_key"
                    ).strip()
                    if cli_auth_mode == "setup_token":
                        cli_oauth_token = host.setting("CLAUDE_CODE_OAUTH_TOKEN") or ""
                    else:
                        # ⚠ _resolve_api_key("claude_code") 는 키 매핑이 없어 항상
                        # "" — CLI 의 키는 anthropic 채널로 해석해야 한다
                        # (_build_cli_runtime 과 동일). 이 불일치가 api_key
                        # 모드에서도 증류가 무음 스킵되던 두 번째 원인.
                        distill_api_key = api_key or host.resolve_api_key("anthropic", kwargs)
                    cli_binary_path = host.setting("CLAUDE_CODE_BINARY_PATH") or ""

                memory_distill_spec = DistillSpec(
                    workflow_id=str(kwargs.get("workflow_id") or ""),
                    interaction_id=interaction_id,
                    provider=provider,
                    model=model,
                    api_key=distill_api_key,
                    base_url=base_url,
                    cli_auth_mode=cli_auth_mode,
                    cli_oauth_token=cli_oauth_token,
                    cli_binary_path=cli_binary_path,
                    credentials=credentials,
                    host=host,
                )

            llm_client = None
            cli_cleanup = None
            if provider in _CLI_BACKENDS:
                # CLI 백엔드는 에이전트 루프를 CLI 가 소유한다 — 파이프라인 Stage 10 이
                # 돌지 않으므로 registry 를 파이프라인에 넘겨도 아무도 보지 않는다.
                # 그래서 여기서 registry 는 떼어낸다 — 서버 CLI 브릿지가 자기 조립으로
                # 같은 표면을 만들어 mcp__connector__* 로 광고한다.
                if registry:
                    logger.warning(
                        "agents/geny: %s 백엔드는 Tools 포트 연결 도구 %d개를 이번 실행에서 "
                        "사용할 수 없습니다 (서버 CLI 브릿지가 자기 표면을 별도 조립)",
                        provider,
                        len(registry),
                    )
                registry = None
            if provider == "claude_code":
                # cloud_skill 은 시그니처로 나르지 않는다 — Geny 규약대로 kwargs
                # 스태시(_cloud_skill)를 브릿지가 읽는다. 여기에 인자로도 넣었다가
                # 시그니처에 없어 CLI 백엔드 전체가 기동 실패했다 (프로드 실증).
                llm_client, cli_cleanup = host.build_cli_runtime(
                    "claude_code",
                    kwargs,
                    cloud_workspace=(_cloud_mount[1] if _cloud_mount else ""),
                    shared_workspaces=[_p for _o, _p, _m in _shared_mounts],
                )
                # 러너가 붙어 있으면 복원·발행의 주체는 러너다. 여기서 표식을
                # 남기면 턴 끝에 이 파드가 자기 로컬 트리로 publish 하게 되고,
                # 그러면 같은 workspace 에 작성자가 둘이 된다.
                _cli_wf = str(kwargs.get("workflow_id") or "")
                if _sandbox is None and _cli_wf and not host.setting("CLAUDE_CODE_WORKSPACE_ROOT"):
                    _hydrated_ws, _hydrated_wf = (
                        host.agent_workspace_dir(_cli_wf, create=False),
                        _cli_wf,
                    )
            elif provider == "codex":
                llm_client, cli_cleanup = host.build_cli_runtime(
                    "codex",
                    kwargs,
                    cloud_workspace=(_cloud_mount[1] if _cloud_mount else ""),
                )
                # 영속 workspace 사용 시 턴 끝 publish 표식 — claude 경로와 동일 규약.
                _codex_wf = str(kwargs.get("workflow_id") or "")
                if _codex_wf and not host.setting("CODEX_WORKSPACE_ROOT"):
                    _hydrated_ws, _hydrated_wf = (
                        host.agent_workspace_dir(_codex_wf, create=False),
                        _codex_wf,
                    )

            # ── 컨텍스트 예산 (컨텍스트 자동 압축 토글) ─────────────────
            # system_prompt 가 최종형이 된 지점 — 여기서 윈도우를 해석하고,
            # 토글 ON 이면 입력측(text/rag)을 예산 안으로 맞춘 뒤 융합한다.
            # 이력(preload)은 자르지 않는다: 그건 파이프라인 Stage 2/4 의 몫.
            # claude_code 는 CLI 가 자체 컨텍스트를 관리하므로 전부 스킵.
            enable_compaction = bool(kwargs.get("enable_compaction", True))
            max_tokens_val = int(kwargs.get("max_tokens", 8192))
            budget_window = 0
            clamped = False
            if provider not in _CLI_BACKENDS:
                from xgen_agent_runtime.host.context_budget import (
                    fit_input_to_budget,
                    resolve_window,
                )

                budget_window = resolve_window(
                    provider,
                    model,
                    int(kwargs.get("context_window") or 0),
                    base_url=base_url,
                    vllm_probe=host.fetch_vllm_max_model_len,
                )
                if enable_compaction and budget_window > 0:
                    fit = fit_input_to_budget(
                        text=text,
                        rag_block=rag_block,
                        system_prompt=system_prompt,
                        history=state.messages,
                        registry=registry,
                        provider=provider,
                        model=model,
                        max_tokens=max_tokens_val,
                        window=budget_window,
                        # 내장 메모리 주입(Stage 2, 최대 10k자≈3k 토큰)은 fit
                        # 이후 system 에 붙는다 — 예약 없이 꽉 채우면 넘친다.
                        reserved_tokens=3_000 if memory_provider is not None else 0,
                    )
                    if fit.clamped:
                        text, rag_block, clamped = fit.text, fit.rag_block, True
                        logger.warning(
                            "agents/geny: 입력 클램프 적용 (window=%d budget=%d before=%d)",
                            fit.window,
                            fit.budget,
                            fit.total_before,
                        )

            user_text = f"{text}\n\n{rag_block}" if rag_block else text

            pipeline = build_pipeline(
                name=node_name,
                provider=provider,
                model=model,
                api_key=api_key,
                base_url=base_url,
                system_prompt=system_prompt,
                registry=registry,
                max_iterations=int(kwargs.get("max_iterations", 20)),
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=max_tokens_val,
                stream=streaming,
                output_schema=schema,
                llm_client=llm_client,
                memory_provider=memory_provider,
                memory_distill_spec=memory_distill_spec,
                tool_context=run_tool_context,
                # 모델의 실제 윈도우 — 압축 임계(80%)·guard·루프 토큰-비 정지의
                # 공통 기준. 0(미해석)이면 executor 기본값(200k) 유지.
                context_window_budget=budget_window,
                # claude_code 는 CLI 가 자체 컨텍스트(오토-컴팩션)를 관리한다 —
                # 파이프라인 층의 압축이 겹치면 같은 전사를 두 주체가 자르게
                # 되므로 CLI 백엔드에서는 항상 끈다 (파라미터 설명과 일치).
                enable_compaction=(enable_compaction and provider not in _CLI_BACKENDS),
                credentials=credentials,
            )
        except Exception as exc:  # noqa: BLE001 - surface build errors as output, never crash the graph
            logger.exception("agents/geny: failed to build pipeline")
            # 파이프라인 조립 실패 시 이미 만든 메모리 provider 가 turn teardown 을
            # 못 타므로 여기서 직접 정리한다 (FD/락 누수 방지).
            try:
                if locals().get("memory_provider") is not None:
                    import asyncio as _asyncio

                    _asyncio.run(locals()["memory_provider"].close())
            except Exception:  # noqa: BLE001
                pass
            err = f"[ERROR] geny agent could not start: {exc}"
            return iter([err]) if streaming else err

        # 호스트가 턴 단위 취소 훅을 줄 수 있다(사이드카 데몬: 같은 interaction 의
        # 다음 턴을 오염시키지 않는 per-turn Event). 없으면 interaction 스코프 레지스트리.
        _extra_cancel = kwargs.get("cancel_check")
        if not callable(_extra_cancel):
            _extra_cancel = None

        def _cancelled() -> bool:
            if _extra_cancel is not None:
                try:
                    if _extra_cancel():
                        return True
                except Exception:  # noqa: BLE001
                    pass
            if not interaction_id:
                return False
            try:
                from xgen_agent_runtime.host.cancel_context import is_cancelled

                return is_cancelled(interaction_id, response_io_id)
            except Exception:  # noqa: BLE001
                return False

        def _teardown() -> None:
            # 턴이 만든/바꾼 파일을 원본(MinIO + DB 인덱스)에 반영한다. 이걸
            # 빠뜨리면 파드가 죽는 순간 산출물이 사라진다.
            #
            # delete_missing 은 **hydrate 가 성공한 턴에서만** 켠다: 복원에
            # 실패한 빈 캐시로 삭제를 전파하면 원본이 통째로 날아간다.
            # 턴이 만진 파일을 원본에 반영한다 — 3-way(커넥터로컬 flush / 러너
            # publish / 파드 publish) + 클라우드/공유. host 가 캡슐화한다: 서버는
            # 실제 반영, 커넥터 사이드카는 자기 동기화 엔진으로 flush. delete_missing
            # 은 hydrate 성공 턴에서만(빈 캐시 삭제 전파 방지) — host 가 판정한다.
            host.finalize_turn(
                sandbox=_sandbox,
                workflow_id=str(kwargs.get("workflow_id") or ""),
                user_id=kwargs.get("user_id"),
                hydrated_wf=_hydrated_wf,
                hydrated_ws=_hydrated_ws,
                shared_mounts=_shared_mounts,
                cloud_mount=_cloud_mount,
            )
            # CLI 워크스페이스/브릿지 토큰 + built-in run workspace 정리 (순서 무관, 둘 다 방어적)
            for fn in (cli_cleanup, run_dir_cleanup):
                if fn is None:
                    continue
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    pass

        if streaming:
            turn_iter = stream_turn(
                pipeline,
                user_text,
                state,
                tool_events=bool(kwargs.get("tool_events", True)),
                result_sink=result_sink,
                cancel_check=_cancelled,
                output_schema=schema,
                on_close=_teardown,
                host=host,
            )
            if clamped and schema is None:
                # 입력이 잘렸음을 사용자에게 알린다 (agent_xgen 의 경고 관행과
                # 동일 — 축약은 실패가 아니라 부분 반영). 구조화 출력(schema)
                # 스트림에는 섞지 않는다: JSON 파싱을 깨뜨린다.
                from xgen_agent_runtime.host.context_budget import CLAMP_NOTICE

                def _with_notice(inner):
                    yield CLAMP_NOTICE
                    yield from inner

                return _with_notice(turn_iter)
            return turn_iter
        try:
            return run_turn(pipeline, user_text, state, output_schema=schema, host=host)
        finally:
            _teardown()
