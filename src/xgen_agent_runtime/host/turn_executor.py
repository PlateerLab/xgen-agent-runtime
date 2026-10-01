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
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

# 본체가 쓰는 모듈-수준 상수/헬퍼(항상 실행) — 서버에서 resolve. lazy 트리거라
# 순환 없음(execute 가 turn_executor 를 지연 import). Phase 2 에서 패키지로 이전.
from xgen_agent_runtime.host._constants import (  # noqa: E402
    _CLI_BACKENDS,
    _self_evolution_policy,
    cli_tool_naming_note,
    default_prompt,
    SELF_EVOLUTION_PROMPT_BLOCK,
)
from xgen_agent_runtime.host import local_folders as _local_folders
from xgen_agent_runtime.host.tool_exposure import registers_core, sends_every_schema
from xgen_agent_runtime.host.turn_input import TurnInput


#: 턴 안에서 MCP 도구 목록을 다시 읽지 않는 CLI 백엔드 — 계층 노출(문 뒤에 숨긴 도구)을 쓸 수 없다.
#:
#: 2026-09-30 실측(실제 CLI + 실제 브릿지 + 가짜 모델 서버): Claude Code 2.1.280 은
#: ``notifications/tools/list_changed`` 를 받으면 tools/list 를 다시 읽는다(셔틀이 그 재조회가 끝날 때까지
#: 호출 응답을 붙들어 같은 턴에 새 도구가 선다). Codex 0.159.2 는 한 번도 다시 읽지 않았다 — ToolSearch 로
#: 연 도구를 부르면 "unsupported call" 이었다.
_CLI_WITHOUT_LIST_REFRESH = ("codex",)


def _thinking_param(value: Any) -> Optional[str]:
    """노드·대화의 ``thinking`` 값 → 파이프라인의 표준 값. 비었거나 ``auto`` 면 None(모델 기본)."""
    text = str(value or "").strip().lower()
    return None if text in ("", "auto", "default") else text


def _env_bytes(name: str, default: int) -> int:
    """``GENY_*`` 를 읽고, 없으면 개명 전 이름(``XGENY_*``)을 읽는다 — 배포에 남은 옛 설정을 살린다.

    뜻은 예전 그대로다: 없으면 기본값, 빈 값이면 0.
    """
    raw = os.getenv(name)
    if raw is None:
        raw = os.getenv("X" + name)
    if raw is None:
        return default
    return int(raw or 0)


#: 세션에서 읽어 오는 이미지 첨부의 예산 — 워크스페이스 첨부(workflow 쪽 GENY_IMAGE_*)와
#: 같은 값. 한 장과 한 턴 합계 둘 다 본다.
_ATTACH_IMAGE_MAX_BYTES = _env_bytes("GENY_IMAGE_MAX_BYTES", 20 * 1024 * 1024)
_ATTACH_TURN_IMAGE_MAX_BYTES = _env_bytes("GENY_TURN_IMAGE_MAX_BYTES", 40 * 1024 * 1024)

logger = logging.getLogger("editor.nodes.xgen.agent.agent_geny")


def _budget_pair(value: Any) -> Optional[Tuple[int, int]]:
    """노드 파라미터 → (soft, hard). None/0/빈 값이면 예산 없음."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        soft, hard = int(value[0] or 0), int(value[1] or 0)
    elif isinstance(value, dict):
        soft, hard = int(value.get("soft") or 0), int(value.get("hard") or 0)
    else:
        hard = int(value or 0)
        soft = hard // 2
    return (soft, hard) if soft > 0 and hard > soft else None


def _prune_threshold(value: Any) -> Optional[int]:
    """노드 파라미터 → 비용 트리거 임계(토큰). None/음수/해석 불가는 런타임 기본,
    0 은 '끔' 을 뜻하므로 그대로 넘긴다."""
    if value is None:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n >= 0 else None


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


def _memory_block_for(host: Any, workflow_id: str, *, write_available: Optional[bool]) -> str:
    """이 턴의 메모리 지침 — 쓰기 도구가 있으면 원래 블록, 없으면 읽기 전용 블록.

    ``write_available`` 이 None(CLI 브릿지처럼 registry 를 여기서 못 보는 경로)이면 호스트의
    선택 훅 ``memory_write_available(workflow_id)`` 에 묻는다. 훅이 없으면 예전대로 쓰기 블록.
    """
    from xgen_agent_runtime.host._constants import MEMORY_PROMPT_BLOCK, MEMORY_READONLY_PROMPT_BLOCK

    if write_available is None:
        probe = getattr(host, "memory_write_available", None)
        try:
            write_available = bool(probe(workflow_id)) if callable(probe) else True
        except Exception:  # noqa: BLE001 — 훅 실패는 예전 동작(쓰기 블록)
            write_available = True
    return MEMORY_PROMPT_BLOCK if write_available else MEMORY_READONLY_PROMPT_BLOCK


def _folder_device_info(host: Any) -> Dict[str, Any]:
    """이 대화의 폴더가 있는 기기 — 호스트가 알면(OPTIONAL 훅). 모르면 빈 dict.

    폴더는 대화에 붙고 물리적으로는 기기 하나에 있다. 턴을 보낸 화면(웹·휴대폰·다른 PC)과
    그 기기가 다를 수 있다 — 안내가 기기 이름과 그 사실을 말한다.
    """
    probe = getattr(host, "folder_device_info", None)
    if not callable(probe):
        return {}
    try:
        info = probe()
    except Exception:  # noqa: BLE001 — 안내 한 줄 때문에 턴을 깨지 않는다
        return {}
    return dict(info) if isinstance(info, dict) else {}


def _local_device_platform(host: Any) -> Optional[str]:
    """이번 턴 기기의 OS — 호스트가 알면(OPTIONAL 훅). 모르면 None(안내는 "device")."""
    probe = getattr(host, "local_device_platform", None)
    if not callable(probe):
        return None
    try:
        return str(probe() or "") or None
    except Exception:  # noqa: BLE001 — 이름 하나 때문에 턴을 깨지 않는다
        return None


def _tool_result_filter(host: Any) -> Optional[Any]:
    """이 턴의 도구 결과 필터 — 호스트가 주면(OPTIONAL 훅 ``tool_result_filter``). 없으면 None.

    턴마다 한 번 묻는다. 훅이 없거나 실패하면 필터 없이 돈다(결과 그대로) — 필터 하나 때문에
    턴을 깨지 않는다. 받은 필터는 SDK·CLI 두 경로의 도구 컨텍스트에 똑같이 실린다.
    """
    probe = getattr(host, "tool_result_filter", None)
    if not callable(probe):
        return None
    try:
        result_filter = probe()
    except Exception:  # noqa: BLE001 — 훅 실패는 필터 없음
        logger.warning("agents/geny: tool_result_filter 훅 실패 — 필터 없이 진행", exc_info=True)
        return None
    return result_filter if callable(result_filter) else None


class AgentTurnExecutor:
    """execute() 의 host-무관 판. 서버·커넥터가 같은 run() 을 돈다."""

    def run(self, host: Any, **kwargs):
        # node_name 은 워크플로 executor 가 kwargs 에 실어 준다(node_info nodeName)
        # = 노드의 self.nodeName. build_pipeline name= 에만 쓰인다.
        node_name = kwargs.get("node_name") or ""
        from xgen_agent_runtime.host.memory import history_messages
        from xgen_agent_runtime.host.rag import collect_rag
        from xgen_agent_runtime.host.runner import (
            DEFAULT_MAX_CONTINUATION_SLICES,
            build_pipeline,
            run_turn,
            stream_turn,
        )
        from xgen_agent_runtime.host.tools import adapt_tools
        from xgen_agent_runtime import PipelineState
        from xgen_agent_runtime.core.shared_keys import SharedKeys

        turn_input = TurnInput.from_raw(kwargs.get("text"))
        text = turn_input.text
        streaming = bool(kwargs.get("streaming", True))
        interaction_id = str(kwargs.get("interaction_id") or "")
        response_io_id = kwargs.get("response_io_id")

        try:
            # host(HostServices)는 run() 인자로 받는다 — 서버는 ServerHostServices.
            # 본체는 인프라에 오직 host.* 로만 닿는다.
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

            # 도구 표면은 **계층적**이다. 기본 도구(웹·파일·셸·기억)와 연결된
            # 지식소스의 검색 도구는 언제나 즉시 보이고, 연결된 API/DB/MCP 노드는
            # 이름과 한 줄로만 알려 둔 뒤 ToolSearch 로 필요할 때 스키마를 끌어온다.
            # 도구 목록은 재고 목록이 아니라 지도다 — 이번 턴에 부르지도 않을 수백 개의
            # 스키마에 컨텍스트를 쓰면, 모델은 더 많이 읽고 더 못 고른다.
            # 'flat' 은 그 계층을 포기하고 전부 선노출하는 탈출구다.
            _flat_tools = sends_every_schema(kwargs.get("tool_exposure"))
            if provider in _CLI_WITHOUT_LIST_REFRESH and not _flat_tools:
                # 이 CLI 는 턴 안에서 도구 목록을 다시 읽지 않는다 — 문이나 ToolSearch 가 연 도구를 이름으로
                # 부를 수 없다("unsupported call"). 계층은 토큰 절약이지 능력의 경계가 아니므로, 이 백엔드는
                # 처음부터 전부 보여 준다(능력은 SDK 와 같고, 숨김 목록이 없을 뿐이다).
                logger.info(
                    "agents/geny: %s 는 턴 중 도구 목록을 다시 읽지 않는다 — 평면 노출", provider
                )
                _flat_tools = True

            def _turn_one(name: str) -> bool:
                """이 도구가 이번 턴 **첫 화면**에 스키마까지 나가는가.

                등록 지점마다 제 판단으로 ``core=True`` 를 쓰면 표면은 아무도
                의도하지 않은 모양이 된다 — 그래서 계획은 tool_exposure 한 곳에
                있고, 여기서는 그 계획에 물을 뿐이다.
                """
                return registers_core(name, flat=_flat_tools)

            result_sink: Dict[str, str] = {}
            # ── 표면은 provider 와 무관하게 **하나**다 ─────────────────────────
            # SDK provider 는 이 레지스트리를 파이프라인(Stage 3/10)으로 쓰고, CLI provider
            # (claude_code·codex)는 같은 레지스트리를 TurnToolSurface 로 묶어 호스트의 MCP 브릿지가
            # 그대로 광고·실행한다(host.tool_surface). 예전엔 CLI 표면을 호스트가 따로 조립해서
            # 같은 에이전트가 provider 에 따라 다른 도구를 받았다(게스트 제외 누락·문 누락·스키마
            # 차이 — 2026-09-30 감사).
            _is_cli = provider in _CLI_BACKENDS
            #: CLI 에 도구가 광고되는 MCP 서버 이름 — 프롬프트 이름 규약 안내가 쓴다.
            _cli_mcp_server = "connector"

            # ── CLI 도구 브릿지 가용성 (claude_code/codex 전용) ───────────
            # CLI 는 도구를 MCP 로만 받는다. 호스트가 브릿지를 못 세우면 이 턴의 도구는 CLI 에 닿지
            # 않는다 — 그때 도구를 약속하면 유령 호출이 된다(감사 #25). host.cli_bridge_available 은
            # OPTIONAL — 없으면 True. 예외도 True(판정 불가 = 브릿지 있음).
            _cli_bridge_ok = True
            _cli_bridge_reason = ""
            if _is_cli:
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
            #: 이 턴에 도구가 모델에게 닿는가 — SDK 는 항상, CLI 는 브릿지가 있을 때.
            _tools_reach_model = (not _is_cli) or _cli_bridge_ok

            registry = adapt_tools(kwargs.get("tools"), result_sink=result_sink, core=_flat_tools)
            rag_block, embedded_tools = collect_rag(
                text,
                kwargs.get("context"),
                context_builder=host.rag_context_builder,
            )
            if embedded_tools:
                registry = adapt_tools(
                    embedded_tools, result_sink=result_sink, registry=registry, core=True
                )
            # 사용자 기기(데스크톱·CLI·VSCode·모바일 앱, 웹 [폴더])가 올린 기기 도구 — 실행자
            # (user_id)의 기기가 연결돼 있으면 그 카탈로그를 이번 턴 도구로 합산한다(그래프 노드 없이
            # 실행 시점 자동). 기기 미연결 시 빈 리스트 → no-op.
            #
            # 폴더는 **대화에 붙는다**(local_folders). 새 앱은 매 턴 이 대화에 연결된 폴더를
            # 보내고, 폴더 도구(파일·셸·열기…)는 폴더가 있을 때만 보인다 — 규칙은
            # host.local_folders 한 곳이다. 옛 앱·웹(None)은 예전 규칙 그대로다.
            _folders = _local_folders.parse_local_folders(kwargs.get("local_folders"))
            _device_tool_names: List[str] = []
            try:
                # client_surface 게이트(host 내부): 대화 출처가 앱일 때만 기기 도구를
                # 준다 (web 대화엔 앱이 연결돼 있어도 no-op).
                connector_tools = host.build_connector_mcp_tools(
                    kwargs.get("user_id"), kwargs.get("client_surface")
                )
                connector_tools = _local_folders.filter_device_tools(
                    connector_tools or [], _folders
                )
                _device_tool_names = [getattr(t, "name", "") or "" for t in connector_tools]
                if connector_tools:
                    # 기기 도구도 계층을 지킨다 — 브라우저 조작 6종은 BrowserGuide 뒤에
                    # 두고 기본 동사만 남긴다. 폴더가 연결된 대화의 폴더 도구는 첫 화면에
                    # 바로 나간다(사용자가 폴더를 붙인 것 자체가 "내 파일을 다뤄라" 다).
                    registry = adapt_tools(
                        connector_tools,
                        result_sink=result_sink,
                        registry=registry,
                        core=lambda name: (
                            _turn_one(name)
                            or _local_folders.folder_tool_is_turn_one(name, _folders)
                        ),
                    )
                    logger.info(
                        "agents/geny: 기기 도구 %d개 자동 주입 (연결 폴더 %s)",
                        len(connector_tools),
                        "-" if _folders is None else len(_folders),
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("agents/geny: 기기 도구 주입 실패 (무시): %s", exc)
            #: 이번 턴에 폴더 도구가 하나도 없는가 — 그러면 기록 속 옛 폴더 도구 호출을
            #: 요청 사본에서 평문으로 바꾼다(없는 도구를 다시 부르거나 옛 경로를 믿지 않게).
            _retire_device_calls = not any(
                _local_folders.is_folder_tool(n) for n in _device_tool_names
            )
            _turn_notes: List[str] = []
            _folder_info = _folder_device_info(host) if _folders else {}
            _folder_platform = _local_device_platform(host) if _folders is not None else None
            _folder_note = _local_folders.turn_note(
                _folders,
                available_tools=_device_tool_names,
                platform=_folder_platform,
                device_name=str(_folder_info.get("name") or "") or None,
                remote=bool(_folder_info.get("remote")),
            )
            if _folder_note:
                _turn_notes.append(_folder_note)
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
            if _turn_notes:
                state.shared[SharedKeys.TURN_NOTES] = _turn_notes
            if _retire_device_calls:
                state.shared[SharedKeys.RETIRED_TOOL_CALLS] = _local_folders.retired_calls_spec()
            _folder_facts = _local_folders.shared_folder_facts(
                _folders,
                device=(
                    f'{_local_folders.device_label(_folder_platform)} "{_folder_info.get("name")}"'
                    if _folder_info.get("name")
                    else _local_folders.device_label(_folder_platform)
                ),
            )
            if _folder_facts:
                # sandbox 도구가 기기 경로를 받거나 "없음" 을 돌려줄 때 기기 도구를 가리키는 안내가 읽는다
                # (stages/s10_tool/second_machine).
                state.shared[_local_folders.SHARED_FOLDERS_KEY] = _folder_facts
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
            # enable_memory 기본 True — 메모리 노드와 무관하게 파일 vault 를 attach 한다.
            # provider 수명은 turn teardown 이 소유(runner._close_memory_provider). 자동 계층
            # (Stage 2 주입, Stage 18 기록)은 provider 와 무관하게 모든 백엔드에서 돈다. 스스로
            # 읽고 쓰는 도구 6종은 다른 도구와 같은 레지스트리에 들어간다(CLI 도 같은 표면).
            memory_provider = None
            # 명시적으로 비운 것("")과 아예 안 준 것(None/키 없음)을 구분한다 —
            # `or default_prompt` 는 둘을 똑같이 취급해, 사용자가 System Prompt 를
            # 의도적으로 비워도 조용히 기본 문구로 되돌아갔다(진짜 "시스템 프롬프트
            # 없음" 을 요청할 방법이 없었다). 키 자체가 없을 때만 기본값을 쓴다.
            system_prompt = kwargs.get("system_prompt")
            if system_prompt is None:
                system_prompt = default_prompt

            # ── 코드 실행 기반 (xgen-workflow-sandbox) ─────────────────
            #
            # 켜져 있으면 파일/셸 도구는 **이 파드가 아니라** 러너 세션에서
            # 돈다. 준비가 곧 복원이라 여기서 한 번만 붙인다.
            # 실패하면 붙이지 않는다 — 반쯤 붙은 상태가 제일 나쁘다.
            # 러너 세션은 host 가 붙인다(GenySandbox 프로토콜).
            _sandbox = host.make_sandbox(
                str(kwargs.get("workflow_id") or ""),
                kwargs.get("user_id"),
            )
            # CLI 런타임 빌더가 여기서 꺼내 쓴다 (같은 세션 — CLI 프로세스의 cwd).
            kwargs["_sandbox_session"] = _sandbox
            # 영구 작업 도구 — 서버 스케줄러에 이 에이전트를 건다. CLI 의
            # 세션 한정 Cron* 은 runner 가 차단하므로, 이게 없으면 반복 요청을
            # 받을 길 자체가 없다.
            _job_tools = []
            if kwargs.get("workflow_id") and kwargs.get("user_id"):
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
            if _job_tools and _tools_reach_model:
                system_prompt = system_prompt + "\n\n" + host.jobs_prompt_block()
            # 호스트가 소유한 스킬 도구들(앱 등). Jobs 처럼 스킬이 늘 때마다
            # 프로토콜을 넓히지 않으려고 일반 훅 하나로 받는다 — 계층 판정은
            # 여기서 이름으로만 한다(TURN_ONE_TOOLS).
            _host_skill_tools: list = []
            try:
                _host_skill_tools = list(host.build_host_skill_tools(**kwargs) or [])
            except Exception as _hexc:  # noqa: BLE001
                logger.warning("agents/geny: 호스트 스킬 도구 실패 (스킵): %s", _hexc)
            # 실행 환경 안내 — 도구가 어디서 도는지 host 가 설명한다(러너 sandbox).
            # 안 알려 주면 에이전트는 자기 코드가 어디서 도는지 모른 채 /tmp 에 쓰고
            # 다음 턴에 잃는다.
            _env_block = host.environment_prompt(_sandbox, provider)
            if _env_block:
                system_prompt = system_prompt + "\n\n" + _env_block
            if _tools_reach_model and kwargs.get("system_prompt") != "":
                # 왕복 수가 곧 비용이다 — 병렬 호출·일괄 스크립트·출력 최소화 원칙.
                # 사용자가 시스템 프롬프트를 명시적으로 비운 턴에는 붙이지 않는다.
                from xgen_agent_runtime.host._constants import EFFICIENCY_PROMPT_BLOCK

                system_prompt = system_prompt + EFFICIENCY_PROMPT_BLOCK
            # ── 자기진화(self-evolution) 판정 ────────────────────────────────
            # ★ 보안: 배포(deploy_)·게스트(guest_)·고정본 실행에서는 절대 허용하지 않는다. 그
            # 실행은 워크플로 OWNER user_id 로 돌아 write-access 검사를 통과하므로,
            # 익명 사용자/문서 프롬프트 인젝션이 라이브 프로덕션 그래프를 영구 변조할
            # 수 있다(감사 CRITICAL). 판정은 모든 provider 공용이다.
            _se_allowed, _se_reason = _self_evolution_policy(kwargs, host.setting)
            kwargs["_self_evolution_allowed"] = _se_allowed
            if not _se_allowed:
                logger.info("agents/geny: self-evolution 미배선 — %s", _se_reason)

            _memory_block_pending = False
            if bool(kwargs.get("enable_memory", True)):
                from xgen_agent_runtime.host._constants import (
                    MEMORY_AUTO_PROMPT_BLOCK,
                )
                from xgen_agent_runtime.host.memory_tools import build_memory_tools

                memory_provider = host.build_memory_provider(
                    str(kwargs.get("workflow_id") or ""), interaction_id
                )
                if memory_provider is not None and not _tools_reach_model:
                    # 브릿지 없는 CLI: 도구는 없고 자동 계층(Stage 2 주입 + Stage 18 기록)만
                    # 돈다 — 그 사실만 알리고 도구는 광고하지 않는다.
                    system_prompt = system_prompt + MEMORY_AUTO_PROMPT_BLOCK
                    logger.info(
                        "agents/geny: CLI 메모리 도구 미광고 — %s (자동 계층만 동작)",
                        _cli_bridge_reason,
                    )
                elif memory_provider is not None:
                    try:
                        from xgen_agent_runtime.tools import ToolRegistry

                        if registry is None:
                            registry = ToolRegistry()
                        for mem_tool in build_memory_tools(memory_provider):
                            registry.register(mem_tool, core=_turn_one(mem_tool.name))
                        # 지침 블록은 **여기서 붙이지 않는다** — 호스트 정책(register_builtin_tools /
                        # register_forged_tools)이 게스트·동결 턴에서 memory_write/pin 을 뺀 뒤,
                        # 남은 도구를 보고 문구를 고른다(아래 _memory_block_pending).
                        _memory_block_pending = True
                        logger.info("agents/geny: 내장 메모리 활성 (self-serve 도구 6개 등록)")
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "agents/geny: 메모리 도구 등록 실패 (자동 계층만 동작): %s", exc
                        )

            # ── built-in 도구 패밀리 (web/parsing/ssh/workflow/filesystem/shell) ──
            # 모든 provider 가 **같은 조립**을 지난다. CLI 네이티브 도구는 전면 차단이므로 파일/셸도
            # 여기서 나온 우리 도구가 유일한 경로다. 파일 도구는 workspace 에 격리된다(path guard).
            # 관리자 차단: GENY_TOOLS_*_ENABLED.
            run_tool_context = None
            run_dir_cleanup = None
            # 턴 종료 시 원본(MinIO+DB)에 반영할 workspace. hydrate 가 **성공한**
            # 경우에만 채운다 — 복원 실패한 빈 캐시로 삭제를 전파하면 원본이
            # 통째로 날아간다.
            _hydrated_ws: Optional[str] = None
            _hydrated_wf: str = ""
            if _tools_reach_model:
                try:
                    import shutil as _shutil
                    import tempfile as _tempfile

                    from xgen_agent_runtime.tools import ToolRegistry

                    if registry is None:
                        registry = ToolRegistry()
                    bt_summary = host.register_builtin_tools(
                        registry,
                        # 불리언을 넘긴다 — 이 함수는 다른 레포(workflow)에 있고,
                        # 시그니처를 바꾸면 두 레포의 배포 순서가 계약이 된다.
                        # 계획(tool_exposure)은 그쪽에서도 import 할 수 있다.
                        core=_flat_tools,
                        user_id=kwargs.get("user_id"),
                        anthropic_api_key=host.resolve_api_key("anthropic", kwargs),
                        ssh_servers=host.load_ssh_servers(),
                    )
                    for _jt in list(_job_tools) + list(_host_skill_tools):
                        if registry.get(_jt.name) is None:
                            # 게이트웨이만 첫 턴에 선다 — 멤버는 그 문 뒤다
                            # (JobGuide→Job*, AppGuide→App*).
                            registry.register(_jt, core=_turn_one(_jt.name))
                    if bt_summary["tools"]:
                        # 영속 workspace (Drive형 동기화의 전제): workflow(에이전트)
                        # 축의 안정 디렉터리 — 턴을 가로질러 파일이 살아남는다.
                        # workflow_id 가 없는 비정형 실행만 임시 디렉터리로 폴백(턴 종료 시 정리).
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
                                # 러너가 알려 준 경로를 그대로 쓴다.
                                run_dir = _sandbox.workdir
                            else:
                                # 원본(host)에서 복원한 뒤 턴을 시작한다. hydrate 가
                                # **성공한** 경우에만 markers 를 세운다 — 복원 실패한
                                # 빈 캐시로 삭제를 전파하면 원본이 통째로 날아간다.
                                _hyd = host.hydrate_workspace(_wf_for_ws, run_dir)
                                if _hyd:
                                    _hydrated_ws, _hydrated_wf = run_dir, _wf_for_ws
                                elif _hyd is None:
                                    # 호스트가 복원 개념이 없다(동기화 폴더가 곧 원본)
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

                        run_tool_context = host.build_run_tool_context(
                            interaction_id=interaction_id,
                            run_dir=run_dir,
                            extras=bt_summary["extras"],
                            storage_dir=_storage_dir,
                            extra_allowed=[],
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
                                    core=_flat_tools,
                                    sandboxed=_sandbox is not None,
                                )
                            except Exception as _fexc:  # noqa: BLE001
                                logger.warning(
                                    "agents/geny: 저장된 도구 복원 실패 (스킵): %s", _fexc
                                )

                except Exception as exc:  # noqa: BLE001 — 내장 도구는 실행을 깨지 않는다
                    logger.warning("agents/geny: built-in 도구 등록 실패 (스킵): %s", exc)

            if _memory_block_pending:
                # 호스트 정책이 끝난 뒤의 registry 가 진실이다 — 쓰기 도구가 남아 있으면 "저장하라",
                # 없으면 "여기서는 저장할 수 없다". 약속과 표면이 어긋나면 모델이 우회한다.
                _has_write = registry is not None and registry.get("memory_write") is not None
                system_prompt = system_prompt + _memory_block_for(
                    host,
                    str(kwargs.get("workflow_id") or ""),
                    write_available=_has_write,
                )
                _memory_block_pending = False

            # ── 자기진화(self-evolution) 등록 — built-in tools 와 독립 ──────────
            # WorkflowSelf 는 registry + workflow_id 만 있으면 되고(편집은 DB, workspace
            # 불필요), 내장 도구 조립과 무관해야 한다. 모든 provider 가 같은 등록기를 지난다.
            if _tools_reach_model and _se_allowed:
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
                    # 프롬프트 블록은 도구가 **실제로 등록된** 경우에만 — 호스트가
                    # WorkflowSelf 를 주지 않았으면 "그래프를 영구 편집할 수 있다"고 말해
                    # 놓고 도구가 없는 유령 안내가 된다.
                    if registry.get("WorkflowSelf") is not None:
                        system_prompt = system_prompt + SELF_EVOLUTION_PROMPT_BLOCK
                    else:
                        logger.info(
                            "agents/geny: self-evolution 미배선 — host 가 WorkflowSelf 를 제공하지 않음"
                        )
                except Exception as _sexc:  # noqa: BLE001
                    logger.warning("agents/geny: self-evolution 도구 등록 실패 (스킵): %s", _sexc)
            elif _is_cli and _se_allowed and not _tools_reach_model:
                logger.info(
                    "agents/geny: self-evolution 미배선 — CLI 브릿지 없음 (%s)",
                    _cli_bridge_reason,
                )

            # 숨긴 도구로 가는 입구(ToolSearch·SelfExtendGuide) — 파이프라인 조립도 같은 함수를
            # 부르지만, CLI 는 레지스트리를 파이프라인에 넘기지 않으므로 여기서 세운다.
            if _tools_reach_model:
                from xgen_agent_runtime.host.runner import ensure_surface_entrances

                ensure_surface_entrances(registry)

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

            # 호스트의 도구 결과 필터(선택 훅) — 턴마다 한 번 받아 SDK·CLI 두 경로의 도구
            # 컨텍스트에 똑같이 싣는다. 적용은 Stage 10 라우터 한 곳(RegistryRouter.route)이다.
            _result_filter = _tool_result_filter(host)
            if _result_filter is not None and run_tool_context is not None:
                run_tool_context.result_filter = _result_filter

            llm_client = None
            cli_cleanup = None
            #: CLI 턴의 도구 표면 — 호스트의 MCP 브릿지가 광고·실행한다(host.tool_surface).
            _tool_surface = None
            if _is_cli:
                # CLI 백엔드는 에이전트 루프를 CLI 가 소유한다 — 파이프라인 Stage 10 이
                # 돌지 않으므로 레지스트리는 파이프라인이 아니라 표면 객체로 간다. 같은 레지스트리·
                # 같은 도구 컨텍스트·같은 턴 상태 — SDK 경로와 도구가 **같다**.
                if registry is not None and len(registry) and _tools_reach_model:
                    from xgen_agent_runtime.host.tool_surface import TurnToolSurface
                    from xgen_agent_runtime.tools.catalog import deferred_catalog_text

                    _tool_surface = TurnToolSurface(
                        registry=registry,
                        tool_context=run_tool_context,
                        state=state,
                        server_name=_cli_mcp_server,
                    )
                    if _result_filter is not None:
                        # run_tool_context 가 없으면 표면이 제 컨텍스트를 만든다 — 거기에도 싣는다.
                        _tool_surface.tool_context.result_filter = _result_filter
                    kwargs["_tool_surface"] = _tool_surface
                    # 숨김 목록 — SDK 는 Stage 3 이 붙이는 글을 같은 함수로 만들어 붙인다.
                    _catalog = deferred_catalog_text(registry)
                    if _catalog:
                        system_prompt = system_prompt + "\n\n" + _catalog
                    system_prompt = system_prompt + cli_tool_naming_note(_cli_mcp_server, provider)
                    logger.info(
                        "agents/geny: %s 백엔드 — 도구 %d개(첫 화면 %d개)를 MCP 브릿지로 넘긴다",
                        provider,
                        len(registry),
                        len(registry.list_exposed()),
                    )
                elif registry is not None and len(registry) and not _tools_reach_model:
                    # 브릿지가 없으면 정말로 못 준다. 그때는 **조용히 넘어가지
                    # 않는다** — 에이전트가 이유를 알아야 사용자에게 정확히 말한다.
                    logger.warning(
                        "agents/geny: 도구 %d개를 전달할 수 없다 — %s",
                        len(registry),
                        _cli_bridge_reason,
                    )
                    system_prompt = system_prompt + (
                        f"\n\n(Note: {len(registry)} tool(s) are wired into this agent"
                        " but cannot be delivered on this backend this turn"
                        f" — {_cli_bridge_reason}. Do not claim they exist; tell the user"
                        " this configuration issue if they ask for those capabilities.)"
                    )
            if provider == "claude_code":
                llm_client, cli_cleanup = host.build_cli_runtime(
                    "claude_code",
                    kwargs,
                )
            elif provider == "codex":
                llm_client, cli_cleanup = host.build_cli_runtime(
                    "codex",
                    kwargs,
                )

            # ── 작은 워크스페이스 fast path (관리자/노드 opt-in) ──────────
            # 전체 입력을 확실히 실을 수 있을 때만 탐색 왕복을 없앤다. 과제명·확장자·
            # 업무 용어는 보지 않고 크기/파일·디렉터리 수/UTF-8/명시적 경로 참조만 본다.
            # 실패하거나 애매하면 기존 agent loop 로 돌아간다. 모든 provider 가 같은 입력을 받는다
            # (CLI 도 같은 도구 컨텍스트·같은 턴 상태를 쓰므로 읽은 파일 장부가 이어진다).
            from xgen_agent_runtime.host.workspace_fast_path import (
                WORKSPACE_FAST_PATH_SETTING,
                flag_enabled,
                prepare_workspace_fast_path,
                register_snapshot_witnesses,
            )

            _fast_path_requested = flag_enabled(kwargs.get("enable_workspace_fast_path"))
            if not _fast_path_requested:
                _fast_path_requested = host.setting_truthy(WORKSPACE_FAST_PATH_SETTING)
            _fast_path = None
            _fast_path_original_text = text
            _fast_path_reason = "disabled"
            if (
                _tools_reach_model
                and _fast_path_requested
                and run_tool_context is not None
                and registry is not None
                and registry.get("Bash") is not None
            ):
                import asyncio as _asyncio

                try:
                    _fast_path = _asyncio.run(
                        prepare_workspace_fast_path(
                            text,
                            turn_input.attachments,
                            run_tool_context,
                            enabled=True,
                        )
                    )
                    if _fast_path.active:
                        text = _fast_path.text
                        _fast_path_reason = "eligible"
                    else:
                        _fast_path_reason = _fast_path.reason
                        logger.info(
                            "agents/geny: workspace fast path 폴백 (%s)",
                            _fast_path.reason,
                        )
                except Exception:  # noqa: BLE001 — 최적화 실패는 기존 루프로 폴백
                    _fast_path_reason = "preparation_error"
                    logger.warning("agents/geny: workspace fast path 준비 실패", exc_info=True)
            elif _fast_path_requested:
                if not _tools_reach_model:
                    _fast_path_reason = "tools_unavailable"
                elif run_tool_context is None:
                    _fast_path_reason = "workspace_unavailable"
                else:
                    _fast_path_reason = "bash_unavailable"

            # ── 요청에 이름이 나온 작업 폴더 파일 붙이기 (host/referenced_files.py) ──
            # 파일을 하나씩 Read 하는 왕복을 없앤다. RAG 블록과 같은 자리(사용자 턴)에 싣고,
            # 예산 맞추기가 함께 자를 수 있게 rag_block 에 합친다. 끝까지 실은 파일은
            # "읽은 파일" 장부에 올려 바로 Edit/Write 할 수 있게 한다.
            # 빠른 경로가 열리면 작업 폴더 전체가 이미 실렸다 — 따로 붙이지 않는다.
            if (
                _tools_reach_model
                and not (_fast_path is not None and _fast_path.active)
                and str(host.setting("GENY_PREFETCH_REFERENCED_FILES", "1")).strip()
                not in (
                    "0",
                    "false",
                    "off",
                )
            ):
                try:
                    import asyncio as _asyncio

                    from xgen_agent_runtime.host.referenced_files import collect as _collect_refs
                    from xgen_agent_runtime.tools.built_in._file_witness import WITNESSED_KEY
                    from xgen_agent_runtime.tools.fs import LocalFS, RunnerFS

                    _ref_fs = None
                    if _sandbox is not None:
                        _ref_fs = RunnerFS(_sandbox, str(getattr(_sandbox, "workdir", "") or ""))
                    elif getattr(run_tool_context, "working_dir", ""):
                        _ref_fs = LocalFS(str(run_tool_context.working_dir))
                    if _ref_fs is not None:
                        _pre = _asyncio.run(_collect_refs(text, _ref_fs))
                        if _pre.block:
                            rag_block = f"{rag_block}\n\n{_pre.block}" if rag_block else _pre.block
                            if _pre.witnessed:
                                _book = list(state.shared.get(WITNESSED_KEY) or [])
                                _book.extend(p for p in _pre.witnessed if p not in _book)
                                state.shared[WITNESSED_KEY] = _book
                            logger.info(
                                "agents/geny: 요청에 나온 파일 %d개 첨부, %d개 이름만",
                                len(_pre.attached),
                                len(_pre.skipped),
                            )
                except Exception:  # noqa: BLE001 — 붙이기 실패는 턴을 깨지 않는다(모델이 직접 읽으면 된다)
                    logger.warning("agents/geny: 요청 파일 첨부 실패 (건너뜀)", exc_info=True)

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
                        # A partial snapshot is not a snapshot. If budgeting
                        # trims any of it, recompute from the original request
                        # and keep the normal agent loop/witness semantics.
                        if _fast_path is not None and _fast_path.active:
                            _fast_path = None
                            _fast_path_reason = "context_budget"
                            fit = fit_input_to_budget(
                                text=_fast_path_original_text,
                                rag_block=rag_block,
                                system_prompt=system_prompt,
                                history=state.messages,
                                registry=registry,
                                provider=provider,
                                model=model,
                                max_tokens=max_tokens_val,
                                window=budget_window,
                                reserved_tokens=(3_000 if memory_provider is not None else 0),
                            )
                            logger.info("agents/geny: workspace fast path 폴백 (context_budget)")
                        text, rag_block, clamped = fit.text, fit.rag_block, fit.clamped
                        if fit.clamped:
                            logger.warning(
                                "agents/geny: 입력 클램프 적용 (window=%d budget=%d before=%d)",
                                fit.window,
                                fit.budget,
                                fit.total_before,
                            )

            if _fast_path is not None and _fast_path.active:
                register_snapshot_witnesses(state, _fast_path)
                logger.info(
                    "agents/geny: workspace fast path 활성 (files=%d bytes=%d)",
                    _fast_path.file_count,
                    _fast_path.total_bytes,
                )
            # 켜지 않은 턴은 판정을 지운다 — 재사용된 state 에 앞 턴 값이 남지 않게.
            state.shared.pop(SharedKeys.WORKSPACE_FAST_PATH, None)
            if _fast_path_requested:
                state.shared[SharedKeys.WORKSPACE_FAST_PATH] = {
                    "active": bool(_fast_path is not None and _fast_path.active),
                    "reason": _fast_path_reason,
                    "file_count": _fast_path.file_count if _fast_path is not None else 0,
                    "total_bytes": _fast_path.total_bytes if _fast_path is not None else 0,
                }

            user_text = f"{text}\n\n{rag_block}" if rag_block else text
            # 첨부 경로는 **여기서** 절대 경로가 된다. 파일 도구는 절대 경로를 요구하는데
            # 첨부는 워크스페이스 상대로 들어오므로, 기준을 모델에게 맡기면 틀린 자리를 만든다
            # (2026-09-21 실측: 작업 폴더가 …/<wf>/workspace 인데 …/<wf>/uploads 로 읽으려다
            # 샌드박스 가드에 막혔다). 기준을 아는 쪽은 턴을 여는 이곳 하나다.
            from xgen_agent_runtime.host.attachment_paths import (
                absolutize_attachments,
                hydrate_sandbox_images,
            )

            _ws_base = str(
                getattr(run_tool_context, "working_dir", "")
                or getattr(_sandbox, "workdir", "")
                or ""
            )
            if turn_input.attachments:
                _atts = (
                    absolutize_attachments(turn_input.attachments, _ws_base)
                    if _ws_base
                    else list(turn_input.attachments)
                )
                # 세션에만 있는 이미지(배포된 고정본의 대화 첨부)는 여기서 읽는다 —
                # 서빙 파드에는 사본이 없고, 세션을 들고 있는 자리가 여기뿐이다.
                # ``run`` 은 동기라(노드가 그대로 부른다) 짧은 루프 하나를 돌린다;
                # 실패는 첨부 하나를 포기할 뿐 턴을 깨지 않는다.
                if _sandbox is not None and any(
                    isinstance(a, dict)
                    and str(a.get("kind") or a.get("type") or "").lower()
                    in ("image", "img", "picture")
                    for a in _atts
                ):
                    import asyncio as _asyncio

                    try:
                        _atts = _asyncio.run(
                            hydrate_sandbox_images(
                                _atts,
                                _sandbox,
                                max_bytes=_ATTACH_IMAGE_MAX_BYTES,
                                turn_max_bytes=_ATTACH_TURN_IMAGE_MAX_BYTES,
                            )
                        )
                    except Exception:  # noqa: BLE001 — 첨부 준비 실패가 턴을 깨지 않는다
                        logger.warning("세션 이미지 첨부 준비 실패 (건너뜀)", exc_info=True)
                turn_input = replace(turn_input, attachments=_atts)
            pipeline_input = turn_input.with_text(user_text).as_pipeline_input()

            # Codex-style durable rollout은 관리자 opt-in이다. 대화/도구 결과를
            # 포함할 수 있으므로 기본 활성화하거나 임시 디렉터리로 fallback하지
            # 않는다. workflow storage가 있는 턴만 host가 파일 수명·보존을
            # 소유하고, Pipeline의 공개 입력/출력에는 아무 필드도 추가하지 않는다.
            rollout_path = None
            from xgen_agent_runtime.host.rollouts import (
                ROLLOUT_ENABLED_SETTING,
                allocate_rollout_path,
            )

            if host.setting_truthy(ROLLOUT_ENABLED_SETTING):
                rollout_workflow_id = str(kwargs.get("workflow_id") or "")
                if not rollout_workflow_id:
                    logger.warning(
                        "agents/geny: rollout recording enabled but skipped — "
                        "workflow_id is unavailable"
                    )
                else:
                    rollout_storage_root = host.workspace_storage_root(rollout_workflow_id)
                    if not str(rollout_storage_root).strip():
                        raise ValueError("rollout recording requires a workspace storage root")
                    rollout_path = allocate_rollout_path(
                        rollout_storage_root,
                        interaction_id,
                    )

            pipeline = build_pipeline(
                name=node_name,
                provider=provider,
                model=model,
                api_key=api_key,
                base_url=base_url,
                system_prompt=system_prompt,
                # CLI 는 레지스트리를 파이프라인이 아니라 표면 객체로 받는다(위 _tool_surface).
                registry=None if _is_cli else registry,
                max_iterations=int(kwargs.get("max_iterations", 20)),
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=max_tokens_val,
                # 생각의 강도(노드 값·대화의 선택) — "auto"·빈 값이면 모델 기본.
                thinking_level=_thinking_param(kwargs.get("thinking")),
                stream=streaming,
                output_schema=schema,
                llm_client=llm_client,
                memory_provider=memory_provider,
                memory_distill_spec=memory_distill_spec,
                tool_context=None if _is_cli else run_tool_context,
                # run_tool_context 가 없는 턴(내장 도구 없음)에도 Stage 10 에 필터가 실리게.
                tool_result_filter=None if _is_cli else _result_filter,
                # 모델의 실제 윈도우 — 압축 임계(80%)·guard·루프 토큰-비 정지의
                # 공통 기준. 0(미해석)이면 executor 기본값(200k) 유지.
                context_window_budget=budget_window,
                # claude_code 는 CLI 가 자체 컨텍스트(오토-컴팩션)를 관리한다 —
                # 파이프라인 층의 압축이 겹치면 같은 전사를 두 주체가 자르게
                # 되므로 CLI 백엔드에서는 항상 끈다 (파라미터 설명과 일치).
                enable_compaction=(enable_compaction and provider not in _CLI_BACKENDS),
                credentials=credentials,
                # 호스트가 캐시 토큰 기록을 갖춘 뒤 명시적으로 켠다 (기본 off).
                enable_prompt_cache=bool(kwargs.get("enable_prompt_cache", False)),
                # 노드가 주면 그대로, 없으면 런타임 기본(3). 0 이면 반복 거부 종료 끔.
                **(
                    {"repeat_stop_after": _prune_threshold(kwargs["repeat_stop_after"])}
                    if "repeat_stop_after" in kwargs
                    and _prune_threshold(kwargs["repeat_stop_after"]) is not None
                    else {}
                ),
                # 노드가 주면 그대로, 없으면 런타임 기본(30,000). 0 이면 비용 트리거 끔.
                **(
                    {"prune_over_tokens": _prune_threshold(kwargs["prune_over_tokens"])}
                    if "prune_over_tokens" in kwargs
                    and _prune_threshold(kwargs["prune_over_tokens"]) is not None
                    else {}
                ),
                # 노드가 (soft, hard) 를 주면 그대로, 없으면 런타임 기본(100만/300만).
                **(
                    {"turn_input_budget_tokens": _budget_pair(kwargs["turn_input_budget_tokens"])}
                    if "turn_input_budget_tokens" in kwargs
                    else {}
                ),
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

        # 호스트가 턴 단위 취소 훅을 줄 수 있다(같은 interaction 의 다음 턴을
        # 오염시키지 않는 per-turn Event). 없으면 interaction 스코프 레지스트리.
        _extra_cancel = kwargs.get("cancel_check")
        if not callable(_extra_cancel):
            _extra_cancel = None

        def _cancelled() -> bool:
            try:
                from xgen_agent_runtime.host.cancel_context import is_cancelled

                return is_cancelled(interaction_id, response_io_id, cancel_check=_extra_cancel)
            except Exception:  # noqa: BLE001
                return False

        def _teardown() -> None:
            # 턴이 만든/바꾼 파일을 원본(MinIO + DB 인덱스)에 반영한다. 이걸
            # 빠뜨리면 파드가 죽는 순간 산출물이 사라진다.
            #
            # delete_missing 은 **hydrate 가 성공한 턴에서만** 켠다: 복원에
            # 실패한 빈 캐시로 삭제를 전파하면 원본이 통째로 날아간다.
            # 러너 publish / 파드 publish 는 host 가 캡슐화한다. delete_missing
            # 은 hydrate 성공 턴에서만(빈 캐시 삭제 전파 방지) — host 가 판정한다.
            host.finalize_turn(
                sandbox=_sandbox,
                workflow_id=str(kwargs.get("workflow_id") or ""),
                user_id=kwargs.get("user_id"),
                hydrated_wf=_hydrated_wf,
                hydrated_ws=_hydrated_ws,
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
                pipeline_input,
                state,
                tool_events=bool(kwargs.get("tool_events", True)),
                result_sink=result_sink,
                cancel_check=_cancelled,
                output_schema=schema,
                on_close=_teardown,
                host=host,
                rollout_path=rollout_path,
                max_continuation_slices=int(
                    kwargs.get("max_continuation_slices", DEFAULT_MAX_CONTINUATION_SLICES)
                ),
                usage_sink=kwargs.get("usage_sink"),
                on_loop=_tool_surface.bind_loop if _tool_surface is not None else None,
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
            return run_turn(
                pipeline,
                pipeline_input,
                state,
                output_schema=schema,
                host=host,
                usage_sink=kwargs.get("usage_sink"),
                rollout_path=rollout_path,
                max_continuation_slices=int(
                    kwargs.get("max_continuation_slices", DEFAULT_MAX_CONTINUATION_SLICES)
                ),
                on_loop=_tool_surface.bind_loop if _tool_surface is not None else None,
            )
        finally:
            _teardown()
