"""Pipeline assembly + sync bridges for the ``agents/geny`` node.

Assembly uses geny-executor's ``PipelineBuilder`` for stage wiring, then
attaches an explicitly-built LLM client via ``attach_runtime`` — that is the
library-recommended production path and the only builder-compatible way to
thread a custom ``base_url`` (vLLM / custom endpoints).

The xgen executor runs ``execute()`` in a worker thread, so both bridges
drive the async engine on a private event loop, exactly like the harness
node. ``stream_turn`` translates engine events into xgen stream chunks:
``text.delta`` → str, tool events → ``{"type": "agent_event", ...}`` dicts
(the shape agent_node_processor forwards to the chat UI), and — once, after
the pipeline finishes — ``{"type": "usage", "data": {...}}`` with the turn's
token/cost totals (:func:`turn_usage`).
"""

from __future__ import annotations

import asyncio
import time
import json
import logging
import re
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from xgen_agent_runtime import ClientRegistry, Pipeline, PipelineBuilder, PipelineState
from xgen_agent_runtime.tools import ToolRegistry

logger = logging.getLogger("editor.geny_bridge.runner")

# xgen provider option → geny-executor ClientRegistry key.
# "vllm" maps to the "custom" OpenAI-compatible profile: geny-executor's own
# vllm profile disables tool calling (conservative default), while xgen's
# vLLM deployments serve tool-capable models behind --enable-auto-tool-choice.
_PROVIDER_MAP = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google",
    "vllm": "custom",
    "bedrock": "bedrock",
    "vertex": "vertex",
}

_DISPLAY_RESULT_LIMIT = 4000  # chars shown in a tool_result agent_event
_DISPLAY_TAIL_KEEP = 800  # of which the last N chars are kept (download markers live at the tail)


def _map_provider(provider: str) -> str:
    return _PROVIDER_MAP.get(provider, provider)


def build_client(
    provider: str,
    api_key: str,
    base_url: Optional[str],
    *,
    credentials: Optional[Dict[str, Any]] = None,
) -> Any:
    """provider 문자열 → 런타임 클라이언트.

    ``credentials`` 는 api_key 하나로 부족한 provider(bedrock 의 AWS 키/리전,
    vertex 의 project/location/서비스계정 JSON)의 다중 필드 자격증명 dict —
    빈 값은 걸러서 생성자 표면에 그대로 전달한다 (keyword-only: 기존
    ``build_client(provider, api_key, base_url)`` 호출·monkeypatch 는 그대로
    유효하다).
    """
    key = _map_provider(provider)
    if key == "custom" and not base_url:
        raise ValueError(
            "vLLM/custom provider requires a Base URL (parameter or VLLM_API_BASE_URL config)"
        )
    client_cls = ClientRegistry.get(key)
    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    for cred_key, cred_val in (credentials or {}).items():
        if cred_val not in (None, ""):
            kwargs[cred_key] = cred_val
    return client_cls(**kwargs)


# Claude Code CLI 의 네이티브 fs/셸 도구 — Geny credentials.py 의
# _NATIVE_FS_SHELL_TOOLS 와 **완전 동일 목록** (웹 도구는 Geny 가 어떤 모드에서도
# 막지 않으므로 여기서도 제외). Geny 검증 로직: 샌드박스(GAPT)가 있을 때만 이
# 목록을 차단하고 브릿지의 executor fs 도구가 격리 실행으로 대체한다 — 샌드박스가
# 없으면(=XGEN) 네이티브 허용이 Geny 의 기본값이다. cli_allow_local_tools=False
# 는 그 "샌드박스 격리 모드"의 차단만 미러한다 (이때 파일 작업은 브릿지의
# path-guard 된 executor Read/Write/Bash 가 workspace 안에서 담당).
_CLI_LOCAL_TOOLS = (
    "Bash",
    "Read",
    "Write",
    "Edit",
    "MultiEdit",
    "NotebookEdit",
    "Glob",
    "Grep",
    "LS",
)

#: CLI 의 **세션 한정 스케줄** 도구 — 항상 차단한다.
#:
#: CronCreate/ScheduleWakeup 류는 CLI 프로세스 메모리에 산다: 대화가 끝나면
#: 큐도 죽고 디스크에도 안 남는다. 에이전트는 "5분마다 실행하도록 걸어놨다"고
#: 답했지만 세션이 끝나는 순간 사라졌다 (프로드 실증 — 사용자에게는 "된다더니
#: 안 도는" 기능이다). XGEN 의 영구 작업은 JobSchedule(서버 스케줄러, DB)이
#: 담당한다 — 같은 일을 하는 반쪽 도구가 곁에 있으면 모델은 반드시 그걸 집는다.
_CLI_SESSION_SCHED_TOOLS = ("CronCreate", "CronDelete", "CronList", "ScheduleWakeup")

_CLI_AUTH_MODES = ("api_key", "setup_token", "oauth", "auto")


def build_cli_client(
    *,
    auth_mode: str = "api_key",
    api_key: str = "",
    oauth_token: str = "",
    binary_path: str = "",
    workspace_dir: Optional[str] = None,
    timeout_s: float = 3600.0,
    max_budget_usd: float = 0.0,
    allow_local_tools: bool = False,
    permission_mode: str = "default",
    mcp_config: Any = None,
    settings_path: str = "",
    allow_tools: Any = (),
    extra_env: Optional[Dict[str, str]] = None,
    disallow_tools_extra: Any = (),
    extra_args: Any = (),
    prewarm_spawn: Optional[bool] = None,
) -> Any:
    """Construct a ``ClaudeCodeCLIClient`` from xgen-resolved settings.

    ``prewarm_spawn`` — hot-spare 프리웜(다음 턴용 CLI 프로세스를 스트림 종료
    직후 미리 띄움) 토글. ``None``(기본)이면 클라이언트 기본값(env
    ``GENY_CLI_PREWARM`` 해석)을 그대로 쓴다 — 기존 동작 불변. 서버(xgen-workflow)
    처럼 파이프라인/클라이언트가 **턴마다 새로** 만들어지는 원샷 호스트는
    ``False`` 를 넘긴다: 프리웜된 프로세스는 다음 턴이 없어 절대 재사용되지
    않고 teardown 에 고아로 남는다.

    Auth wiring mirrors Geny's bundle builder — the two footguns it guards:
    ``--bare``(=bare_mode) is only valid on the api_key channel (it bypasses
    the OAuth credential file), and subscription modes must NOT forward an
    API key (a stale key would 401 an otherwise healthy OAuth session).
    ``setup_token`` injects the long-lived token as ``CLAUDE_CODE_OAUTH_TOKEN``
    via env_extras — the channel that is safe for server/container use.
    """
    if auth_mode not in _CLI_AUTH_MODES:
        raise ValueError(f"unsupported Claude Code auth_mode: {auth_mode!r}")
    if auth_mode == "api_key" and not api_key:
        raise ValueError("Claude Code(api_key 모드): ANTHROPIC_API_KEY 가 설정되어 있지 않습니다")
    if auth_mode == "setup_token" and not oauth_token:
        raise ValueError(
            "Claude Code(setup_token 모드): CLAUDE_CODE_OAUTH_TOKEN 이 설정되어 있지 않습니다"
        )

    from xgen_agent_runtime.llm_client.claude_code import ClaudeCodeCLIClient

    kwargs: Dict[str, Any] = {
        "auth_mode": auth_mode,
        "timeout_s": float(timeout_s),
        "default_permission_mode": permission_mode,
        # CLI 자체 자동업데이트 차단 — 버전은 service/claude_code/cli_installer 가 관리
        "env_extras": {"DISABLE_AUTOUPDATER": "1"},
    }
    if auth_mode == "api_key":
        kwargs["api_key"] = api_key
        kwargs["bare_mode"] = True
    else:
        kwargs["api_key"] = ""
        kwargs["bare_mode"] = False
        if auth_mode == "setup_token":
            kwargs["env_extras"] = {**kwargs["env_extras"], "CLAUDE_CODE_OAUTH_TOKEN": oauth_token}
    if binary_path:
        kwargs["binary_path"] = binary_path
    if workspace_dir:
        kwargs["workspace_dir"] = workspace_dir
    if max_budget_usd and float(max_budget_usd) > 0:
        kwargs["max_budget_usd"] = float(max_budget_usd)
    # 차단 목록 = (로컬도구 차단 시 _CLI_LOCAL_TOOLS) + 호출자 추가분.
    # disallow_tools_extra: agent_geny 가 위임 배선 시 CLI 의 자체 서브에이전트
    # 도구(Task/Agent)를 차단하는 데 쓴다 — CLI 내부 위임은 우리 매니저/작업
    # 내역/완료 트리거를 전부 우회하므로(추적 불가), 위임은 반드시
    # mcp__connector__* 표면으로만 흐르게 강제한다.
    disallowed = list(_CLI_LOCAL_TOOLS) if not allow_local_tools else []
    disallowed.extend(t for t in _CLI_SESSION_SCHED_TOOLS if t not in disallowed)
    for name in tuple(disallow_tools_extra or ()):
        if name and name not in disallowed:
            disallowed.append(str(name))
    if disallowed:
        kwargs["disallow_tools"] = tuple(disallowed)
    # Connector Local MCP bridge (agent_geny wires this for the claude_code
    # backend): mcp_config → --mcp-config <json> + --strict-mcp-config, so the
    # CLI's ONLY MCP surface is our per-user connector bridge; settings_path →
    # --settings <json> pre-allows the bridge server so --print (non-interactive)
    # mode doesn't block every tool call on a permission prompt; allow_tools →
    # --allowedTools.
    if mcp_config:
        kwargs["mcp_config"] = mcp_config
    if settings_path:
        kwargs["settings_path"] = settings_path
    if allow_tools:
        kwargs["allow_tools"] = tuple(allow_tools)
    if extra_args:
        # 그대로 argv 뒤에 붙는다. 지금 쓰는 곳은 `--add-dir <연결된 클라우드>`
        # 하나뿐이다 — CLI 네이티브 도구(Read/Glob/Bash)가 cwd 밖을 못 보므로,
        # 이게 없으면 연결해 둔 클라우드가 CLI 에게는 존재하지 않는다.
        kwargs["extra_args"] = tuple(str(a) for a in extra_args)
    if extra_env:
        # 병합 — setup_token 의 CLAUDE_CODE_OAUTH_TOKEN 등 기존 값을 덮지 않는다.
        merged = dict(kwargs["env_extras"])
        for k, v in extra_env.items():
            merged.setdefault(str(k), str(v))
        kwargs["env_extras"] = merged
    if prewarm_spawn is not None:
        # None 은 전달하지 않는다 — 클라이언트가 env 기본값을 해석하게 둔다.
        kwargs["prewarm_spawn"] = bool(prewarm_spawn)
    return ClaudeCodeCLIClient(**kwargs)


_CODEX_AUTH_MODES = ("api_key", "oauth")


def build_codex_cli_client(
    *,
    auth_mode: str = "api_key",
    api_key: str = "",
    binary_path: str = "",
    workspace_dir: Optional[str] = None,
    timeout_s: float = 3600.0,
    mcp_config: Any = None,
    extra_args: Any = (),
    env_extras: Optional[Dict[str, str]] = None,
) -> Any:
    """xgen 설정으로 ``CodexCLIClient`` 를 구성한다 (claude 의 build_cli_client 짝).

    인증 채널 배타 계약은 런타임이 집행한다: api_key 모드만 OPENAI_API_KEY 를
    subprocess 환경에 주입하고, oauth(ChatGPT 구독) 모드는 절대 키를 흘리지
    않는다 — 키가 남아 있으면 청구 채널이 조용히 뒤집힌다. MCP 서버는
    ``-c mcp_servers.*`` 오버라이드로 주입되어 파드의 $CODEX_HOME(auth.json)을
    건드리지 않는다.
    """
    if auth_mode not in _CODEX_AUTH_MODES:
        raise ValueError(f"unsupported Codex auth_mode: {auth_mode!r}")
    if auth_mode == "api_key" and not api_key:
        raise ValueError("Codex(api_key 모드): OPENAI_API_KEY 가 설정되어 있지 않습니다")

    from xgen_agent_runtime.llm_client.codex import CodexCLIClient

    kwargs: Dict[str, Any] = {
        "auth_mode": auth_mode,
        "api_key": api_key if auth_mode == "api_key" else "",
        "timeout_s": float(timeout_s),
    }
    if binary_path:
        kwargs["binary_path"] = binary_path
    if workspace_dir:
        kwargs["workspace_dir"] = workspace_dir
    if mcp_config:
        kwargs["mcp_config"] = mcp_config
    if extra_args:
        kwargs["extra_args"] = tuple(str(a) for a in extra_args)
    if env_extras:
        kwargs["env_extras"] = {str(k): str(v) for k, v in env_extras.items()}
    return CodexCLIClient(**kwargs)


def _schema_instruction(schema: Dict[str, Any]) -> str:
    return (
        "\n\n# Output format\n"
        "Respond with a single JSON object that conforms to the JSON Schema below. "
        "Output ONLY the JSON object — no explanations, no markdown fences.\n"
        + json.dumps(schema, ensure_ascii=False)
    )


def build_pipeline(
    *,
    name: str,
    provider: str,
    model: str,
    api_key: str,
    base_url: Optional[str] = None,
    system_prompt: str = "",
    registry: Optional[ToolRegistry] = None,
    max_iterations: int = 20,
    temperature: float = 0.7,
    max_tokens: int = 8192,
    stream: bool = True,
    output_schema: Optional[Dict[str, Any]] = None,
    llm_client: Optional[Any] = None,
    memory_provider: Optional[Any] = None,
    memory_distill_spec: Optional[Any] = None,
    tool_context: Optional[Any] = None,
    context_window_budget: int = 0,
    enable_compaction: bool = True,
    credentials: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """Assemble a one-shot pipeline for a single node execution.

    ``llm_client`` overrides the provider/api_key/base_url wiring (tests).
    The System stage is always registered — it is what publishes the tool
    registry onto ``state.tools``, so it must exist even for an empty prompt.

    When the registry holds deferred tools (2.42.0 exposure model), the
    built-in ``ToolSearch`` is registered as core so the discovery path is
    never stranded — mirrors the library's ``_ensure_tool_search_reachable``,
    which only runs on the manifest build path.

    ``memory_provider`` (agents/geny 내장 메모리) — 전달되면 executor 의
    manifest-memory attach 경로와 동일하게 배선한다: Stage 2(Context) 에
    ``MemoryAwareRetriever`` (Pinned Facts/Relevant Knowledge 주입), Stage
    15(Memory) 에 ``ConversationArchivingStrategy`` (STM 기록 + vault
    conversations/ rollup — 브라우저에 대화가 보이는 경로). provider 수명은
    호출자 소유 — turn teardown 에서 ``pipeline._memory_provider.close()``
    가 호출된다 (stream_turn/run_turn 의 finally).

    ``context_window_budget`` — 모델의 실제 컨텍스트 윈도우(토큰). 0 이면
    executor 기본값(200k)이 쓰이는데, 작은 윈도우 모델(vLLM 32k 등)에서는
    압축이 트리거되기 전에 provider 400 이 먼저 난다 — 호출자(agent_geny)가
    token_budget 헬퍼로 해석한 실측/카탈로그 값을 반드시 넘겨야 한다.
    Stage 2(80% proactive)·Stage 4 guard·Stage 16 토큰-비 루프 정지가 전부
    이 값을 기준으로 동작한다.

    ``enable_compaction`` — 노드의 "컨텍스트 자동 압축" 토글.
    True(기본): Stage 2 가 **항상** 등록되고(메모리 유무 무관 — 압축은 메모리
    기능이 아니다) LLMSummaryCompactor 로 80% 초과 시 같은 모델 요약-압축,
    Stage 4 에 TokenBudgetGuard 가 등록되어 예산 부족 시 compact→1회 재검사
    (auto-wire 는 Pipeline._init_state). False: 압축 전면 꺼짐 — Stage 2 는
    메모리 배선용으로만 등록되고(compaction_enabled=False → 프루닝·요약·
    guard 회복까지 전부 스킵, executor 3.3.0 계약), guard 미등록.
    """
    if registry is not None and registry.list_deferred():
        from xgen_agent_runtime.tools.built_in import ToolSearchTool

        if registry.get("ToolSearch") is None:
            registry.register(ToolSearchTool(), core=True)

    system = system_prompt or ""
    if output_schema:
        system += _schema_instruction(output_schema)

    model_opts: Dict[str, Any] = {
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": bool(stream),
        "max_iterations": int(max_iterations),
    }
    if context_window_budget and int(context_window_budget) > 0:
        # PipelineConfig 필드 → attach 시 state.context_window_budget 로 전파.
        model_opts["context_window_budget"] = int(context_window_budget)

    builder = (
        PipelineBuilder(name, api_key=api_key, model=model)
        # 빌더의 모델명 추론 아티팩트(gpt-*→openai, gemini-*→google)에는 api_key
        # 필수 검증이 딸려 있다. 이 파이프라인은 항상 attach_runtime(llm_client,
        # override_manifest=True) 로 명시 클라이언트를 붙이므로 빌더 기본
        # 스테이지는 스캐폴딩일 뿐 — vertex(SA/ADC, 키 없음)·codex(빌더 층 키
        # 없음)가 유령 검증에 죽지 않도록 기본 아티팩트로 고정한다.
        .with_artifact("s06_api", "default")
        .with_model(model, **model_opts)
        .with_system(prompt=system)
        .with_loop(max_turns=int(max_iterations))
    )
    if registry is not None and len(registry):
        builder.with_tools(registry=registry)

    # ── Stage 2(Context) — 항상 등록 ─────────────────────────────
    # 이전에는 memory_provider 가 있을 때만 등록해서, 기억을 끈 에이전트는
    # 이력이 아무리 커져도 **어떤 압축도 없이** provider 400 으로 죽었다.
    # 압축은 메모리 기능이 아니라 컨텍스트 위생이다 — 항상 등록한다.
    # LLMSummaryCompactor 는 state.model/llm_client 로 자가-배선되어 진짜
    # 요약을 만들고, 클라이언트가 없으면 정적 플레이스홀더로 강등된다.
    from xgen_agent_runtime.stages.s02_context.artifact.default.compactors import (
        LLMSummaryCompactor,
    )

    builder.with_context(
        compactor=LLMSummaryCompactor(),
        compaction_enabled=bool(enable_compaction),
        # 원샷 호스트 계약 (executor 3.3.1): 이 파이프라인은 턴마다 새로
        # 만들어져 "다음 턴" 이 없다 — 80~90% 구간의 백그라운드 요약 유예는
        # 결과가 항상 버려지고(낭비 LLM 콜) 태스크가 teardown 에 샌다.
        # False → 80% 트리거가 항상 동기로 압축한다.
        background_compaction=False,
    )

    if enable_compaction:
        # Stage 4 guard — 다음 요청(system+messages+tools 추정)이 응답 헤드룸을
        # 남기지 못하면 "compact" 신호 → GuardStage 가 Stage 2 의 압축기로
        # 이력을 줄이고 1회 재검사, 그래도 안 되면 명확한 메시지로 거절
        # (provider 400 보다 진단 가능). 헤드룸은 출력 max_tokens 예약분 —
        # 단, 비정상 설정(max_tokens ≥ 윈도우)에서 매 턴 거절이 되지 않도록
        # 윈도우의 절반을 상한으로 둔다.
        from xgen_agent_runtime.stages.s04_guard.artifact.default.guards import (
            TokenBudgetGuard,
        )

        headroom = max(4096, int(max_tokens) + 2048)
        if context_window_budget and int(context_window_budget) > 0:
            headroom = min(headroom, max(1024, int(context_window_budget) // 2))
        builder.with_guard(guards=[TokenBudgetGuard(min_remaining_tokens=headroom)])

    if memory_provider is not None:
        # ⚠ 스테이지 등록이 선행 조건 — attach_runtime 의 슬롯 배선은 스테이지가
        # 없으면 **무음 no-op** 이다 (pipeline._set_stage_slot_strategy).
        # ContextStage(2)는 위에서 항상 등록되므로 MemoryStage(18)만 추가한다.
        # 이 줄이 빠지면 STM 기록이 조용히 죽는다 (2026-07-13 프로드에서 실제 발생).
        builder.with_memory()
    pipeline = builder.build()

    if output_schema:
        # Swap the default parser for the schema-validating one (register_stage
        # replaces by order). Validation lands on ParsedResponse; the terminal
        # settle for the node's text output happens in settle_structured().
        from xgen_agent_runtime.stages.s09_parse import ParseStage
        from xgen_agent_runtime.stages.s09_parse.artifact.default.parsers import (
            StructuredOutputParser,
        )

        pipeline.register_stage(ParseStage(parser=StructuredOutputParser(schema=output_schema)))

    if llm_client is not None:
        client = llm_client
    elif credentials and any(v not in (None, "") for v in credentials.values()):
        client = build_client(provider, api_key, base_url, credentials=credentials)
    else:
        # 다중 필드 자격증명이 없으면 기존 3-인자 시그니처 그대로 호출 —
        # 테스트/외부 monkeypatch(lambda provider, api_key, base_url) 보존.
        client = build_client(provider, api_key, base_url)
    pipeline.attach_runtime(llm_client=client, override_manifest=True)

    if tool_context is not None:
        # Stage 10(Tool) 의 ToolContext — working_dir/allowed_paths/extras(ssh·docs)
        # 를 built-in 도구들에 전달한다 (executor 공식 주입점: attach_runtime).
        pipeline.attach_runtime(tool_context=tool_context)

    if memory_provider is not None:
        # executor 의 from_manifest memory attach 경로 미러 (pipeline.py L1394~):
        # runtime 객체가 슬롯 선언을 이긴다 — 여기서 직접 슬롯에 배선한다.
        #
        # 3-피스 배선 (하나라도 빠지면 기억이 "조용히" 죽는다):
        #  1) ContextStage.retriever  ← MemoryAwareRetriever
        #       → state.metadata["memory_pinned"/"memory_context"] 를 채움
        #  2) MemoryStage.strategy    ← ConversationArchivingStrategy
        #       → 턴 종료 시 STM(transcripts) 기록
        #  3) SystemStage.builder     ← ComposablePromptBuilder
        #       → 기본 StaticPromptBuilder 는 metadata 를 렌더하지 않으므로,
        #         base 프롬프트 + PinnedFactsBlock(# Pinned Facts) +
        #         RetrievedMemoryBlock(# Relevant Knowledge) 조합으로 교체.
        #         (Geny 의 MemoryContextBlock 경로와 동일 — 2.50 분리형 블록 사용)
        try:
            from xgen_agent_runtime.memory.retriever import MemoryAwareRetriever

            from xgen_agent_runtime.host.conversation_archive import ConversationArchivingStrategy
            from xgen_agent_runtime.stages.s03_system.artifact.default.builders import (
                ComposablePromptBuilder,
                CustomBlock,
                PinnedFactsBlock,
                RetrievedMemoryBlock,
            )

            pipeline._memory_provider = memory_provider
            pipeline.attach_runtime(
                memory_retriever=MemoryAwareRetriever(memory_provider),
                memory_strategy=ConversationArchivingStrategy(memory_provider),
                system_builder=ComposablePromptBuilder(
                    blocks=[
                        CustomBlock("base", system),
                        PinnedFactsBlock(),
                        RetrievedMemoryBlock(),
                    ]
                ),
            )
            # 턴-종료 증류(distillation) 스펙 — teardown 이 백그라운드로 발사.
            pipeline._memory_distill_spec = memory_distill_spec
            logger.info(
                "geny_bridge: memory wired (context retriever + memory strategy + prompt blocks)"
            )
        except Exception:  # noqa: BLE001 — 메모리는 실행을 깨지 않는다
            logger.exception("geny_bridge: memory attach failed (memoryless run)")
    return pipeline


# ── structured output settle ────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def settle_structured(text: str, schema: Dict[str, Any]) -> str:
    """Validate the final text against the output schema.

    Success → canonical compact JSON (what downstream nodes parse).
    Failure → the raw text unchanged, with a warning — a malformed answer
    must still reach the user (same graceful rule as agent_xgen's parser).
    """
    candidate = (text or "").strip()
    match = _FENCE_RE.search(candidate)
    if match:
        candidate = match.group(1).strip()
    if not candidate.startswith(("{", "[")):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end > start:
            candidate = candidate[start : end + 1]
    try:
        import jsonschema

        parsed = json.loads(candidate)
        jsonschema.validate(parsed, schema)
        return json.dumps(parsed, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "geny_bridge: structured output settle failed (%s) — returning raw text", exc
        )
        return text


# ── xgen agent_event builders ───────────────────────────────────────────────


def _indicator(tool_name: str) -> Optional[Dict[str, Any]]:
    try:
        from xgen_agent_runtime.host.tool_indicators import get_indicator

        return get_indicator(tool_name)
    except Exception:  # noqa: BLE001 - indicator metadata is optional UI sugar
        return None


def _tool_call_event(name: str, tool_input: Any) -> Dict[str, Any]:
    event: Dict[str, Any] = {
        "type": "tool_call",
        "tool_name": name,
        "tool_input": tool_input
        if isinstance(tool_input, str)
        else json.dumps(tool_input or {}, ensure_ascii=False, default=str),
        "timestamp": datetime.now().isoformat(),
    }
    indicator = _indicator(name)
    if indicator:
        event["indicator"] = indicator
    return event


def _display_result(text: str) -> str:
    """tool_result 표시용 축약 — 머리+꼬리를 남긴다.

    문서 도구는 결과 **끝**에 다운로드 마커를 붙인다(서버가 download_artifact 로 승격).
    단순 head 절단이면 4000자를 넘는 결과의 마커가 잘려 다운로드 버튼이 사라진다.
    """
    if len(text) <= _DISPLAY_RESULT_LIMIT:
        return text
    head = _DISPLAY_RESULT_LIMIT - _DISPLAY_TAIL_KEEP
    omitted = len(text) - head - _DISPLAY_TAIL_KEEP
    return f"{text[:head]}\n…[{omitted} chars truncated]…\n{text[-_DISPLAY_TAIL_KEEP:]}"


def _tool_end_event(
    name: str,
    result_text: str,
    *,
    is_error: bool = False,
    duration_ms: Optional[int] = None,
) -> Dict[str, Any]:
    if is_error:
        event: Dict[str, Any] = {
            "type": "tool_error",
            "tool_name": name,
            "error": result_text or "tool execution failed",
        }
    else:
        event = {
            "type": "tool_result",
            "tool_name": name,
            "result": _display_result(result_text),
            "result_length": len(result_text),
            "citations": None,
        }
    event["timestamp"] = datetime.now().isoformat()
    if duration_ms is not None:
        event["duration_ms"] = duration_ms
    indicator = _indicator(name)
    if indicator:
        event["indicator"] = indicator
    return event


def _stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(content)


# ── turn usage (토큰/비용 집계 → ``usage`` 청크) ─────────────────────────────


def turn_usage(pipeline: Pipeline, state: PipelineState) -> Optional[Dict[str, Any]]:
    """턴 1회의 토큰/비용 집계 — 호스트 간 공통 ``usage`` 페이로드.

    출처는 Stage 7(TokenStage) 이 API 호출마다 ``state.turn_token_usage`` 에
    쌓는 per-call ``TokenUsage`` (SDK 경로·CLI 경로 모두 Stage 6 이
    ``last_api_response`` 로 올려 Stage 7 이 추적한다; ``begin_turn`` 이
    턴 시작마다 비우므로 합계 = 이 턴의 총량). 비용은 provider 가 직접
    보고한 값(Claude Code result envelope 의 ``total_cost_usd`` →
    ``TokenUsage.cost_usd``)을 우선하고, 없으면 Stage 7 계산기의 per-turn
    누적(``state.total_cost_usd``, 0 이면 미상)을 쓴다.

    반환 shape (크로스-레포 계약 — 커넥터 TurnReport.usage / 서버 report-turn
    output_data.usage / trace.record_llm_usage 가 그대로 읽는다)::

        {"input_tokens": int, "output_tokens": int,
         "cache_read_tokens": int|None, "cache_creation_tokens": int|None,
         "total_cost_usd": float|None, "model": str|None, "provider": str|None}

    사용량이 전혀 기록되지 않은 턴(API 호출 0회 — 가드 거절·즉시 오류)은
    ``None`` — 호출자는 이때 usage 청크를 내지 않는다.
    """
    from xgen_agent_runtime.core.state import TokenUsage

    calls = list(getattr(state, "turn_token_usage", None) or [])
    if not calls:
        return None
    total = TokenUsage()
    for u in calls:
        if isinstance(u, TokenUsage):
            total += u
    cost: Optional[float] = total.cost_usd
    if cost is None:
        turn_cost = getattr(state, "total_cost_usd", 0.0) or 0.0
        cost = float(turn_cost) if turn_cost > 0 else None
    last = getattr(state, "last_api_response", None)
    model = str(getattr(last, "model", "") or "") or str(getattr(state, "model", "") or "")
    provider = ""
    try:
        resolver = getattr(pipeline, "_resolved_provider_name", None)
        if callable(resolver):
            provider = str(resolver(state) or "")
    except Exception:  # noqa: BLE001 — 진단용 라벨일 뿐
        provider = ""
    if not provider:
        provider = str(getattr(getattr(state, "llm_client", None), "provider", "") or "")
    return {
        "input_tokens": int(total.input_tokens),
        "output_tokens": int(total.output_tokens),
        "cache_read_tokens": int(total.cache_read_input_tokens),
        "cache_creation_tokens": int(total.cache_creation_input_tokens),
        "total_cost_usd": float(cost) if cost is not None else None,
        "model": model or None,
        "provider": provider or None,
    }


def _should_record_execution(host: Any, *, produced_output: bool, failed: bool) -> bool:
    """메모리 실행 기록 여부 — 출력 0 으로 실패한 턴은 host 정책에 따른다.

    ``host.record_failed_starts``(기본 True) 가 False 인 호스트(커넥터 로컬
    LocalHostServices)는 "시작도 못 한" 턴(텍스트 0 + 오류/취소)을 vault 에
    남기지 않는다 — 서버 폴백이 같은 턴을 다시 돌려 기록하므로 중복 실패
    기록이 쌓이던 경로. 성공 턴·출력이 있었던 실패 턴은 항상 기록.
    """
    if produced_output or not failed:
        return True
    return bool(getattr(host, "record_failed_starts", True)) if host is not None else True


# ── sync bridges (executor runs execute() in a worker thread) ───────────────


def _record_execution(
    pipeline: Pipeline,
    loop: asyncio.AbstractEventLoop,
    *,
    input_text: str,
    state: PipelineState,
    output_text: str,
    success: bool,
    duration_ms: int,
    error: str = "",
    cancelled: bool = False,
) -> None:
    """턴 실행 1회를 메모리에 기록 (Geny record_execution 미러) — teardown 직전.

    provider 가 살아 있는 마지막 지점(_close_memory_provider 직전)에서
    동기 호출, 상한 10s. 실패는 로그만 — 턴 결과를 절대 바꾸지 않는다.
    """
    provider = getattr(pipeline, "_memory_provider", None)
    if provider is None:
        return
    spec = getattr(pipeline, "_memory_distill_spec", None)
    try:
        from xgen_agent_runtime.host.execution_record import record_turn_execution

        loop.run_until_complete(
            asyncio.wait_for(
                record_turn_execution(
                    provider,
                    input_text=input_text,
                    output_text=output_text,
                    success=success,
                    duration_ms=duration_ms,
                    session_id=str(getattr(state, "session_id", "") or ""),
                    provider_name=str(getattr(spec, "provider", "") or "") if spec else "",
                    model=str(getattr(spec, "model", "") or "") if spec else "",
                    error=error,
                    cancelled=cancelled,
                ),
                timeout=10.0,
            )
        )
    except Exception:  # noqa: BLE001 — 기록은 best-effort
        logger.debug("geny_bridge: execution record failed (turn unaffected)", exc_info=True)


def _close_memory_provider(pipeline: Pipeline, loop: asyncio.AbstractEventLoop) -> None:
    """turn teardown 에서 내장 메모리 provider 를 닫는다.

    ``pipeline.aclose()`` 는 memory provider 를 닫지 않는다(수명은 호스트
    소유). 같은 스레드/루프 안에서 생성·사용·정리하는 계약을 지키기 위해
    반드시 turn 의 finally (loop.close() 직전)에서 호출한다.
    """
    provider = getattr(pipeline, "_memory_provider", None)
    if provider is None:
        return
    try:
        loop.run_until_complete(provider.close())
    except Exception:  # noqa: BLE001 - teardown must not mask the run
        logger.debug("geny_bridge: memory provider close failed", exc_info=True)
    # 턴-종료 증류 — Geny compact_now 케이던스의 XGEN 판. 백그라운드 데몬
    # 스레드가 자기 루프·자기 provider 로 facts/rollup 을 돌리므로 여기(턴
    # 스레드)는 즉시 반환한다. 실패/미설정은 조용히 스킵.
    spec = getattr(pipeline, "_memory_distill_spec", None)
    if spec is not None:
        try:
            from xgen_agent_runtime.host.distill import launch_distillation

            launch_distillation(spec)
        except Exception:  # noqa: BLE001
            logger.debug("geny_bridge: distillation launch failed", exc_info=True)


def stream_turn(
    pipeline: Pipeline,
    text: str,
    state: PipelineState,
    *,
    tool_events: bool = True,
    result_sink: Optional[Dict[str, str]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    output_schema: Optional[Dict[str, Any]] = None,
    on_close: Optional[Callable[[], None]] = None,
    host: Optional[Any] = None,
) -> Iterator[Union[str, Dict[str, Any]]]:
    """Drive ``run_stream`` from a sync generator.

    Yields assistant text chunks (str) and, when ``tool_events`` is on, xgen
    ``agent_event`` dicts for tool progress — both pipeline-dispatched tools
    (``tool.call_*``) and CLI-internal executions announced by subprocess
    backends (``api.cli_tool_call`` / ``api.tool_result`` with source="cli").
    A terminal engine error surfaces as a readable ``[ERROR]`` chunk. Closing
    the generator mid-stream (client cancel → agent_node_processor calls
    ``.close()``) tears the pipeline down via the ``finally`` block;
    ``cancel_check`` adds a cooperative stop between events. ``on_close``
    runs last in teardown (e.g. per-run CLI workspace cleanup).

    ``output_schema`` 는 스트리밍에서는 모델이 생성한 JSON 텍스트가 그대로
    흐른다 — 사후 검증·정규화는 non-stream(run_turn) 경로에서만 가능하다.

    **usage 청크** — 파이프라인이 끝난 뒤(성공·오류 무관, 취소 제외) 사용량이
    기록돼 있으면 ``{"type": "usage", "data": turn_usage(...)}`` 를 **정확히
    한 번**, 제너레이터 종료 직전에 yield 한다. 소비자(사이드카·서버
    agent_geny)가 토큰/비용을 집계하는 단일 출처.

    ``host`` — 선택. ``host.record_failed_starts`` (기본 True) 가 False 이면
    "출력 0 + 실패/취소" 턴의 메모리 실행 기록을 건너뛴다
    (:func:`_should_record_execution`).
    """
    loop = asyncio.new_event_loop()
    agen = pipeline.run_stream(text, state)
    streamed_text = False
    cli_tool_names: Dict[str, str] = {}  # tool_use_id → name (CLI 내부 실행 짝맞춤)
    turn_started = time.monotonic()
    out_parts: List[str] = []
    turn_error = ""
    turn_completed = False
    try:
        while True:
            if cancel_check is not None and cancel_check():
                logger.info("geny_bridge: cancellation requested — stopping stream")
                break
            try:
                event = loop.run_until_complete(agen.__anext__())
            except StopAsyncIteration:
                turn_completed = True
                break
            if event.type == "text.delta":
                chunk = event.data.get("text", "")
                if chunk:
                    streamed_text = True
                    out_parts.append(chunk)
                    yield chunk
            elif event.type == "pipeline.complete":
                # Degraded-streaming fallback: a client without true SSE
                # (BaseClient default) emits no text.delta at all — the final
                # text then arrives only on pipeline.complete. Never fires
                # when deltas flowed, so vendor streams are not double-sent.
                result = event.data.get("result", "")
                if not streamed_text and result:
                    out_parts.append(result)
                    yield result
            elif event.type == "pipeline.error":
                turn_error = str(event.data.get("error", "unknown error"))
                yield f"\n[ERROR] {turn_error}"
            elif tool_events and event.type == "tool.call_start":
                yield {
                    "type": "agent_event",
                    "data": _tool_call_event(event.data.get("name", ""), event.data.get("input")),
                }
            elif tool_events and event.type == "tool.call_complete":
                name = event.data.get("name", "")
                yield {
                    "type": "agent_event",
                    "data": _tool_end_event(
                        name,
                        (result_sink or {}).get(name, ""),
                        is_error=bool(event.data.get("is_error")),
                        duration_ms=event.data.get("duration_ms"),
                    ),
                }
            elif event.type == "canvas_command":
                # WorkflowSelf(self-evolution) 라이브 캔버스 편집 — 그래프 변경을
                # 프론트 캔버스에 즉시 반영시키는 사이드채널 이벤트 (tool_events 무관).
                yield {"type": "canvas_command", "data": event.data}
            elif tool_events and event.type == "api.cli_tool_call":
                # Claude Code CLI 가 내부에서 실행한 도구의 공지 — 파이프라인
                # Stage 10 을 거치지 않으므로 여기서 직접 UI 이벤트로 변환한다.
                name = event.data.get("name", "") or "cli_tool"
                tool_use_id = event.data.get("id") or ""
                if tool_use_id:
                    cli_tool_names[tool_use_id] = name
                yield {
                    "type": "agent_event",
                    "data": _tool_call_event(name, event.data.get("input")),
                }
            elif (
                tool_events
                and event.type == "api.tool_result"
                and event.data.get("source") == "cli"
            ):
                name = cli_tool_names.pop(event.data.get("tool_use_id", ""), "") or "cli_tool"
                yield {
                    "type": "agent_event",
                    "data": _tool_end_event(
                        name,
                        _stringify_content(event.data.get("content")),
                        is_error=bool(event.data.get("is_error")),
                    ),
                }
        if turn_completed:
            # 파이프라인 종료 후 정확히 1회 — 취소(break)로 나온 턴은 백그라운드
            # 태스크가 아직 돌고 있어 집계가 확정되지 않았으므로 내지 않는다.
            usage = turn_usage(pipeline, state)
            if usage is not None:
                yield {"type": "usage", "data": usage}
    finally:
        try:
            loop.run_until_complete(agen.aclose())
        except Exception:  # noqa: BLE001 - teardown must not mask the run
            pass
        try:
            loop.run_until_complete(pipeline.aclose())
        except Exception:  # noqa: BLE001
            pass
        turn_failed = bool(turn_error) or not turn_completed
        if _should_record_execution(host, produced_output=bool(out_parts), failed=turn_failed):
            _record_execution(
                pipeline,
                loop,
                input_text=text,
                state=state,
                output_text="".join(out_parts),
                success=turn_completed and not turn_error,
                duration_ms=int((time.monotonic() - turn_started) * 1000),
                error=turn_error or ("cancelled" if not turn_completed else ""),
                cancelled=not turn_completed and not turn_error,
            )
        else:
            logger.debug(
                "geny_bridge: execution record skipped — failed before any output "
                "(host.record_failed_starts=False)"
            )
        _close_memory_provider(pipeline, loop)
        loop.close()
        if on_close is not None:
            try:
                on_close()
            except Exception:  # noqa: BLE001
                pass


def run_turn(
    pipeline: Pipeline,
    text: str,
    state: PipelineState,
    *,
    output_schema: Optional[Dict[str, Any]] = None,
    host: Optional[Any] = None,
    usage_sink: Optional[Dict[str, Any]] = None,
) -> str:
    """Run one turn to completion and return the final text (non-streaming).

    반환값은 문자열이라 usage 를 실어 보낼 결과 객체가 없다 — 호출자가
    ``usage_sink`` (dict) 를 넘기면 실행 후 :func:`turn_usage` 페이로드
    (stream_turn 의 ``usage`` 청크 ``data`` 와 동일 shape)로 채워 준다.
    ``host`` 는 stream_turn 과 같은 record_failed_starts 게이트.
    """
    loop = asyncio.new_event_loop()
    turn_started = time.monotonic()
    turn_output = ""
    turn_success = False
    turn_error = ""
    produced_output = False
    try:
        result = loop.run_until_complete(pipeline.run(text, state))
        produced_output = bool(getattr(result, "text", "") or "")
        if usage_sink is not None:
            try:
                usage = turn_usage(pipeline, state)
                if usage is not None:
                    usage_sink.update(usage)
            except Exception:  # noqa: BLE001 — 집계는 턴 결과를 바꾸지 않는다
                logger.debug("geny_bridge: usage aggregation failed", exc_info=True)
        if not result.success:
            turn_error = str(result.error or "unknown error")
            return f"[ERROR] {result.error}"
        final = result.text or ""
        if output_schema:
            final = settle_structured(final, output_schema)
        turn_output = final
        turn_success = True
        return final
    except BaseException as exc:
        turn_error = f"{type(exc).__name__}: {exc}"[:300]
        raise
    finally:
        try:
            loop.run_until_complete(pipeline.aclose())
        except Exception:  # noqa: BLE001
            pass
        if _should_record_execution(
            host, produced_output=produced_output or bool(turn_output), failed=not turn_success
        ):
            _record_execution(
                pipeline,
                loop,
                input_text=text,
                state=state,
                output_text=turn_output,
                success=turn_success,
                duration_ms=int((time.monotonic() - turn_started) * 1000),
                error=turn_error,
            )
        else:
            logger.debug(
                "geny_bridge: execution record skipped — failed before any output "
                "(host.record_failed_starts=False)"
            )
        _close_memory_provider(pipeline, loop)
        loop.close()
