"""보조 호출(대화 압축) 사용량이 turn_usage 의 auxiliary 로 나간다.

메인 루프 값은 그대로 두고(반복 한도·예산이 읽는 값), 호스트가 과금·한도에 더할 수 있게 따로 싣는다.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from xgen_agent_runtime.core.config import ModelConfig
from xgen_agent_runtime.core.state import PipelineState, TokenUsage
from xgen_agent_runtime.host.runner import turn_usage
from xgen_agent_runtime.llm_client import BaseClient, ClientCapabilities
from xgen_agent_runtime.llm_client.types import APIResponse, ContentBlock
from xgen_agent_runtime.stages.s02_context.artifact.default.compactors import (
    CHECKPOINT_MARKER,
    LLMSummaryCompactor,
)


class _Client(BaseClient):
    provider = "anthropic"
    capabilities = ClientCapabilities()

    async def _send(self, request, *, purpose=""):
        raise RuntimeError("unused")

    async def create_message(self, **kwargs):
        return APIResponse(
            content=[ContentBlock(type="text", text="RECAP")],
            stop_reason="end_turn",
            model=kwargs["model_config"].model,
            usage=TokenUsage(input_tokens=900, output_tokens=80, cache_read_input_tokens=50),
        )


class _Pipeline:
    def _resolved_provider_name(self, state):
        return "anthropic"


@pytest.mark.asyncio
async def test_compaction_usage_is_reported_as_auxiliary():
    state = PipelineState()
    for i in range(25):
        state.messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"})
    state.llm_client = _Client()
    comp = LLMSummaryCompactor(
        keep_recent=10,
        resolve_cfg=lambda s: ModelConfig(model="claude-x"),
        has_override=lambda: True,
        client_getter=lambda s: s.llm_client,
    )
    await comp.compact(state)
    state.turn_token_usage.append(TokenUsage(input_tokens=100, output_tokens=10))

    usage = turn_usage(_Pipeline(), state)
    assert (usage["input_tokens"], usage["output_tokens"], usage["calls"]) == (100, 10, 1), "메인 값은 그대로"
    aux = usage["auxiliary"]
    assert (aux["input_tokens"], aux["output_tokens"], aux["cache_read_tokens"], aux["calls"]) == (900, 80, 50, 1)
    assert aux["by_purpose"]["compact"]["input_tokens"] == 900

    state.begin_turn()
    assert state.aux_usage_entries() == [], "턴마다 비운다"
    assert turn_usage(_Pipeline(), state) is None


def test_aux_only_turn_still_reports():
    state = PipelineState()
    state.add_aux_usage("compact", TokenUsage(input_tokens=7, output_tokens=3))
    usage = turn_usage(_Pipeline(), state)
    assert usage["input_tokens"] == 0 and usage["auxiliary"]["input_tokens"] == 7


@pytest.mark.asyncio
async def test_background_compaction_usage_moves_to_the_real_state():
    from xgen_agent_runtime.stages.s02_context.artifact.default.stage import _CompactionShadow

    state = PipelineState()
    for i in range(25):
        state.messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"})
    state.llm_client = _Client()
    shadow = _CompactionShadow(state)
    comp = LLMSummaryCompactor(
        keep_recent=10,
        resolve_cfg=lambda s: ModelConfig(model="claude-x"),
        has_override=lambda: True,
        client_getter=lambda s: s.llm_client,
    )
    await comp.compact(shadow)
    head = shadow.messages[0]["content"]
    # 4.89.0: the summary sits inside the checkpoint message, after its intro.
    assert head.startswith(CHECKPOINT_MARKER) and "RECAP" in head, "기록이 압축을 깨지 않는다"
    assert [p for p, _ in shadow.aux_usage] == ["compact"]
