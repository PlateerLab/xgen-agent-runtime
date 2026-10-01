"""생각(thinking) 조절 — 모델마다 다른 방식을 하나의 값으로.

표의 ``verified`` 칸은 2026-10-01 dev 실측이다. 여기서 못 박는 것:
  · 모델마다 화면에 내놓는 선택지(받는 값만)
  · 고른 값이 그 모델이 받는 값으로 옮겨지는 규칙(가까운 값·끌 수 없는 모델)
  · 각 클라이언트가 실제로 보내는 모양(실측에서 받아들여진 모양 그대로)
"""

from __future__ import annotations

import pytest

from xgen_agent_runtime.core.config import ModelConfig
from xgen_agent_runtime.llm_client.anthropic import AnthropicClient
from xgen_agent_runtime.llm_client.openai import OpenAIClient
from xgen_agent_runtime.llm_client.profiles import get_profiled_client_class
from xgen_agent_runtime.llm_client.thinking import (
    NONE,
    normalize_thinking,
    thinking_spec,
)
from xgen_agent_runtime.llm_client.translators._cli import claude_code_argv
from xgen_agent_runtime.llm_client.translators._codex import codex_argv
from xgen_agent_runtime.llm_client.types import APIRequest


def _opts(provider: str, model: str):
    return thinking_spec(provider, model).to_dict()


# ── 표: 화면에 내놓는 선택지 ─────────────────────────────────────────


@pytest.mark.parametrize(
    "provider,model,kind,options",
    [
        # Anthropic — dev 실측
        ("anthropic", "claude-haiku-4-5-20251001", "levels", ["off", "low", "medium", "high", "max"]),
        ("anthropic", "claude-sonnet-4-5-20250929", "levels", ["off", "low", "medium", "high", "max"]),
        ("anthropic", "claude-sonnet-4-6", "levels", ["off", "low", "medium", "high", "max"]),
        ("anthropic", "claude-opus-4-7", "levels", ["off", "low", "medium", "high", "xhigh", "max"]),
        ("anthropic", "claude-sonnet-5", "levels", ["off", "low", "medium", "high", "xhigh", "max"]),
        ("anthropic", "claude-sonnet-5-5", "levels", ["off", "low", "medium", "high", "xhigh", "max"]),
        ("anthropic", "claude-opus-5-5", "levels", ["low", "medium", "high", "xhigh", "max"]),
        # Bedrock 표기는 Anthropic 모델로
        ("bedrock", "anthropic.claude-haiku-4-5-20251001-v1:0", "levels", ["off", "low", "medium", "high", "max"]),
        # OpenAI — dev 실측
        ("openai", "gpt-4.1", "none", []),
        ("openai", "gpt-4o-mini", "none", []),
        ("openai", "gpt-5.4", "levels", ["off", "low", "medium", "high", "xhigh"]),
        ("openai", "gpt-5.6-luna", "levels", ["off", "low", "medium", "high", "xhigh"]),
        ("openai", "gpt-6-sol", "levels", ["off", "low", "medium", "high", "xhigh"]),
        ("openai", "gpt-6-astra", "levels", ["low", "medium", "high", "xhigh"]),
        # Gemini — 문서
        ("vertex", "gemini-2.5-pro", "levels", ["low", "medium", "high"]),
        ("vertex", "gemini-2.5-flash", "levels", ["off", "low", "medium", "high"]),
        # vLLM — qwen3.8 은 실측(켜기/끄기)
        ("vllm", "qwen3.8-27b", "toggle", ["off", "on"]),
        ("vllm", "Qwen/Qwen3-4B", "toggle", ["off", "on"]),
        ("vllm", "Qwen/Qwen3-30B-A3B-Thinking-2507", "none", []),
        ("vllm", "openai/gpt-oss-120b", "levels", ["low", "medium", "high"]),
        ("vllm", "meta-llama/Llama-3.3-70B-Instruct", "none", []),
        # CLI
        ("claude_code", "haiku", "levels", ["off", "low", "medium", "high", "xhigh", "max"]),
        ("claude_code", "opus", "levels", ["low", "medium", "high", "xhigh", "max"]),
        ("codex", "gpt-5.2", "levels", ["off", "low", "medium", "high", "xhigh"]),
        ("codex", "gpt-5.3-codex", "levels", ["low", "medium", "high", "xhigh"]),
    ],
)
def test_options_per_model(provider, model, kind, options):
    spec = _opts(provider, model)
    assert spec["kind"] == kind
    assert spec["options"] == options


def test_prefix_does_not_swallow_a_newer_model():
    """``gpt-5`` 가 ``gpt-5.4`` 를, ``claude-opus-5`` 가 ``claude-opus-5-5`` 를 먹으면 안 된다."""
    assert thinking_spec("openai", "gpt-5.4").default == "off"
    assert thinking_spec("openai", "gpt-5-mini").levels[0] == "minimal"
    assert thinking_spec("anthropic", "claude-opus-5-5").can_disable is False
    assert thinking_spec("anthropic", "claude-opus-5").can_disable is True


def test_unknown_models_cannot_be_controlled():
    assert thinking_spec("anthropic", "claude-sonnet-9") is NONE
    assert thinking_spec("openai", "gpt-7-nova") is NONE
    assert thinking_spec("vllm", "") is NONE
    assert thinking_spec("mystery", "x") is NONE


# ── 값 옮기기 ────────────────────────────────────────────────────────


def test_normalize_rules():
    opus55 = thinking_spec("anthropic", "claude-opus-5-5")
    qwen = thinking_spec("vllm", "qwen3.8-27b")
    s46 = thinking_spec("anthropic", "claude-sonnet-4-6")
    assert normalize_thinking(opus55, "auto") is None
    assert normalize_thinking(opus55, "") is None
    assert normalize_thinking(opus55, "off") == "low", "끌 수 없으면 가장 약하게"
    assert normalize_thinking(opus55, "on") == "medium", "켜기는 그 모델의 기본 강도"
    assert normalize_thinking(qwen, "high") == "on", "켜기/끄기 모델에 강도는 켜기"
    assert normalize_thinking(qwen, "off") == "off"
    assert normalize_thinking(s46, "xhigh") in ("high", "max"), "없는 강도는 가장 가까운 것"
    assert normalize_thinking(s46, "xhigh") == "high", "같은 거리면 약한 쪽"
    assert normalize_thinking(s46, "none") == "off", "OpenAI 의 끄기 이름도 받는다"
    assert normalize_thinking(thinking_spec("openai", "gpt-4.1"), "high") is None
    assert normalize_thinking(s46, "nonsense") is None


# ── 보내는 모양 ──────────────────────────────────────────────────────


def _areq(model: str, level: str, max_tokens: int = 8192, **kw) -> APIRequest:
    return APIRequest(model=model, messages=[{"role": "user", "content": "hi"}], max_tokens=max_tokens,
                      temperature=0.0, thinking_level=level, **kw)


def test_anthropic_budget_model_uses_budget_tokens_and_keeps_answer_room():
    kwargs = AnthropicClient(api_key="k")._build_kwargs(_areq("claude-haiku-4-5-20251001", "high"))
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 16384}
    assert kwargs["max_tokens"] > 16384, "budget_tokens < max_tokens"
    assert "temperature" not in kwargs, "생각을 켜면 temperature 를 받지 않는다"
    assert "output_config" not in (kwargs.get("extra_body") or {})


def test_anthropic_adaptive_model_uses_effort():
    kwargs = AnthropicClient(api_key="k")._build_kwargs(_areq("claude-opus-4-7", "xhigh"))
    assert kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert kwargs["extra_body"]["output_config"] == {"effort": "xhigh"}
    assert kwargs["max_tokens"] >= 32000


def test_anthropic_off_values_differ_per_model():
    c = AnthropicClient(api_key="k")
    assert c._build_kwargs(_areq("claude-sonnet-5", "off"))["thinking"] == {"type": "disabled"}
    assert c._build_kwargs(_areq("claude-sonnet-5-5", "off"))["thinking"] == {"type": "between_tools"}
    off46 = c._build_kwargs(_areq("claude-sonnet-4-6", "off"))
    assert off46["thinking"] == {"type": "disabled"}
    assert off46["temperature"] == 0.0, "생각을 끄면 temperature 를 지킨다"


def test_base_request_normalizes_for_the_model():
    """ModelConfig 의 값이 그 모델이 받는 값으로 맞춰져 요청에 실린다 — 끌 수 없는 모델의 off 는 low."""
    c = AnthropicClient(api_key="k")
    req = c._build_request(
        model_config=ModelConfig(model="claude-opus-5-5", thinking_level="off", thinking_enabled=True),
        messages=[{"role": "user", "content": "hi"}], system="", tools=None, tool_choice=None, stream=False,
    )
    assert req.thinking_level == "low"
    assert req.thinking is None, "표준 값이 옛 필드를 대신한다"
    none_req = c._build_request(
        model_config=ModelConfig(model="claude-opus-5-5", thinking_level="auto"),
        messages=[{"role": "user", "content": "hi"}], system="", tools=None, tool_choice=None, stream=False,
    )
    assert none_req.thinking_level is None


def test_openai_reasoning_effort():
    c = OpenAIClient(api_key="k")
    on = c._build_kwargs(_areq("gpt-6-sol", "high"))
    assert on["reasoning_effort"] == "high"
    assert "temperature" not in on
    assert c._build_kwargs(_areq("gpt-5.4", "off"))["reasoning_effort"] == "none"


def test_vllm_chat_template_kwargs():
    custom = get_profiled_client_class("custom")(api_key="EMPTY", base_url="http://localhost:1/v1")
    on = custom._build_kwargs(_areq("qwen3.8-27b", "on"))
    assert on["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert "reasoning_effort" not in on
    off = custom._build_kwargs(_areq("qwen3.8-27b", "off"))
    assert off["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    oss = custom._build_kwargs(_areq("openai/gpt-oss-120b", "high"))
    assert oss["reasoning_effort"] == "high"


def test_gemini_thinking_config():
    from xgen_agent_runtime.llm_client.google import GoogleClient

    c = GoogleClient(api_key="k")
    assert c._build_kwargs(_areq("gemini-2.5-flash", "off"))["config"]["thinking_config"] == {"thinking_budget": 0}
    assert c._build_kwargs(_areq("gemini-2.5-pro", "high"))["config"]["thinking_config"] == {"thinking_budget": 32768}
    assert c._build_kwargs(_areq("gemini-3-pro-preview", "low"))["config"]["thinking_config"] == {"thinking_level": "low"}


def test_claude_code_effort_flag_and_off_env():
    on = claude_code_argv(_areq("haiku", "xhigh"))
    assert on[on.index("--effort") + 1] == "xhigh"
    assert "--effort" not in claude_code_argv(_areq("haiku", "off")), "끄기는 argv 가 아니라 env"

    from xgen_agent_runtime.llm_client.claude_code import ClaudeCodeCLIClient

    client = ClaudeCodeCLIClient(binary_path="/bin/true", workspace_dir="/tmp", api_key="k")
    runner = client._make_runner(request=_areq("haiku", "off"))
    assert runner.env_extras.get("MAX_THINKING_TOKENS") == "0"
    assert "MAX_THINKING_TOKENS" not in client._make_runner(request=_areq("haiku", "low")).env_extras


def test_codex_reasoning_effort_config():
    argv = codex_argv(_areq("gpt-5.2", "off"))
    assert "model_reasoning_effort=\"none\"" in argv or "model_reasoning_effort=none" in argv
    argv = codex_argv(_areq("gpt-5.3-codex", "high"))
    assert any(a.startswith("model_reasoning_effort=") and "high" in a for a in argv)


def test_thinking_level_flows_from_config_to_state_and_stage():
    from xgen_agent_runtime.core.config import PipelineConfig
    from xgen_agent_runtime.core.state import PipelineState
    from xgen_agent_runtime.host.turn_executor import _thinking_param

    cfg = PipelineConfig(model=ModelConfig(model="claude-sonnet-5", thinking_level="high"))
    state = PipelineState()
    cfg.apply_to_state(state)
    assert state.thinking_level == "high"
    assert ModelConfig.from_dict(cfg.model.to_dict()).thinking_level == "high"
    assert _thinking_param("auto") is None and _thinking_param("") is None and _thinking_param("High") == "high"
