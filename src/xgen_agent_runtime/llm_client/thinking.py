"""생각(thinking·reasoning) 조절 — 모델마다 다른 방식을 하나의 값으로.

호출자는 **한 가지 값**만 고른다:

    None        모델 기본 — 아무것도 보내지 않는다
    "off"       생각하지 않는다
    "on"        생각한다(켜기/끄기만 되는 모델)
    "minimal" · "low" · "medium" · "high" · "xhigh" · "max"   생각의 강도

그 값을 **이 모델이 실제로 받는 요청**으로 바꾸는 일은 여기 한 곳이 한다. 모델마다 받는 것이 다르다:

  · Anthropic  Haiku 4.5·Sonnet 4.5 는 토큰 예산(``thinking.type=enabled`` + ``budget_tokens``)만,
               4.6 이후는 ``thinking.type=adaptive`` + ``output_config.effort``. 끄는 값도 다르다 —
               대부분 ``disabled``, Sonnet 5.5 는 ``between_tools``, Opus 5.5·Fable 은 끌 수 없다.
  · OpenAI     ``reasoning_effort`` — 끄기는 ``none``. gpt-4.1·4o 는 받지 않고, gpt-6-astra 는 끌 수 없다.
  · Gemini     2.5 는 ``thinking_budget``(0 이면 끔, 2.5 Pro 는 못 끔), 3 계열은 ``thinking_level``.
  · vLLM       서빙하는 모델의 채팅 템플릿이 정한다 — Qwen3 계열은 ``enable_thinking``, DeepSeek V3.x 는
               ``thinking``, gpt-oss 는 ``reasoning_effort``.
  · CLI        Claude Code 는 ``--effort``(끄기는 ``MAX_THINKING_TOKENS=0``), Codex 는 ``model_reasoning_effort``.

표의 출처는 두 가지다. ``verified=True`` 는 2026-10-01 dev 에 등록된 모델에 **실제로 보내 본 것**(받는 값·거절
문구·생각 토큰 수), 나머지는 vendor 공식 문서다. 표에 없는 모델은 조절할 수 없는 것으로 본다(``kind="none"``) —
모르는 모델에 아무 값이나 보내면 400 이거나(받지 않는 파라미터), 조용히 무시되거나(템플릿이 모르는 키) 둘 중
하나라 "조절했다" 고 말할 수 없다.

화면은 :meth:`ThinkingSpec.to_dict` 의 ``options`` 를 그대로 선택지로 쓴다. 고른 값이 그 모델에 없으면
:func:`normalize_thinking` 이 가장 가까운 값으로 옮긴다(모델을 바꿔도 선택이 깨지지 않게).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

#: 강도의 순서 — 가까운 값을 찾을 때 쓴다.
LEVEL_ORDER: Tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh", "max")
#: 받는 값 전부.
VALUES: Tuple[str, ...] = ("off", "on") + LEVEL_ORDER

_LV5 = ("low", "medium", "high", "xhigh", "max")
_LV4 = ("low", "medium", "high", "max")
_LV3 = ("low", "medium", "high")
_EFF = ("low", "medium", "high", "xhigh")


@dataclass(frozen=True)
class ThinkingSpec:
    """한 모델이 생각을 어떻게 조절받는가.

    ``kind``      ``"none"`` 조절할 수 없다 · ``"toggle"`` 켜기/끄기만 · ``"levels"`` 강도
    ``levels``    강도 모델이 받는 값(약한 것부터)
    ``can_disable`` 끌 수 있는가(강도 모델에서 False 면 늘 생각한다)
    ``default``   아무것도 보내지 않을 때 모델이 하는 것: ``"off"`` · ``"on"`` · 강도 · ``""``(서버마다 다르다)
    ``via``       보내는 방식(내부)
    ``verified``  dev 에서 실제로 보내 확인했는가
    """

    kind: str = "none"
    levels: Tuple[str, ...] = ()
    can_disable: bool = True
    default: str = ""
    via: str = ""
    off_type: str = "disabled"
    verified: bool = False
    #: Gemini 2.5 처럼 강도를 토큰 예산으로 보내는 모델의 표.
    budgets: Tuple[Tuple[str, int], ...] = field(default_factory=tuple)

    def options(self) -> Tuple[str, ...]:
        """화면의 선택지(모델 기본 제외) — 이 모델이 실제로 받는 값만."""
        if self.kind == "toggle":
            return ("off", "on") if self.can_disable else ()
        if self.kind == "levels":
            return (("off",) if self.can_disable else ()) + tuple(self.levels)
        return ()

    @property
    def controllable(self) -> bool:
        return bool(self.options())

    def budget(self, level: str) -> int:
        table = dict(self.budgets)
        if level in table:
            return table[level]
        return _ANTHROPIC_BUDGETS.get(level, 8192)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind if self.controllable else "none",
            "options": list(self.options()),
            "default": self.default,
            "can_disable": bool(self.can_disable) if self.kind != "none" else False,
            "verified": self.verified,
        }


NONE = ThinkingSpec()


def _levels(
    levels: Tuple[str, ...],
    *,
    can_disable: bool = True,
    default: str = "",
    via: str,
    verified: bool = False,
    off_type: str = "disabled",
    budgets: Tuple[Tuple[str, int], ...] = (),
) -> ThinkingSpec:
    return ThinkingSpec(
        kind="levels",
        levels=levels,
        can_disable=can_disable,
        default=default,
        via=via,
        off_type=off_type,
        verified=verified,
        budgets=budgets,
    )


def _toggle(*, default: str = "", via: str, verified: bool = False) -> ThinkingSpec:
    return ThinkingSpec(
        kind="toggle", can_disable=True, default=default, via=via, verified=verified
    )


# ── Anthropic (직접·Bedrock) ──────────────────────────────────────────
#
# 2026-10-01 dev 실측(같은 질문, 변형마다 한 번):
#   haiku-4-5·sonnet-4-5   enabled+budget ✓, adaptive·effort ✗("adaptive thinking is not supported")
#   sonnet-4-6·opus-4-6    adaptive+effort low|medium|high|max ✓, xhigh ✗, disabled ✓, 기본 생각 안 함
#   opus-4-7               adaptive+effort low..max(xhigh 포함) ✓, enabled ✗, 기본 생각 안 함
#   sonnet-5·opus-5        위와 같고 기본이 생각함(adaptive)
#   sonnet-5-5             disabled ✗ — "To turn thinking off on this model, send between_tools", between_tools ✓
#   opus-5-5               disabled·between_tools 모두 ✗ — 끌 수 없다
# effort "none" 은 모든 모델이 거절한다.

#: 토큰 예산으로 생각하는 모델의 강도별 예산(1024 이상, max_tokens 보다 작아야 한다 — 보낼 때 맞춘다).
_ANTHROPIC_BUDGETS: Dict[str, int] = {
    "minimal": 1024,
    "low": 2048,
    "medium": 8192,
    "high": 16384,
    "xhigh": 24576,
    "max": 32000,
}

_ANTHROPIC: Tuple[Tuple[str, ThinkingSpec], ...] = (
    (
        "claude-opus-5-5",
        _levels(_LV5, can_disable=False, default="medium", via="anthropic_adaptive", verified=True),
    ),
    (
        "claude-sonnet-5-5",
        _levels(
            _LV5, default="high", via="anthropic_adaptive", off_type="between_tools", verified=True
        ),
    ),
    (
        "claude-fable-5-1",
        _levels(_LV5, can_disable=False, default="high", via="anthropic_adaptive"),
    ),
    ("claude-fable-5", _levels(_LV5, can_disable=False, default="high", via="anthropic_adaptive")),
    ("claude-mythos", _levels(_LV5, can_disable=False, default="high", via="anthropic_adaptive")),
    ("claude-opus-5", _levels(_LV5, default="high", via="anthropic_adaptive", verified=True)),
    ("claude-sonnet-5", _levels(_LV5, default="high", via="anthropic_adaptive", verified=True)),
    ("claude-opus-4-8", _levels(_LV5, default="off", via="anthropic_adaptive")),
    ("claude-opus-4-7", _levels(_LV5, default="off", via="anthropic_adaptive", verified=True)),
    ("claude-opus-4-6", _levels(_LV4, default="off", via="anthropic_adaptive", verified=True)),
    ("claude-sonnet-4-6", _levels(_LV4, default="off", via="anthropic_adaptive", verified=True)),
    ("claude-opus-4-5", _levels(_LV4, default="off", via="anthropic_budget")),
    ("claude-sonnet-4-5", _levels(_LV4, default="off", via="anthropic_budget", verified=True)),
    ("claude-haiku-4-5", _levels(_LV4, default="off", via="anthropic_budget", verified=True)),
)

#: Claude Code CLI 의 별칭이 가리키는 모델(CLI 2.1.285 실측 — modelUsage 로 확인).
_CLAUDE_CODE_ALIASES = {
    "haiku": "claude-haiku-4-5",
    "sonnet": "claude-sonnet-5-5",
    "opus": "claude-opus-5-5",
}
#: ``MAX_THINKING_TOKENS=0`` 이 생각을 끄지 못하는 모델(Claude Code 문서).
_CLI_CANNOT_DISABLE = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-fable", "claude-mythos")


# ── OpenAI (Chat Completions ``reasoning_effort``) ───────────────────
#
# 2026-10-01 dev 실측: gpt-4.1·gpt-4o-mini 는 파라미터 자체를 거절("Unrecognized request argument"),
# gpt-5.4·5.6-*·6-luna·6-sol 은 none|low|medium|high|xhigh, gpt-6-astra 는 none 을 거절(끌 수 없다).
# minimal·max 는 이 모델들이 모두 거절했다.

_OPENAI: Tuple[Tuple[str, ThinkingSpec], ...] = (
    (
        "gpt-6-astra",
        _levels(_EFF, can_disable=False, default="medium", via="openai_effort", verified=True),
    ),
    ("gpt-6.1-sol", _levels(_EFF, can_disable=False, default="medium", via="openai_effort")),
    ("gpt-6-sol", _levels(_EFF, default="medium", via="openai_effort", verified=True)),
    ("gpt-6-luna", _levels(_EFF, default="medium", via="openai_effort", verified=True)),
    ("gpt-5.6", _levels(_EFF, default="medium", via="openai_effort", verified=True)),
    ("gpt-5.5-pro", NONE),
    ("gpt-5.5", _levels(_EFF, default="medium", via="openai_effort")),
    ("gpt-5.4-pro", NONE),
    ("gpt-5.4", _levels(_EFF, default="off", via="openai_effort", verified=True)),
    ("gpt-5.3-codex", _levels(_EFF, can_disable=False, via="openai_effort")),
    ("gpt-5.2-pro", NONE),
    ("gpt-5.2-codex", _levels(_EFF, can_disable=False, via="openai_effort")),
    ("gpt-5.2", _levels(_EFF, default="off", via="openai_effort")),
    ("gpt-5.1", _levels(_LV3, default="off", via="openai_effort")),
    ("gpt-5-pro", NONE),
    ("gpt-5-chat", NONE),
    (
        "gpt-5",
        _levels(("minimal",) + _LV3, can_disable=False, default="medium", via="openai_effort"),
    ),
    ("o4-mini", _levels(_LV3, can_disable=False, default="medium", via="openai_effort")),
    ("o3", _levels(_LV3, can_disable=False, default="medium", via="openai_effort")),
    ("o1", _levels(_LV3, can_disable=False, default="medium", via="openai_effort")),
)


# ── Gemini (Gemini API·Vertex) ────────────────────────────────────────

_G25_FLASH = (("low", 1024), ("medium", 8192), ("high", 24576))
_G25_PRO = (("low", 1024), ("medium", 8192), ("high", 32768))

_GEMINI: Tuple[Tuple[str, ThinkingSpec], ...] = (
    (
        "gemini-2.5-pro",
        _levels(_LV3, can_disable=False, default="on", via="gemini_budget", budgets=_G25_PRO),
    ),
    (
        "gemini-2.5-flash-lite",
        _levels(_LV3, default="off", via="gemini_budget", budgets=_G25_FLASH),
    ),
    ("gemini-2.5-flash", _levels(_LV3, default="on", via="gemini_budget", budgets=_G25_FLASH)),
    (
        "gemini-3-pro",
        _levels(("low", "high"), can_disable=False, default="high", via="gemini_level"),
    ),
    ("gemini-3.1-pro", _levels(_LV3, can_disable=False, default="high", via="gemini_level")),
    (
        "gemini-3-flash",
        _levels(("minimal",) + _LV3, can_disable=False, default="high", via="gemini_level"),
    ),
    (
        "gemini-3.1-flash-lite",
        _levels(("minimal",) + _LV3, can_disable=False, default="minimal", via="gemini_level"),
    ),
    (
        "gemini-3.5-flash-lite",
        _levels(("minimal",) + _LV3, can_disable=False, default="minimal", via="gemini_level"),
    ),
    (
        "gemini-3.5-flash",
        _levels(("minimal",) + _LV3, can_disable=False, default="medium", via="gemini_level"),
    ),
    (
        "gemini-3.6-flash",
        _levels(("minimal",) + _LV3, can_disable=False, default="medium", via="gemini_level"),
    ),
    ("gemini-3.7-flash", _levels(_LV3, can_disable=False, default="medium", via="gemini_level")),
    ("gemini-3.8-flash", _levels(_LV3, can_disable=False, default="medium", via="gemini_level")),
)


# ── vLLM (서빙 모델의 채팅 템플릿) ────────────────────────────────────
#
# 같은 vLLM 이라도 모델의 템플릿이 받는 변수가 다르다. 템플릿이 모르는 키는 **조용히 버려진다** — 그래서
# 모르는 모델은 조절할 수 없는 것으로 둔다. 2026-10-01 dev(qwen3.8-27b) 실측: enable_thinking true/false 가
# 생각을 켜고 끈다. reasoning_effort low|medium|xhigh 도 받지만(high 는 거절) 세 문제를 두 번씩 재도 강도와
# 생각 길이가 함께 움직이지 않았다 — 켜기/끄기로만 내놓는다.

_VLLM: Tuple[Tuple["re.Pattern[str]", ThinkingSpec], ...] = (
    (re.compile(r"qwen3\.8.*2\.4t"), NONE),
    (re.compile(r"qwen3\.8"), _toggle(via="ctk_enable_thinking", verified=True)),
    (re.compile(r"qwen3.*thinking-2507"), NONE),
    (re.compile(r"qwen3.*instruct-2507"), NONE),
    (re.compile(r"qwen3\.[5-7]"), _toggle(default="on", via="ctk_enable_thinking")),
    (re.compile(r"^qwen3(-|$)"), _toggle(default="on", via="ctk_enable_thinking")),
    (re.compile(r"^qwq"), NONE),
    (re.compile(r"deepseek-?v3[.-]?[12]"), _toggle(default="off", via="ctk_thinking")),
    (re.compile(r"deepseek-?r1"), NONE),
    (re.compile(r"gpt-oss"), _levels(_LV3, can_disable=False, default="medium", via="vllm_effort")),
    (
        re.compile(r"glm-?5[.-]?3"),
        _levels(("low", "high", "max"), can_disable=False, default="max", via="vllm_effort"),
    ),
    (
        re.compile(r"glm-?(4[.-]?[5-7]|5)(\b|[.-]|$)"),
        _toggle(default="on", via="ctk_enable_thinking"),
    ),
    (re.compile(r"kimi-?k2[.-]?thinking"), NONE),
    (re.compile(r"kimi-?k2[.-]?[56]"), _toggle(default="on", via="ctk_thinking")),
    (re.compile(r"gemma-?4"), _toggle(default="off", via="ctk_enable_thinking")),
)


# ── Codex CLI (``-c model_reasoning_effort``) ────────────────────────

_CODEX: Tuple[Tuple[str, ThinkingSpec], ...] = (
    ("gpt-5.3-codex", _levels(_EFF, can_disable=False, via="codex_effort")),
    ("gpt-5.2-codex", _levels(_EFF, can_disable=False, via="codex_effort")),
    ("gpt-5.2", _levels(_EFF, default="off", via="codex_effort")),
)


#: 런타임 클라이언트 이름 → 표를 고르는 provider 이름.
_PROVIDER_ALIASES = {
    "custom": "vllm",
    "openai_compatible": "vllm",
    "deepseek": "vllm",
    "azure_foundry": "azure",
    "gemini": "google",
}


def _match(table: Tuple[Tuple[str, ThinkingSpec], ...], model: str) -> Optional[ThinkingSpec]:
    """접두사 표에서 가장 긴 것 — ``gpt-5`` 가 ``gpt-5.4`` 를, ``claude-opus-5`` 가 ``claude-opus-5-5`` 를
    먹지 않게 경계(끝 또는 ``-``)까지 본다."""
    best: Optional[Tuple[int, ThinkingSpec]] = None
    for prefix, spec in table:
        if model == prefix or model.startswith(prefix + "-") or model.startswith(prefix + "@"):
            if best is None or len(prefix) > best[0]:
                best = (len(prefix), spec)
    return best[1] if best else None


def _anthropic_core(model: str) -> str:
    """Bedrock 표기(``us.anthropic.…-v1:0``)·짧은 별칭을 Anthropic 의 모델 이름으로."""
    from xgen_agent_runtime.llm_client.anthropic import _resolve_anthropic_model
    from xgen_agent_runtime.llm_client.bedrock import core_model_id

    return core_model_id(_resolve_anthropic_model(str(model or "").strip()))


def _basename(model: str) -> str:
    return str(model or "").strip().lower().rsplit("/", 1)[-1]


def thinking_spec(provider: str, model: str) -> ThinkingSpec:
    """이 provider 의 이 모델이 생각을 어떻게 조절받는가. 모르면 :data:`NONE`."""
    provider = _PROVIDER_ALIASES.get(
        str(provider or "").strip().lower(), str(provider or "").strip().lower()
    )
    name = str(model or "").strip()
    if not name:
        return NONE
    if provider in ("anthropic", "bedrock"):
        return _match(_ANTHROPIC, _anthropic_core(name)) or NONE
    if provider == "claude_code":
        core = _CLAUDE_CODE_ALIASES.get(name.lower()) or _anthropic_core(name)
        base = _match(_ANTHROPIC, core)
        if base is None:
            return NONE
        can_disable = base.can_disable and not any(core.startswith(p) for p in _CLI_CANNOT_DISABLE)
        # CLI 는 모델마다 맞는 방식으로 옮긴다 — --effort 다섯 단계를 모든 별칭이 받았다(실측).
        return _levels(
            _LV5,
            can_disable=can_disable,
            default=base.default,
            via="cli_effort",
            verified=name.lower() in _CLAUDE_CODE_ALIASES,
        )
    if provider in ("openai", "azure"):
        return _match(_OPENAI, name.lower()) or NONE
    if provider == "codex":
        found = _match(_CODEX, name.lower())
        if found is not None:
            return found
        base = _match(_OPENAI, name.lower())
        if base is None or base.kind == "none":
            return NONE
        return _levels(
            base.levels, can_disable=base.can_disable, default=base.default, via="codex_effort"
        )
    if provider in ("google", "vertex"):
        return _match(_GEMINI, _basename(name)) or NONE
    if provider == "vllm":
        served = _basename(name)
        for pattern, spec in _VLLM:
            if pattern.search(served):
                return spec
        return NONE
    return NONE


def normalize_thinking(spec: ThinkingSpec, value: Any) -> Optional[str]:
    """고른 값 → 이 모델이 받는 값. ``None`` 이면 아무것도 보내지 않는다(모델 기본).

    - 비었거나 ``auto``·``default`` → None
    - 조절할 수 없는 모델 → None
    - ``off`` 를 끌 수 없는 모델 → 가장 약한 강도(생각을 줄이는 쪽이 사용자의 뜻에 가깝다)
    - 켜기/끄기 모델에 강도 → ``on``
    - 강도 모델에 ``on`` → 그 모델의 기본 강도(없으면 ``medium``, 그것도 없으면 가운데)
    - 그 모델에 없는 강도 → 가장 가까운 강도(같은 거리면 약한 쪽)
    """
    text = str(value or "").strip().lower()
    if text == "none":  # OpenAI 의 끄기 이름
        text = "off"
    if text in ("", "auto", "default") or not spec.controllable:
        return None
    if text == "off":
        if spec.can_disable:
            return "off"
        return spec.levels[0] if spec.kind == "levels" and spec.levels else None
    if spec.kind == "toggle":
        return "on" if text == "on" or text in LEVEL_ORDER else None
    levels = spec.levels
    if text == "on":
        if spec.default in levels:
            return spec.default
        return "medium" if "medium" in levels else levels[len(levels) // 2]
    if text in levels:
        return text
    if text not in LEVEL_ORDER:
        return None
    want = LEVEL_ORDER.index(text)
    return min(levels, key=lambda lv: (abs(LEVEL_ORDER.index(lv) - want), LEVEL_ORDER.index(lv)))


# ── 보낼 모양 ─────────────────────────────────────────────────────────


#: adaptive 로 높은 강도를 고르면 생각이 max_tokens 안에서 자리를 먹는다 — 답이 남을 만큼 둔다.
_ADAPTIVE_MIN_MAX_TOKENS = {"high": 16000, "xhigh": 32000, "max": 64000}


def anthropic_request(spec: ThinkingSpec, level: str, max_tokens: int) -> Dict[str, Any]:
    """Anthropic Messages 의 ``thinking``·``output_config``·``max_tokens``. 생각을 켜면 표시는 요약으로."""
    if level == "off":
        return {"thinking": {"type": spec.off_type}, "max_tokens": max_tokens}
    if spec.via == "anthropic_budget":
        budget = max(1024, spec.budget(level))
        # budget_tokens < max_tokens — 답이 쓸 자리(원래 max_tokens)를 예산 위에 둔다.
        return {
            "thinking": {"type": "enabled", "budget_tokens": budget},
            "max_tokens": max(max_tokens, budget + max(1024, max_tokens)),
        }
    return {
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": level},
        "max_tokens": max(max_tokens, _ADAPTIVE_MIN_MAX_TOKENS.get(level, 0)),
    }


def openai_effort(level: str) -> str:
    """``reasoning_effort`` 값 — 끄기는 ``none``."""
    return "none" if level == "off" else level


def gemini_thinking_config(spec: ThinkingSpec, level: str) -> Dict[str, Any]:
    """google-genai ``thinking_config``."""
    if spec.via == "gemini_budget":
        return {"thinking_budget": 0 if level == "off" else spec.budget(level)}
    return {"thinking_level": level}


def vllm_request(spec: ThinkingSpec, level: str) -> Dict[str, Any]:
    """vLLM OpenAI 호환 서버로 보낼 것 — ``reasoning_effort`` 또는 ``extra_body.chat_template_kwargs``."""
    on = level != "off"
    if spec.via == "ctk_enable_thinking":
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": on}}}
    if spec.via == "ctk_thinking":
        return {"extra_body": {"chat_template_kwargs": {"thinking": on}}}
    if spec.via == "vllm_effort":
        return {"reasoning_effort": "none" if level == "off" else level}
    return {}


def claude_code_flags(spec: ThinkingSpec, level: str) -> Tuple[List[str], Dict[str, str]]:
    """Claude Code CLI 의 (argv 추가분, env 추가분)."""
    if level == "off":
        return [], {"MAX_THINKING_TOKENS": "0"}
    return ["--effort", level], {}


def codex_effort(level: str) -> str:
    return "none" if level == "off" else level


__all__ = [
    "LEVEL_ORDER",
    "NONE",
    "ThinkingSpec",
    "VALUES",
    "anthropic_request",
    "claude_code_flags",
    "codex_effort",
    "gemini_thinking_config",
    "normalize_thinking",
    "openai_effort",
    "thinking_spec",
    "vllm_request",
]
