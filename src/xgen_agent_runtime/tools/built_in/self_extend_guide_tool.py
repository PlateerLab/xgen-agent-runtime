"""SelfExtendGuide — 자기확장 스킬의 문. 부르면 그 방(제작·환경·자기진화 도구)이 열린다.

왜 문을 세우나 (2026-09-20 실측, dev 28일 · Qwen 토크나이저)
------------------------------------------------------------
고정 프리픽스(시스템 프롬프트 + 턴1 도구 스키마)는 호출당 약 8,600토큰이고 그중
**도구 스키마가 6,200토큰(72%)** 이었다. 자기확장 도구 여섯(WorkflowSelf 1,552 ·
ForgeTool 769 · PythonEnv 243 · SystemPackages 150 · DeleteForgedTool 83 ·
ListForgedTools 57 = 2,854토큰, 프리픽스의 33%)이 **모든 모델 호출**에 실렸는데,
실제로 쓰인 턴은 각각 0.7% · 1.4% · 2.4% · 0% · 0% · 0.5% 였다.

예전에 이 도구들을 ToolSearch 뒤에 숨겼더니 에이전트가 자기 환경에 패키지를 깔 수
있다는 걸 모른 채 후퇴했다(2026-08-18) — 문제는 **숨긴 것**이 아니라 **문이 없던
것**이다. 위임(DelegationGuide)·브라우저(BrowserGuide)·작업(JobGuide)과 같은
규약으로 문 하나를 세운다: 문의 설명이 "무엇을 할 수 있는가" 를 말하고(인식은
유지), 부르면 그 도구들이 이 턴에 열린다(왕복 1회, 쓰는 턴의 ≤3%). 스키마 여섯을
매 호출에 싣는 대신 문 하나(~110토큰)만 싣는다.
"""

from __future__ import annotations

from typing import Any, Dict

from xgen_agent_runtime.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult
from xgen_agent_runtime.tools.built_in._skill_gateway import open_family, with_opened

#: 이 문이 여는 방 — 런타임 제작 도구(ForgeTool·ListForgedTools·DeleteForgedTool·
#: PythonEnv)와 호스트가 등록하는 둘(SystemPackages·WorkflowSelf). 등록돼 있지
#: 않은 이름은 조용히 건너뛴다(open_family 규약).
SELF_EXTEND_FAMILY = (
    "ForgeTool",
    "ListForgedTools",
    "DeleteForgedTool",
    "PythonEnv",
    "SystemPackages",
    "WorkflowSelf",
)

_MAP = (
    "Self-extension tools (now callable):\n"
    "- PythonEnv — install/declare Python packages for this sandbox so they persist "
    "(plain `pip install` in Bash also works for the current session).\n"
    "- SystemPackages — apt packages that must survive pod recycling (declare once, "
    "re-applied every session).\n"
    "- ForgeTool / ListForgedTools / DeleteForgedTool — make a TOOL, which is what the user "
    "means by making or adding a tool: turn a script in your workspace into a reusable tool "
    "(stdin JSON → stdout JSON; verified once with test_input). A tool is not an app; if the "
    "user also wants a screen for it, an artifact page can call it (ArtifactGuide).\n"
    "- WorkflowSelf — edit your own workflow graph to permanently gain capabilities "
    "(attach document search / tool nodes, set params, test_run, apply). Call "
    "WorkflowSelf(action='guidance') first.\n"
    "Pick the smallest step that solves the problem: a one-off `pip install` in Bash "
    "needs none of these."
)


class SelfExtendGuideTool(Tool):
    """자기확장 스킬 게이트웨이 — 문을 열면 방이 열린다(_skill_gateway 참조)."""

    @property
    def name(self) -> str:
        return "SelfExtendGuide"

    @property
    def description(self) -> str:
        # 이 설명이 곧 "인식" 이다 — 여기서 능력을 말하지 않으면 모델은 그런 길이
        # 있다는 것 자체를 모른다(2026-08-18 회귀). 짧게, 그러나 전부.
        #
        # 첫 문장이 **"도구를 만들어 달라" 는 요청의 주인**을 밝힌다. 예전 첫 문장은
        # "START HERE to extend yourself" 였고, 옆의 ArtifactGuide 는 "사람이 열어서 쓰는
        # 것" 이었다. 사람이 쓸 도구를 시키면 모델은 "나를 넓히는 일" 보다 "사람이 쓰는 것"
        # 에 끌려 앱을 만들었다 — 실측(2026-09-24, 턴-1 표면 재현): "요약 도구", "환율 계산
        # 도구", "주식 시세 툴" 등 도구 요청 10개 중 5개가 ArtifactCreate 로 갔다.
        # XGEN 화면이 부르는 "도구" 는 ForgeTool 이 만든 것이다([Agent 생성 도구]·[도구] 탭).
        return (
            "START HERE when the user asks you to make, build or add a TOOL — a tool here is "
            "a script you register with ForgeTool: it joins your tool list, persists across "
            "sessions, and the user uses it by asking you. Also extends your environment: "
            "packages that persist (PythonEnv / SystemPackages) and edits to your own workflow "
            "graph, e.g. document search (WorkflowSelf). Calling this OPENS those tools and "
            "returns a short map. Free, instant."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(concurrency_safe=True, read_only=True, idempotent=True)

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        opened = open_family(context, SELF_EXTEND_FAMILY)
        return ToolResult(content=with_opened(_MAP, opened), metadata={"opened": opened})
