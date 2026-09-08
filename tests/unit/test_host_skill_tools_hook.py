"""호스트가 소유한 스킬 도구를 얹는 일반 훅.

Jobs 처럼 서버가 소유하는 스킬이 늘 때마다 프로토콜을 넓히면, 두 레포의 배포
순서가 매번 계약이 된다. 훅은 하나로 두고 **계층 판정만 런타임이** 한다.
"""
from xgen_agent_runtime.host.host import HostServices


def test_기본_구현은_빈_목록이다():
    """런타임이 먼저 배포돼도 옛 호스트가 깨지지 않는다."""

    class _OldHost:
        pass

    assert HostServices.build_host_skill_tools(_OldHost(), workflow_id="wf") == []


def test_호스트가_얹으면_그대로_온다():
    class _NewHost:
        def build_host_skill_tools(self, **kwargs):
            return [("ArtifactGuide", kwargs.get("workflow_id"))]

    assert _NewHost().build_host_skill_tools(workflow_id="wf-1") == [("ArtifactGuide", "wf-1")]
