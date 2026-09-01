"""세션 명령은 **선언된 환경**에서 돈다.

러너는 선언된 파이썬 환경(``PythonEnv``)의 ``bin`` 과 세션 HOME 의
``.local/bin`` 을 PATH 앞에 얹어 준다. 로그인 셸(`-l`)은 ``/etc/profile`` 을
읽고, Debian 계열(러너 이미지는 python:3.14-slim)의 그 파일은 PATH 를 통째로
덮어쓴다 — 선언된 환경이 조용히 사라지고 기본 인터프리터가 돈다.

그래서 프로드에서 에이전트는 ``PythonEnv`` 로 설치한 패키지를 다음 ``Bash``
에서 못 찾고, ``pip install`` 로 다시 깔아야 했다(후자는 기본 인터프리터가 읽는
곳에 앉는다). 눈으로 확인되지 않는 종류다 — 두 명령 다 rc=0 으로 끝난다.
"""
import asyncio
import inspect

from xgen_agent_runtime.tools._xgeny_sandbox import sb_run


class _Sandbox:
    workdir = "/w"

    def __init__(self):
        self.argv = None

    async def ensure(self):
        return None

    async def exec(self, argv, **kwargs):
        self.argv = list(argv)

        class _R:
            rc = 0
            stdout = b""
            stderr = b""

        return _R()


def test_셸_명령은_로그인_셸로_돌지_않는다():
    sb = _Sandbox()
    asyncio.run(sb_run(sb, "python3 x.py"))
    assert sb.argv[0] == "bash"
    assert "-lc" not in sb.argv and "-l" not in sb.argv, sb.argv
    assert sb.argv[1] == "-c"
    assert sb.argv[2] == "python3 x.py"


def test_이유가_코드에_적혀_있다():
    """다음 사람이 -l 을 되돌리지 않도록 — 되돌리면 조용히 깨진다."""
    src = inspect.getsource(sb_run)
    assert "/etc/profile" in src and "PATH" in src


# ── PythonEnv 는 "설치했다" 를 확인하고 답한다 ────────────────────────
#
# 이 도구의 유일한 실패 방식은 조용한 것이었다: "적용 완료" 라고 답해 놓고 다음
# 명령에서 ModuleNotFoundError 가 났다. 에이전트는 원인을 짚지 못해 재설치를
# 반복했다. 이제 환경의 인터프리터에게 직접 물어보고, 아니면 아니라고 말한다.

class _EnvSandbox:
    """ensure_env / exec / read_bytes / write_bytes 를 갖춘 세션 대역."""

    workdir = "/w"

    def __init__(self, *, installed=(), verify_rc=0, verify_out=None, explode=False):
        self.env_id = ""
        self.files = {}
        self._installed = set(installed)
        self._verify_rc = verify_rc
        self._verify_out = verify_out
        self._explode = explode
        self.verified_with = None

    async def ensure(self):
        return None

    async def read_bytes(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    async def write_bytes(self, path, data):
        self.files[path] = data
        return len(data)

    async def ensure_env(self, packages):
        return "sha256:abcdef123456", list(packages)

    async def exec(self, argv, **kwargs):
        self.verified_with = kwargs.get("env_id")
        if self._explode:
            raise RuntimeError("러너 불통")
        import ast as _ast
        import json as _json
        import re as _re

        missing = self._verify_out
        if missing is None:
            # 도구가 물어본 이름만 본다 — 실제 인터프리터가 하는 일과 같다.
            code = argv[-1]
            asked = _ast.literal_eval(_re.search(r"names = (\[.*?\])", code).group(1))
            missing = [n for n in asked if n not in self._installed]

        class _R:
            rc = self._verify_rc
            stdout = _json.dumps(missing).encode()
            stderr = b""

        return _R()


def _install(sb, pkgs=("httpx",)):
    from xgen_agent_runtime.host.forged_tools import _bind_tool_base
    from xgen_agent_runtime.host.python_env import PythonEnvTool

    tool = _bind_tool_base(PythonEnvTool(workflow_id="wf", workspace_dir="/w"))

    class _Ctx:
        sandbox = sb
        working_dir = "/w"

    return asyncio.run(tool.execute({"action": "install", "packages": list(pkgs)}, _Ctx()))


def test_적용됐으면_적용됐다고_답한다():
    sb = _EnvSandbox(installed=("httpx",))
    res = _install(sb, ("httpx",))
    assert not res.is_error, res.content
    assert sb.env_id == "sha256:abcdef123456"
    # 확인은 그 환경의 인터프리터에게 직접 묻는다 — 셸을 거치지 않는다.
    assert sb.verified_with == "sha256:abcdef123456"


def test_적용_안_됐으면_성공이라고_답하지_않는다():
    """이게 프로드에서 조용히 넘어가던 자리다."""
    sb = _EnvSandbox(installed=())
    res = _install(sb, ("httpx",))
    assert res.is_error
    assert "httpx" in res.content


def test_확인을_못했다고_실패로_단정하지_않는다():
    """모르는 것을 실패로 읽으면 멀쩡한 설치가 고장 난 것처럼 보인다."""
    for sb in (_EnvSandbox(explode=True), _EnvSandbox(verify_rc=1)):
        res = _install(sb, ("httpx",))
        assert not res.is_error, res.content
