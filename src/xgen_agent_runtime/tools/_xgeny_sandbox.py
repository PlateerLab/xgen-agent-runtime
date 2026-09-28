"""옛 import 경로 — :mod:`xgen_agent_runtime.tools._geny_sandbox` 로 옮겼다(XGeny → Geny 개명, 4.65.0).

호스트(xgen-workflow)가 이 경로로 import 하던 기간의 코드가 새 런타임에서도 그대로 돌게 **같은 모듈
객체**를 돌려준다(별도 사본이 아니다 — 한쪽을 monkeypatch 하면 다른 쪽도 바뀐다). 새 코드는 쓰지 말 것.
"""

import sys

from xgen_agent_runtime.tools import _geny_sandbox as _module

if not hasattr(_module, "XgenySandbox"):
    _module.XgenySandbox = _module.GenySandbox

sys.modules[__name__] = _module
