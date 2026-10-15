"""도구 호출의 능력 · 출처(capabilities · origin)와 그것을 싣는 tool.call_start / tool.call_complete."""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest
from langchain_core.tools import StructuredTool

from xgen_agent_runtime.host.device_tools import build_device_tool, capabilities_from_annotations
from xgen_agent_runtime.host.tool_exposure import TURN_ONE_DISCOVERY, TURN_ONE_TOOLS
from xgen_agent_runtime.host.tools import adapt_tools
from xgen_agent_runtime.memory.provider import MemoryHooks, NoteDraft, Scope, provider_hooks
from xgen_agent_runtime.memory.providers.file.provider import FileMemoryProvider
from xgen_agent_runtime.memory.retriever import MemoryAwareRetriever
from xgen_agent_runtime.stages.s10_tool.artifact.default.executors import PartitionExecutor, SequentialExecutor
from xgen_agent_runtime.stages.s10_tool.artifact.default.routers import RegistryRouter
from xgen_agent_runtime.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult, tool_origin
from xgen_agent_runtime.tools.built_in.bash_tool import BashTool, classify_shell_command
from xgen_agent_runtime.tools.registry import ToolRegistry


# ── Bash: 명령마다 (실패-닫힘) ──

READ_ONLY_COMMANDS = [
    "LANG=C ls",
    "LC_ALL=C sort f",
    "less f",
    "rg x",
    "man ls",
    "xxd f",
    "tree -L 2",
    "ls -la",
    "cat a.csv | wc -l",
    "grep -n foo *.py",
    "python -c 'print(1 > 0)'",
    "wc -l data.csv 2>/dev/null",
    "echo hi",
    "find . -name '*.md'",
    "awk '{print $1}' f.txt",
    "sed -n '1,10p' f",
    "git log --oneline -5",
    "git branch -a",
    "git config --get user.name",
    "docker ps --format '{{.Names}}'",
    "cd /tmp && ls",
    "head -3 f.csv >&2",
    "curl -s https://x/y",
    "wget -qO- https://x",
    "sort -u f",
    "uniq f",
    "date",
    "kubectl get pods",
    "pip list",
    "tar -tzf a.tar.gz",
    "env",
    "timeout 5 cat f",
    "Get-ChildItem -Recurse | Select-Object Name",
    "grep 'DELETE FROM' app.log",
    "echo 'rm -rf /'",
    "cat <<EOF\nhi\nEOF",
    "python3 - <<'EOF'\nimport pandas as pd\nprint(pd.read_csv('a.csv').shape)\nEOF",
]

NOT_READ_ONLY_COMMANDS = [
    "Get-ChildItem | ForEach-Object Delete",
    "FOO=1 grep x f",
    "LD_PRELOAD=/tmp/x.so ls",
    "LESSOPEN='|x %s' less f",
    "env LD_PRELOAD=x ls",
    "PAGER=x git log",
    "less +!x f",
    "tar -tf a.tar --checkpoint-action=exec=x",
    "tar -tvf a.tar -I x",
    "sort --compress-program=x f",
    "git grep -Ox y",
    "git diff --ext-diff",
    "man -P x ls",
    "rg --pre x y",
    "xxd a b",
    "tree -o out",
    "file -C -m m",
    "python gen.py > report.md",
    "cat a | tee out.md",
    "rm -rf build",
    "mkdir x",
    "sed -i 's/a/b/' f.txt",
    "sed 'w out' f",
    "pip install pandas",
    "git commit -am x",
    "python -c \"df.to_excel('o.xlsx')\"",
    "python summarize.py",
    "find . -name '*.py' | xargs sed -i 's/a/b/'",
    "perl -pi -e 's/a/b/' f",
    "node -e \"require('fs').writeFileSync('x','1')\"",
    "bash -c \"touch x\"",
    "env sh -c 'touch x'",
    "pwsh -Command \"Remove-Item x\"",
    "echo $(touch x)",
    "ls \"$(mkdir y)\"",
    "grep foo f 2> err.txt",
    "echo hi 1> out.txt",
    "echo x &> out.log",
    "ls >| out",
    "sort -o out in",
    "uniq in out",
    "iconv -f utf8 -t ascii -o out in",
    "yq -i '.a=1' f",
    "find . -fls out",
    "awk 'BEGIN{system(\"touch x\")}'",
    "git log --output=x",
    "git branch new",
    "git reflog expire --expire=now --all",
    "tar --delete -f a.tar m",
    "ip route del default",
    "mount -o remount,rw /",
    "date -s '2020-01-01'",
    "hostname x",
    "npm config set registry x",
    "go env -w X=1",
    "curl -X POST https://x",
    "curl --data a=1 https://x",
    "curl -sSLo file https://x",
    "wget https://x/file",
    "ssh host 'touch x'",
    "Invoke-RestMethod -Method Post https://x",
    "Get-ChildItem | ForEach-Object { Remove-Item $_ }",
    "sudo cat /etc/shadow",
    "{ reboot; }",
    "bash <<'EOF'\nrm -rf /\nEOF",
    "python3 - <<'EOF'\nimport os\nos.remove('x')\nEOF",
    "cat <<EOF > out.txt\nhi\nEOF",
    "ls 'unterminated",
]


@pytest.mark.parametrize("command", READ_ONLY_COMMANDS)
def test_bash_read_only_commands(command: str) -> None:
    assert classify_shell_command(command).read_only, command


@pytest.mark.parametrize("command", NOT_READ_ONLY_COMMANDS)
def test_bash_commands_that_write_or_run_code_are_not_read_only(command: str) -> None:
    assert not classify_shell_command(command).read_only, command


def test_bash_reports_only_read_only_and_egress() -> None:
    for command in READ_ONLY_COMMANDS + NOT_READ_ONLY_COMMANDS:
        caps = classify_shell_command(command)
        assert not caps.destructive and not caps.concurrency_safe and not caps.idempotent, command
    assert classify_shell_command("curl -s https://x/y").network_egress
    assert classify_shell_command("pip install x").network_egress
    assert classify_shell_command("git clone https://x").network_egress
    assert classify_shell_command("python -c 'import requests; requests.get(1)'").network_egress
    for command in ("kubectl get pods", "npm view x", "pip download x", "docker pull x"):
        assert classify_shell_command(command).network_egress, command
    assert not classify_shell_command("ls").network_egress
    assert not classify_shell_command("pip list").network_egress
    assert BashTool().capabilities({"command": "ls"}).read_only


def test_bash_classification_is_bounded() -> None:
    for big in ("open(" * 20000, "Remove-Item " * 20000, "x" * 1_000_000):
        t0 = time.monotonic()
        assert not classify_shell_command(big).read_only
        assert time.monotonic() - t0 < 1.0


# ── 기기 도구: MCP 주석 ──

def test_device_annotations_become_read_only_and_egress() -> None:
    assert capabilities_from_annotations(None) is None
    assert capabilities_from_annotations({}) is None
    ro = capabilities_from_annotations({"readOnlyHint": True, "openWorldHint": False})
    assert ro.read_only and not ro.network_egress
    assert not ro.concurrency_safe and not ro.destructive, "병렬 · 파괴는 바꾸지 않는다"
    assert not capabilities_from_annotations({"readOnlyHint": True}).network_egress, "MCP 어댑터와 같은 규칙"
    assert capabilities_from_annotations({"readOnlyHint": True, "idempotentHint": True}).idempotent
    rw = capabilities_from_annotations({"readOnlyHint": False, "destructiveHint": True, "openWorldHint": False})
    assert not rw.read_only and not rw.destructive

    async def _call(tool, args):
        return {"content": [{"type": "text", "text": "x"}]}

    read = build_device_tool(server="local", tool="ReadFile", description="d", input_schema={}, call=_call,
                             annotations={"readOnlyHint": True, "openWorldHint": False})
    shell = build_device_tool(server="local", tool="Shell", description="d", input_schema={}, call=_call)
    assert read.capabilities({}).read_only and not shell.capabilities({}).read_only
    assert tool_origin(read) == "device"


# ── 포트 도구: 노드가 적은 능력 ──

def test_port_tool_capabilities_from_metadata() -> None:
    get = StructuredTool.from_function(func=lambda q="": "ok", name="rates_api", description="GET rates")
    get.metadata = {"read_only": True, "network_egress": True}
    post = StructuredTool.from_function(func=lambda q="": "ok", name="order_api", description="POST order")
    post.metadata = {"read_only": False, "network_egress": True}
    plain = StructuredTool.from_function(func=lambda q="": "ok", name="plain", description="no metadata")
    other = StructuredTool.from_function(func=lambda q="": "ok", name="other", description="other metadata")
    other.metadata = {"opens_family": "web"}
    reg = adapt_tools([get, post, plain, other, {"name": "calc", "func": lambda **_: 1, "description": "c",
                                                 "capabilities": {"read_only": True}}])
    assert reg.get("rates_api").capabilities({}).read_only and reg.get("rates_api").capabilities({}).network_egress
    assert not reg.get("rates_api").capabilities({}).concurrency_safe, "병렬은 바꾸지 않는다"
    assert not reg.get("order_api").capabilities({}).read_only
    assert not reg.get("plain").capabilities({}).read_only, "모르면 쓰기 가능(실패-닫힘)"
    assert not reg.get("other").capabilities({}).read_only, "read_only 를 적지 않은 메타데이터는 능력이 아니다"
    assert reg.get("calc").capabilities({}).read_only
    assert tool_origin(reg.get("rates_api")) == "adapted"


# ── 실행기: 사건에 성격 ──

class _RoTool(Tool):
    input_schema = {"type": "object", "properties": {}}

    def __init__(self, name: str, read_only: bool) -> None:
        self._name, self._ro = name, read_only

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "t"

    def capabilities(self, input):
        return ToolCapabilities(concurrency_safe=self._ro, read_only=self._ro, idempotent=self._ro)

    async def execute(self, input, context):  # noqa: A002
        return ToolResult(content="ok")


@pytest.mark.parametrize("executor", ["partition", "sequential"])
def test_executor_events_carry_capabilities_and_origin(executor: str) -> None:
    reg = ToolRegistry()
    reg.register(_RoTool("look", True))
    reg.register(_RoTool("make", False))
    router = RegistryRouter(reg)
    events: List[tuple[str, Dict[str, Any]]] = []
    calls = [{"tool_use_id": "t1", "tool_name": "look", "tool_input": {}},
             {"tool_use_id": "t2", "tool_name": "make", "tool_input": {}}]
    ex = PartitionExecutor(registry=reg) if executor == "partition" else SequentialExecutor()
    asyncio.run(ex.execute_all(calls, router, ToolContext(), on_event=lambda e, p: events.append((e, p))))
    starts = {p["name"]: p for e, p in events if e == "tool.call_start"}
    ends = {p["name"]: p for e, p in events if e == "tool.call_complete"}
    assert starts["look"]["capabilities"]["read_only"] is True and starts["make"]["capabilities"]["read_only"] is False
    assert ends["look"]["capabilities"]["read_only"] is True
    assert "origin" in starts["look"]


# ── 턴 1 표: 도구 발견 ──

def test_turn_one_discovery_is_part_of_the_table() -> None:
    assert "ToolSearch" in TURN_ONE_DISCOVERY
    assert TURN_ONE_DISCOVERY <= TURN_ONE_TOOLS
    assert "ToolBatch" in TURN_ONE_TOOLS and "ToolBatch" not in TURN_ONE_DISCOVERY


# ── 검색기 hooks ──

def test_provider_hooks_reads_what_set_hooks_stored() -> None:
    class _P:
        def __init__(self):
            self._hooks = None

        def set_hooks(self, hooks):
            self._hooks = hooks

    p = _P()
    assert provider_hooks(p) is None
    p.set_hooks(MemoryHooks(search_exclude_categories=("memory-map",)))
    assert provider_hooks(p).search_exclude_categories == ("memory-map",)

    class _NotHooks:
        max_results = 3

    p._hooks = _NotHooks()
    assert provider_hooks(p) is None, "MemoryHooks 가 아니면 쓰지 않는다"


@pytest.mark.asyncio
async def test_excluded_categories_do_not_crowd_out_the_top_k() -> None:
    # 뺀 분류(대화 기록)가 저장소의 대부분이어도 나머지 노트가 결과에 남는다
    with tempfile.TemporaryDirectory() as td:
        p = FileMemoryProvider(root=Path(td), scope=Scope.SESSION)
        await p.initialize()
        for i in range(8):
            await p.notes().write(NoteDraft(title=f"대화 {i}", body="드보트 드보트 드보트 대화 기록", category="conversations",
                                            filename=f"c{i}.md"))
        await p.notes().write(NoteDraft(title="규정", body="드보트 규정", category="topics", filename="t.md"))
        hooks = MemoryHooks(max_results=2, search_exclude_categories=("conversations",))
        pf = await MemoryAwareRetriever(p, hooks=hooks)._prefetch_layers("드보트", hooks)
        kw = pf.get("kw_notes") or []
        cats = [(c.metadata or {}).get("category") for c in kw]
        assert "conversations" not in cats
        assert "topics" in cats
        assert len(kw) <= 2
