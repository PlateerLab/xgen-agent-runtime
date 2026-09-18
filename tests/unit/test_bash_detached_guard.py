"""Bash in the sandbox refuses to detach a process — it would die with the command."""

from xgen_agent_runtime.tools.built_in.bash_tool import _detached_process_reason


def test_detach_verbs_and_trailing_ampersand_are_caught():
    for cmd in (
        "nohup npx serve -s build -l 3000 &",
        "cd app && setsid python server.py > log 2>&1 &",
        "python -m http.server 8080 &",
        "(sleep 30; echo hi) &",
        "disown %1",
    ):
        reason = _detached_process_reason(cmd)
        assert reason and "ArtifactCreate" in reason, cmd


def test_ordinary_commands_pass():
    for cmd in (
        "npm run build && npm test",
        "cat a.txt | grep x | wc -l",
        "ls -la; echo done",
        "python train.py --epochs 3",
        "grep -r 'foo && bar' src",
        "docker logs x 2>&1 | tail",
    ):
        assert _detached_process_reason(cmd) is None, cmd
