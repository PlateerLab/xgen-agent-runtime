"""점프 호스트(bastion) 경로 — 해석·거부·실제 다이얼 순서.

이 파일이 지키는 것은 하나다: **선언한 경로대로, 그 순서로 연결한다.**
경로를 잘못 세우면 증상이 잔인하다 — 직접 다이얼이 우연히 성공해 엉뚱한 장비에
명령을 쏘거나, 순환 경로가 홉마다 타임아웃을 다 쓰고 나서야 실패한다. 그래서
해석 단계에서 미리 거절하는 규칙들을 값으로 못 박는다.
"""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from xgen_agent_runtime.tools._ssh import (
    MAX_JUMP_DEPTH,
    SSHConfigError,
    jump_names,
    resolve_chain,
)
from xgen_agent_runtime.tools.built_in._ssh_store import SSHServerStore


def _srv(name, **kw):
    base = {"name": name, "host": f"{name}.example", "user": "u", "password": "p"}
    base.update(kw)
    return base


# ── jump_names — 사람이 실제로 입력하는 형태를 견딘다 ──────────────

@pytest.mark.parametrize(
    "value,expected",
    [
        (None, []),
        ("", []),
        ([], []),
        ("bastion", ["bastion"]),
        ("a, b ,, c", ["a", "b", "c"]),          # 콤마 문자열 + 빈 칸
        (["a", "  ", "b"], ["a", "b"]),           # 빈 항목은 조회하지 않는다
    ],
)
def test_jump_names_shapes(value, expected):
    assert jump_names(_srv("t", jump=value)) == expected


def test_jump_names_rejects_nonsense():
    with pytest.raises(SSHConfigError):
        jump_names(_srv("t", jump={"host": "x"}))


# ── resolve_chain — 다이얼 순서 ────────────────────────────────────

def test_no_jump_resolves_to_itself():
    t = _srv("t")
    assert [h["name"] for h in resolve_chain(t, None)] == ["t"]


def test_chain_is_nearest_hop_first_then_target():
    servers = {s["name"]: s for s in [
        _srv("bastion"), _srv("inner"), _srv("db", jump=["bastion", "inner"]),
    ]}
    chain = resolve_chain(servers["db"], servers.get)
    assert [h["name"] for h in chain] == ["bastion", "inner", "db"]


def test_nested_jump_is_expanded_transitively():
    """bastion 자신이 또 경유가 필요하면 그것도 앞에 온다 — 한 단계만 보면 안 된다."""
    servers = {s["name"]: s for s in [
        _srv("edge"), _srv("bastion", jump=["edge"]), _srv("db", jump=["bastion"]),
    ]}
    assert [h["name"] for h in resolve_chain(servers["db"], servers.get)] == [
        "edge", "bastion", "db",
    ]


def test_missing_jump_host_is_refused_by_name():
    servers = {"db": _srv("db", jump=["ghost"])}
    with pytest.raises(SSHConfigError) as exc:
        resolve_chain(servers["db"], servers.get)
    assert "ghost" in str(exc.value)


def test_loop_is_refused_before_dialling():
    a = _srv("a", jump=["b"])
    b = _srv("b", jump=["a"])
    servers = {"a": a, "b": b}
    with pytest.raises(SSHConfigError) as exc:
        resolve_chain(a, servers.get)
    assert "loop" in str(exc.value)


def test_depth_is_capped():
    names = [f"h{i}" for i in range(MAX_JUMP_DEPTH + 3)]
    servers = {}
    for i, n in enumerate(names):
        servers[n] = _srv(n, jump=[names[i + 1]] if i + 1 < len(names) else [])
    with pytest.raises(SSHConfigError) as exc:
        resolve_chain(servers[names[0]], servers.get)
    assert "deeper than" in str(exc.value)


def test_jump_without_resolver_is_refused_not_dialled_directly():
    """resolver 가 없다고 조용히 직결하면, 평평한 망에서는 **성공**해 버린다."""
    with pytest.raises(SSHConfigError) as exc:
        resolve_chain(_srv("db", jump=["bastion"]), None)
    assert "resolver" in str(exc.value)


# ── 스토어가 곧 resolver ───────────────────────────────────────────

def test_store_resolve_is_a_usable_resolver():
    store = SSHServerStore([_srv("bastion"), _srv("db", jump=["bastion"])])
    chain = resolve_chain(store.resolve("db"), store.resolve)
    assert [h["name"] for h in chain] == ["bastion", "db"]


def test_list_public_exposes_via_and_never_secrets():
    store = SSHServerStore([
        _srv("bastion"), _srv("db", jump=["bastion"], password="secret-pass", private_key="KEY", passphrase="P"),
    ])
    pub = {s["name"]: s for s in store.list_public()}
    assert pub["db"]["via"] == ["bastion"]
    assert pub["bastion"]["via"] == []
    # 검사 대상은 **값**이다. 'auth': 'password' 는 인증 *종류* 라벨이라 남아야 한다
    # — 그게 모델이 sudo 가능 여부를 아는 근거다.
    blob = repr(pub)
    assert "secret-pass" not in blob
    assert "KEY" not in blob
    assert not any(key in row for row in pub.values()
                   for key in ("password", "private_key", "passphrase"))


# ── 실제 다이얼: tunnel 이 이전 홉으로 채워지는가 ──────────────────

class _FakeConn:
    def __init__(self, host):
        self.host = host
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        self.closed = True
        return False


@pytest.fixture
def fake_asyncssh(monkeypatch):
    calls = []
    mod = types.ModuleType("asyncssh")

    async def connect(host, **kw):
        calls.append((host, kw.get("tunnel")))
        return _FakeConn(host)

    mod.connect = connect
    mod.import_private_key = lambda *a, **k: object()
    monkeypatch.setitem(sys.modules, "asyncssh", mod)
    return calls


def test_dial_order_and_tunnel_wiring(fake_asyncssh):
    """각 홉은 **직전 홉의 연결을 tunnel 로** 물고 열려야 한다."""
    from xgen_agent_runtime.tools import _ssh

    store = SSHServerStore([
        _srv("bastion"), _srv("inner", jump=["bastion"]), _srv("db", jump=["inner"]),
    ])

    async def go():
        async with _ssh._open(
            store.resolve("db"), connect_timeout=5, resolver=store.resolve
        ) as conn:
            return conn

    conn = asyncio.run(go())
    hosts = [h for h, _ in fake_asyncssh]
    assert hosts == ["bastion.example", "inner.example", "db.example"]
    # 첫 홉은 터널 없음, 이후는 직전 연결을 물고 있다.
    assert fake_asyncssh[0][1] is None
    assert fake_asyncssh[1][1].host == "bastion.example"
    assert fake_asyncssh[2][1].host == "inner.example"
    assert conn.host == "db.example"


def test_all_hops_are_closed_on_exit(fake_asyncssh):
    """asyncssh 는 tunnel 로 넘긴 연결을 소유하지 않는다 — 우리가 닫아야 한다."""
    from xgen_agent_runtime.tools import _ssh

    store = SSHServerStore([_srv("bastion"), _srv("db", jump=["bastion"])])
    opened = []

    async def go():
        async with _ssh._open(
            store.resolve("db"), connect_timeout=5, resolver=store.resolve
        ) as conn:
            opened.append(conn)

    asyncio.run(go())
    # 열린 연결 전부(터널 포함)가 닫혔는지 — 터널은 fake 의 tunnel 인자로 추적.
    tunnels = [t for _, t in fake_asyncssh if t is not None]
    assert tunnels and all(t.closed for t in tunnels)
    assert all(c.closed for c in opened)


def test_failing_jump_host_names_itself(fake_asyncssh, monkeypatch):
    """중간 홉이 죽으면 '어느 홉'인지 말해야 한다 — 아니면 최종 호스트를 의심하게 된다."""
    from xgen_agent_runtime.tools import _ssh

    store = SSHServerStore([_srv("bastion"), _srv("db", jump=["bastion"])])
    mod = sys.modules["asyncssh"]

    async def connect(host, **kw):
        if host == "bastion.example":
            raise OSError("Connection refused")
        return _FakeConn(host)

    mod.connect = connect

    async def go():
        async with _ssh._open(
            store.resolve("db"), connect_timeout=5, resolver=store.resolve
        ):
            pass

    with pytest.raises(ConnectionError) as exc:
        asyncio.run(go())
    assert "bastion" in str(exc.value)


# ── 경유 전용 서버(listable=False) ─────────────────────────────────
#
# 사용자가 bastion 하나를 잠시 꺼도, 그 뒤의 목적지들은 계속 닿아야 한다.
# 그렇다고 꺼 둔 bastion 자체에 명령을 쏠 수 있으면 "끈" 것이 아니다.

def _hop_only(name, **kw):
    return _srv(name, listable=False, **kw)


def test_hop_only_server_is_hidden_from_the_agent():
    store = SSHServerStore([_hop_only("bastion"), _srv("db", jump=["bastion"])])
    assert [s["name"] for s in store.list_public()] == ["db"]
    assert store.names() == ["db"]


def test_hop_only_server_is_not_a_valid_target():
    store = SSHServerStore([_hop_only("bastion"), _srv("db", jump=["bastion"])])
    assert store.target("bastion") is None      # 명령을 쏠 수 없고
    assert store.resolve("bastion") is not None  # 경로로는 여전히 쓰인다


def test_route_through_a_disabled_bastion_still_resolves():
    store = SSHServerStore([_hop_only("bastion"), _srv("db", jump=["bastion"])])
    chain = resolve_chain(store.target("db"), store.resolve)
    assert [h["name"] for h in chain] == ["bastion", "db"]


# ── 세션 저장소: 호스트가 말할 때는 호스트가 진실 ──────────────────
#
# 개인 설정이 대화 중에 바뀔 수 있게 되면서 생긴 요구다. 턴마다 새로 주입되는
# 목록이 유일한 진실이어야 하고, 디스크에 남은 예전 사본이 그걸 이기면 안 된다 —
# 지운 서버가 계속 살아 있거나, 바꾼 비밀번호가 반영되지 않는다.

import json as _json
from pathlib import Path as _Path


class _Ctx:
    def __init__(self, storage, extras=None):
        self.storage_path = str(storage)
        self.extras = extras or {}


def _servers_file(tmp_path):
    return _Path(tmp_path) / "ssh" / "servers.json"


def test_injected_list_wins_over_a_stale_file(tmp_path):
    _servers_file(tmp_path).parent.mkdir(parents=True)
    _servers_file(tmp_path).write_text(_json.dumps([_srv("old", password="revoked")]))
    ctx = _Ctx(tmp_path, {"ssh": {"servers": [_srv("new")]}})
    store = SSHServerStore.from_context(ctx)
    assert store.names() == ["new"]


def test_an_empty_injection_means_zero_servers_not_fall_back_to_disk(tmp_path):
    """SSH 를 끄면 디스크의 예전 목록이 되살아나면 안 된다."""
    _servers_file(tmp_path).parent.mkdir(parents=True)
    _servers_file(tmp_path).write_text(_json.dumps([_srv("old")]))
    ctx = _Ctx(tmp_path, {"ssh": {"servers": []}})
    assert SSHServerStore.from_context(ctx).names() == []


def test_credentials_are_not_left_on_disk_by_default(tmp_path):
    ctx = _Ctx(tmp_path, {"ssh": {"servers": [_srv("web", password="s3cr3t")]}})
    SSHServerStore.from_context(ctx)
    assert not _servers_file(tmp_path).exists()


def test_a_file_left_by_an_older_version_is_cleaned_up(tmp_path):
    _servers_file(tmp_path).parent.mkdir(parents=True)
    _servers_file(tmp_path).write_text(_json.dumps([_srv("old", password="s3cr3t")]))
    ctx = _Ctx(tmp_path, {"ssh": {"servers": [_srv("web")]}})
    SSHServerStore.from_context(ctx)
    assert not _servers_file(tmp_path).exists()


def test_persistence_is_available_when_a_host_explicitly_asks(tmp_path):
    ctx = _Ctx(tmp_path, {"ssh": {"servers": [_srv("web")], "persist": True}})
    SSHServerStore.from_context(ctx)
    assert _servers_file(tmp_path).exists()


def test_without_any_host_injection_the_file_is_the_fallback(tmp_path):
    """standalone 실행 — 아무도 말해 주지 않을 때만 디스크를 읽는다."""
    _servers_file(tmp_path).parent.mkdir(parents=True)
    _servers_file(tmp_path).write_text(_json.dumps([_srv("web")]))
    assert SSHServerStore.from_context(_Ctx(tmp_path)).names() == ["web"]
