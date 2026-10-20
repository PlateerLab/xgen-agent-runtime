"""Real Redis atomic approval proof; opt in with an isolated test Redis URL."""
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from xgen_agent_runtime.permission.approval import ApprovalContext, ApprovalEnvelope, ApprovalError, RedisApprovalStore


@pytest.fixture
def store():
    url = os.getenv("P1_TEST_REDIS_URL")
    if not url:
        pytest.skip("isolated test Redis not configured")
    redis = pytest.importorskip("redis")
    client = redis.Redis.from_url(url, socket_timeout=2, socket_connect_timeout=2, decode_responses=True)
    client.ping()
    prefix = "p1-runtime-approval-test:" + uuid.uuid4().hex + ":"
    try:
        yield RedisApprovalStore(client, prefix=prefix)
    finally:
        keys = list(client.scan_iter(prefix + "*"))
        if keys:
            client.delete(*keys)
        client.close()


def test_durable_cross_instance_decision_is_bound_and_consumed_once(store):
    actor = ApprovalContext("synthetic-user", "workflow", "run", "session")
    envelope = ApprovalEnvelope.issue(actor, "record", {"value": "no-secret"})
    store.issue(envelope)
    peer = RedisApprovalStore(store.client, prefix=store.prefix)
    assert not peer.decide(envelope, replace(actor, run_id="wrong-run"), "approve")
    assert not peer.consume(envelope)
    assert peer.decide(envelope, actor, "approve")
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(lambda _: store.consume(envelope), range(8))) == 1
    assert not peer.consume(envelope)


def test_expiry_and_redis_fault_never_approve(store):
    actor = ApprovalContext("synthetic-user", "workflow", "run", "session")
    envelope = ApprovalEnvelope.issue(actor, "record", {}, ttl_s=0.05)
    store.issue(envelope)
    time.sleep(0.06)
    assert not store.decide(envelope, actor, "approve")

    class Unavailable:
        def eval(self, *args):
            raise RuntimeError("secret-store-canary")

    with pytest.raises(ApprovalError) as caught:
        RedisApprovalStore(Unavailable()).consume(envelope)
    assert "secret-store-canary" not in str(caught.value)
