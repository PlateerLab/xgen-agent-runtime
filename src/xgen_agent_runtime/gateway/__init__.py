"""Inbound chat gateway — receive messages from chat platforms, run an agent
turn, reply.

Built-in: the executor owns the gateway framework + platform adapters —
**Telegram** (HTTP long-poll), **Discord** (Gateway WebSocket), and **Slack**
(Socket Mode WebSocket), none needing a public endpoint. A host declares
platforms in config and supplies a handler (``message in → reply text out``);
it ships no transport code. Run :class:`GatewayRunner` from the app lifespan.

    from xgen_agent_runtime.gateway import build_gateway

    async def handler(msg):                 # msg: InboundMessage
        return await run_my_agent(msg.chat_id, msg.text)   # -> reply str

    runner = build_gateway(
        [{"platform": "telegram", "config": {"token": "123:abc"}}],
        handler,
    )
    await runner.start()
    ...
    await runner.shutdown()
"""

from xgen_agent_runtime.gateway.adapter import PlatformAdapter
from xgen_agent_runtime.gateway.discord import DiscordGatewayAdapter
from xgen_agent_runtime.gateway.factory import (
    BUILTIN_GATEWAY_PLATFORMS,
    build_gateway,
    build_platform_adapter,
)
from xgen_agent_runtime.gateway.runner import GatewayHandler, GatewayRunner
from xgen_agent_runtime.gateway.slack import SlackGatewayAdapter
from xgen_agent_runtime.gateway.telegram import TelegramGatewayAdapter
from xgen_agent_runtime.gateway.types import GatewayReply, InboundMessage

__all__ = [
    "InboundMessage",
    "GatewayReply",
    "PlatformAdapter",
    "GatewayRunner",
    "GatewayHandler",
    "TelegramGatewayAdapter",
    "DiscordGatewayAdapter",
    "SlackGatewayAdapter",
    "BUILTIN_GATEWAY_PLATFORMS",
    "build_gateway",
    "build_platform_adapter",
]
