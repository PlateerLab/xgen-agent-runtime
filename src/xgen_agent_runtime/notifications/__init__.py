"""Notification endpoints registry — host-supplied webhook targets.

The registry is service-instantiated. ``PushNotificationTool`` reads
endpoints from ``ToolContext.extras["notification_endpoints"]``.
"""

from xgen_agent_runtime.notifications.registry import (
    NotificationEndpoint,
    NotificationEndpointRegistry,
)

__all__ = ["NotificationEndpoint", "NotificationEndpointRegistry"]
