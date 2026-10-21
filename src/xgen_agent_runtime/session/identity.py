"""Server-issued identity carried through a single runtime execution.

The runtime does not authenticate this identity. A host must construct it only
from a verified Gateway principal and an authorized Agent Session lookup.
"""

from dataclasses import dataclass
from typing import Literal
from uuid import UUID


PlatformType = Literal["web", "desktop", "mobile", "cli", "vscode"]
_PLATFORMS = frozenset(("web", "desktop", "mobile", "cli", "vscode"))


@dataclass(frozen=True, slots=True)
class CanonicalSessionIdentity:
    """Account, platform session, device and Agent Session for tool routing.

    ``platform_session_id`` is the Gateway ``sid``. ``agent_session_id`` is
    the canonical server-issued conversation ID, distinct from the runtime's
    local ``PipelineState.session_id`` execution identifier.
    """

    tenant_id: str
    user_id: int
    platform_session_id: UUID
    device_id: UUID
    agent_session_id: UUID
    platform_type: PlatformType

    def __post_init__(self) -> None:
        if not isinstance(self.tenant_id, str) or not self.tenant_id.strip():
            raise ValueError("tenant_id must be a non-empty string")
        if type(self.user_id) is not int or self.user_id <= 0:
            raise ValueError("user_id must be a positive integer")
        for name in ("platform_session_id", "device_id", "agent_session_id"):
            if not isinstance(getattr(self, name), UUID):
                raise ValueError(f"{name} must be a UUID")
        if not isinstance(self.platform_type, str) or self.platform_type not in _PLATFORMS:
            raise ValueError("platform_type is unsupported")
