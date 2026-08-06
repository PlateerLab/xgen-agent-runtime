"""CompositeMemoryProvider — per-layer routing across providers.

Re-exports the public surface so callers can do::

    from xgen_agent_runtime.memory.composite import (
        CompositeMemoryProvider,
        LayerRouting,
    )
"""

from xgen_agent_runtime.memory.composite.provider import CompositeMemoryProvider
from xgen_agent_runtime.memory.composite.routing import LayerRouting

__all__ = ["CompositeMemoryProvider", "LayerRouting"]
