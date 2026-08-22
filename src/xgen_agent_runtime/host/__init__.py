"""xgen_agent_runtime.host — host-agnostic agent-turn executor.

Extracted from ``agent_geny.AgentGenyNode.execute`` so the xgen-workflow server
(web) and the desktop connector (local Python sidecar) run the SAME turn logic
and can never diverge. The host supplies infrastructure through
:class:`HostServices`; the executor supplies the (identical) orchestration.

This layer lives INSIDE the runtime (not a separate package): the engine core
stays pure, and this submodule holds the turn orchestration + the ``HostServices``
protocol that both hosts implement. ``ServerHostServices`` lives in xgen-workflow;
:class:`~xgen_agent_runtime.host.local_host.LocalHostServices` (server-backed
state, local execution) ships here for the connector sidecar. Every module here
is import-clean of xgen-workflow — the product coupling is injected via the host.
"""

from __future__ import annotations

from xgen_agent_runtime.host.host import CliRuntime, CloudMount, HostServices

__all__ = ["HostServices", "CloudMount", "CliRuntime"]
__version__ = "0.1.0"
