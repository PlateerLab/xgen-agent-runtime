"""4.24.2 — tool indicator registry contract.

The A2UI Live Canvas was removed from XGEN (2026-09-14): no host registers the
``a2ui_*`` tools any more, so their indicator entries and the ``genui``
category are gone. Unknown tools must still fall back to the default indicator.
"""

from __future__ import annotations

from xgen_agent_runtime.host.tool_indicators import (
    DEFAULT_INDICATOR,
    TOOL_INDICATORS,
    get_indicator,
    is_hidden,
)

REMOVED = (
    "a2ui_list_components",
    "a2ui_inspect_component",
    "a2ui_validate_messages",
    "a2ui_finalize_surface",
)


def test_removed_a2ui_tools_are_not_registered():
    for name in REMOVED:
        assert name not in TOOL_INDICATORS
    assert all(ind.get("category") != "genui" for ind in TOOL_INDICATORS.values())


def test_removed_names_fall_back_to_the_default_indicator():
    for name in REMOVED:
        ind = get_indicator(name)
        assert ind["category"] == DEFAULT_INDICATOR["category"]
        assert ind["tool_name"] == name
        assert is_hidden(name) is False


def test_registered_entries_keep_their_metadata():
    ind = get_indicator("vectordb_retrieval_tool")
    assert ind["display_label"] == "지식 검색"
    assert ind["category"] == "retrieval"
    assert get_indicator("WorkflowSelf")["category"] == "self"
