"""
Flow: A JSON-native query engine for football data.

This module provides the `Flow` class — a powerful, composable way to work with
nested football data structures without flattening. Built for real-world use
cases like StatsBomb event files, it supports filtering, grouping, joining, and
streaming directly from JSON, APIs, or folders of files.

Part of the `penaltyblog` ecosystem.

Usage:
    from penaltyblog.matchflow import Flow

    shots = (
        Flow.statsbomb.events(match_id=12345)
        .filter(lambda r: r.get("type", {}).get("name") == "Shot")
        .select("player.name", "location", "shot.outcome.name")
    )
"""

from .flow import Flow
from .flowgroup import FlowGroup
from .helpers import (
    coalesce,
    combine_fields,
    get_field,
    get_index,
    where_equals,
    where_exists,
    where_in,
    where_not_none,
)
from .parallel import folder_flow
