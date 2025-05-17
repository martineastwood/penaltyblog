"""
Flow presets for StatsBomb event data.
These are convenience transforms you can use with Flow.pipe(...).
"""

from typing import Callable

from ..flow import Flow
from ..helpers import where_exists


def shots_only(flow: Flow) -> Flow:
    """
    Filter the event stream to only include shots.
    """
    return flow.filter(lambda r: r.get("type", {}).get("name") == "Shot")


def passes_only(flow: Flow) -> Flow:
    """
    Filter the event stream to only include passes.
    """
    return flow.filter(lambda r: r.get("type", {}).get("name") == "Pass")


def fouls_only(flow: Flow) -> Flow:
    """
    Filter the event stream to only include fouls.
    """
    return flow.filter(lambda r: r.get("type", {}).get("name") == "Foul Committed")


def cards_only(flow: Flow) -> Flow:
    """
    Filter the event stream to only include cards.
    """
    return flow.filter(where_exists("foul_committed.card"))


def goals(flow: Flow) -> Flow:
    """
    Filter the event stream to only include shots that resulted in goals.
    """
    return flow.filter(
        lambda r: r.get("type", {}).get("name") == "Shot"
        and r.get("shot", {}).get("outcome", {}).get("name") == "Goal"
    )


def assists(flow: Flow) -> Flow:
    """
    Filter to only passes that are marked as assists.
    """
    return flow.filter(
        lambda r: r.get("type", {}).get("name") == "Pass"
        and r.get("pass", {}).get("goal_assist") is True
    )


def xg_above(threshold: float) -> Callable[[Flow], Flow]:
    """
    Returns a pipeable function to filter shots above a certain xG threshold.
    """

    def _filter(flow: Flow) -> Flow:
        return flow.filter(
            lambda r: r.get("type", {}).get("name") == "Shot"
            and (r.get("shot", {}).get("statsbomb_xg") or 0.0) >= threshold
        )

    return _filter
