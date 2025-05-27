import pytest

from penaltyblog.matchflow.flow import Flow


def test_group_summary():
    data = [
        {"team": "A", "minute": 1, "value": 1},
        {"team": "A", "minute": 2, "value": 3},
        {"team": "B", "minute": 1, "value": 2},
        {"team": "B", "minute": 2, "value": 4},
    ]

    flow = Flow.from_records(data)
    result = (
        flow.group_by("team")
        .summary(lambda rows: {"total_value": sum(r["value"] for r in rows)})
        .collect()
    )

    result = sorted(result, key=lambda x: x["team"])

    assert result == [
        {"team": "A", "total_value": 4},
        {"team": "B", "total_value": 6},
    ]


def test_group_cumulative():
    data = [
        {"team": "A", "minute": 1, "value": 1},
        {"team": "A", "minute": 2, "value": 3},
        {"team": "B", "minute": 1, "value": 2},
        {"team": "B", "minute": 2, "value": 4},
    ]

    flow = Flow.from_records(data)
    result = flow.group_by("team").cumulative("value").collect()

    # Sort by team then minute to make order deterministic
    result = sorted(result, key=lambda x: (x["team"], x["minute"]))

    assert result == [
        {"team": "A", "minute": 1, "value": 1, "cumulative_value": 1},
        {"team": "A", "minute": 2, "value": 3, "cumulative_value": 4},
        {"team": "B", "minute": 1, "value": 2, "cumulative_value": 2},
        {"team": "B", "minute": 2, "value": 4, "cumulative_value": 6},
    ]


def test_group_by_respects_optimize_flag():
    flow = (
        Flow.from_records([{"x": 1}, {"x": 2}], optimize=False)
        .select("x")
        .filter(lambda r: r["x"] > 0)
        .filter(lambda r: r["x"] > 0)
        .filter(lambda r: r["x"] > 0)
    )

    group = flow.group_by("x")
    plan = group._get_plan()

    # Ensure no fused steps appear
    assert all(
        step["op"] != "fused" for step in plan
    ), "Plan was optimized despite optimize=False"

    # Ensure FlowGroup inherited the optimize=False flag
    assert group.optimize is False
