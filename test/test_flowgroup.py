from datetime import datetime, timedelta

import numpy as np

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


def test_row_based_rolling_sum():
    records = [
        {"team": "A", "value": 1},
        {"team": "A", "value": 2},
        {"team": "A", "value": 3},
        {"team": "A", "value": 4},
    ]

    result = (
        Flow.from_records(records)
        .group_by("team")
        .rolling_summary(window=2, aggregators={"sum_last_2": ("sum", "value")})
        .collect()
    )

    values = [r["sum_last_2"] for r in result]
    assert values == [1, 3, 5, 7]


def test_time_based_rolling_mean():
    base_time = datetime(2023, 1, 1, 0, 0, 0)
    records = [
        {"player": "X", "ts": base_time + timedelta(seconds=0), "score": 10},
        {"player": "X", "ts": base_time + timedelta(seconds=20), "score": 20},
        {"player": "X", "ts": base_time + timedelta(seconds=20), "score": 35},
        {"player": "X", "ts": base_time + timedelta(seconds=40), "score": 30},
        {"player": "X", "ts": base_time + timedelta(seconds=70), "score": 40},
        {"player": "X", "ts": base_time + timedelta(seconds=100), "score": 50},
    ]

    result = (
        Flow.from_records(records)
        .group_by("player")
        .sort_by("ts")
        .rolling_summary(
            window="30s",
            time_field="ts",
            aggregators={"mean_score_30s": ("mean", "score")},
        )
        .collect()
    )

    # Only rows within 30s from current should be counted
    expected = [
        np.float64(10.0),
        np.float64(15.0),
        np.float64(21.666666666666668),
        np.float64(28.333333333333332),
        np.float64(35.0),
        np.float64(45.0),
    ]
    assert [r["mean_score_30s"] for r in result] == expected


def test_min_periods_applied():
    records = [
        {"team": "B", "val": 10},
        {"team": "B", "val": 20},
        {"team": "B", "val": 30},
    ]

    result = (
        Flow.from_records(records)
        .group_by("team")
        .rolling_summary(
            window=2, min_periods=2, aggregators={"avg_last_2": ("mean", "val")}
        )
        .collect()
    )

    assert result[0]["avg_last_2"] is None  # not enough records
    assert result[1]["avg_last_2"] == 15
    assert result[2]["avg_last_2"] == 25


def test_multiple_groups():
    records = [
        {"team": "A", "val": 1},
        {"team": "B", "val": 10},
        {"team": "A", "val": 2},
        {"team": "B", "val": 20},
        {"team": "A", "val": 3},
        {"team": "B", "val": 30},
    ]

    result = (
        Flow.from_records(records)
        .group_by("team")
        .rolling_summary(window=2, aggregators={"sum_last_2": ("sum", "val")})
        .collect()
    )

    a = [r for r in result if r["team"] == "A"]
    b = [r for r in result if r["team"] == "B"]

    assert [r["sum_last_2"] for r in a] == [1, 3, 5]
    assert [r["sum_last_2"] for r in b] == [10, 30, 50]
