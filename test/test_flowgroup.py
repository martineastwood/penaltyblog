from datetime import datetime, timedelta

import numpy as np

from penaltyblog.matchflow.flow import Flow
from penaltyblog.matchflow.group import FlowGroup


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


def test_time_bucket():
    base_time = datetime(2023, 1, 1, 0, 0, 0)
    records = [
        {"player": "X", "ts": base_time + timedelta(seconds=0), "score": 10},
        {"player": "X", "ts": base_time + timedelta(seconds=20), "score": 20},
        {"player": "X", "ts": base_time + timedelta(seconds=40), "score": 30},
        {"player": "X", "ts": base_time + timedelta(seconds=50), "score": 10},
        {"player": "X", "ts": base_time + timedelta(seconds=70), "score": 45},
        {"player": "X", "ts": base_time + timedelta(seconds=100), "score": 50},
    ]

    result = (
        Flow.from_records(records)
        .group_by("player")
        .time_bucket(
            freq="30s",
            aggregators={"sum_score": ("sum", "score")},
            time_field="ts",
            label="left",
        )
        .collect()
    )

    expected = [
        {"player": "X", "bucket": base_time + timedelta(seconds=0), "sum_score": 30},
        {"player": "X", "bucket": base_time + timedelta(seconds=30), "sum_score": 40},
        {"player": "X", "bucket": base_time + timedelta(seconds=60), "sum_score": 45},
        {"player": "X", "bucket": base_time + timedelta(seconds=90), "sum_score": 50},
    ]

    assert result == expected
