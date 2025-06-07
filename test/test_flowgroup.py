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


def test_time_bucket_with_missing_time_field():
    base_time = datetime(2023, 1, 1, 0, 0, 0)
    records = [
        {"player": "X", "ts": base_time + timedelta(seconds=0), "score": 10},
        {"player": "X", "score": 20},  # Missing time field
        {"player": "X", "ts": base_time + timedelta(seconds=40), "score": 30},
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
        {"player": "X", "bucket": base_time + timedelta(seconds=0), "sum_score": 10},
        {"player": "X", "bucket": base_time + timedelta(seconds=30), "sum_score": 30},
    ]

    assert result == expected


def test_time_bucket_with_non_datetime_field():
    records = [
        {"player": "X", "ts": 0, "score": 10},
        {"player": "X", "ts": 20, "score": 20},
        {"player": "X", "ts": 40, "score": 30},
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
        {"player": "X", "bucket": timedelta(seconds=0), "sum_score": 30},
        {"player": "X", "bucket": timedelta(seconds=30), "sum_score": 30},
    ]

    assert result == expected


def test_time_bucket_with_right_label():
    base_time = datetime(2023, 1, 1, 0, 0, 0)
    records = [
        {"player": "X", "ts": base_time + timedelta(seconds=0), "score": 10},
        {"player": "X", "ts": base_time + timedelta(seconds=20), "score": 20},
        {"player": "X", "ts": base_time + timedelta(seconds=40), "score": 30},
    ]

    result = (
        Flow.from_records(records)
        .group_by("player")
        .time_bucket(
            freq="30s",
            aggregators={"sum_score": ("sum", "score")},
            time_field="ts",
            label="right",
        )
        .collect()
    )

    expected = [
        {"player": "X", "bucket": base_time + timedelta(seconds=30), "sum_score": 30},
        {"player": "X", "bucket": base_time + timedelta(seconds=60), "sum_score": 30},
    ]

    assert result == expected

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


def test_time_bucket_multiple_groups():
    base_time = datetime(2023, 1, 1, 0, 0, 0)
    records = [
        {"team": "A", "ts": base_time + timedelta(seconds=0), "score": 10},
        {"team": "A", "ts": base_time + timedelta(seconds=20), "score": 20},
        {"team": "B", "ts": base_time + timedelta(seconds=10), "score": 5},
        {"team": "B", "ts": base_time + timedelta(seconds=35), "score": 15},
    ]
    result = (
        Flow.from_records(records)
        .group_by("team")
        .time_bucket(
            freq="30s",
            aggregators={"sum_score": ("sum", "score")},
            time_field="ts",
            label="left",
        )
        .collect()
    )
    expected = [
        {"team": "A", "bucket": base_time, "sum_score": 30},
        {"team": "B", "bucket": base_time + timedelta(seconds=10), "sum_score": 20},
    ]
    assert sorted(result, key=lambda x: (x["team"], x["bucket"])) == sorted(expected, key=lambda x: (x["team"], x["bucket"]))


def test_time_bucket_empty_input():
    result = (
        Flow.from_records([])
        .group_by("team")
        .time_bucket(
            freq="30s",
            aggregators={"sum_score": ("sum", "score")},
            time_field="ts",
            label="left",
        )
        .collect()
    )
    assert result == []


def test_time_bucket_all_missing_time_field():
    records = [
        {"team": "A", "score": 10},
        {"team": "A", "score": 20},
    ]
    result = (
        Flow.from_records(records)
        .group_by("team")
        .time_bucket(
            freq="30s",
            aggregators={"sum_score": ("sum", "score")},
            time_field="ts",
            label="left",
        )
        .collect()
    )
    assert result == []


def test_time_bucket_non_uniform_buckets():
    base_time = datetime(2023, 1, 1, 0, 0, 0)
    records = [
        {"team": "A", "ts": base_time + timedelta(seconds=0), "score": 5},
        {"team": "A", "ts": base_time + timedelta(seconds=5), "score": 10},
        {"team": "A", "ts": base_time + timedelta(seconds=65), "score": 20},
    ]
    result = (
        Flow.from_records(records)
        .group_by("team")
        .time_bucket(
            freq="60s",
            aggregators={"sum_score": ("sum", "score")},
            time_field="ts",
            label="left",
        )
        .collect()
    )
    expected = [
        {"team": "A", "bucket": base_time, "sum_score": 15},
        {"team": "A", "bucket": base_time + timedelta(seconds=60), "sum_score": 20},
    ]
    assert result == expected


def test_time_bucket_numeric_time_field():
    records = [
        {"team": "A", "ts": 0, "score": 1},
        {"team": "A", "ts": 15, "score": 2},
        {"team": "A", "ts": 45, "score": 3},
    ]
    result = (
        Flow.from_records(records)
        .group_by("team")
        .time_bucket(
            freq="30s",
            aggregators={"sum_score": ("sum", "score")},
            time_field="ts",
            label="left",
        )
        .collect()
    )
    expected = [
        {"team": "A", "bucket": timedelta(seconds=0), "sum_score": 3},
        {"team": "A", "bucket": timedelta(seconds=30), "sum_score": 3},
    ]
    assert result == expected


def test_time_bucket_custom_aggregator():
    base_time = datetime(2023, 1, 1, 0, 0, 0)
    records = [
        {"team": "A", "ts": base_time + timedelta(seconds=0), "score": 10},
        {"team": "A", "ts": base_time + timedelta(seconds=10), "score": 30},
    ]
    result = (
        Flow.from_records(records)
        .group_by("team")
        .time_bucket(
            freq="30s",
            aggregators={"max_score": ("max", "score")},
            time_field="ts",
            label="left",
        )
        .collect()
    )
    expected = [
        {"team": "A", "bucket": base_time, "max_score": 30},
    ]
    assert result == expected

def test_time_bucket_with_right_label():
    base_time = datetime(2023, 1, 1, 0, 0, 0)
    records = [
        {"player": "X", "ts": base_time + timedelta(seconds=0), "score": 10},
        {"player": "X", "ts": base_time + timedelta(seconds=20), "score": 20},
        {"player": "X", "ts": base_time + timedelta(seconds=40), "score": 30},
    ]

    result = (
        Flow.from_records(records)
        .group_by("player")
        .time_bucket(
            freq="30s",
            aggregators={"sum_score": ("sum", "score")},
            time_field="ts",
            label="right",
        )
        .collect()
    )

    expected = [
        {"player": "X", "bucket": base_time + timedelta(seconds=30), "sum_score": 30},
        {"player": "X", "bucket": base_time + timedelta(seconds=60), "sum_score": 30},
    ]

    assert result == expected

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
