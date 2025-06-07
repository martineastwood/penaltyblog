from datetime import datetime, timedelta

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
    try:
        (
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
    except ValueError as e:
        assert str(e) == "time_bucket: field 'ts' has type int; when freq has a time suffix you must provide datetime or timedelta values."


def test_time_bucket_invalid_frequency_string():
    records = [
        {"player": "X", "ts": 0, "score": 10},
    ]

    try:
        (
            Flow.from_records(records)
            .group_by("player")
            .time_bucket(
                freq="5x",  # Invalid frequency
                aggregators={"sum_score": ("sum", "score")},
                time_field="ts",
                label="left",
            )
            .collect()
        )
    except ValueError as e:
        assert str(e) == "Invalid window '5x': use int for row-count or str ending in s/m/h/d for time."


def test_time_bucket_empty_group_after_filtering():
    records = [
        {"player": "X", "score": 10},  # Missing time field
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
    assert result == []


def test_time_bucket_mixed_valid_invalid_time_fields():
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


def test_time_bucket_boundary_conditions():
    base_time = datetime(2023, 1, 1, 0, 0, 0)
    records = [
        {"player": "X", "ts": base_time + timedelta(seconds=30), "score": 10},
        {"player": "X", "ts": base_time + timedelta(seconds=60), "score": 20},
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
        {"player": "X", "bucket": base_time + timedelta(seconds=30), "sum_score": 10},
        {"player": "X", "bucket": base_time + timedelta(seconds=60), "sum_score": 20},
    ]

    assert result == expected


def test_time_bucket_large_time_gaps():
    base_time = datetime(2023, 1, 1, 0, 0, 0)
    records = [
        {"player": "X", "ts": base_time + timedelta(seconds=0), "score": 10},
        {"player": "X", "ts": base_time + timedelta(seconds=120), "score": 20},
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
        {"player": "X", "bucket": base_time + timedelta(seconds=120), "sum_score": 20},
    ]

    assert result == expected


def test_time_bucket_multiple_aggregators():
    base_time = datetime(2023, 1, 1, 0, 0, 0)
    records = [
        {
            "player": "X",
            "ts": base_time + timedelta(seconds=0),
            "score": 10,
            "assists": 1,
        },
        {
            "player": "X",
            "ts": base_time + timedelta(seconds=20),
            "score": 20,
            "assists": 2,
        },
    ]

    result = (
        Flow.from_records(records)
        .group_by("player")
        .time_bucket(
            freq="30s",
            aggregators={
                "sum_score": ("sum", "score"),
                "sum_assists": ("sum", "assists"),
            },
            time_field="ts",
            label="left",
        )
        .collect()
    )

    expected = [
        {"player": "X", "bucket": base_time, "sum_score": 30, "sum_assists": 3},
    ]

    assert result == expected


def test_time_bucket_negative_time_values():
    records = [
        {"player": "X", "ts": -10, "score": 10},
        {"player": "X", "ts": 20, "score": 20},
    ]
    try:
        (
            Flow.from_records(records)
            .group_by("player")
            .time_bucket(
                freq=30,  # Use numeric freq for numeric time fields
                aggregators={"sum_score": ("sum", "score")},
                time_field="ts",
                label="left",
            )
            .collect()
        )
    except ValueError as e:
        assert str(e) == "Invalid time value '-10' in record."


def test_time_bucket_non_uniform_time_intervals():
    base_time = datetime(2023, 1, 1, 0, 0, 0)
    records = [
        {"player": "X", "ts": base_time + timedelta(seconds=0), "score": 10},
        {"player": "X", "ts": base_time + timedelta(seconds=15), "score": 20},
        {"player": "X", "ts": base_time + timedelta(seconds=45), "score": 30},
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
        {"player": "X", "bucket": base_time, "sum_score": 30},
        {"player": "X", "bucket": base_time + timedelta(seconds=30), "sum_score": 30},
    ]

    assert result == expected


def test_time_bucket_with_custom_bucket_name():
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
            label="left",
            bucket_name="custom_bucket",
        )
        .collect()
    )

    expected = [
        {"player": "X", "custom_bucket": base_time, "sum_score": 30},
        {"player": "X", "custom_bucket": base_time + timedelta(seconds=30), "sum_score": 30},
    ]

    assert result == expected

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
    assert sorted(result, key=lambda x: (x["team"], x["bucket"])) == sorted(
        expected, key=lambda x: (x["team"], x["bucket"])
    )


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
            freq=30,  # Use numeric freq for numeric time fields
            aggregators={"sum_score": ("sum", "score")},
            time_field="ts",
            label="left",
        )
        .collect()
    )
    expected = [
        {"team": "A", "bucket": 0.0, "sum_score": 3},
        {"team": "A", "bucket": 30.0, "sum_score": 3},
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
