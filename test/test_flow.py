import json

import numpy as np
import pytest

from penaltyblog.matchflow.aggregates import (
    all_,
    any_,
    count,
    count_nonnull,
    first_,
    iqr_,
    last_,
    list_,
    max_,
    mean_,
    median_,
    min_,
    mode_,
    percentile_,
    range_,
    std_,
    sum_,
    unique,
)
from penaltyblog.matchflow.flow import Flow
from penaltyblog.matchflow.predicates_helpers import (
    and_,
    not_,
    or_,
    where_contains,
    where_equals,
    where_exists,
    where_gt,
    where_in,
    where_is_null,
    where_not_in,
)


@pytest.fixture
def data():
    return [
        {"player": {"name": "Kane"}, "xg": 0.4},
        {"player": {"name": "Son"}, "xg": 0.1},
    ]


@pytest.fixture
def record():
    return {
        "type": {"name": "Shot"},
        "team": {"name": "Barcelona"},
        "xg": 0.18,
        "player": {"name": None},
        "tags": ["goal", "header"],
    }


@pytest.fixture
def data2():
    return [
        {"x": 10, "nested": {"y": 1}},
        {"x": 20, "nested": {"y": 2}},
        {"x": None, "nested": {"y": 3}},
        {"x": 40},
    ]


@pytest.fixture
def scalar_record():
    return {"team": {"name": "Barcelona"}}


@pytest.fixture
def list_record():
    return {"tags": ["goal", "header", "cross"]}


@pytest.fixture
def dict_record():
    return {"meta": {"info": {"provider": "StatsBomb"}}}


@pytest.fixture
def list_of_dicts_record():
    return {"events": [{"type": "pass"}, {"type": "shot"}]}


@pytest.fixture
def dict_record():
    return {"meta": {"provider": {"name": "StatsBomb"}}}


@pytest.fixture
def list_of_dicts_record():
    return {"players": [{"id": 1}, {"id": 2}]}


def test_plan(data):
    flow = Flow()
    assert flow.plan == []

    f = Flow().filter(lambda r: r["value"] > 10)
    assert len(f.plan) == 1
    step = f.plan[0]
    assert step["op"] == "filter"
    assert callable(step["predicate"])

    flow = (
        Flow.from_records(data)
        .filter(lambda r: r["xg"] > 0.2)
        .select("player.name", "xg")
    )

    results = flow.collect()
    assert results == [{"player": {"name": "Kane"}, "xg": 0.4}]


def test_flow_plan_structure(data):
    flow = (
        Flow.from_records(data)
        .filter(lambda r: r["xg"] > 0.2)
        .select("player.name", "xg")
    )

    plan = flow.plan

    assert len(plan) == 3

    assert plan[0]["op"] == "from_materialized"
    assert isinstance(plan[0]["records"], list)

    assert plan[1]["op"] == "filter"
    assert callable(plan[1]["predicate"])

    assert plan[2]["op"] == "select"
    assert plan[2]["fields"] == ["player.name", "xg"]


def test_rename(data):
    # Flat rename
    flow = Flow.from_records(data).rename(xg="expected_goals")
    results = flow.collect()
    assert results == [
        {"player": {"name": "Kane"}, "expected_goals": 0.4},
        {"player": {"name": "Son"}, "expected_goals": 0.1},
    ]
    plan = flow.plan
    assert plan[-1]["op"] == "rename"
    assert plan[-1]["mapping"] == {"xg": "expected_goals"}

    # Nested rename
    flow2 = Flow.from_records(data).rename(**{"player.name": "name"})
    results2 = flow2.collect()
    assert results2 == [
        {"player": {}, "xg": 0.4, "name": "Kane"},
        {"player": {}, "xg": 0.1, "name": "Son"},
    ]
    plan2 = flow2.plan
    assert plan2[-1]["op"] == "rename"
    assert plan2[-1]["mapping"] == {"player.name": "name"}


def test_deeply_nested_rename():
    data = [
        {"a": {"b": {"c": 42}}, "other": 1},
        {"a": {"b": {"c": 99}}, "other": 2},
    ]
    flow = Flow.from_records(data).rename(**{"a.b.c": "x.y.z"})
    results = flow.collect()
    assert results == [
        {"a": {"b": {}}, "other": 1, "x": {"y": {"z": 42}}},
        {"a": {"b": {}}, "other": 2, "x": {"y": {"z": 99}}},
    ]
    plan = flow.plan
    assert plan[-1]["op"] == "rename"
    assert plan[-1]["mapping"] == {"a.b.c": "x.y.z"}


def test_sort_by_ascending():
    data = [
        {"id": 3, "value": "c"},
        {"id": 1, "value": "a"},
        {"id": 2, "value": "b"},
    ]
    flow = Flow.from_records(data)
    result = flow.sort_by("id").collect()
    ids = [r["id"] for r in result]
    assert ids == [1, 2, 3]


def test_sort_by_descending():
    data = [
        {"score": 10},
        {"score": 30},
        {"score": 20},
    ]
    flow = Flow.from_records(data)
    result = flow.sort_by("score", ascending=False).collect()
    scores = [r["score"] for r in result]
    assert scores == [30, 20, 10]


def test_sort_by_mixed_directions():
    data = [
        {"team": "A", "score": 5},
        {"team": "A", "score": 10},
        {"team": "B", "score": 7},
        {"team": "B", "score": 2},
    ]
    result = (
        Flow.from_records(data)
        .sort_by("team", "score", ascending=[True, False])
        .collect()
    )
    assert result == [
        {"team": "A", "score": 10},
        {"team": "A", "score": 5},
        {"team": "B", "score": 7},
        {"team": "B", "score": 2},
    ]


def test_limit_records():
    data = [{"x": i} for i in range(100)]
    result = Flow.from_records(data).limit(10).collect()
    assert len(result) == 10
    assert result == [{"x": i} for i in range(10)]


def test_drop_fields():
    data = [
        {"a": 1, "b": 2, "meta": {"id": 123, "type": "x"}},
        {"a": 3, "meta": {"type": "y"}},
        {"a": 4},
    ]
    result = (
        Flow.from_records(data)
        .drop("b", "meta.id", "meta.type", "nonexistent")
        .collect()
    )
    assert result == [
        {"a": 1, "meta": {}},
        {"a": 3, "meta": {}},
        {"a": 4},
    ]


def test_select_preserves_structure():
    data = [
        {"a": 1, "meta": {"id": 42, "type": "goal"}},
        {"a": 2, "meta": {"id": 99, "type": "pass"}},
    ]

    result = Flow.from_records(data).select("a", "meta.id").collect()

    assert result == [
        {"a": 1, "meta": {"id": 42}},
        {"a": 2, "meta": {"id": 99}},
    ]


def test_flatten_nested_record():
    data = [{"a": 1, "meta": {"id": 42, "details": {"type": "goal", "x": 12}}}]

    result = Flow.from_records(data).flatten().collect()

    assert result == [
        {
            "a": 1,
            "meta.id": 42,
            "meta.details.type": "goal",
            "meta.details.x": 12,
        }
    ]


def test_distinct_keep_last():
    data = [
        {"x": 1, "y": 1},
        {"x": 1, "y": 99},  # this should be kept if keep='last'
        {"x": 2, "y": 2},
        {"x": 2, "y": 88},  # this should be kept
    ]

    result = Flow.from_records(data).distinct("x", keep="last").collect()

    assert result == [
        {"x": 1, "y": 99},
        {"x": 2, "y": 88},
    ]


def test_distinct_keep_first():
    data = [
        {"x": 1, "y": 1},
        {"x": 1, "y": 99},  # should be skipped
        {"x": 2, "y": 2},
        {"x": 2, "y": 88},  # should be skipped
    ]

    result = Flow.from_records(data).distinct("x", keep="first").collect()

    assert result == [
        {"x": 1, "y": 1},
        {"x": 2, "y": 2},
    ]


def test_dropna_specific_fields():
    data = [
        {"id": 1, "score": 5},
        {"id": 2, "score": None},
        {"id": 3},
        {"id": None, "score": 7},
    ]
    result = Flow.from_records(data).dropna("id", "score").collect()
    assert result == [{"id": 1, "score": 5}]


def test_dropna_any_top_level():
    data = [
        {"id": 1, "score": 5},
        {"id": 2, "score": None},
        {"id": None, "score": 3},
        {"id": 3},  # missing score
    ]
    result = Flow.from_records(data).dropna().collect()
    assert result == [{"id": 1, "score": 5}]


def test_explode_flat():
    data = [
        {"id": 1, "tags": ["a", "b"]},
        {"id": 2, "tags": []},
        {"id": 3, "tags": None},
        {"id": 4},
    ]
    result = Flow.from_records(data).explode("tags").collect()

    expected = [
        {"id": 1, "tags": "a"},
        {"id": 1, "tags": "b"},
        {"id": 2, "tags": []},
        {"id": 3, "tags": None},
        {"id": 4},
    ]

    for r, e in zip(result, expected):
        assert r == e, f"Mismatch:\nactual:   {r}\nexpected: {e}"

    assert result == expected


def test_explode_nested():
    data = [
        {"event": {"id": 1, "tags": ["a", "b"]}},
        {"event": {"id": 2, "tags": ["c"]}},
    ]
    result = Flow.from_records(data).explode("event.tags").collect()

    assert result == [
        {"event": {"id": 1, "tags": "a"}},
        {"event": {"id": 1, "tags": "b"}},
        {"event": {"id": 2, "tags": "c"}},
    ]


def test_explode_multiple_fields():
    data = [
        {"id": 1, "tags": ["a", "b"], "players": ["alice", "bob"]},
        {"id": 2, "tags": ["c"], "players": ["charlie"]},
        {"id": 3, "tags": None, "players": None},
        {"id": 4},  # no fields
    ]
    result = Flow.from_records(data).explode("tags", "players").collect()

    assert result == [
        {"id": 1, "tags": "a", "players": "alice"},
        {"id": 1, "tags": "b", "players": "bob"},
        {"id": 2, "tags": "c", "players": "charlie"},
        {"id": 3, "tags": None, "players": None},
        {"id": 4},
    ]


def test_left_join():
    left = [
        {"id": 1, "x": 10},
        {"id": 2, "x": 20},
        {"id": 3, "x": 30},
    ]

    right = [
        {"id": 1, "y": 100},
        {"id": 2, "y": 200},
    ]

    flow1 = Flow.from_records(left)
    flow2 = Flow.from_records(right)

    result = flow1.join(flow2, on="id").collect()

    assert result == [
        {"id": 1, "x": 10, "y": 100},
        {"id": 2, "x": 20, "y": 200},
        {"id": 3, "x": 30},  # no match
    ]


def test_join_with_conflicting_keys():
    left = [{"id": 1, "value": "left"}]
    right = [{"id": 1, "value": "right"}]

    flow1 = Flow.from_records(left)
    flow2 = Flow.from_records(right)

    result = flow1.join(flow2, on="id").collect()

    assert result == [{"id": 1, "value": "left", "value_right": "right"}]


def test_inner_join():
    left = [
        {"id": 1, "x": 10},
        {"id": 2, "x": 20},
        {"id": 3, "x": 30},
    ]

    right = [
        {"id": 1, "y": 100},
        {"id": 2, "y": 200},
        {"id": 4, "y": 400},
    ]

    result = (
        Flow.from_records(left)
        .join(Flow.from_records(right), on="id", how="inner")
        .collect()
    )

    assert result == [
        {"id": 1, "x": 10, "y": 100},
        {"id": 2, "x": 20, "y": 200},
    ]


def test_split_array_to_fields():
    data = [
        {"id": 1, "location": [10, 50]},
        {"id": 2, "location": [30]},
        {"id": 3, "location": None},
        {"id": 4},
    ]

    result = Flow.from_records(data).split_array("location", into=["x", "y"]).collect()

    assert result == [
        {"id": 1, "location": [10, 50], "x": 10, "y": 50},
        {"id": 2, "location": [30], "x": 30, "y": None},
        {"id": 3, "location": None},
        {"id": 4},  # <- passed through unchanged
    ]


def test_pivot_basic():
    data = [
        {"player": "Alice", "metric": "goals", "value": 3},
        {"player": "Alice", "metric": "assists", "value": 2},
        {"player": "Bob", "metric": "goals", "value": 1},
        {"player": "Bob", "metric": "assists", "value": 4},
    ]

    result = (
        Flow.from_records(data)
        .pivot(index="player", columns="metric", values="value")
        .collect()
    )

    assert result == [
        {"player": "Alice", "goals": 3, "assists": 2},
        {"player": "Bob", "goals": 1, "assists": 4},
    ]


def test_ungrouped_summary():
    data = [
        {"x": 10},
        {"x": 20},
        {"x": 30},
    ]

    result = (
        Flow.from_records(data)
        .summary(
            {
                "count": count(),
                "sum_x": sum_("x"),
                "mean_x": mean_("x"),
            }
        )
        .collect()
    )

    assert result == [{"count": 3, "sum_x": 60, "mean_x": 20.0}]


def test_invalid_summary_output():
    with pytest.raises(ValueError, match="summary function must return a dict"):
        Flow.from_records([{"x": 1}, {"x": 2}]).summary(
            lambda rows: [{"x": 1}, {"x": 2}]
        ).collect()


def test_count(data2):
    assert count()(data2) == 4


def test_count_nonnull():
    data = [
        {"nested": {"v": 100}},
        {"nested": {"v": 200}},
        {"nested": {"v": None}},
        {},  # missing entirely
    ]

    assert count_nonnull("nested.v")(data) == 2
    assert count_nonnull("nested.v")(data) == 2


def test_sum_(data2):
    assert sum_("x")(data2) == 70


def test_mean_(data2):
    assert mean_("x")(data2) == pytest.approx(70 / 3)


def test_min_(data2):
    assert min_("x")(data2) == 10


def test_max_(data2):
    assert max_("x")(data2) == 40


def test_std_(data2):
    assert std_("x")(data2) == pytest.approx(np.std([10, 20, 40], ddof=0))


def test_median_(data2):
    assert median_("x")(data2) == 20


def test_range_(data2):
    assert range_("x")(data2) == 30


def test_iqr_(data2):
    assert iqr_("x")(data2) == pytest.approx(15)


def test_percentile_(data2):
    assert percentile_("x", 90)(data2) == pytest.approx(36)


def test_mode_():
    data = [
        {"y": 1},
        {"y": 1},
        {"y": 2},
        {"y": 3},
    ]
    assert mode_("y")(data) == 1


def test_first_last(data2):
    assert first_("x")(data2) == 10
    assert last_("x")(data2) == 40


def test_unique():
    data = [
        {"y": 1},
        {"y": 1},
        {"y": 2},
        {"y": 3},
    ]
    assert set(unique("y")(data)) == {1, 2, 3}


def test_list_():
    data = [
        {"y": 1},
        {"y": 1},
        {"y": 2},
        {"y": 3},
    ]
    assert list_("y")(data) == [1, 1, 2, 3]


def test_all_():
    data = [
        {"x": 1},
        {"x": 2},
        {"x": 3},
    ]
    assert all_("x")(data) == True


def test_all_any(data2):
    assert all_("x")(data2) is False
    assert any_("x")(data2) is True


def test_nested_field_access(data2):
    assert sum_("nested.y")(data2) == 6
    assert count_nonnull("x")(data2) == 3
    assert first_("nested.y")(data2) == 1


def test_empty_data():
    empty = []

    assert count()(empty) == 0
    assert sum_("x")(empty) == 0
    assert mean_("x")(empty) is None
    assert median_("x")(empty) is None
    assert percentile_("x", 90)(empty) is None
    assert list_("x")(empty) == []


def test_groupby_summary_with_sum():
    data = [
        {"team": "A", "points": 10},
        {"team": "A", "points": 20},
        {"team": "B", "points": 15},
    ]

    flow = Flow.from_records(data)
    result = flow.group_by("team").summary({"points": ("sum", "points")}).collect()

    assert any(r["team"] == "A" and r["points"] == 30 for r in result)
    assert any(r["team"] == "B" and r["points"] == 15 for r in result)


def test_summary_on_nested_field():
    data = [
        {"player": {"stats": {"points": 10}}},
        {"player": {"stats": {"points": 20}}},
        {"player": {"stats": {"points": 15}}},
    ]

    result = (
        Flow.from_records(data)
        .summary({"total_points": ("sum", "player.stats.points")})
        .collect()
    )

    assert len(result) == 1
    assert result[0]["total_points"] == 45


def test_groupby_summary_on_nested_fields():
    data = [
        {"player": {"id": "a", "stats": {"points": 10}}},
        {"player": {"id": "a", "stats": {"points": 20}}},
        {"player": {"id": "b", "stats": {"points": 15}}},
    ]

    result = (
        Flow.from_records(data)
        .group_by("player.id")
        .summary({"total_points": ("sum", "player.stats.points")})
        .collect()
    )

    assert len(result) == 2
    grouped = {r["player.id"]: r["total_points"] for r in result}

    assert grouped["a"] == 30
    assert grouped["b"] == 15


def test_summary_with_custom_callable():
    data = [
        {"player": {"name": "alice"}, "score": 10},
        {"player": {"name": "bob"}, "score": 15},
        {"player": {"name": "alice"}, "score": 25},
    ]

    # Custom summary: compute the average score manually
    def compute_metrics(rows):
        scores = [r["score"] for r in rows]
        return {
            "avg_score": sum(scores) / len(scores),
            "max_score": max(scores),
        }

    result = Flow.from_records(data).summary(compute_metrics).collect()

    assert len(result) == 1
    summary = result[0]
    assert summary["avg_score"] == 50 / 3
    assert summary["max_score"] == 25


def test_flow_summary_with_callable_and_field():
    data = [
        {"player": {"stats": {"points": 10}}},
        {"player": {"stats": {"points": 20}}},
        {"player": {"stats": {"points": 5}}},
    ]

    def scaled_sum(rows, field):
        return sum(r["player"]["stats"]["points"] for r in rows) * 10

    result = (
        Flow.from_records(data)
        .summary({"scaled_points": (scaled_sum, "player.stats.points")})
        .collect()
    )

    assert len(result) == 1
    assert result[0]["scaled_points"] == (10 + 20 + 5) * 10  # 350


def test_group_summary_with_callable_and_field():
    data = [
        {"player": {"id": "a", "stats": {"points": 10}}},
        {"player": {"id": "a", "stats": {"points": 5}}},
        {"player": {"id": "b", "stats": {"points": 7}}},
    ]

    def range_fn(rows, field):
        values = [r["player"]["stats"]["points"] for r in rows]
        return max(values) - min(values)

    result = (
        Flow.from_records(data)
        .group_by("player.id")
        .summary({"point_range": (range_fn, "player.stats.points")})
        .collect()
    )

    grouped = {r["player.id"]: r["point_range"] for r in result}

    assert grouped["a"] == 5  # 10 - 5
    assert grouped["b"] == 0  # single row


def test_summary_with_invalid_tuple_format():
    data = [{"a": 1}]

    with pytest.raises(TypeError):
        Flow.from_records(data).summary({"bad": (123, "a")}).collect()


def test_schema_inference_flat_fields():
    data = [
        {"id": 1, "user": {"name": "Alice", "age": 30}},
        {"id": 2, "user": {"name": "Bob", "age": 25}},
        {"id": 3, "user": {"name": "Charlie", "age": None}},
    ]

    flow = Flow.from_records(data)
    schema = flow.schema()

    assert schema == {
        "id": {int},
        "user.name": {str},
        "user.age": {int, type(None)},
    }


def test_cast_fields_to_types():
    data = [
        {"score": "10", "player": {"age": "25"}},
        {"score": 12, "player": {"age": 30}},
        {"score": None, "player": {"age": "unknown"}},
    ]

    flow = Flow.from_records(data).cast(
        score=int, **{"player.age": lambda x: int(x) if x.isdigit() else None}
    )

    result = flow.flatten().collect()

    assert result == [
        {"score": 10, "player.age": 25},
        {"score": 12, "player.age": 30},
        {"score": None, "player.age": None},
    ]


def test_from_json_lazy(tmp_path):
    data = [{"x": i} for i in range(3)]
    path = tmp_path / "test.json"
    path.write_text(json.dumps(data))

    flow = Flow.from_json(str(path))
    assert flow.plan[0]["op"] == "from_json"
    assert flow.collect() == data


def test_from_jsonl_lazy(tmp_path):
    data = [{"x": i} for i in range(3)]
    path = tmp_path / "test.jsonl"
    with path.open("w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")

    flow = Flow.from_jsonl(str(path))
    assert flow.plan[0]["op"] == "from_jsonl"
    assert flow.collect() == data


def test_to_json_and_jsonl(tmp_path):
    records = [{"a": 1}, {"a": 2}]
    f = Flow.from_records(records)

    json_path = tmp_path / "data.json"
    jsonl_path = tmp_path / "data.jsonl"

    f.to_json(str(json_path))
    assert json.loads(json_path.read_text()) == records

    f.to_jsonl(str(jsonl_path))
    lines = jsonl_path.read_text().splitlines()
    assert [json.loads(line) for line in lines] == records


def test_to_pandas():
    data = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    f = Flow.from_records(data)
    df = f.to_pandas()

    assert list(df.columns) == ["a", "b"]
    assert df.shape == (2, 2)
    assert df["a"].tolist() == [1, 2]


def test_sample_fraction_deterministic():
    data = [{"id": i} for i in range(100)]
    sampled = Flow.from_records(data).sample_fraction(0.25, seed=42).collect()
    assert 15 <= len(sampled) <= 35  # Rough bounds
    assert all(r in data for r in sampled)


def test_sample_fraction_zero():
    data = [{"id": i} for i in range(10)]
    out = Flow.from_records(data).sample_fraction(0.0).collect()
    assert out == []


def test_sample_fraction_one():
    data = [{"id": i} for i in range(10)]
    out = Flow.from_records(data).sample_fraction(1.0).collect()
    assert out == data


def test_map_basic():
    flow = Flow.from_records([{"x": 1}, {"x": 2}])
    out = flow.map(lambda r: {"y": r["x"] + 1}).collect()
    assert out == [{"y": 2}, {"y": 3}]


def test_map_drop_none():
    flow = Flow.from_records([{"x": 1}, {"x": 2}, {"x": 3}])
    out = flow.map(lambda r: {"x": r["x"]} if r["x"] % 2 == 1 else None).collect()
    assert out == [{"x": 1}, {"x": 3}]


def test_map_raises_on_non_dict():
    flow = Flow.from_records([{"x": 1}])
    with pytest.raises(TypeError, match="map function must return a dict"):
        list(flow.map(lambda r: [1, 2, 3]).collect())


def test_map_raises_on_none_when_strict():
    flow = Flow.from_records([{"x": 1}])
    assert list(flow.map(lambda r: None).collect()) == []


def test_pipe_is_lazy():
    called = []

    def mark_call(flow):
        called.append("called")
        return flow.assign(x2=lambda r: r["x"] * 2)

    flow = Flow.from_records([{"x": 4}]).pipe(mark_call)

    # Nothing called yet
    assert called == []

    result = flow.collect()
    assert result == [{"x": 4, "x2": 8}]
    assert called == ["called"]


def test_keys_flat_records():
    records = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob", "age": 30},
        {"id": 3, "country": "UK"},
    ]
    flow = Flow.from_records(records)
    keys = flow.keys()
    assert keys == {"id", "name", "age", "country"}


def test_keys_nested_records():
    records = [
        {"user": {"id": 1, "name": "Alice"}, "score": 10},
        {"user": {"id": 2}, "score": 15},
        {"user": {"id": 3, "name": "Charlie"}, "extra": {"active": True}},
    ]
    flow = Flow.from_records(records)
    keys = flow.keys()
    assert keys == {"user.id", "user.name", "score", "extra.active"}


def test_keys_with_limit():
    records = [{"a": 1}, {"b": 2}, {"c": 3}]
    flow = Flow.from_records(records)
    keys = flow.keys(limit=1)
    assert keys == {"a"}  # Only first record scanned


def test_keys_empty_flow():
    flow = Flow.from_records([])
    keys = flow.keys()
    assert keys == set()


def test_is_empty_true():
    flow = Flow.from_records([])
    assert flow.is_empty() is True


def test_is_empty_false():
    flow = Flow.from_records([{"a": 1}])
    assert flow.is_empty() is False


def test_is_empty_lazy_behavior():
    called = {"count": 0}

    def generator():
        called["count"] += 1
        yield {"a": 1}

    flow = Flow.from_records(list(generator()))  # Materialized version
    assert flow.is_empty() is False
    assert called["count"] == 1  # generator evaluated exactly once


def test_where_equals(record):
    pred = where_equals("type.name", "Shot")
    assert pred(record) is True

    pred2 = where_equals("team.name", "Real Madrid")
    assert pred2(record) is False


def test_where_in(record):
    pred = where_in("team.name", ["Barcelona", "Real"])
    assert pred(record) is True

    pred2 = where_in("team.name", ["Real", "City"])
    assert pred2(record) is False


def test_where_gt(record):
    pred = where_gt("xg", 0.15)
    assert pred(record) is True

    pred2 = where_gt("xg", 0.3)
    assert pred2(record) is False


def test_where_exists_and_null(record):
    assert where_exists("player.name")(record) is False
    assert where_is_null("player.name")(record) is True
    assert where_exists("team.name")(record) is True


def test_where_contains(record):
    assert where_contains("team.name", "Barça")(record) is False
    assert where_contains("team.name", "Barc")(record) is True


def test_and_predicate(record):
    p1 = where_equals("type.name", "Shot")
    p2 = where_gt("xg", 0.1)
    combined = and_(p1, p2)
    assert combined(record) is True

    combined = and_(p1, where_gt("xg", 0.3))
    assert combined(record) is False


def test_or_predicate(record):
    p1 = where_equals("type.name", "Pass")
    p2 = where_gt("xg", 0.1)
    combined = or_(p1, p2)
    assert combined(record) is True

    combined = or_(
        where_equals("type.name", "Foul"), where_equals("team.name", "Chelsea")
    )
    assert combined(record) is False


def test_not_predicate(record):
    pred = where_equals("type.name", "Shot")
    assert not_(pred)(record) is False

    pred2 = where_gt("xg", 0.3)
    assert not_(pred2)(record) is True


def test_missing_field_safe():
    record = {}
    pred = where_equals("foo.bar", "baz")
    assert pred(record) is False

    assert where_is_null("foo.bar")(record) is True
    assert where_exists("foo.bar")(record) is False


# === where_in tests ===
def test_where_in_scalar(scalar_record):
    pred = where_in("team.name", ["Barcelona", "Real Madrid"])
    assert pred(scalar_record) is True

    pred2 = where_in("team.name", ["Man City"])
    assert pred2(scalar_record) is False


def test_where_in_list(list_record):
    pred = where_in("tags", ["goal", "assist"])
    assert pred(list_record) is True

    pred2 = where_in("tags", ["penalty"])
    assert pred2(list_record) is False


def test_where_in_raises_on_dict(dict_record):
    pred = where_in("meta.info", ["StatsBomb"])(dict_record)
    with pytest.raises(TypeError):
        pred(dict_record)


def test_where_in_raises_on_list_of_dicts(list_of_dicts_record):
    pred = where_in("events", ["pass"])(list_of_dicts_record)
    with pytest.raises(TypeError):
        pred(list_of_dicts_record)


# === where_not_in tests ===
def test_where_not_in_scalar(scalar_record):
    pred = where_not_in("team.name", ["Man City", "PSG"])(scalar_record)
    assert pred is True

    pred2 = where_not_in("team.name", ["Barcelona"])(scalar_record)
    assert pred2 is False


def test_where_not_in_list(list_record):
    pred = where_not_in("tags", ["penalty"])(list_record)
    assert pred is True

    pred2 = where_not_in("tags", ["goal", "header"])(list_record)
    assert pred2 is False


def test_where_not_in_raises_on_dict(dict_record):
    pred = where_not_in("meta.info", ["StatsBomb"])(dict_record)
    with pytest.raises(TypeError):
        pred(dict_record)


def test_where_not_in_raises_on_list_of_dicts(list_of_dicts_record):
    pred = where_not_in("events", ["pass"])(list_of_dicts_record)
    with pytest.raises(TypeError):
        pred(list_of_dicts_record)


# === where_equals ===
def test_where_equals_scalar(scalar_record):
    assert where_equals("team.name", "Barcelona")(scalar_record) is True
    assert where_equals("team.name", "Madrid")(scalar_record) is False


def test_where_equals_missing_field(scalar_record):
    assert where_equals("team.coach", "Xavi")(scalar_record) is False


# === where_gt ===
def test_where_gt_pass():
    record = {"xg": 0.25}
    assert where_gt("xg", 0.2)(record) is True
    assert where_gt("xg", 0.3)(record) is False


def test_where_gt_missing_field(scalar_record):
    assert where_gt("nonexistent", 0.1)(scalar_record) is False


def test_where_gt_invalid_type_ignored(dict_record):
    pred = where_gt("meta.provider", 1.0)
    with pytest.raises(TypeError):
        assert pred(dict_record) is False


# === where_exists / where_is_null ===
def test_where_exists(scalar_record):
    assert where_exists("team.name")(scalar_record) is True
    assert where_exists("player.name")(scalar_record) is False
    assert where_exists("missing.field")(scalar_record) is False


def test_where_is_null(scalar_record):
    assert where_is_null("player.name")(scalar_record) is True
    assert where_is_null("team.name")(scalar_record) is False
    assert where_is_null("missing.field")(scalar_record) is True


def test_with_schema_basic_cast():
    data = [{"x": "1", "y": "2.5"}, {"x": "3", "y": "4.0"}]
    flow = Flow.from_records(data).with_schema({"x": int, "y": float})
    result = flow.collect()
    assert result == [{"x": 1, "y": 2.5}, {"x": 3, "y": 4.0}]


def test_with_schema_nested_field():
    data = [{"user": {"age": "20"}}]
    flow = Flow.from_records(data).with_schema({"user.age": int})
    result = flow.collect()
    assert result[0]["user"]["age"] == 20


def test_with_schema_strict_mode_raises():
    data = [{"x": "abc"}]
    with pytest.raises(ValueError):
        Flow.from_records(data).with_schema({"x": int}, strict=True).collect()


def test_with_schema_fallback_on_error():
    data = [{"x": "abc"}]
    flow = Flow.from_records(data).with_schema({"x": int}, strict=False)
    result = flow.collect()
    assert result[0]["x"] == "abc"


def test_with_schema_drop_extra_fields():
    data = [{"a": "1", "b": "2", "c": "3"}]
    flow = Flow.from_records(data).with_schema({"a": int}, drop_extra=True)
    result = flow.collect()
    assert result == [{"a": 1}]


def test_without_schema_drop_extra_fields():
    data = [{"a": "1", "b": "2", "c": "3"}]
    flow = Flow.from_records(data).with_schema({"a": int})
    result = flow.collect()
    assert result == [{"a": 1, "b": "2", "c": "3"}]
