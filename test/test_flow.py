import json

import pandas as pd
import pytest

from penaltyblog.matchflow.flow import Flow


@pytest.fixture
def sample_records():
    return [
        {"id": 1, "value": 10, "tags": ["a", "b"], "nested": {"x": 1, "y": {"z": 2}}},
        {"id": 2, "value": 20, "tags": ["b"], "nested": {"x": 3, "y": {"z": 4}}},
        {"id": 3, "value": 10, "tags": [], "nested": {"x": 5}},
    ]


def test_init(sample_records):
    assert Flow(sample_records).collect() == sample_records
    assert Flow.from_records(sample_records).collect() == sample_records
    assert Flow(sample_records[0]).collect() == [sample_records[0]]
    assert Flow.from_records([sample_records[0]]).collect() == [sample_records[0]]

    def gen():
        yield {"id": 1}
        yield {"id": 2}

    assert Flow(gen()).collect() == [{"id": 1}, {"id": 2}]
    assert Flow.from_generator(gen()).collect() == [{"id": 1}, {"id": 2}]

    with pytest.raises(TypeError):
        Flow.from_records(123)

    with pytest.raises(TypeError):
        Flow.from_records("123")


def test_len(sample_records):
    assert len(Flow(sample_records)) == len(sample_records)

    flow = Flow(sample_records)
    len(flow)
    results = flow.collect()
    assert len(results) == len(sample_records)
    assert results == sample_records


def test_iter(sample_records):
    flow = Flow(sample_records)
    n = 0
    for _ in flow:
        n += 1
    assert n == len(sample_records)


def test_eq(sample_records):
    assert Flow(sample_records) == Flow(sample_records)
    assert Flow(sample_records) == sample_records
    assert Flow(sample_records) != Flow(sample_records[1:])
    assert Flow(sample_records) != sample_records[1:]


def test_collect(sample_records):
    assert Flow(sample_records).collect() == sample_records


def test_selecting():
    data = [
        {
            "id": 1,
            "value": 10,
            "tags": ["a", "b"],
            "nested": {"x": 1, "y": {"z": 2}},
            "extra.field": "foo",
            "another.extra.field": {"that.is.nested": "foo"},
        }
    ]

    assert Flow(data).select("id", "value").collect() == [{"id": 1, "value": 10}]
    assert Flow(data).select("nested.x").collect() == [{"x": 1}]
    assert Flow(data).select("extra.field").collect() == [{"extra.field": "foo"}]
    assert Flow(data).select("does_not_exist").collect() == [{"does_not_exist": None}]
    assert Flow(data).select("nested.y.z").collect() == [{"z": 2}]


def test_filter(sample_records):
    out = Flow(sample_records).filter(lambda r: r["value"] == 10).collect()
    assert out == [sample_records[0], sample_records[2]]

    out = Flow(sample_records).filter(lambda r: r["value"] > 1000).collect()
    assert out == []

    with pytest.raises(KeyError):
        Flow(sample_records).filter(lambda r: r["does_not_exist"] > 1000).collect()

    out = (
        Flow(sample_records)
        .filter(lambda r: r.get("does_not_exist", -1) > 1000)
        .collect()
    )
    assert out == []


def test_assign(sample_records):
    out = (
        Flow(sample_records)
        .assign(double=lambda r: r["value"] * 2)
        .select("id", "double")
        .collect()
    )
    assert out == [
        {"id": 1, "double": 20},
        {"id": 2, "double": 40},
        {"id": 3, "double": 20},
    ]

    out = (
        Flow(sample_records)
        .assign(double=lambda r: r["value"] * 2, double_id=lambda r: r["id"] * 2)
        .select("double_id", "double")
        .collect()
    )
    assert out == [
        {"double_id": 2, "double": 20},
        {"double_id": 4, "double": 40},
        {"double_id": 6, "double": 20},
    ]

    out = Flow(sample_records).assign(id=lambda r: r["id"] * 2).select("id").collect()
    assert out == [
        {"id": 2},
        {"id": 4},
        {"id": 6},
    ]

    assert Flow([]).assign(new_field=lambda r: 1).collect() == []

    recs = Flow(sample_records).assign(b=lambda r: r["id"]).collect()
    assert recs[0]["b"] == 1

    recs = Flow([{"a": 1, "b": 2}]).assign(c=lambda r: r["a"] + r["b"]).collect()
    assert recs == [{"a": 1, "b": 2, "c": 3}]

    recs = (
        Flow([{"a": 1, "b": 2}])
        .assign(a=lambda r: r["b"], b=lambda r: r["a"])
        .collect()
    )
    assert recs == [{"a": 2, "b": 2}]

    recs = Flow([{"a": 1, "b": 2}]).assign(c=lambda r: 1).collect()
    assert recs[0]["c"] == 1

    with pytest.raises(TypeError):
        Flow([{"a": 1, "b": 2}]).assign(c=1).collect()


def test_drop(sample_records):
    recs = Flow(sample_records).drop("id").collect()
    assert recs == [
        {"value": 10, "tags": ["a", "b"], "nested": {"x": 1, "y": {"z": 2}}},
        {"value": 20, "tags": ["b"], "nested": {"x": 3, "y": {"z": 4}}},
        {"value": 10, "tags": [], "nested": {"x": 5}},
    ]

    recs = Flow(sample_records).drop("id", "value").collect()
    assert recs == [
        {"tags": ["a", "b"], "nested": {"x": 1, "y": {"z": 2}}},
        {"tags": ["b"], "nested": {"x": 3, "y": {"z": 4}}},
        {"tags": [], "nested": {"x": 5}},
    ]

    assert Flow([]).drop("id").collect() == []

    recs = Flow(sample_records).drop("does_not_exist").collect()
    assert recs == sample_records


def test_sort_limit_head(sample_records):
    top2 = Flow(sample_records).sort("value", reverse=True).limit(2).collect()
    assert [r["id"] for r in top2] == [2, 1]

    top2 = Flow(sample_records).sort("value", reverse=False).limit(2).collect()
    assert [r["id"] for r in top2] == [1, 3]

    top2 = Flow(sample_records).sort("value", reverse=True).head(2).collect()
    assert [r["id"] for r in top2] == [2, 1]

    assert Flow(sample_records).head(1) == [sample_records[0]]
    assert Flow([]).sort("id") == []


def test_split_array(sample_records):
    recs = Flow(sample_records).split_array("tags", into=["t0", "t1"]).collect()
    assert recs[0]["t0"] == "a" and recs[0]["t1"] == "b"

    with pytest.raises(TypeError):
        Flow(sample_records).split_array("tags").collect()

    recs = (
        Flow(sample_records).split_array("does_not_exist", into=["t0", "t1"]).collect()
    )
    assert recs[0]["t0"] == None and recs[0]["t1"] == None

    recs = (
        Flow([{"id": 1, "tags": "not_a_list"}])
        .split_array("tags", into=["t0"])
        .collect()
    )
    assert recs[0]["t0"] is None

    recs = (
        Flow([{"id": 1, "tags": ["a", "b", "c"]}])
        .split_array("tags", into=["t0", "t1"])
        .collect()
    )
    assert recs[0]["t0"] == "a" and recs[0]["t1"] == "b"
    assert "t2" not in recs[0]

    recs = Flow(sample_records).split_array("tags", into=[]).collect()
    assert recs == sample_records
    assert Flow([]).split_array("tags", into=["t0"]).collect() == []


def test_explode(sample_records):
    recs = Flow(sample_records).explode("tags").collect()
    assert all(isinstance(r["tags"], str) for r in recs if r["tags"] != [])

    recs = Flow(sample_records).explode("non_existent_tags").collect()
    assert recs == sample_records

    recs = Flow(sample_records).explode("tags").collect()
    assert len(recs) == 4

    data = [{"id": 1, "tags": "abc"}]
    assert Flow(data).explode("tags").collect() == data

    assert Flow([]).explode("tags").collect() == []


def test_group_by_summary_ungroup(sample_records):
    recs = (
        Flow(sample_records).group_by("value").summary(count=("id", "count")).collect()
    )
    assert {row["value"]: row["count"] for row in recs} == {10: 2, 20: 1}

    recs = Flow(sample_records).group_by("value").summary(count="count").collect()
    assert {row["value"]: row["count"] for row in recs} == {10: 2, 20: 1}

    recs = (
        Flow(sample_records)
        .group_by("id")
        .summary(count="count", max_value=("value", "max"))
        .collect()
    )
    assert [x["max_value"] for x in recs] == [10, 20, 10]

    with pytest.raises(AttributeError):
        Flow(sample_records).ungroup().collect()

    recs = Flow(sample_records).group_by("value").ungroup().collect()
    assert len(recs) == len(sample_records)

    recs = (
        Flow(sample_records)
        .group_by("id")
        .summary(count="count", max_value=("value", "max"))
        .select("id")
        .collect()
    )

    assert len(recs) == len(sample_records)
    assert all(r["id"] is not None for r in recs)
    assert all(r.get("max_value") is None for r in recs)


def test_row_number_drop_duplicates_take_last(sample_records):
    rn = Flow(sample_records).row_number("value", new_field="rn").collect()
    # for value=10 you'd get rn=1,2 (in insertion order)
    ids_for_10 = [r["id"] for r in rn if r["value"] == 10]
    assert set(ids_for_10) == {1, 3}
    # drop duplicates by value
    unique = Flow([{"a": 1}, {"a": 1}, {"a": 2}]).drop_duplicates("a").collect()
    assert unique == [{"a": 1}, {"a": 2}]
    # take_last
    last1 = Flow([{"x": i} for i in range(5)]).take_last(1).collect()
    assert last1 == [{"x": 4}]


def test_concat():
    assert Flow([]).concat(Flow([])).collect() == []
    assert Flow([{"x": 1}]).concat(Flow([{"x": 2}])).collect() == [{"x": 1}, {"x": 2}]
    assert Flow([{"x": 1}]).concat(Flow([{"x": 2}])).concat(
        Flow([{"x": 3}])
    ).collect() == [
        {"x": 1},
        {"x": 2},
        {"x": 3},
    ]


def test_join_and_errors(sample_records):
    right = [{"key": 10, "foo": "bar"}, {"key": 99, "foo": "nope"}]
    joined = (
        Flow(sample_records)
        .join(right, left_on="value", right_on="key", fields=["foo"], how="inner")
        .collect()
    )
    assert all(r.get("foo") == "bar" for r in joined)
    assert len(joined) == 2

    joined = (
        Flow(sample_records)
        .join(right, left_on="value", right_on="key", fields=["foo"], how="left")
        .collect()
    )
    assert any(r.get("foo") == "bar" for r in joined)
    assert len(joined) == 3

    joined = (
        Flow(sample_records)
        .join(right, left_on="value", right_on="key", fields=["foo"])
        .collect()
    )
    assert any(r.get("foo") == "bar" for r in joined)
    assert len(joined) == 3

    with pytest.raises(ValueError):
        flow = Flow(sample_records)
        flow.join(right, "value", "key", how="weird").collect()

    right = [{"key": 100, "foo": "bar"}, {"key": 99, "foo": "nope"}]
    joined = (
        Flow(sample_records)
        .join(right, left_on="value", right_on="key", fields=["foo"], how="inner")
        .collect()
    )
    assert len(joined) == 0

    right = [{"does_not_exist": 100, "foo": "bar"}]
    joined = (
        Flow(sample_records)
        .join(
            right,
            left_on="value",
            right_on="does_not_exist",
            fields=["foo"],
            how="inner",
        )
        .collect()
    )
    assert len(joined) == 0

    right = [{"key": 10, "foo": "bar"}]
    joined = (
        Flow(sample_records)
        .join(
            right,
            left_on="does_not_exist",
            right_on="does_not_exist",
            fields=["foo"],
            how="inner",
        )
        .collect()
    )
    assert len(joined) == 0


def test_first_last_is_empty_keys(sample_records):
    flow = Flow(sample_records)
    assert flow.first()["id"] == 1
    assert flow.last()["id"] == 3
    assert Flow([]).is_empty() is True
    assert Flow(sample_records).is_empty() is False
    # keys union
    ks = flow.keys(limit=1)
    assert "id" in ks and "value" in ks


def test_flatten_and_to_pandas(sample_records):
    flow = Flow(sample_records)
    flat = flow.flatten().collect()
    # nested.x should become top‐level
    assert all("nested.x" in r for r in flat)
    df = Flow(sample_records).to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {"id", "value", "tags", "nested"}


def test_json_file_roundtrip(tmp_path, sample_records):
    folder = tmp_path / "out"
    Flow(sample_records).to_json_files(folder, by="id")
    # ensure files exist
    files = sorted(folder.iterdir())
    assert any(f.name == "1.json" for f in files)
    # test jsonl
    jl = tmp_path / "data.jsonl"
    Flow(sample_records).to_jsonl(jl)
    reloaded = Flow.from_jsonl(jl).collect()
    assert reloaded == sample_records
    # test single‐json
    js = tmp_path / "data.json"
    Flow(sample_records).to_json_single(js)
    re2 = Flow.from_file(js).collect()
    assert re2 == sample_records


def test_from_folder_and_glob(tmp_path, sample_records):
    # write two json files
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "b.json"
    f1.write_text(json.dumps(sample_records[0]))
    f2.write_text(json.dumps([sample_records[1], sample_records[2]]))
    all_recs = Flow.from_folder(tmp_path).collect()
    assert len(all_recs) == 3

    # glob
    all2 = Flow.from_glob(str(tmp_path / "*.json")).collect()
    assert len(all2) == 3


def test_describe(sample_records):
    df = Flow(sample_records).describe()
    unique_values = {r["value"] for r in sample_records}
    assert len(unique_values) == 2
    assert "25%" in df.index


def test_pipe(sample_records):
    def custom_filter_and_limit(flow_instance, val_gt, num):
        return flow_instance.filter(lambda r: r.get("value", 0) == val_gt).limit(num)

    recs = Flow(sample_records).pipe(custom_filter_and_limit, 10, 10)
    assert len(recs.collect()) == 2

    def returns_list(flow):
        return flow.collect()

    recs = Flow(sample_records).pipe(returns_list)
    assert isinstance(recs, list)


def test_explode_multi():
    data = [
        {"id": 1, "names": ["A", "B"], "scores": [100, 200, 300]},
        {"id": 2, "names": ["C", "D", "E"]},
        {"id": 3, "scores": [400]},
    ]

    recs = Flow(data).explode_multi(["names", "scores"]).collect()
    expected = [
        {"id": 1, "names": "A", "scores": 100},
        {"id": 1, "names": "B", "scores": 200},
        {"id": 1, "names": None, "scores": 300},
        {"id": 2, "names": "C", "scores": None},
        {"id": 2, "names": "D", "scores": None},
        {"id": 2, "names": "E", "scores": None},
        {"id": 3, "names": None, "scores": 400},
    ]
    assert recs == expected

    # Custom fillvalue
    recs = Flow(data).explode_multi(["names", "scores"], fillvalue="MISSING").collect()
    assert recs[2]["names"] == "MISSING"

    # Empty keys list
    with pytest.raises(ValueError):
        Flow(data).explode_multi([]).collect()

    # missing key from all records
    recs = Flow(data).explode_multi(["does_not_exist"]).collect()
    assert recs == data


def test_sample_and_sample_frac_edge_cases(sample_records):
    # sample
    assert len(Flow(sample_records).sample(0).collect()) == 0
    assert len(Flow(sample_records).sample(100).collect()) == len(sample_records)

    sampled = Flow(sample_records).sample(3, seed=42).collect()
    assert len(sampled) == 3
    sampled2 = Flow(sample_records).sample(3, seed=42).collect()
    assert sampled == sampled2  # Reproducible with seed
    assert Flow([]).sample(5).collect() == []

    # sample_frac
    assert len(Flow(sample_records).sample_frac(0.0).collect()) == 0
    all_selected = Flow(sample_records).sample_frac(1.0).collect()
    assert all_selected == sample_records
    # Approx fraction
    frac_sample = (
        Flow(sample_records * 20).sample_frac(0.5, seed=123).collect()
    )  # Larger N
    assert (
        len(sample_records * 20) * 0.2
        < len(frac_sample)
        < len(sample_records * 20) * 0.7
    )

    assert Flow([]).sample_frac(0.5).collect() == []


def test_flatten_edge_cases():
    assert Flow([]).flatten().collect() == []
    assert Flow([{}]).flatten().collect() == [{}]
    no_nest = [{"a": 1, "b": 2}]
    assert Flow(no_nest).flatten().collect() == no_nest

    # Multiple levels
    deep_nest = [{"a": {"b": {"c": 10, "d": 15}}, "e": 20}]
    assert Flow(deep_nest).flatten().collect() == [{"a.b.c": 10, "a.b.d": 15, "e": 20}]
    assert Flow(deep_nest).flatten(sep="_").collect() == [
        {"a_b_c": 10, "a_b_d": 15, "e": 20}
    ]
    test_data = [{"a": {"b": 10}, "a.b": 20}]
    assert Flow(test_data).flatten().collect() == [{"a.b": 20}]

    test_data_rev = [{"a.b": 20, "a": {"b": 10}}]
    assert Flow(test_data_rev).flatten().collect() == [{"a.b": 10}]
