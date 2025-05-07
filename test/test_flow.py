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


def test_selecting():
    data = [
        {
            "id": 1,
            "value": 10,
            "tags": ["a", "b"],
            "nested": {"x": 1, "y": {"z": 2}},
            "extra.field": "foo",
        }
    ]

    flow = Flow(data)
    assert flow.select("id", "value").collect() == [{"id": 1, "value": 10}]

    flow = Flow(data)
    assert flow.select("nested.x").collect() == [{"x": 1}]

    flow = Flow(data)
    assert flow.select("extra.field").collect() == [{"extra.field": "foo"}]


def test_filter_and_collect(sample_records):
    flow = Flow(sample_records)
    out = flow.filter(lambda r: r["value"] == 10).collect()
    assert out == [sample_records[0], sample_records[2]]


def test_assign_select_drop(sample_records):
    flow = Flow(sample_records)
    # assign a new field, then select only that
    out = flow.assign(double=lambda r: r["value"] * 2).select("id", "double").collect()
    assert out == [
        {"id": 1, "double": 20},
        {"id": 2, "double": 40},
        {"id": 3, "double": 20},
    ]


def test_sort_limit_head(sample_records):
    flow = Flow(sample_records)
    # sort descending by value, take top 2
    top2 = flow.sort("value", reverse=True).limit(2).collect()
    assert [r["id"] for r in top2] == [2, 1]
    # head() is just limit + collect
    assert Flow(sample_records).head(1) == [sample_records[0]]


def test_split_array_explode(sample_records):
    flow = Flow(sample_records)
    # split tags into tag0, tag1
    recs = flow.split_array("tags", into=["t0", "t1"]).collect()
    assert recs[0]["t0"] == "a" and recs[0]["t1"] == "b"
    # explode back into one‐tag‐per‐record
    exploded = Flow(sample_records).explode("tags").collect()
    assert all(isinstance(r["tags"], str) for r in exploded if r["tags"] != [])


def test_group_by_summary_ungroup(sample_records):
    flow = Flow(sample_records)
    fg = flow.group_by("value")
    # summarize count per value
    summary = fg.summary(count=("id", "count")).collect()
    # two groups: value=10 has 2 ids, value=20 has 1
    assert {row["value"]: row["count"] for row in summary} == {10: 2, 20: 1}


def test_row_number_drop_duplicates_take_last(sample_records):
    flow = Flow(sample_records)
    # row numbers by value
    rn = flow.row_number("value", new_field="rn").collect()
    # for value=10 you'd get rn=1,2 (in insertion order)
    ids_for_10 = [r["id"] for r in rn if r["value"] == 10]
    assert set(ids_for_10) == {1, 3}
    # drop duplicates by value
    unique = Flow([{"a": 1}, {"a": 1}, {"a": 2}]).drop_duplicates("a").collect()
    assert unique == [{"a": 1}, {"a": 2}]
    # take_last
    last1 = Flow([{"x": i} for i in range(5)]).take_last(1).collect()
    assert last1 == [{"x": 4}]


def test_join_and_errors(sample_records):
    left = Flow(sample_records)
    right = [{"key": 10, "foo": "bar"}, {"key": 99, "foo": "nope"}]
    joined = left.join(right, left_on="value", right_on="key", fields=["foo"]).collect()
    # only value==10 matches key==10
    assert any(r.get("foo") == "bar" for r in joined)
    with pytest.raises(ValueError):
        # invalid how should error
        left.join(right, "value", "key", how="weird").collect()


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
