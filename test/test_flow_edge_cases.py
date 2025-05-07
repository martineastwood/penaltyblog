import numpy as np
import pytest

from penaltyblog.matchflow import Flow, FlowGroup


@pytest.fixture
def sample_records_complex():
    return [
        {
            "id": 1,
            "value": 10,
            "cat": "A",
            "tags": ["a", "b"],
            "nested": {"x": 1, "y": 2},
        },
        {"id": 2, "value": 20, "cat": "B", "tags": ["b", "c"], "nested": {"x": 3}},
        {"id": 3, "value": 10, "cat": "A", "tags": [], "nested": {"y": 4}},
        {"id": 4, "value": None, "cat": "C", "tags": ["d"], "nested": {}},
        {
            "id": 5,
            "value": 30,
            "cat": "B",
            "tags": ["a", "c", "e"],
            "nested": {"x": 5, "y": 6, "z": 7},
        },
    ]


@pytest.fixture
def empty_flow():
    return Flow([])


@pytest.fixture
def sample_records():
    return [
        {"id": 1, "value": 10, "group": "A", "nested": {"x": 1}, "tags": ["a", "b"]},
        {"id": 2, "value": 20, "group": "B", "nested": {"x": 2}, "tags": ["b"]},
        {"id": 3, "value": None, "group": "A", "nested": {"x": 3}, "tags": []},
    ]


def test_filter_and_select(sample_records):
    flow = Flow.from_records(sample_records)
    out = (
        flow.filter(lambda r: r["value"] is not None and r["value"] > 10)
        .select("id", "value")
        .collect()
    )
    assert out == [{"id": 2, "value": 20}]


def test_limit_and_head(sample_records):
    # Test limit
    flow1 = Flow.from_records(sample_records)
    assert flow1.limit(2).collect() == sample_records[:2]

    # Test head on a new Flow instance to ensure it starts from the beginning
    flow2 = Flow.from_records(sample_records)
    assert flow2.head(1).collect() == [sample_records[0]]


def test_sort_and_row_number(sample_records):
    flow = Flow.from_records(sample_records)
    sorted_flow = flow.sort(by="value", reverse=True)
    vals = [r["value"] for r in sorted_flow.collect()]
    assert vals == [20, 10, None]
    # row_number should attach order
    rn = (
        Flow.from_records(sample_records)
        .row_number(by="value", new_field="rank")
        .collect()
    )
    ranks = [r["rank"] for r in sorted(rn, key=lambda r: r["id"])]
    assert ranks == [1, 2, None]


def test_drop_and_keys(sample_records):
    flow = Flow.from_records(sample_records)
    dropped = flow.drop("tags", "nested").collect()
    assert all("tags" not in r and "nested" not in r for r in dropped)
    # keys() inspects union of keys
    k = Flow.from_records(sample_records).keys(limit=2)
    assert set(k) >= {"id", "value", "group", "nested", "tags"}


def test_unique(sample_records):
    data = [
        {"a": 1},
        {"a": 2},
        {"a": 1},
        {"a": 3},
    ]
    flow = Flow.from_records(data)
    uniq = flow.unique("a").collect()
    assert uniq == [{"a": 1}, {"a": 2}, {"a": 3}]


def test_summary_and_group_by_ungroup(sample_records):
    flow = Flow.from_records(sample_records)
    # overall summary
    overall = flow.summary(
        total_value=("value", "sum"),
        count="count",
        mean_value=("value", "mean"),
    ).first()
    assert overall["total_value"] == 30  # 10 + 20 + None→None is skipped
    assert overall["count"] == 3
    assert pytest.approx(overall["mean_value"], rel=1e-3) == (10 + 20) / 2

    # group‐by summary
    fg = Flow.from_records(sample_records).group_by("group")
    summary_flow = fg.summary(
        sum_value=("value", "sum"),
        cnt="count",
    )
    rows = sorted(summary_flow.collect(), key=lambda r: r["group"])
    assert rows == [
        {"group": "A", "sum_value": 10, "cnt": 2},
        {"group": "B", "sum_value": 20, "cnt": 1},
    ]


def test_flatten(sample_records):
    flow = Flow.from_records(sample_records)
    flat = flow.flatten(sep="__").select("id", "nested__x").collect()
    # nested.x should become nested__x
    assert flat == [
        {"id": 1, "nested__x": 1},
        {"id": 2, "nested__x": 2},
        {"id": 3, "nested__x": 3},
    ]


def test_explode_and_explode_multi():
    data = [
        {"id": 1, "tags": ["a", "b"], "scores": [10, 20]},
        {"id": 2, "tags": ["c"], "scores": [30]},
    ]
    flow = Flow.from_records(data)
    exploded = flow.explode("tags").collect()
    # each record with one tag
    assert {r["tags"] for r in exploded} == {"a", "b", "c"}

    # explode_multi should zip tags and scores
    multi = Flow.from_records(data).explode_multi(["tags", "scores"]).collect()
    assert multi == [
        {"id": 1, "tags": "a", "scores": 10},
        {"id": 1, "tags": "b", "scores": 20},
        {"id": 2, "tags": "c", "scores": 30},
    ]


def test_is_empty_last_first(sample_records):
    empty = Flow.from_records([])
    assert empty.is_empty()
    assert empty.first() is None
    assert empty.last() is None

    nonempty = Flow.from_records(sample_records)
    assert not nonempty.is_empty()
    assert nonempty.first() == sample_records[0]
    assert nonempty.last() == sample_records[-1]

    # ========================


# --- Test __init__, from_records, from_generator ---
def test_flow_instantiation(sample_records_complex):
    assert Flow(sample_records_complex).collect() == sample_records_complex
    assert Flow.from_records(sample_records_complex).collect() == sample_records_complex
    assert Flow(sample_records_complex[0]).collect() == [sample_records_complex[0]]
    assert Flow.from_records(sample_records_complex[0]).collect() == [
        sample_records_complex[0]
    ]

    def gen():
        yield {"id": 1}
        yield {"id": 2}

    assert Flow(gen()).collect() == [{"id": 1}, {"id": 2}]
    # # Assuming Flow.from_generator exists and works as expected
    assert Flow.from_generator(gen()).collect() == [{"id": 1}, {"id": 2}]

    with pytest.raises(
        TypeError,
        match="Expected a list of dicts, a single dict, or iterable of dicts",
    ):
        Flow.from_records(123)

    # Test that internal records are shallow copies
    original_rec = {"id": 1, "data": [10, 20]}
    flow = Flow([original_rec])
    collected_rec = flow.collect()[0]
    assert collected_rec is not original_rec  # Different dict objects
    collected_rec["id"] = 99
    assert original_rec["id"] == 1  # Original unchanged for immutable values
    collected_rec["data"].append(30)
    assert original_rec["data"] == [
        10,
        20,
        30,
    ]  # Shallow copy means mutable internals are shared


# --- Test Iterator Consumption & Peeking ---
def test_len_and_consumption(sample_records_complex):
    flow = Flow(sample_records_complex)
    assert len(flow) == 5
    # After len(), the flow should still be fully iterable because of tee
    assert flow.collect() == sample_records_complex

    flow_gen = Flow(({"i": i} for i in range(3)))
    assert len(flow_gen) == 3
    assert flow_gen.collect() == [{"i": 0}, {"i": 1}, {"i": 2}]


def test_first_last_is_empty_keys_consumption(sample_records_complex, empty_flow):
    flow = Flow(sample_records_complex)
    assert flow.first() == sample_records_complex[0]
    # Flow should still be iterable from the start after first()
    assert flow.collect()[0] == sample_records_complex[0]

    flow = Flow(sample_records_complex)  # Re-init
    assert flow.last() == sample_records_complex[-1]
    # Flow is consumed by last() as it iterates through one leg of tee
    # flow._records is now the *other* leg of tee, starting from the beginning
    assert flow.collect() == sample_records_complex

    assert not Flow(sample_records_complex).is_empty()
    assert (
        Flow(sample_records_complex).collect() == sample_records_complex
    )  # is_empty should not fully consume

    assert empty_flow.is_empty()
    assert empty_flow.first() is None

    flow_keys = Flow(sample_records_complex)
    keys_sample = flow_keys.keys(limit=2)
    assert "id" in keys_sample and "value" in keys_sample and "cat" in keys_sample
    assert "nested" in keys_sample
    # Check flow_keys is still fully iterable
    assert len(flow_keys.collect()) == 5

    assert Flow([]).keys() == set()


def test_collect_multiple_times(sample_records_complex):
    flow = Flow(sample_records_complex)
    data1 = flow.collect()
    data2 = flow.collect()  # Should be empty if _records was a generator and consumed
    assert data1 == sample_records_complex
    assert data2 == []  # Original iterator is consumed


# --- Test cache ---
def test_cache(sample_records_complex):
    flow = Flow(sample_records_complex)
    cached_flow = flow.cache()

    # Original flow is consumed by .cache()
    assert flow.collect() == []

    assert cached_flow.collect() == sample_records_complex

    # Cached flow previously consumed by the .collect()
    assert len(cached_flow) == 5

    assert cached_flow.collect() == sample_records_complex  # Can collect multiple times
    assert len(cached_flow) == 5  # Len also works multiple times

    # Operations on cached_flow shouldn't affect a new flow from original data
    filtered_cached = cached_flow.filter(lambda r: r["id"] == 1)
    assert len(filtered_cached.collect()) == 1

    flow_new = Flow(sample_records_complex)  # New flow from original data
    assert len(flow_new.collect()) == 5

    assert Flow([]).cache().collect() == []


# --- Test filter ---
def test_filter_edge_cases(sample_records_complex, empty_flow):
    flow = Flow(sample_records_complex)
    assert flow.filter(lambda r: r["id"] > 100).collect() == []  # Empty result

    flow = Flow(sample_records_complex)
    assert flow.filter(lambda r: True).collect() == sample_records_complex  # All match


# --- Test assign ---
def test_assign_edge_cases(sample_records_complex, empty_flow):
    flow = Flow(sample_records_complex)
    # Overwrite existing field
    assigned = flow.assign(value=lambda r: r["id"] * 100).collect()
    assert assigned[0]["value"] == 100
    # Assign multiple
    assigned_multi = (
        Flow(sample_records_complex)
        .assign(
            new_val=lambda r: r["value"] * 2 if r["value"] else 0,
            id_str=lambda r: str(r["id"]),
        )
        .collect()
    )
    assert assigned_multi[0]["new_val"] == 20
    assert assigned_multi[0]["id_str"] == "1"
    assert assigned_multi[3]["new_val"] == 0  # Handles None in value

    # Function raises error
    with pytest.raises(ZeroDivisionError):
        Flow(sample_records_complex).assign(error_field=lambda r: r["id"] / 0).collect()

    assert empty_flow.assign(new_field=lambda r: 1).collect() == []


# --- Test select ---
def test_select_edge_cases(sample_records_complex, empty_flow):
    flow = Flow(sample_records_complex)
    # Select non-existent fields
    selected = flow.select("id", "non_existent_field").collect()
    assert selected[0] == {"id": 1, "non_existent_field": None}
    # Select no fields
    assert Flow(sample_records_complex).select().collect() == [{}, {}, {}, {}, {}]
    # Select duplicate fields
    selected_dup = Flow(sample_records_complex).select("id", "id", "value").collect()
    assert selected_dup[0] == {"id": 1, "value": 10}  # Behaves like set
    assert empty_flow.select("id").collect() == []


# --- Test drop ---
def test_drop_edge_cases(sample_records_complex, empty_flow):
    flow = Flow(sample_records_complex)
    # Drop non-existent field
    dropped = flow.drop("non_existent").collect()
    assert dropped == sample_records_complex  # No change
    # Drop all fields (by listing them)
    all_keys = list(sample_records_complex[0].keys())
    assert Flow(sample_records_complex).drop(*all_keys).collect()[0] == {}
    assert empty_flow.drop("id").collect() == []


# --- Test sort ---
def test_sort_edge_cases(sample_records_complex, empty_flow):
    # Sorting by field with Nones (already covered by your code, but good to have specific test)
    flow_val_sort = Flow(sample_records_complex).sort("value").collect()
    assert flow_val_sort[-1]["value"] is None  # Nones at the end for ascending
    assert flow_val_sort[0]["value"] == 10

    flow_val_sort_rev = (
        Flow(sample_records_complex).sort("value", reverse=True).collect()
    )
    assert flow_val_sort_rev[-1]["value"] is None  # Nones still at the end
    assert flow_val_sort_rev[0]["value"] == 30

    # Sorting by field not in all records (None for missing key)
    data_mixed_keys = [{"id": 1, "sortkey": 10}, {"id": 2}, {"id": 3, "sortkey": 5}]
    flow_mixed = Flow(data_mixed_keys).sort("sortkey").collect()
    assert [r["id"] for r in flow_mixed] == [3, 1, 2]  # id:2 (None) goes last

    assert empty_flow.sort("id").collect() == []
    single_item_flow = Flow([{"id": 1}])
    assert single_item_flow.sort("id").collect() == [{"id": 1}]

    # Sorting by mixed data types (should raise TypeError)
    data_mixed_types = [{"id": 1, "val": 10}, {"id": 2, "val": "apple"}]
    with pytest.raises(TypeError):
        Flow(data_mixed_types).sort("val").collect()


# --- Test split_array ---
def test_split_array_edge_cases(sample_records_complex, empty_flow):
    flow = Flow(sample_records_complex)
    # Key not present
    recs_key_missing = flow.split_array(
        "non_existent_tags", into=["t0", "t1"]
    ).collect()
    assert recs_key_missing[0]["t0"] is None and recs_key_missing[0]["t1"] is None
    # Value for key is not a list
    recs_not_list = (
        Flow([{"id": 1, "tags": "not_a_list"}])
        .split_array("tags", into=["t0"])
        .collect()
    )
    assert recs_not_list[0]["t0"] is None

    # Array longer than into list
    recs_longer = (
        Flow([{"id": 1, "tags": ["a", "b", "c"]}])
        .split_array("tags", into=["t0", "t1"])
        .collect()
    )
    assert recs_longer[0]["t0"] == "a" and recs_longer[0]["t1"] == "b"
    assert "t2" not in recs_longer[0]
    # `into` list is empty
    flow = Flow(sample_records_complex)
    recs_empty_into = flow.split_array("tags", into=[]).collect()
    assert recs_empty_into == sample_records_complex  # No change if `into` is empty
    assert empty_flow.split_array("tags", into=["t0"]).collect() == []


# --- Test summary (standalone) ---
def test_summary_standalone_edge_cases(sample_records_complex, empty_flow):
    # Assuming _resolve_agg handles empty lists / None values gracefully for common aggs
    # e.g., sum of empty = 0, count of empty = 0, mean of empty = None/NaN
    summary = empty_flow.summary(
        total_value=("value", "sum"), num_records="count", avg_value=("value", "mean")
    ).first()
    assert float(summary["total_value"]) == 0  # Or None, depending on _resolve_agg
    assert float(summary["num_records"]) == 0
    assert bool(np.isnan(summary["avg_value"])) == True

    # Custom callable
    def count_As(records):
        return sum(1 for r in records if r.get("cat") == "A")

    summary_custom = Flow(sample_records_complex).summary(num_A=count_As).first()
    assert summary_custom["num_A"] == 2

    # No aggregates
    assert Flow(sample_records_complex).summary().first() == {}


# --- Test concat ---
def test_concat_edge_cases(sample_records_complex, empty_flow):
    flow1 = Flow(sample_records_complex[:2])
    flow2 = Flow(sample_records_complex[2:])

    concatenated = flow1.concat(flow2).collect()
    assert concatenated == sample_records_complex

    flow1 = Flow(sample_records_complex[:2])
    assert flow1.concat(empty_flow).collect() == sample_records_complex[:2]

    flow1 = Flow(sample_records_complex[:2])
    assert empty_flow.concat(flow1).collect() == sample_records_complex[:2]
    assert empty_flow.concat(empty_flow).collect() == []

    # Concatenating multiple
    f1 = Flow([{"id": 1}])
    f2 = Flow([{"id": 2}])
    f3 = Flow([{"id": 3}])
    assert f1.concat(f2, f3).collect() == [{"id": 1}, {"id": 2}, {"id": 3}]

    # Ensure shallow copies are made from concatenated flows' records
    original_rec = {"id": 100, "mutable": [1]}
    f_orig = Flow([original_rec])
    f_concat = Flow([]).concat(f_orig)
    collected = f_concat.collect()[0]
    assert collected is not original_rec
    collected["mutable"].append(2)
    assert original_rec["mutable"] == [1, 2]  # Shared mutable internal


# --- Test row_number ---
def test_row_number_edge_cases(sample_records_complex, empty_flow):
    assert empty_flow.row_number("id").collect() == []
    # Field not present in all records (None rank)
    data_mixed_keys = [{"id": 1, "val": 10}, {"id": 2}, {"id": 3, "val": 5}]
    flow_mixed = Flow(data_mixed_keys).row_number("val", new_field="rank").collect()
    # Expected order by id after sorting by val (5, 10, None)
    # id:3 (val:5) -> rank 1
    # id:1 (val:10) -> rank 2
    # id:2 (val:None) -> rank None
    ranks_by_id = {r["id"]: r["rank"] for r in flow_mixed}
    assert ranks_by_id == {3: 1, 1: 2, 2: None}


# --- Test drop_duplicates ---
def test_drop_duplicates_edge_cases(sample_records_complex, empty_flow):
    data_dups = [
        {"a": 1, "b": 10},
        {"a": 1, "b": 11},
        {"a": 2, "b": 12},
        {"a": 1, "b": 13},
    ]
    flow = Flow(data_dups)
    # No fields (duplicate entire records - none here)
    assert flow.drop_duplicates().collect() == data_dups

    # Keep last
    flow = Flow(data_dups)
    assert flow.drop_duplicates("a", keep="last").collect() == [
        {"a": 1, "b": 13},
        {"a": 2, "b": 12},
    ]

    # Keep False (drop all occurrences of duplicates for key 'a')
    flow = Flow(data_dups)
    assert flow.drop_duplicates("a", keep=False).collect() == [
        {"a": 2, "b": 12}
    ]  # Only a=2 is truly unique by 'a'

    assert empty_flow.drop_duplicates().collect() == []
    flow_all_unique = Flow([{"id": 1}, {"id": 2}])
    assert flow_all_unique.drop_duplicates("id").collect() == [{"id": 1}, {"id": 2}]

    flow_all_dup_val = Flow([{"id": 1, "val": "x"}, {"id": 2, "val": "x"}])
    assert flow_all_dup_val.drop_duplicates("val").collect() == [
        {"id": 1, "val": "x"}
    ]  # keep='first'

    # Records with unhashable values (e.g., list) when fields is empty
    data_unhashable = [{"a": [1, 2]}, {"a": [1, 2]}]
    with pytest.raises(TypeError):  # tuple(sorted(record.items())) will fail
        Flow(data_unhashable).drop_duplicates().collect()


# --- Test take_last ---
def test_take_last_edge_cases(sample_records_complex, empty_flow):
    assert Flow(sample_records_complex).take_last(0).collect() == []
    assert (
        Flow(sample_records_complex).take_last(100).collect() == sample_records_complex
    )
    assert empty_flow.take_last(5).collect() == []


# --- Test unique ---
def test_unique_edge_cases(sample_records_complex, empty_flow):
    # No fields (acts like drop_duplicates on whole record)
    data_full_dups = [{"a": 1, "b": 2}, {"a": 1, "b": 2}, {"a": 3, "b": 4}]
    assert Flow(data_full_dups).unique().collect() == [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ]

    # Field not present in some records
    data_missing_key = [{"a": 1}, {"b": 2}, {"a": 1}]
    # Key tuple for {"b":2} will be (None,) if unique("a")
    # Key tuple for {"b":2} will be (None,2) if unique("a","b")
    # Key tuple for {"b":2} will be (2,None) if unique("b","a")
    assert Flow(data_missing_key).unique("a").collect() == [
        {"a": 1},
        {"a": None},
    ]  # Assuming get defaults to None for key
    assert Flow(data_missing_key).unique("a", "b").collect() == [
        {"a": 1, "b": None},
        {"a": None, "b": 2},
    ]

    assert empty_flow.unique("id").collect() == []


# --- Test rename ---
def test_rename_edge_cases(sample_records_complex, empty_flow):
    flow = Flow(sample_records_complex)
    # Map key that doesn't exist
    renamed = flow.rename(non_existent="new_name").collect()
    assert "new_name" not in renamed[0]
    # Map to existing key name (should overwrite if old key existed)
    # This needs care: if old='id', new='value', and 'value' exists, pop('id') happens first.
    renamed_conflict = Flow([{"id": 1, "value": 10}]).rename(id="value").collect()
    assert renamed_conflict == [{"value": 1}]  # 'value':10 is gone
    # Empty mapping
    flow = Flow(sample_records_complex)
    assert flow.rename().collect() == sample_records_complex
    assert empty_flow.rename(id="new_id").collect() == []


# --- Test pipe ---
def test_pipe(sample_records_complex):
    def custom_filter_and_limit(flow_instance, val_gt, num):
        return flow_instance.filter(lambda r: r.get("value", 0) or 0 > val_gt).limit(
            num
        )

    result_flow = Flow(sample_records_complex).pipe(custom_filter_and_limit, 15, 1)
    assert len(result_flow.collect()) == 1  # Should be record with id 2 or 5

    def returns_list(flow_instance):
        return flow_instance.collect()

    # If pipe's return type hint is Flow, this would be a type violation for static analysis,
    # but runtime it will return what the func returns.
    result_list = Flow(sample_records_complex).pipe(returns_list)
    assert isinstance(result_list, list)


# --- Test explode, explode_multi ---
def test_explode_edge_cases(sample_records_complex, empty_flow):
    # Key not present
    exploded_key_missing = (
        Flow(sample_records_complex).explode("non_existent_tags").collect()
    )
    assert exploded_key_missing == sample_records_complex  # Yields record as-is

    # Value for key is not a list
    data_tag_not_list = [{"id": 1, "tags": "abc"}]
    assert Flow(data_tag_not_list).explode("tags").collect() == data_tag_not_list

    # Value for key is an empty list
    data_empty_tags = [{"id": 1, "tags": []}]

    assert Flow(data_empty_tags).explode("tags").collect() == []
    assert empty_flow.explode("tags").collect() == []


def test_explode_multi_edge_cases(empty_flow):
    data = [
        {"id": 1, "names": ["A", "B"], "scores": [100, 200, 300]},  # scores longer
        {"id": 2, "names": ["C", "D", "E"]},  # scores missing
        {"id": 3, "scores": [400]},  # names missing
    ]
    flow = Flow(data)
    # Default fillvalue=None
    exploded = flow.explode_multi(["names", "scores"]).collect()
    expected = [
        {"id": 1, "names": "A", "scores": 100},
        {"id": 1, "names": "B", "scores": 200},
        {"id": 1, "names": None, "scores": 300},  # names filled with None
        {"id": 2, "names": "C", "scores": None},  # scores filled with None
        {"id": 2, "names": "D", "scores": None},
        {"id": 2, "names": "E", "scores": None},
        {"id": 3, "names": None, "scores": 400},  # names filled with None
    ]
    assert exploded == expected

    # Custom fillvalue
    flow = Flow(data)
    exploded_fill = flow.explode_multi(
        ["names", "scores"], fillvalue="MISSING"
    ).collect()
    assert exploded_fill[2]["names"] == "MISSING"

    # Empty keys list
    with pytest.raises(ValueError):
        Flow(data).explode_multi([]).collect()


# --- Test sample, sample_frac ---
def test_sample_and_sample_frac_edge_cases(sample_records_complex, empty_flow):
    # sample
    assert len(Flow(sample_records_complex).sample(0).collect()) == 0
    assert len(Flow(sample_records_complex).sample(100).collect()) == len(
        sample_records_complex
    )

    sampled = Flow(sample_records_complex).sample(3, seed=42).collect()
    assert len(sampled) == 3
    sampled2 = Flow(sample_records_complex).sample(3, seed=42).collect()
    assert sampled == sampled2  # Reproducible with seed
    assert empty_flow.sample(5).collect() == []

    # sample_frac
    assert len(Flow(sample_records_complex).sample_frac(0.0).collect()) == 0
    all_selected = Flow(sample_records_complex).sample_frac(1.0).collect()
    assert all_selected == sample_records_complex
    # Approx fraction
    frac_sample = (
        Flow(sample_records_complex * 20).sample_frac(0.5, seed=123).collect()
    )  # Larger N
    assert (
        len(sample_records_complex * 20) * 0.2
        < len(frac_sample)
        < len(sample_records_complex * 20) * 0.7
    )

    assert empty_flow.sample_frac(0.5).collect() == []


# --- Test flatten ---
def test_flatten_edge_cases(empty_flow):
    assert empty_flow.flatten().collect() == []
    assert Flow([{}]).flatten().collect() == [{}]  # Empty record
    no_nest = [{"a": 1, "b": 2}]
    assert Flow(no_nest).flatten().collect() == no_nest  # No nested dicts

    # Multiple levels
    deep_nest = [{"a": {"b": {"c": 10}}, "d": 20}]
    assert Flow(deep_nest).flatten().collect() == [{"a.b.c": 10, "d": 20}]
    assert Flow(deep_nest).flatten(sep="_").collect() == [{"a_b_c": 10, "d": 20}]

    test_data = [{"a": {"b": 10}, "a.b": 20}]  # "a" then "a.b"
    assert Flow(test_data).flatten().collect() == [{"a.b": 20}]

    test_data_rev = [{"a.b": 20, "a": {"b": 10}}]  # "a.b" then "a"
    assert Flow(test_data_rev).flatten().collect() == [{"a.b": 10}]
