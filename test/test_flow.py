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


def test_materialize_basic(sample_records):
    flow = Flow(sample_records)
    # Materialize returns a new Flow backed by a list
    mat = flow.materialize()
    assert isinstance(mat.collect(), list)
    # The original flow is not exhausted
    assert flow.collect() == sample_records
    # The materialized flow is equal to the original
    assert mat.collect() == sample_records
    # Mutating the materialized flow does not affect the original
    mat_list = mat.collect()
    mat_list[0]["id"] = 999
    assert flow.collect()[0]["id"] == 1


def test_materialize_empty():
    flow = Flow([])
    mat = flow.materialize()
    assert mat.collect() == []
    assert flow.collect() == []


def test_materialize_single():
    record = {"id": 1}
    flow = Flow([record])
    mat = flow.materialize()
    assert mat.collect() == [record]
    assert flow.collect() == [record]


def test_fork_basic(sample_records):
    flow = Flow(sample_records)
    f1, f2 = flow.fork()
    # Both forks yield the same records
    assert f1.collect() == sample_records
    assert f2.collect() == sample_records
    # # Both are one-shot: after collecting, further iteration yields empty list
    assert f1.collect() == []
    assert f2.collect() == []


def test_fork_empty():
    flow = Flow([])
    f1, f2 = flow.fork()
    assert f1.collect() == []
    assert f2.collect() == []


def test_fork_single():
    record = {"id": 1}
    flow = Flow([record])
    f1, f2 = flow.fork()
    assert f1.collect() == [record]
    assert f2.collect() == [record]


def test_assign_does_not_mutate_original():
    original = [{"a": 1}]
    flow1 = Flow(original)
    flow1.assign(b=lambda r: 2)
    assert "b" not in original[0]


def test_init(sample_records):
    # 1. Flow.collect returns all records as-is
    assert Flow(sample_records).collect() == sample_records
    # 2. Flow.from_records returns all records as-is
    assert Flow.from_records(sample_records).collect() == sample_records
    # 3. Single record input returns single record
    assert Flow(sample_records[0]).collect() == [sample_records[0]]
    # 4. Single record with from_records
    assert Flow.from_records([sample_records[0]]).collect() == [sample_records[0]]

    # 5. Flow from generator yields correct records
    def gen():
        yield {"id": 1}
        yield {"id": 2}

    # 6. Flow from generator yields correct records
    assert Flow(gen()).collect() == [{"id": 1}, {"id": 2}]
    # 7. Flow.from_generator yields correct records
    assert Flow.from_generator(gen()).collect() == [{"id": 1}, {"id": 2}]

    # 8. TypeError if not iterable (int)
    with pytest.raises(TypeError):
        Flow.from_records(123)
    # 9. TypeError if not iterable (str)
    with pytest.raises(TypeError):
        Flow.from_records("123")

    # 10. Flow from sample_records again
    assert Flow(sample_records).collect() == sample_records
    # 11. Flow.from_records from sample_records again
    assert Flow.from_records(sample_records).collect() == sample_records
    # 12. Single record input returns single record again
    assert Flow(sample_records[0]).collect() == [sample_records[0]]
    # 13. Single record with from_records again
    assert Flow.from_records([sample_records[0]]).collect() == [sample_records[0]]

    # 14. Flow from generator yields correct records again
    def gen():
        yield {"id": 1}
        yield {"id": 2}

    # 15. Flow from generator yields correct records again
    assert Flow(gen()).collect() == [{"id": 1}, {"id": 2}]
    # 16. Flow.from_generator yields correct records again
    assert Flow.from_generator(gen()).collect() == [{"id": 1}, {"id": 2}]

    # 17. TypeError if not iterable (int) again
    with pytest.raises(TypeError):
        Flow.from_records(123)
    # 18. TypeError if not iterable (str) again
    with pytest.raises(TypeError):
        Flow.from_records("123")


def test_len(sample_records):
    # 1. Length of Flow is length of records
    assert len(Flow(sample_records)) == len(sample_records)

    # 2. Length after collect is unchanged
    flow = Flow(sample_records)
    len(flow)
    results = flow.collect()
    # 3. Length of collected results
    assert len(results) == len(sample_records)
    # 4. Collected results match input
    assert results == sample_records

    # 5. Length of Flow is length of records again
    assert len(Flow(sample_records)) == len(sample_records)

    # 6. Length after collect is unchanged again
    flow = Flow(sample_records)
    len(flow)
    results = flow.collect()
    # 7. Length of collected results again
    assert len(results) == len(sample_records)
    # 8. Collected results match input again
    assert results == sample_records


def test_iter(sample_records):
    # 1. Iterating counts all records
    flow = Flow(sample_records)
    n = 0
    for _ in flow:
        n += 1
    # 2. Iterating counts all records
    assert n == len(sample_records)
    # # 3. Iterating again yields original records
    assert list(flow) == sample_records

    # 4. Iterating counts all records again
    flow = Flow(sample_records)
    n = 0
    for _ in flow:
        n += 1
    # 5. Iterating counts all records again
    assert n == len(sample_records)
    # 6. Iterating again yields original records again
    assert list(flow) == sample_records


def test_eq(sample_records):
    # 1. Flow equals itself
    assert Flow(sample_records) == Flow(sample_records)
    # 2. Flow equals list of same records
    assert Flow(sample_records) == sample_records
    # 3. Flow not equal to different Flow
    assert Flow(sample_records) != Flow(sample_records[1:])
    # 4. Flow not equal to different list
    assert Flow(sample_records) != sample_records[1:]

    # 5. Flow equals itself again
    assert Flow(sample_records) == Flow(sample_records)
    # 6. Flow equals list of same records again
    assert Flow(sample_records) == sample_records
    # 7. Flow not equal to different Flow again
    assert Flow(sample_records) != Flow(sample_records[1:])
    # 8. Flow not equal to different list again
    assert Flow(sample_records) != sample_records[1:]


def test_collect(sample_records):
    # 1. Collect returns all records
    assert Flow(sample_records).collect() == sample_records
    # 2. Collect single record
    assert Flow(sample_records[0]).collect() == [sample_records[0]]

    # 3. Collect from generator
    def gen():
        yield {"id": 1}
        yield {"id": 2}

    # 4. Collect from generator
    assert Flow(gen()).collect() == [{"id": 1}, {"id": 2}]

    # 5. Collect returns all records again
    assert Flow(sample_records).collect() == sample_records
    # 6. Collect single record again
    assert Flow(sample_records[0]).collect() == [sample_records[0]]

    # 7. Collect from generator again
    def gen():
        yield {"id": 1}
        yield {"id": 2}

    # 8. Collect from generator again
    assert Flow(gen()).collect() == [{"id": 1}, {"id": 2}]


def test_selecting():
    data = [
        {
            "id": 1,
            "value": 10,
            "tags": ["a", "b"],
            "nested": {"x": 1, "y": {"z": 2}},
            "extra.field": "foo",
            "another.extra.field": {"that.is.nested": "foo"},
            "player.info": {"name.full": "Jane Doe"},
        }
    ]
    # 1. Select single and multiple fields
    assert Flow(data).select("id", "value").collect() == [{"id": 1, "value": 10}]
    # 2. Select nested field
    assert Flow(data).select("nested.x").collect() == [{"nested.x": 1}]
    assert Flow(data).select("nested.x", leaf_names=True).collect() == [{"x": 1}]
    # 3. Select dotted field
    assert Flow(data).select("extra.field").collect() == [{"extra.field": "foo"}]
    # 4. Select missing field
    assert Flow(data).select("does_not_exist").collect() == [{"does_not_exist": None}]
    # 5. Select deep nested
    assert Flow(data).select("nested.y.z").collect() == [{"nested.y.z": 2}]
    assert Flow(data).select("nested.y.z", leaf_names=True).collect() == [{"z": 2}]
    # 6. Select duplicate leaf names
    with pytest.warns(UserWarning):
        _ = Flow(data).select("foo.bar", "other.bar", leaf_names=True).collect()
    # 6. Flatten and select
    recs = Flow(data).flatten().select("another.extra.field.that.is.nested").collect()
    assert recs == [{"another.extra.field.that.is.nested": "foo"}]
    # 7. Rename and assign
    recs = (
        Flow(data)
        .rename(**{"player.info": "player_info"})
        .assign(name_full=lambda r: r["player_info"].get("name.full"))
        .select("name_full")
    )
    # 8. Rename and assign
    assert recs.collect() == [{"name_full": "Jane Doe"}]

    data = [
        {
            "id": 1,
            "value": 10,
            "tags": ["a", "b"],
            "nested": {"x": 1, "y": {"z": 2}},
            "extra.field": "foo",
            "another.extra.field": {"that.is.nested": "foo"},
            "player.info": {"name.full": "Jane Doe"},
        }
    ]

    # 9. Select single and multiple fields again
    assert Flow(data).select("id", "value").collect() == [{"id": 1, "value": 10}]
    # 10. Select nested field again
    assert Flow(data).select("nested.x").collect() == [{"nested.x": 1}]
    # 11. Select dotted field again
    assert Flow(data).select("extra.field").collect() == [{"extra.field": "foo"}]
    # 12. Select missing field again
    assert Flow(data).select("does_not_exist").collect() == [{"does_not_exist": None}]
    # 13. Select deep nested again
    assert Flow(data).select("nested.y.z").collect() == [{"nested.y.z": 2}]

    # 14. Flatten and select
    recs = Flow(data).flatten().select("another.extra.field.that.is.nested").collect()
    assert recs == [{"another.extra.field.that.is.nested": "foo"}]

    recs = (
        Flow(data)
        .rename(**{"player.info": "player_info"})
        .assign(name_full=lambda r: r["player_info"].get("name.full"))
        .select("name_full")
    )
    # 15. Rename and assign
    assert recs.collect() == [{"name_full": "Jane Doe"}]


def test_filter(sample_records):
    # 1. Filter by value
    out = Flow(sample_records).filter(lambda r: r["value"] == 10).collect()
    assert out == [sample_records[0], sample_records[2]]
    # 2. Filter with no matches
    out = Flow(sample_records).filter(lambda r: r["value"] > 1000).collect()
    assert out == []
    # 3. Filter with get and missing field
    out = (
        Flow(sample_records)
        .filter(lambda r: r.get("does_not_exist", -1) > 1000)
        .collect()
    )
    assert out == []
    # 4. Raise KeyError if field missing
    with pytest.raises(KeyError):
        Flow(sample_records).filter(lambda r: r["does_not_exist"] > 1000).collect()
    # 5. Filter with get and missing field
    out = (
        Flow(sample_records)
        .filter(lambda r: r.get("does_not_exist", -1) > 1000)
        .collect()
    )
    # 6. Filter with get and missing field
    assert out == []

    out = Flow(sample_records).filter(lambda r: r["value"] == 10).collect()
    # 7. Filter by value again
    assert out == [sample_records[0], sample_records[2]]

    out = Flow(sample_records).filter(lambda r: r["value"] > 1000).collect()
    # 8. Filter with no matches again
    assert out == []

    # 9. Raise KeyError if field missing again
    with pytest.raises(KeyError):
        Flow(sample_records).filter(lambda r: r["does_not_exist"] > 1000).collect()

    out = (
        Flow(sample_records)
        .filter(lambda r: r.get("does_not_exist", -1) > 1000)
        .collect()
    )
    # 10. Filter with get and missing field again
    assert out == []


def test_assign(sample_records):
    # 1. Assign new field
    out = (
        Flow(sample_records)
        .assign(double=lambda r: r["value"] * 2)
        .select("id", "double")
        .collect()
    )
    # 1. Assign new field
    assert out == [
        {"id": 1, "double": 20},
        {"id": 2, "double": 40},
        {"id": 3, "double": 20},
    ]
    # 2. Assign multiple fields
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
    # 3. Assign overwrites field
    out = Flow(sample_records).assign(id=lambda r: r["id"] * 2).select("id").collect()
    assert out == [
        {"id": 2},
        {"id": 4},
        {"id": 6},
    ]
    # 4. Assign on empty Flow
    assert Flow([]).assign(new_field=lambda r: 1).collect() == []
    # 5. Assign value from record
    recs = Flow(sample_records).assign(b=lambda r: r["id"]).collect()
    # 5. Assign value from record
    assert recs[0]["b"] == 1
    # 6. Assign sum of fields
    recs = Flow([{"a": 1, "b": 2}]).assign(c=lambda r: r["a"] + r["b"]).collect()
    # 6. Assign sum of fields
    assert recs == [{"a": 1, "b": 2, "c": 3}]
    # 7. Assign order matters
    recs = (
        Flow([{"a": 1, "b": 2}])
        .assign(a=lambda r: r["b"], b=lambda r: r["a"])
        .collect()
    )
    # 7. Assign order matters
    assert recs == [{"a": 2, "b": 2}]
    # 8. Assign constant value
    recs = Flow([{"a": 1, "b": 2}]).assign(c=lambda r: 1).collect()
    # 8. Assign constant value
    assert recs[0]["c"] == 1
    # 9. Assign with non-callable raises TypeError
    with pytest.raises(TypeError):
        Flow([{"a": 1, "b": 2}]).assign(c=1).collect()

    out = (
        Flow(sample_records)
        .assign(double=lambda r: r["value"] * 2)
        .select("id", "double")
        .collect()
    )
    # 1. Assign new field
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
    # 2. Assign multiple fields
    assert out == [
        {"double_id": 2, "double": 20},
        {"double_id": 4, "double": 40},
        {"double_id": 6, "double": 20},
    ]

    # 3. Assign overwrites field
    out = Flow(sample_records).assign(id=lambda r: r["id"] * 2).select("id").collect()
    assert out == [
        {"id": 2},
        {"id": 4},
        {"id": 6},
    ]
    # 4. Assign overwrites field
    assert Flow([]).assign(new_field=lambda r: 1).collect() == []

    recs = Flow(sample_records).assign(b=lambda r: r["id"]).collect()
    # 5. Assign overwrites field
    assert recs[0]["b"] == 1

    recs = Flow([{"a": 1, "b": 2}]).assign(c=lambda r: r["a"] + r["b"]).collect()
    # 6. Assign sum of fields
    assert recs == [{"a": 1, "b": 2, "c": 3}]

    recs = (
        Flow([{"a": 1, "b": 2}])
        .assign(a=lambda r: r["b"], b=lambda r: r["a"])
        .collect()
    )
    # 7. Assign order matters
    assert recs == [{"a": 2, "b": 2}]

    recs = Flow([{"a": 1, "b": 2}]).assign(c=lambda r: 1).collect()
    # 8. Assign constant value
    assert recs[0]["c"] == 1

    with pytest.raises(TypeError):
        Flow([{"a": 1, "b": 2}]).assign(c=1).collect()


def test_drop(sample_records):
    # 1. Drop a single field
    recs = Flow(sample_records).drop("id").collect()
    # 2. Drop multiple fields
    assert recs == [
        {"value": 10, "tags": ["a", "b"], "nested": {"x": 1, "y": {"z": 2}}},
        {"value": 20, "tags": ["b"], "nested": {"x": 3, "y": {"z": 4}}},
        {"value": 10, "tags": [], "nested": {"x": 5}},
    ]
    # 2. Drop multiple fields
    recs = Flow(sample_records).drop("id", "value").collect()
    assert recs == [
        {"tags": ["a", "b"], "nested": {"x": 1, "y": {"z": 2}}},
        {"tags": ["b"], "nested": {"x": 3, "y": {"z": 4}}},
        {"tags": [], "nested": {"x": 5}},
    ]
    # 3. Drop on empty Flow
    assert Flow([]).drop("id").collect() == []
    # 4. Drop missing field does nothing
    recs = Flow(sample_records).drop("does_not_exist").collect()
    # 5. Drop missing field does nothing
    assert recs == sample_records

    recs = Flow(sample_records).drop("id").collect()
    # 6. Drop a single field again
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
    # 7. Drop missing field does nothing again
    assert recs == sample_records


def test_sort_limit_head(sample_records):
    # 1. Sorting by a field missing in some records
    records = [
        {"id": 1, "value": 2},
        {"id": 2},
        {"id": 3, "value": 1},
    ]
    sorted_recs = Flow(records).sort("value").collect()
    # 1. Sorting by a field missing in some records
    assert [r["id"] for r in sorted_recs] == [3, 1, 2]  # missing value at end

    # 2. Sorting by a field with some None values
    records = [
        {"id": 1, "value": 2},
        {"id": 2, "value": None},
        {"id": 3, "value": 1},
    ]
    sorted_recs = Flow(records).sort("value").collect()
    # 2. Sorting by a field with some None values
    assert [r["id"] for r in sorted_recs] == [3, 1, 2]

    # 3. Sorting by a field with mixed types (should raise TypeError)
    records = [
        {"id": 1, "value": 2},
        {"id": 2, "value": "a"},
        {"id": 3, "value": 1},
    ]
    with pytest.raises(TypeError):
        Flow(records).sort("value").collect()

    # 4. Sorting by a field where all values are None
    records = [
        {"id": 1, "value": None},
        {"id": 2, "value": None},
    ]
    sorted_recs = Flow(records).sort("value").collect()
    # 4. Sorting by a field where all values are None
    assert [r["id"] for r in sorted_recs] == [1, 2]

    # 5. Sorting by a non-existent field (all missing)
    records = [
        {"id": 1},
        {"id": 2},
    ]
    sorted_recs = Flow(records).sort("nonexistent").collect()
    assert [r["id"] for r in sorted_recs] == [1, 2]

    # 6. Sorting with reverse=True and None/missing values
    records = [
        {"id": 1, "value": 2},
        {"id": 2},
        {"id": 3, "value": 1},
    ]
    sorted_recs = Flow(records).sort("value", reverse=True).collect()
    # 6. Sorting with reverse=True and None/missing values
    assert [r["id"] for r in sorted_recs] == [1, 3, 2]

    # 7. Sorting when field values are not comparable (should raise TypeError)
    records = [
        {"id": 1, "value": object()},
        {"id": 2, "value": 1},
    ]
    with pytest.raises(TypeError):
        Flow(records).sort("value").collect()

    # 8. Sorting a single-record Flow
    records = [{"id": 1, "value": 2}]
    sorted_recs = Flow(records).sort("value").collect()
    # 8. Sorting a single-record Flow
    assert sorted_recs == records

    # 9. Sorting by a field with duplicate values (stability)
    records = [
        {"id": 1, "value": 2},
        {"id": 2, "value": 1},
        {"id": 3, "value": 2},
    ]
    sorted_recs = Flow(records).sort("value").collect()
    # ids with value=2 should preserve order (1, then 3)
    # 9. Sorting by a field with duplicate values (stability)
    assert [r["id"] for r in sorted_recs] == [2, 1, 3]

    top2 = Flow(sample_records).sort("value", reverse=True).limit(2).collect()
    # 10. Limit after sorting in descending order
    assert [r["id"] for r in top2] == [2, 1]

    top2 = Flow(sample_records).sort("value", reverse=False).limit(2).collect()
    # 11. Limit after sorting in ascending order
    assert [r["id"] for r in top2] == [1, 3]

    top2 = Flow(sample_records).sort("value", reverse=True).head(2).collect()
    # 12. Head after sorting in descending order
    assert [r["id"] for r in top2] == [2, 1]

    assert Flow(sample_records).head(1).collect() == [sample_records[0]]
    # 13. Sort on empty Flow
    assert Flow([]).sort("id").collect() == []

    with pytest.raises(ValueError):
        Flow(sample_records).head(-1)

    with pytest.raises(ValueError):
        Flow(sample_records).limit(-1)


def test_sort_multiple_fields():
    records = [
        {"id": 1, "a": 2, "b": 3},
        {"id": 2, "a": 1, "b": 2},
        {"id": 3, "a": 2, "b": 2},
        {"id": 4, "a": 1, "b": 3},
        {"id": 5, "a": None, "b": 1},
        {"id": 6, "a": 2, "b": None},
        {"id": 7, "a": None, "b": None},
    ]
    # Sort by a, then b
    sorted_recs = Flow(records).sort(["a", "b"]).collect()
    # ids: 2 (a=1,b=2), 4 (a=1,b=3), 3 (a=2,b=2), 1 (a=2,b=3), 5 (a=None,b=1), 6 (a=2,b=None), 7 (a=None,b=None)
    assert [r["id"] for r in sorted_recs] == [2, 4, 3, 1, 5, 6, 7]

    # Sort by b, then a
    sorted_recs = Flow(records).sort(["b", "a"]).collect()
    # ids: 2 (b=2,a=1), 3 (b=2,a=2), 4 (b=3,a=1), 1 (b=3,a=2), 5 (b=1,a=None), 6 (b=None,a=2), 7 (b=None,a=None)
    assert [r["id"] for r in sorted_recs] == [2, 3, 4, 1, 5, 6, 7]

    # Sort by a, then b, reverse
    sorted_recs = Flow(records).sort(["a", "b"], reverse=True).collect()
    # ids: 1 (a=2,b=3), 3 (a=2,b=2), 4 (a=1,b=3), 2 (a=1,b=2), 5 (a=None,b=1), 6 (a=2,b=None), 7 (a=None,b=None)
    assert [r["id"] for r in sorted_recs] == [1, 3, 4, 2, 5, 6, 7]

    # Sort by a single field (should match previous tests)
    sorted_recs = Flow(records).sort("a").collect()
    # ids: 2, 4, 1, 3, 6, 5, 7 (by a, then original order for ties)
    assert [r["id"] for r in sorted_recs if r["a"] is not None] == [2, 4, 1, 3, 6]
    assert [r["id"] for r in sorted_recs if r["a"] is None] == [5, 7]

    # Stability: records with same a and b keep original order
    records_dup = [
        {"id": 1, "a": 1, "b": 2},
        {"id": 2, "a": 1, "b": 2},
        {"id": 3, "a": 1, "b": 2},
    ]
    sorted_recs = Flow(records_dup).sort(["a", "b"]).collect()
    assert [r["id"] for r in sorted_recs] == [1, 2, 3]

    # All None in one field
    records_none = [
        {"id": 1, "a": None, "b": 2},
        {"id": 2, "a": None, "b": 1},
    ]
    sorted_recs = Flow(records_none).sort(["a", "b"]).collect()
    assert [r["id"] for r in sorted_recs] == [1, 2]


@pytest.mark.filterwarnings(
    "ignore:'tags' has only.*elements but expected.*:UserWarning"
)
def test_split_array(sample_records):
    # 1. 'into' is not a list (should raise ValueError)
    with pytest.warns(UserWarning):
        Flow(sample_records).split_array("tags", into="notalist").collect()

    # 2. input list is shorter than 'into'
    recs = Flow([{"tags": ["a"]}]).split_array("tags", into=["t0", "t1"]).collect()
    # 2. Input list is shorter than 'into'
    assert recs[0]["t0"] == "a" and recs[0]["t1"] is None

    # 3. array field is None
    recs = Flow([{"tags": None}]).split_array("tags", into=["t0", "t1"]).collect()
    # 4. Array field is None or empty: keys may not exist, so check accordingly
    assert ("t0" not in recs[0] or recs[0]["t0"] is None) and (
        "t1" not in recs[0] or recs[0]["t1"] is None
    )

    # 5. array field is an empty list
    recs = Flow([{"tags": []}]).split_array("tags", into=["t0", "t1"]).collect()
    assert recs[0]["t0"] is None and recs[0]["t1"] is None

    # 6. original record has extra fields
    recs = (
        Flow([{"tags": ["a", "b"], "extra": 123}])
        .split_array("tags", into=["t0", "t1"])
        .collect()
    )
    # 6. Original record has extra fields
    assert recs[0]["extra"] == 123

    # 7. 'into' has duplicate names
    recs = Flow([{"tags": ["a", "b"]}]).split_array("tags", into=["x", "x"]).collect()
    # The last one should overwrite
    # 7. 'Into' has duplicate names
    assert recs[0]["x"] == "b"

    # 8. array field with mixed types
    recs = (
        Flow([{"tags": [1, "b", None]}])
        .split_array("tags", into=["t0", "t1", "t2"])
        .collect()
    )
    # 8. Array field with mixed types
    assert recs[0]["t0"] == 1 and recs[0]["t1"] == "b" and recs[0]["t2"] is None

    recs = Flow(sample_records).split_array("tags", into=["t0", "t1"]).collect()
    # 9. Split array into fields
    assert recs[0]["t0"] == "a" and recs[0]["t1"] == "b"

    with pytest.raises(TypeError):
        Flow(sample_records).split_array("tags").collect()

    recs = (
        Flow(sample_records).split_array("does_not_exist", into=["t0", "t1"]).collect()
    )
    assert ("t0" not in recs[0] or recs[0]["t0"] is None) and (
        "t1" not in recs[0] or recs[0]["t1"] is None
    )

    recs = (
        Flow([{"id": 1, "tags": "not_a_list"}])
        .split_array("tags", into=["t0"])
        .collect()
    )
    # 10. Split array with non-list field
    assert "t0" not in recs[0] or recs[0]["t0"] is None

    recs = (
        Flow([{"id": 1, "tags": ["a", "b", "c"]}])
        .split_array("tags", into=["t0", "t1"])
        .collect()
    )
    # 11. Split array with more elements than 'into'
    assert recs[0]["t0"] == "a" and recs[0]["t1"] == "b"
    assert "t2" not in recs[0]

    recs = Flow(sample_records).split_array("tags", into=[]).collect()
    # 12. Split array with empty 'into'
    assert recs == sample_records
    assert Flow([]).split_array("tags", into=["t0"]).collect() == []


def test_explode(sample_records):
    recs = Flow(sample_records).explode("tags").collect()
    # 1. Explode list field into multiple records
    assert all(isinstance(r["tags"], str) for r in recs if r["tags"] != [])

    recs = Flow(sample_records).explode("non_existent_tags").collect()
    assert recs == sample_records

    recs = Flow(sample_records).explode("tags").collect()
    # 3. Explode list field and check length
    assert len(recs) == 4

    data = [{"id": 1, "tags": "abc"}]
    # 4. Explode non-list field
    assert Flow(data).explode("tags").collect() == data

    # 5. Explode on empty Flow
    assert Flow([]).explode("tags").collect() == []


def test_group_by_summary_scalar_enforcement(sample_records):
    import pytest

    from penaltyblog.matchflow.flow import Flow

    # Valid custom aggregate: returns scalar
    def scalar_agg(records):
        return 42

    result = Flow(sample_records).group_by("value").summary(x=scalar_agg).collect()
    assert all(row["x"] == 42 for row in result)

    # Invalid custom aggregate: returns list
    def list_agg(records):
        return [1, 2]

    with pytest.raises(ValueError, match="non-scalar"):
        Flow(sample_records).group_by("value").summary(x=list_agg).collect()

    # Invalid custom aggregate: returns tuple
    def tuple_agg(records):
        return (1, 2)

    with pytest.raises(ValueError, match="non-scalar"):
        Flow(sample_records).group_by("value").summary(x=tuple_agg).collect()

    # Invalid custom aggregate: returns dict
    def dict_agg(records):
        return {"a": 1}

    with pytest.raises(ValueError, match="non-scalar"):
        Flow(sample_records).group_by("value").summary(x=dict_agg).collect()

    # Invalid custom aggregate: returns set
    def set_agg(records):
        return {1, 2}

    with pytest.raises(ValueError, match="non-scalar"):
        Flow(sample_records).group_by("value").summary(x=set_agg).collect()


def test_group_by_summary_ungroup(sample_records):
    recs = (
        Flow(sample_records).group_by("value").summary(count=("id", "count")).collect()
    )
    # 1. Group by value and summarize count
    # 2. Group by value and summarize count with shorthand
    assert {row["value"]: row["count"] for row in recs} == {10: 2, 20: 1}

    recs = Flow(sample_records).group_by("value").summary(count="count").collect()
    # 3. Group by value and summarize count with shorthand
    assert {row["value"]: row["count"] for row in recs} == {10: 2, 20: 1}

    recs = (
        Flow(sample_records)
        .group_by("id")
        .summary(count="count", max_value=("value", "max"))
        .collect()
    )
    # 4. Group by id and summarize count and max value
    assert [x["max_value"] for x in recs] == [10, 20, 10]

    with pytest.raises(AttributeError):
        Flow(sample_records).ungroup().collect()

    recs = Flow(sample_records).group_by("value").ungroup().collect()
    # 5. Ungroup after grouping by value
    assert len(recs) == len(sample_records)

    recs = (
        Flow(sample_records)
        .group_by("id")
        .summary(count="count", max_value=("value", "max"))
        .select("id")
        .collect()
    )

    assert len(recs) == len(sample_records)
    # 6. Check all ids are present after select
    assert all(r["id"] is not None for r in recs)
    # 7. Check max_value is not present after select
    assert all(r.get("max_value") is None for r in recs)


def test_row_number_drop_duplicates_take_last(sample_records):
    rn = Flow(sample_records).row_number("value", new_field="rn").collect()
    # for value=10 you'd get rn=1,2 (in insertion order)
    ids_for_10 = [r["id"] for r in rn if r["value"] == 10]
    # 1. Row number by value
    assert set(ids_for_10) == {1, 3}
    # drop duplicates by value
    unique = Flow([{"a": 1}, {"a": 1}, {"a": 2}]).drop_duplicates("a").collect()
    # 2. Drop duplicates by field
    assert unique == [{"a": 1}, {"a": 2}]
    # take_last
    last1 = Flow([{"x": i} for i in range(5)]).take_last(1).collect()
    # 3. Take last n records
    assert last1 == [{"x": 4}]


def test_concat():
    # 1. Concatenate two empty Flows
    assert Flow([]).concat(Flow([])).collect() == []
    # 2. Concatenate two non-empty Flows
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
    # 1. Inner join with matching keys
    assert all(r.get("foo") == "bar" for r in joined)
    # 2. Check length of joined records
    assert len(joined) == 2

    joined = (
        Flow(sample_records)
        .join(right, left_on="value", right_on="key", fields=["foo"], how="left")
        .collect()
    )
    # 3. Left join with matching keys
    assert any(r.get("foo") == "bar" for r in joined)
    # 4. Check length of joined records
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
    # 5. Inner join with no matching keys
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
    # 6. Inner join with non-existent keys
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
    # 7. Inner join with non-existent keys on both sides
    assert len(joined) == 0


def test_first_last_is_empty_keys(sample_records):
    flow = Flow(sample_records)
    # 1. First record by id
    assert flow.first()["id"] == 1
    # 2. Last record by id
    assert flow.last()["id"] == 3
    # 3. Check if empty Flow is empty
    assert Flow([]).is_empty() is True
    # 4. Check if non-empty Flow is not empty
    assert Flow(sample_records).is_empty() is False
    # keys union
    ks = flow.keys(limit=1)
    # 5. Check keys in Flow
    assert "id" in ks and "value" in ks


def test_flatten_and_to_pandas(sample_records):
    flow = Flow(sample_records)
    flat = flow.flatten().collect()
    # nested.x should become top‐level
    # 1. Flatten nested fields
    assert all("nested.x" in r for r in flat)
    df = Flow(sample_records).to_pandas()
    # 2. Convert to pandas DataFrame
    assert isinstance(df, pd.DataFrame)
    # 3. Check DataFrame columns
    assert set(df.columns) >= {"id", "value", "tags", "nested"}


def test_json_file_roundtrip(tmp_path, sample_records):
    folder = tmp_path / "out"
    Flow(sample_records).to_json_files(folder, by="id")
    # ensure files exist
    files = sorted(folder.iterdir())
    # 1. Check JSON files exist
    assert any(f.name == "1.json" for f in files)
    # test jsonl
    jl = tmp_path / "data.jsonl"
    Flow(sample_records).to_jsonl(jl)
    reloaded = Flow.from_jsonl(jl).collect()
    # 2. Reload from JSONL and check records
    assert reloaded == sample_records
    # test single‐json
    js = tmp_path / "data.json"
    Flow(sample_records).to_json_single(js)
    re2 = Flow.from_file(js).collect()
    # 3. Reload from single JSON and check records
    assert re2 == sample_records


def test_from_folder_and_glob(tmp_path, sample_records):
    # write two json files
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "b.json"
    f1.write_text(json.dumps(sample_records[0]))
    f2.write_text(json.dumps([sample_records[1], sample_records[2]]))
    all_recs = Flow.from_folder(tmp_path).collect()
    # 1. Load from folder and check record count
    assert len(all_recs) == 3

    # glob
    all2 = Flow.from_glob(str(tmp_path / "*.json")).collect()
    # 2. Load from glob pattern and check record count
    assert len(all2) == 3


def test_describe(sample_records):
    df = Flow(sample_records).describe()
    unique_values = {r["value"] for r in sample_records}
    # 1. Check unique values in description
    assert len(unique_values) == 2
    # 2. Check '25%' percentile in description
    assert "25%" in df.index


def test_pipe(sample_records):
    def custom_filter_and_limit(flow_instance, val_gt, num):
        return flow_instance.filter(lambda r: r.get("value", 0) == val_gt).limit(num)

    recs = Flow(sample_records).pipe(custom_filter_and_limit, 10, 10)
    # 1. Pipe custom filter and limit
    assert len(recs.collect()) == 2

    def returns_list(flow):
        return flow.collect()

    recs = Flow(sample_records).pipe(returns_list)
    # 2. Pipe function returning list
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
    # 1. Explode multiple fields with different lengths
    assert recs == expected

    # Custom fillvalue
    recs = Flow(data).explode_multi(["names", "scores"], fillvalue="MISSING").collect()
    # 2. Explode with custom fillvalue
    assert recs[2]["names"] == "MISSING"

    # Empty keys list
    with pytest.raises(ValueError):
        Flow(data).explode_multi([]).collect()

    # missing key from all records
    recs = Flow(data).explode_multi(["does_not_exist"]).collect()
    # 4. Explode with missing key from all records
    assert recs == data


def test_sample_and_sample_frac_edge_cases(sample_records):
    # sample
    # 1. Sample with zero count
    assert len(Flow(sample_records).sample(0).collect()) == 0
    # 2. Sample with count greater than available
    assert len(Flow(sample_records).sample(100).collect()) == len(sample_records)

    sampled = Flow(sample_records).sample(3, seed=42).collect()
    # 3. Sample with seed for reproducibility
    assert len(sampled) == 3
    sampled2 = Flow(sample_records).sample(3, seed=42).collect()
    # 4. Check reproducibility with seed
    assert sampled == sampled2
    # 5. Sample from empty Flow
    assert Flow([]).sample(5).collect() == []

    # sample_frac
    # 6. Sample fraction with zero fraction
    assert len(Flow(sample_records).sample_frac(0.0).collect()) == 0
    all_selected = Flow(sample_records).sample_frac(1.0).collect()
    # 7. Sample fraction with full fraction
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

    # 9. Sample fraction from empty Flow
    assert Flow([]).sample_frac(0.5).collect() == []


def test_flatten_edge_cases():
    # 1. Flatten empty Flow
    assert Flow([]).flatten().collect() == []
    # 2. Flatten Flow with empty dict
    assert Flow([{}]).flatten().collect() == [{}]
    no_nest = [{"a": 1, "b": 2}]
    # 3. Flatten Flow with no nested fields
    assert Flow(no_nest).flatten().collect() == no_nest

    # Multiple levels
    deep_nest = [{"a": {"b": {"c": 10, "d": 15}}, "e": 20}]
    # 4. Flatten multiple levels of nesting
    assert Flow(deep_nest).flatten().collect() == [{"a.b.c": 10, "a.b.d": 15, "e": 20}]
    # 5. Flatten with custom separator
    assert Flow(deep_nest).flatten(sep="_").collect() == [
        {"a_b_c": 10, "a_b_d": 15, "e": 20}
    ]
    test_data = [{"a": {"b": 10}, "a.b": 20}]
    # 6. Flatten with conflicting keys
    assert Flow(test_data).flatten().collect() == [{"a.b": 20}]

    test_data_rev = [{"a.b": 20, "a": {"b": 10}}]
    # 7. Flatten with reversed conflicting keys
    assert Flow(test_data_rev).flatten().collect() == [{"a.b": 10}]
