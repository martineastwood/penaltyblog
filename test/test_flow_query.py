import pytest

from penaltyblog.matchflow.flow import Flow

RECORDS = [
    {"id": 1, "age": 25, "name": "Alice", "score": 90.5},
    {"id": 2, "age": 30, "name": "Bob", "score": 85.0},
    {"id": 3, "age": 35, "name": "Charlie", "score": None},
    {"id": 4, "age": 40, "name": "Alice", "score": 95.0},
]


def make_flow():
    return Flow.from_records(RECORDS)


def test_query_equals():
    f = make_flow().query("name == 'Alice'")
    rows = f.collect()
    assert len(rows) == 2
    assert all(r["name"] == "Alice" for r in rows)


def test_query_greater_than():
    f = make_flow().query("age > 30")
    rows = f.collect()
    assert len(rows) == 2
    assert all(r["age"] > 30 for r in rows)


def test_query_combined_and():
    f = make_flow().query("age > 30 and name == 'Alice'")
    rows = f.collect()
    assert len(rows) == 1
    assert rows[0]["id"] == 4


def test_query_combined_or():
    f = make_flow().query("name == 'Bob' or score > 90")
    rows = f.collect()
    assert {r["id"] for r in rows} == {2, 1, 4}


def test_query_not():
    f = make_flow().query("not (name == 'Alice')")
    rows = f.collect()
    assert all(r["name"] != "Alice" for r in rows)


def test_query_in_list():
    f = make_flow().query("name in ['Alice', 'Charlie']")
    rows = f.collect()
    assert {r["name"] for r in rows} == {"Alice", "Charlie"}


def test_query_not_in():
    f = make_flow().query("name not in ['Bob']")
    rows = f.collect()
    assert all(r["name"] != "Bob" for r in rows)


def test_query_invalid_syntax():
    with pytest.raises(ValueError):
        make_flow().query("name === 'Alice'")


def test_query_unsupported_expr():
    with pytest.raises(ValueError):
        make_flow().query("len(name) > 3")


def test_query_chained_comparisons_disallowed():
    with pytest.raises(ValueError):
        make_flow().query("20 < age < 40")


def test_query_dot_notation():
    data = [
        {"player": {"name": "Saka"}, "age": 22},
        {"player": {"name": "Martinelli"}, "age": 21},
    ]
    f = Flow.from_records(data).query("player.name == 'Saka'")
    rows = f.collect()
    assert len(rows) == 1
    assert rows[0]["player"]["name"] == "Saka"


def test_query_contains():
    f = make_flow().query("name.contains('Ali')")
    rows = f.collect()
    assert all("Ali" in r["name"] for r in rows)


def test_query_is_null():
    f = make_flow().query("score is None")
    rows = f.collect()
    assert len(rows) == 1
    assert rows[0]["id"] == 3


def test_query_is_not_null():
    f = make_flow().query("score is not None")
    rows = f.collect()
    assert all(r["score"] is not None for r in rows)
