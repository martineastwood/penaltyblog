from datetime import date, datetime

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
    with pytest.raises(ValueError, match="Unsupported field expression"):
        make_flow().query("unsupported_func(name) > 3").collect()


def test_query_len_string():
    f = make_flow().query("len(name) > 3")
    rows = f.collect()
    assert {r["name"] for r in rows} == {"Alice", "Charlie"}


def test_query_len_list():
    data = [
        {"id": 1, "tags": ["A", "B"]},
        {"id": 2, "tags": ["A", "B", "C"]},
        {"id": 3, "tags": ["A"]},
    ]
    f = Flow.from_records(data).query("len(tags) >= 2")
    rows = f.collect()
    assert {r["id"] for r in rows} == {1, 2}


def test_query_len_on_non_len_field():
    f = make_flow().query("len(age) > 3")
    assert f.is_empty()


def test_query_chained_comparisons():
    f = make_flow().query("20 < age < 40")
    rows = f.collect()
    assert len(rows) == 3
    assert {r["id"] for r in rows} == {1, 2, 3}


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
    f = make_flow().query("name.contains('li')")
    rows = f.collect()
    assert {r["name"] for r in rows} == {"Alice", "Charlie"}


def test_query_startswith():
    f = make_flow().query("name.startswith('A')")
    rows = f.collect()
    assert {r["name"] for r in rows} == {"Alice"}


def test_query_endswith():
    f = make_flow().query("name.endswith('e')")
    rows = f.collect()
    assert {r["name"] for r in rows} == {"Alice", "Charlie"}


def test_query_lower_equals():
    f = make_flow().query("name.lower() == 'alice'")
    rows = f.collect()
    assert len(rows) == 2
    assert all(r["name"] == "Alice" for r in rows)


def test_query_upper_not_equals():
    f = make_flow().query("name.upper() != 'BOB'")
    rows = f.collect()
    assert len(rows) == 3
    assert all(r["name"] != "Bob" for r in rows)


def test_query_lower_in_list():
    f = make_flow().query("name.lower() in ['alice', 'charlie']")
    rows = f.collect()
    assert {r["name"] for r in rows} == {"Alice", "Charlie"}


def test_query_upper_not_in_list():
    f = make_flow().query("name.upper() not in ['ALICE', 'CHARLIE']")
    rows = f.collect()
    assert {r["name"] for r in rows} == {"Bob"}


def test_query_case_transform_on_non_string():
    # This should not raise an error, but simply not match
    f = make_flow().query("age.lower() == 'something'")
    assert f.is_empty()


def test_query_case_transform_unsupported_comparison():
    with pytest.raises(ValueError, match="Unsupported field expression"):
        make_flow().query("name.lower() is 'alice'").collect()


def test_query_case_transform_as_predicate():
    with pytest.raises(ValueError, match="cannot be used as a predicate"):
        make_flow().query("name.lower()").collect()


def test_query_is_null():
    f = make_flow().query("score is None")
    rows = f.collect()
    assert len(rows) == 1
    assert rows[0]["id"] == 3


def test_query_is_not_null():
    f = make_flow().query("score is not None")
    rows = f.collect()
    assert all(r["score"] is not None for r in rows)


# === DATE FILTERING TESTS ===

DATE_RECORDS = [
    {"id": 1, "event": "Event A", "date": datetime(2024, 1, 15), "value": 100},
    {"id": 2, "event": "Event B", "date": datetime(2024, 6, 20), "value": 200},
    {"id": 3, "event": "Event C", "date": datetime(2024, 12, 5), "value": 300},
    {"id": 4, "event": "Event D", "date": date(2024, 3, 10), "value": 150},
    {"id": 5, "event": "Event E", "date": None, "value": 50},
]


def make_date_flow():
    return Flow.from_records(DATE_RECORDS)


def test_query_datetime_literal_greater_than():
    """Test filtering with datetime() literal"""
    f = make_date_flow().query("date > datetime(2024, 6, 1)")
    rows = f.collect()
    assert len(rows) == 2
    assert {r["id"] for r in rows} == {2, 3}


def test_query_date_literal_greater_than():
    """Test filtering with date() literal"""
    f = make_date_flow().query("date > date(2024, 3, 1)")
    rows = f.collect()
    assert len(rows) == 3
    assert {r["id"] for r in rows} == {2, 3, 4}


def test_query_date_variable():
    """Test filtering with date variable using @syntax"""
    cutoff_date = datetime(2024, 5, 1)
    # Call query directly in this scope where cutoff_date is defined
    rows = make_date_flow().query("date >= @cutoff_date").collect()
    assert len(rows) == 2
    assert {r["id"] for r in rows} == {2, 3}


def test_query_date_and_numeric_combined():
    """Test combining date and numeric filters"""
    f = make_date_flow().query("date > datetime(2024, 1, 1) and value > 150")
    rows = f.collect()
    assert len(rows) == 2
    assert {r["id"] for r in rows} == {2, 3}


def test_query_date_null_handling():
    """Test that null dates are handled properly"""
    f = make_date_flow().query("date is not None")
    rows = f.collect()
    assert len(rows) == 4
    assert all(r["date"] is not None for r in rows)


def test_query_date_less_than():
    """Test date less than comparison"""
    f = make_date_flow().query("date < datetime(2024, 6, 1)")
    rows = f.collect()
    assert len(rows) == 2
    assert {r["id"] for r in rows} == {1, 4}


def test_query_date_equals():
    """Test exact date matching"""
    f = make_date_flow().query("date == date(2024, 3, 10)")
    rows = f.collect()
    assert len(rows) == 1
    assert rows[0]["id"] == 4


def test_query_datetime_vs_date_comparison():
    """Test that datetime and date objects can be compared"""
    f = make_date_flow().query("date >= date(2024, 6, 1)")
    rows = f.collect()
    assert len(rows) == 2
    assert {r["id"] for r in rows} == {2, 3}


def test_query_date_type_mismatch_error():
    """Test that incompatible type comparisons raise helpful errors"""
    with pytest.raises(TypeError, match="Cannot compare.*date.*with.*str"):
        make_date_flow().query("date > 'not-a-date'").collect()


# === REGEX TESTS ===


def test_query_regex_basic():
    """Test basic regex matching"""
    f = make_flow().query("name.regex('li')")  # Contains 'li'
    rows = f.collect()
    assert {r["name"] for r in rows} == {"Alice", "Charlie"}


def test_query_regex_pattern():
    """Test regex pattern matching"""
    f = make_flow().query("name.regex('^A.*e$')")  # Starts with A, ends with e
    rows = f.collect()
    assert {r["name"] for r in rows} == {"Alice"}


def test_query_regex_match_alias():
    """Test .match() as an alias for .regex()"""
    f = make_flow().query("name.match('B.*')")  # Starts with B
    rows = f.collect()
    assert {r["name"] for r in rows} == {"Bob"}


def test_query_regex_with_flags():
    """Test regex with flags (case insensitive)"""
    # Using 2 which is the value of re.IGNORECASE
    f = make_flow().query(
        "name.regex('alice', 2)"
    )  # Case insensitive (2 = re.IGNORECASE)
    rows = f.collect()
    assert {r["name"] for r in rows} == {"Alice"}


def test_query_regex_on_non_string():
    """Test regex on non-string field (should convert to string)"""
    f = make_flow().query("age.regex('3')")  # Age contains digit 3
    rows = f.collect()
    assert {r["age"] for r in rows} == {30, 35}


def test_query_regex_on_null():
    """Test regex on null value (should not match)"""
    f = make_flow().query("score.regex('.*')")  # Any pattern on score
    rows = f.collect()
    assert len(rows) == 3  # Should not match the record with score=None
    assert all(r["score"] is not None for r in rows)


def test_query_regex_invalid_pattern():
    """Test invalid regex pattern raises ValueError"""
    with pytest.raises(ValueError, match="Invalid regex pattern"):
        make_flow().query("name.regex('[')").collect()  # Unbalanced bracket


def test_query_regex_no_args():
    """Test regex with no arguments raises ValueError"""
    with pytest.raises(ValueError, match="requires at least one argument"):
        make_flow().query("name.regex()").collect()
