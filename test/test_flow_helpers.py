from penaltyblog.matchflow import helpers
from penaltyblog.matchflow.steps.utils import fast_get_field, get_index


# --- Tests for Accessors ---
def test_get_field_flat():
    f = helpers.get_field("name")
    d = {"name": "Bukayo Saka"}
    assert f(d) == "Bukayo Saka"


def test_get_index_flat():
    f = helpers.get_index("end_location", 1)
    d = {"end_location": [100, 40]}
    assert f(d) == 40


def test_get_field_basic():
    f = helpers.get_field("player.name")
    d = {"player": {"name": "Bukayo Saka"}}
    assert f(d) == "Bukayo Saka"


def test_get_field_missing():
    f = helpers.get_field("player.age", default=21)
    d = {"player": {}}
    assert f(d) == 21


def test_get_field_non_dict():
    f = helpers.get_field("player.name", default="Unknown")
    d = {"player": None}
    assert f(d) == "Unknown"


def test_get_index_basic():
    f = helpers.get_index("pass.end_location", 0)
    d = {"pass": {"end_location": [100, 40]}}
    assert f(d) == 100


def test_get_index_out_of_bounds():
    f = helpers.get_index("pass.end_location", 2, default=-1)
    d = {"pass": {"end_location": [100, 40]}}
    assert f(d) == -1


def test_get_index_non_list():
    f = helpers.get_index("pass.end_location", 0, default=None)
    d = {"pass": {"end_location": None}}
    assert f(d) is None


# --- Tests for Predicates ---
def test_where_equals_flat():
    pred = helpers.where_equals("name", "Bukayo Saka")
    d = {"name": "Bukayo Saka"}
    assert pred(d)


def test_where_in_flat():
    pred = helpers.where_in("name", ["Bukayo Saka", "Gabriel"])
    d = {"name": "Gabriel"}
    assert pred(d)


def test_where_exists_flat():
    pred = helpers.where_exists("name")
    d = {"name": "Bukayo Saka"}
    assert pred(d)


def test_where_not_none_flat():
    pred = helpers.where_not_none("name")
    d = {"name": None}
    assert not pred(d)


def test_where_equals_true():
    pred = helpers.where_equals("player.name", "Bukayo Saka")
    d = {"player": {"name": "Bukayo Saka"}}
    assert pred(d)


def test_where_equals_false():
    pred = helpers.where_equals("player.name", "Bukayo Saka")
    d = {"player": {"name": "Gabriel"}}
    assert not pred(d)


def test_where_in_true():
    pred = helpers.where_in("player.name", ["Bukayo Saka", "Gabriel"])
    d = {"player": {"name": "Gabriel"}}
    assert pred(d)


def test_where_in_false():
    pred = helpers.where_in("player.name", ["Bukayo Saka", "Gabriel"])
    d = {"player": {"name": "Zinchenko"}}
    assert not pred(d)


def test_where_exists_true():
    pred = helpers.where_exists("player.name")
    d = {"player": {"name": "Bukayo Saka"}}
    assert pred(d)


def test_where_exists_false():
    pred = helpers.where_exists("player.name")
    d = {"player": {}}
    assert not pred(d)


def test_where_not_none_true():
    pred = helpers.where_not_none("player.name")
    d = {"player": {"name": "Bukayo Saka"}}
    assert pred(d)


def test_where_not_none_false():
    pred = helpers.where_not_none("player.name")
    d = {"player": {}}
    assert not pred(d)


# --- Tests for Transformers ---
def test_combine_fields_flat():
    f = helpers.combine_fields("full_name", "first_name", "last_name")
    d = {"first_name": "Bukayo", "last_name": "Saka"}
    assert f(d) == {"full_name": "Bukayo Saka"}


def test_coalesce_flat():
    f = helpers.coalesce("nickname", "name", default="Unknown")
    d = {"name": "Bukayo"}
    assert f(d) == "Bukayo"


def test_combine_fields_basic():
    f = helpers.combine_fields("full_name", "first_name", "last_name")
    d = {"first_name": "Bukayo", "last_name": "Saka"}
    assert f(d) == {"full_name": "Bukayo Saka"}


def test_combine_fields_with_join():
    f = helpers.combine_fields("coords", "x", "y", join_str=",")
    d = {"x": 10, "y": 20}
    assert f(d) == {"coords": "10,20"}


def test_combine_fields_missing():
    f = helpers.combine_fields("full_name", "first_name", "last_name")
    d = {"first_name": "Bukayo"}
    assert f(d) == {"full_name": "Bukayo "}


def test_coalesce_first():
    f = helpers.coalesce("player.name", "player.alias", default="Unknown")
    d = {"player": {"name": "Bukayo Saka", "alias": "BS7"}}
    assert f(d) == "Bukayo Saka"


def test_coalesce_second():
    f = helpers.coalesce("player.name", "player.alias", default="Unknown")
    d = {"player": {"alias": "BS7"}}
    assert f(d) == "BS7"


def test_coalesce_default():
    f = helpers.coalesce("player.name", "player.alias", default="Unknown")
    d = {"player": {}}
    assert f(d) == "Unknown"


def test_fast_get_field_simple_dict():
    record = {"a": {"b": {"c": 42}}}
    assert fast_get_field(record, ["a", "b", "c"]) == 42


def test_fast_get_field_missing_key():
    record = {"a": {"b": {}}}
    assert fast_get_field(record, ["a", "b", "x"]) is None


def test_fast_get_field_list_index():
    record = {"coords": [10, 20, 30]}
    assert fast_get_field(record, ["coords", "0"]) == 10
    assert fast_get_field(record, ["coords", "1"]) == 20
    assert fast_get_field(record, ["coords", "2"]) == 30


def test_fast_get_field_list_index_out_of_bounds():
    record = {"coords": [10, 20]}
    assert fast_get_field(record, ["coords", "5"]) is None


def test_fast_get_field_nested_list():
    record = {"frames": [{"player": "A"}, {"player": "B"}]}
    assert fast_get_field(record, ["frames", "0", "player"]) == "A"
    assert fast_get_field(record, ["frames", "1", "player"]) == "B"


def test_fast_get_field_not_a_dict_or_list():
    record = {"x": 5}
    assert fast_get_field(record, ["x", "y"]) is None


def test_fast_get_field_list_then_dict():
    record = {"players": [{"name": "Alice"}, {"name": "Bob"}]}
    assert fast_get_field(record, ["players", "1", "name"]) == "Bob"


def test_fast_get_field_invalid_type_for_index():
    record = {"coords": "notalist"}
    assert fast_get_field(record, ["coords", "0"]) is None


def test_get_index_basic():
    f = get_index("location", 0)
    assert f({"location": [10, 20]}) == 10


def test_get_index_out_of_bounds():
    f = get_index("location", 5)
    assert f({"location": [10, 20]}) is None


def test_get_index_not_a_list():
    f = get_index("location", 0)
    assert f({"location": "notalist"}) is None


def test_get_index_missing_field():
    f = get_index("location", 0)
    assert f({}) is None


def test_get_index_nested_path():
    f = get_index("pass.end_location", 1)
    assert f({"pass": {"end_location": [30, 40]}}) == 40
