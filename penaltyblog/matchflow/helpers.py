"""
Helpers for handling a streaming data pipeline, specifically the Flow class.
"""

__all__ = ["get_field", "resolve_path", "get_index"]

from typing import Any, Callable, Optional

from tabulate import tabulate

from .steps.utils import flatten_dict

# --- Accessors ---


def get_field(
    path: str, default: Any = None
) -> Callable[[Optional[dict[Any, Any]]], Any]:
    """
    Safely access a nested field using dot notation.

    Args:
        path (str): Dot-separated path to a field, e.g. "player.name".
        default (Any, optional): Value to return if the path is invalid.

    Returns:
        Callable[[dict], Any]: A function that retrieves the field value or default.

    Example:
        >>> f = get_field("player.name")
        >>> f({"player": {"name": "Bukayo Saka"}})  # → "Bukayo Saka"
    """
    keys = path.split(".")

    def accessor(d: Optional[dict[Any, Any]]) -> Any:
        for key in keys:
            if not isinstance(d, dict):
                return default
            d = d.get(key, default)
        return d

    return accessor


def resolve_path(record: dict, path: str, default=None):
    """
    Safely access a nested field using dot notation.

    Args:
        record (dict): The record to resolve the path from.
        path (str): The path to resolve.
        default (Any, optional): Value to return if the path is invalid.

    Returns:
        Any: The resolved value or default if not found.

    Example:
        >>> resolve_path({"player": {"name": "Bukayo Saka"}}, "player.name")
        "Bukayo Saka"
    """
    return get_field(path, default)(record)


def get_index(
    path: str, index: int, default: Any = None
) -> Callable[[Optional[dict[Any, Any]]], Any]:
    """
    Safely access an index in a nested list using dot-separated path.

    Args:
        path (str): Dot-separated path to a list field, e.g. "pass.end_location".
        index (int): The index to extract from the list.
        default (Any, optional): Value to return if the path is invalid or index out of bounds.

    Returns:
        Callable[[dict], Any]: A function that retrieves the indexed value or default.

    Example:
        >>> f = get_index("pass.end_location", 0)
        >>> f({"pass": {"end_location": [100, 40]}})  # → 100
    """

    keys = path.split(".")

    def accessor(d: Optional[dict[Any, Any]]) -> Any:
        for key in keys:
            if not isinstance(d, dict):
                return default
            d = d.get(key)
        if d is None:
            return default
        if isinstance(d, list):
            try:
                return d[index]
            except IndexError:
                return default
        return default

    return accessor


# --- Predicates ---


def where_equals(path: str, value: Any) -> Callable[[dict], bool]:
    """
    Create a predicate that checks if a nested field equals a given value.

    Args:
        path (str): Dot-separated path to a field, e.g. "player.name".
        value (Any): The value to compare against.

    Returns:
        Callable[[dict], bool]: A function that returns True if the field equals the value.

    Example:
        >>> f = where_equals("player.name", "Bukayo Saka")
        >>> f({"player": {"name": "Bukayo Saka"}})  # → True
    """
    accessor = get_field(path)

    def predicate(record: dict) -> bool:
        return accessor(record) == value

    return predicate


def where_in(path: str, values: set | list | tuple) -> Callable[[dict], bool]:
    """
    Create a predicate that checks if a nested field is in a list of values.

    Args:
        path (str): Dot-separated path to a field, e.g. "player.name".
        values (set | list | tuple): The values to check against.

    Returns:
        Callable[[dict], bool]: A function that returns True if the field is in the values.

    Example:
        >>> f = where_in("player.name", {"Bukayo Saka", "Mohamed Salah"})
        >>> f({"player": {"name": "Bukayo Saka"}})  # → True
    """
    accessor = get_field(path)
    values = set(values)

    def predicate(record: dict) -> bool:
        return accessor(record) in values

    return predicate


def where_exists(path: str) -> Callable[[dict], bool]:
    """
    Check if a nested field exists and is not None.

    Args:
        path (str): Dot-separated path to a field, e.g. "player.name".

    Returns:
        Callable[[dict], bool]: A function that returns True if the field exists and is not None.

    Example:
        >>> f = where_exists("player.name")
        >>> f({"player": {"name": "Bukayo Saka"}})  # → True
    """
    accessor = get_field(path)

    def predicate(record: dict) -> bool:
        return accessor(record) is not None

    return predicate


def where_not_none(path: str) -> Callable[[dict], bool]:
    """
    Check if a nested field exists and is not None (alias of where_exists).

    Args:
        path (str): Dot-separated path to a field, e.g. "player.name".

    Returns:
        Callable[[dict], bool]: A function that returns True if the field exists and is not None.

    Example:
        >>> f = where_not_none("player.name")
        >>> f({"player": {"name": "Bukayo Saka"}})  # → True
    """
    return where_exists(path)


# --- Transformers ---


def combine_fields(out_field: str, *paths: str, join_str: str = " ") -> Callable:
    """
    Combine multiple fields into a single field.

    Args:
        out_field (str): The name of the output field.
        *paths (str): The paths to the fields to combine.
        join_str (str, optional): The string to join the fields with. Defaults to " ".

    Returns:
        Callable: A function that takes a record and returns a new record with the combined fields.

    Example:
        >>> f = combine_fields("full_name", "first_name", "last_name")
        >>> f({"first_name": "Bukayo", "last_name": "Saka"})  # → {"full_name": "Bukayo Saka"}
    """

    def transformer(d):
        parts = [str(get_field(p)(d) or "") for p in paths]
        return {out_field: join_str.join(parts)}

    return transformer


def coalesce(*paths: str, default=None) -> Callable:
    """
    Return the first non-None value from a list of paths.

    Args:
        *paths (str): The paths to the fields to check.
        default (Any, optional): The default value to return if all paths are None. Defaults to None.

    Returns:
        Callable: A function that takes a record and returns the first non-None value from the paths.

    Example:
        >>> f = coalesce("player.name", "player.alias", default="Unknown")
        >>> f({"player": {"name": "Bukayo Saka"}})  # → "Bukayo Saka"
    """

    def fn(d):
        for path in paths:
            val = get_field(path)(d)
            if val is not None:
                return val
        return default

    return fn


def set_path(record: dict, path: str, value: Any):
    """
    Set a nested field using dot notation. Creates intermediate dicts.

    Args:
        record (dict): The record to set the path on.
        path (str): The path to set.
        value (Any): The value to set.
    """
    keys = path.split(".")
    current = record
    # walk/create intermediate dicts
    for k in keys[:-1]:
        if not isinstance(current.get(k), dict):
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value


def delete_path(record: dict, path: str):
    """
    Delete a nested field using dot notation. Silently does nothing if path doesn't exist.

    Args:
        record (dict): The record to delete the path from.
        path (str): The path to delete.
    """
    keys = path.split(".")
    current: Any = record
    for k in keys[:-1]:
        if not isinstance(current, dict):
            return
        current = current.get(k)
    if isinstance(current, dict):
        current.pop(keys[-1], None)


def explain_plan(
    raw_plan: list[dict],
    optimized_plan: Optional[list[dict]] = None,
    compare: bool = False,
    indent: int = 2,
):
    """
    Print a plan. If compare=True and optimized_plan is given,
    show both raw and optimized side by side.
    """

    def _print(title: str, plan: list[dict]):
        print(title)
        for i, step in enumerate(plan, 1):
            op = step["op"]
            # strip out internal keys
            details = {k: v for k, v in step.items() if k not in ("op", "_notes")}
            notes = step.get("_notes", [])
            line = f"{i:>2}. {op:<15} {details}"
            if notes:
                line += "    // " + "; ".join(notes)
            print(" " * indent + line)

    if compare and optimized_plan is not None:
        _print("=== Pre-optimization plan ===", raw_plan)
        print()
        _print("=== Post-optimization plan ===", optimized_plan)
    else:
        title = "Optimized Plan" if optimized_plan is not None else "Plan"
        plan = optimized_plan or raw_plan
        _print(f"=== {title} ===", plan)


def show_tabular(sample: list[dict]) -> None:
    """
    Show a tabular representation of a sample of records.

    Args:
        sample (list[dict]): A sample of records to show.
    """
    # flatten each record to a flat dict
    flat = [flatten_dict(r) for r in sample]
    if not flat:
        print("<no rows>")
        return

    # collect columns in sorted order
    cols = sorted({k for row in flat for k in row})

    # build rows
    rows = [[row.get(c, "") for c in cols] for row in flat]
    print(tabulate(rows, headers=cols, tablefmt="github"))
