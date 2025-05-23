import random
from collections import defaultdict
from typing import Any, Callable, List, Union


def fast_get_field(record: dict, parts: list[str]) -> Any:
    """
    Retrieve a nested field or list index from a record using dot notation.
    Accepts a precompiled list of parts (split by '.').
    Supports numeric strings as list indices.
    """
    current: Any = record
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit():
            idx = int(part)
            if 0 <= idx < len(current):
                current = current[idx]
            else:
                return None
        else:
            return None
    return current


def get_field(record: dict, path: Union[str, List[str]]):
    if isinstance(path, str):
        parts = path.split(".")
    else:
        parts = path

    current: Any = record
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit():
            idx = int(part)
            if 0 <= idx < len(current):
                current = current[idx]
            else:
                return None
        else:
            return None
    return current


def get_index(field: str, index: int) -> Callable[[dict], Any]:
    """
    Returns a function that extracts the `index`th value from a list field (dot path).

    Example:
        f = get_index("location", 0)
        f({"location": [50, 40]}) → 50

    If the field or index is missing, returns None.
    """
    path = field.split(".")

    def _getter(record: dict) -> Any:
        current: Any = record
        for part in path:
            if not isinstance(current, dict):
                return None
            current = current.get(part)
        if not isinstance(current, list):
            return None
        try:
            return current[index]
        except IndexError:
            return None

    return _getter


def set_nested_field(record: dict, path: str, value: Any):
    """
    Set a nested field in a record using dot notation.

    Args:
        record (dict): The record to modify.
        path (str): The path to the field (dot notation).
        value (Any): The value to set.
    """
    parts = path.split(".")
    d = record
    for part in parts[:-1]:
        if part not in d or not isinstance(d[part], dict):
            d[part] = {}
        d = d[part]
    d[parts[-1]] = value


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """
    Flatten a nested dictionary into a single-level dictionary using dot notation.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The prefix for nested keys.
        sep (str): The separator to use (default is ".").

    Returns:
        dict: Flattened dictionary with dot-notated keys.
    """
    items: dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def schema(records: list[dict], sample_size: int = 50):
    type_map = defaultdict(set)
    for i, row in enumerate(records):
        if i >= sample_size:
            break
        flat = flatten_dict(row)
        for key, value in flat.items():
            type_map[key].add(type(value))
    return dict(type_map)


def unify_types(types: set[type]) -> type:
    if len(types) == 1:
        return next(iter(types))
    elif types.issubset({int, float}):
        return float
    return object  # fallback for mixed/unknown


def reservoir_sample(iterable, k, seed=None):
    rng = random.Random(seed)
    result = []
    for i, item in enumerate(iterable):
        if i < k:
            result.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                result[j] = item
    return result
