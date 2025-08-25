import random
from collections import defaultdict
from typing import Any, Callable, List, Union


def fast_get_field(record: dict, parts: list[str]) -> Any:
    """Retrieve a nested value from a mapping or list using a pre-split path.

    This is a slightly optimized helper for looking up nested values when the
    caller has already split a dotted path into parts. Each element of
    ``parts`` is used to traverse dictionaries by key. If the current value is
    a ``list`` and the part is a numeric string it will be treated as an
    integer index. If any step cannot be resolved the function returns ``None``.

    Args:
        record (dict): The mapping to search. May contain nested dicts/lists.
        parts (list[str]): Pre-split path components (for example ``["a", "0", "b"]``).

    Returns:
        Any: The found value, or ``None`` if the path does not exist or an index is out of range.

    Example:
        >>> fast_get_field({"a": [{"b": 1}]}, ["a", "0", "b"])
        1
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


def get_field(record: dict, path: Union[str, List[str]]) -> Any:
    """Retrieve a nested value from a mapping or list using dot-notation.

    The ``path`` may either be a dotted string (for example ``"a.0.b"``) or
    a pre-split list of parts (``["a", "0", "b"]``). Dictionary keys are
    looked up by name. If a traversal step yields a list and the corresponding
    part is a numeric string, it will be used as a list index. Any lookup
    failure returns ``None`` rather than raising.

    Args:
        record (dict): The mapping to search. May contain nested dicts/lists.
        path (Union[str, List[str]]): Dotted path or list of path components.

    Returns:
        Any: The found value, or ``None`` if the path does not exist or an index is invalid.

    Example:
        >>> get_field({"loc": [50, 40]}, "loc.0")
        50
    """
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
    """Factory that produces a getter which returns an indexed element.

    The returned callable accepts a single ``record`` argument and follows a
    dotted path described by ``field`` to reach a list, then returns the
    element at ``index``. If any step is not a mapping, the final value is not
    a list, or the index is out of range, the callable returns ``None``.

    Args:
        field (str): Dotted path to the list (for example ``"a.b.list"``).
        index (int): The integer index to retrieve from the list.

    Returns:
        Callable[[dict], Any]: A function which when given a record will return
        the requested list element or ``None`` on failure.

    Example:
        >>> getter = get_index("items", 2)
        >>> getter({"items": [0, 1, 2, 3]})
        2
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


def set_nested_field(record: dict, path: str, value: Any) -> None:
    """
    Set a nested field in a record using dot notation.

    Args:
        record (dict): The record to modify.
        path (str): The path to the field (dot notation).
        value (Any): The value to set.
    """
    """Set a value into a nested dictionary structure creating intermediate dicts.

    Given a dotted ``path`` this function will walk or create nested
    dictionaries on ``record`` and assign ``value`` to the final key. Existing
    non-dict values encountered along the path will be overwritten with a new
    dict to allow the assignment to succeed.

    This function mutates ``record`` in-place and returns ``None``.

    Args:
        record (dict): The mapping to modify.
        path (str): Dotted path to assign (for example ``"a.b.c"``).
        value (Any): The value to store at the destination key.

    Returns:
        None

    Example:
        >>> r = {}
        >>> set_nested_field(r, "a.b", 1)
        >>> r
        {'a': {'b': 1}}
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
    """Flatten nested dicts into a single mapping with dotted keys.

    Nested dictionaries are recursively expanded and their keys are joined
    using ``sep``. Non-dict values (including lists) are kept as-is at the
    flattened key.

    Args:
        d (dict): The mapping to flatten.
        parent_key (str, optional): Prefix for keys during recursion.
        sep (str, optional): Separator placed between nested key parts.

    Returns:
        dict: A new flat dictionary where nested keys are represented as
        ``parent.child`` style strings.

    Example:
        >>> flatten_dict({"a": {"b": 1}})
        {'a.b': 1}
    """
    items: dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def schema(records: list[dict], sample_size: int = 50) -> dict:
    """
    Get the schema of a list of records.

    Args:
        records (list[dict]): The list of records to get the schema from.
        sample_size (int): The number of records to sample.

    Returns:
        dict: The schema of the records.
    """
    """Infer a simple schema from an iterable of mapping records.

    This function inspects up to ``sample_size`` records from ``records`` and
    records the Python types observed at each flattened dotted key. The result
    maps dotted key strings to a ``set`` of python ``type`` objects seen for
    that key across the sampled records.

    Args:
        records (list[dict]): Sequence of mapping records to inspect.
        sample_size (int, optional): Maximum number of records to sample. If
            the sequence is shorter, all entries are inspected.

    Returns:
        dict: Mapping of flattened key -> set of observed types. Example:
        ``{"a.b": {int, type(None)}}``.

    Note:
        The function uses ``flatten_dict`` internally so nested dict keys are
        represented with dotted names.
    """
    type_map = defaultdict(set)
    for i, row in enumerate(records):
        if i >= sample_size:
            break
        flat = flatten_dict(row)
        for key, value in flat.items():
            type_map[key].add(type(value))
    return dict(type_map)


def unify_types(types: set[type]) -> type:
    """
    Unify a set of types into a single type.

    Args:
        types (set[type]): The set of types to unify.

    Returns:
        type: The unified type.
    """
    """Choose a single representative type for a set of observed types.

    Rules implemented:
    - If the set contains exactly one type, return that type.
    - If all types are numeric (``int``/``float``), return ``float`` to
      represent a common numeric super-type.
    - Otherwise return ``object`` as a generic fallback.

    Args:
        types (set[type]): Set of python types observed for a field.

    Returns:
        type: Representative python type (for example ``float`` or ``object``).
    """
    if len(types) == 1:
        return next(iter(types))
    elif types.issubset({int, float}):
        return float
    return object  # fallback for mixed/unknown


def reservoir_sample(iterable, k, seed=None) -> list:
    """
    Sample a fixed number of records from an iterable using the reservoir sampling algorithm.

    Args:
        iterable (iterable): The iterable to sample from.
        k (int): The number of records to sample.
        seed (int, optional): The seed for the random number generator.

    Returns:
        list: The sampled records.
    """
    """Return k items sampled uniformly-at-random from ``iterable``.

    This implements reservoir sampling (Algorithm R) which allows uniform
    sampling from an iterable of unknown or large size using constant memory
    (O(k)). If the iterable has fewer than ``k`` items the returned list will
    contain all items in their original order.

    Args:
        iterable (iterable): Source of items to sample from.
        k (int): Desired sample size. Must be >= 0.
        seed (int, optional): Optional random seed for deterministic results.

    Returns:
        list: A list with up to ``k`` sampled items. If ``k`` is zero an
        empty list is returned.

    Raises:
        ValueError: If ``k`` is negative.

    Example:
        >>> reservoir_sample(range(100), 5, seed=1)
        [17, 72, 97, 8, 32]
    """
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
