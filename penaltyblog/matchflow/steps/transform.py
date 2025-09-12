import copy
import itertools
import random
from collections import OrderedDict, defaultdict
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Union

from .utils import (
    fast_get_field,
    flatten_dict,
    get_field,
    reservoir_sample,
    set_nested_field,
)

# Type aliases for records and streams
Record = Dict[str, Any]
RecordStream = Iterator[Record]

if TYPE_CHECKING:
    from ..flow import Flow


def _coerce_join_key(value, strategy="strict"):
    """
    Coerce a join key value for consistent comparison across data types.

    Args:
        value: The join key value to coerce
        strategy: Coercion strategy - 'strict', 'auto', or 'string'

    Returns:
        Coerced value for comparison
    """
    if value is None:
        return None

    if strategy == "strict":
        return value
    elif strategy == "string":
        return str(value)
    elif strategy == "auto":
        # Smart coercion - numeric values get consistent string representation
        if isinstance(value, (int, float)):
            # Convert to string for consistent comparison
            if isinstance(value, float) and value.is_integer():
                # 1.0 -> "1" to match with int 1
                return str(int(value))
            else:
                return str(value)
        elif isinstance(value, str):
            # Try to convert numeric strings for consistency
            try:
                # Check if it's a valid integer
                if value.lstrip("-").isdigit():
                    return str(int(value))
                # Check if it's a valid float
                float_val = float(value)
                if float_val.is_integer():
                    return str(int(float_val))
                else:
                    return str(float_val)
            except ValueError:
                # Not a number, return as-is
                return value
        else:
            # Other types converted to string
            return str(value)
    else:
        raise ValueError(f"Unknown coercion strategy: {strategy}")


def apply_filter(records: RecordStream, step: dict) -> RecordStream:
    """
    Filter records based on a predicate.

    Args:
        records: A stream of records to filter.
        step: Configuration dict containing the predicate to apply.

    Returns:
        A stream of filtered records.
    """
    pred = step["predicate"]
    for r in records:
        if pred(r):
            yield r


def apply_assign(records: RecordStream, step: dict) -> RecordStream:
    """
    Assign new fields to each record.

    Args:
        records: A stream of records to assign fields to.
        step: Configuration dict containing the fields to assign.

    Returns:
        A stream of records with assigned fields.
    """
    fields = step["fields"]
    for r in records:
        new = dict(r)
        for k, func in fields.items():
            new[k] = func(r)
        yield new


def apply_select(records: RecordStream, step: dict) -> RecordStream:
    """
    Select specific fields from each record.

    Args:
        records: A stream of records to select fields from.
        step: Configuration dict containing the fields to select.

    Returns:
        A stream of records with selected fields.
    """
    field_names = step["fields"]

    if all("." not in f for f in field_names):
        for record in records:
            yield {k: record.get(k) for k in field_names}
    else:
        compiled_paths = step.get("_compiled_fields")
        if not compiled_paths:
            compiled_paths = [f.split(".") for f in field_names]
            step["_compiled_fields"] = compiled_paths

        for record in records:
            out: dict[str, Any] = {}
            for path in compiled_paths:
                value = fast_get_field(record, path)
                set_nested_field(out, ".".join(path), value)
            yield out


def _del_nested(d, parts) -> None:
    """Delete a nested key from a dict given a list of parts."""
    for part in parts[:-1]:
        d = d.get(part, {})
    d.pop(parts[-1], None)


def _set_nested(d, parts, value) -> None:
    """Set a nested key in a dict given a list of parts."""
    for part in parts[:-1]:
        if part not in d or not isinstance(d[part], dict):
            d[part] = {}
        d = d[part]
    d[parts[-1]] = value


def apply_rename(records: RecordStream, step: dict) -> RecordStream:
    """
    Rename fields in each record.

    Args:
        records: A stream of records to rename fields in.
        step: Configuration dict containing the mapping of old to new field names.

    Returns:
        A stream of records with renamed fields.
    """
    mapping = step["mapping"]
    for r in records:
        new = dict(r)
        for old, new_key in mapping.items():
            # Try flat key match first
            if old in new:
                new[new_key] = new.pop(old)
            else:
                # Try nested (dot-path) access
                old_parts = old.split(".")
                new_parts = new_key.split(".")
                val = get_field(new, old_parts)
                if val is not None:
                    _del_nested(new, old_parts)
                    _set_nested(new, new_parts, val)
        yield new


def apply_sort(records: RecordStream, step: dict) -> RecordStream:
    """
    Sort records based on a list of keys.

    Args:
        records: A stream of records to sort.
        step: Configuration dict containing the keys to sort by.

    Returns:
        A stream of sorted records.
    """
    keys = step["keys"]
    ascending = step.get("ascending", [True] * len(keys))
    records_list: List[Record] = list(records)

    def sort_key(r):
        key_parts = []
        for k, asc in zip(keys, ascending):
            v = r.get(k)
            # Reverse sort by negating numeric or inverting sortable value
            key_parts.append(v if asc else _invert(v))
        return tuple(key_parts)

    return iter(sorted(records_list, key=sort_key))


def _invert(value):
    if isinstance(value, (int, float)):
        return -value
    elif isinstance(value, str):
        return "".join(chr(255 - ord(c)) for c in value)  # crude inverse for strings
    else:
        return value  # fallback: not reversed


def apply_limit(records: RecordStream, step: dict) -> RecordStream:
    """
    Limit the number of records.

    Args:
        records: A stream of records to limit.
        step: Configuration dict containing the count to limit by.

    Returns:
        A stream of limited records.
    """
    count = step["count"]
    for i, record in enumerate(records):
        if i >= count:
            break
        yield record


def apply_drop(records: RecordStream, step: dict) -> RecordStream:
    """
    Drop specified fields from each record.

    Args:
        records: A stream of records to drop fields from.
        step: Configuration dict containing the keys to drop.

    Returns:
        A stream of records with dropped fields.
    """
    keys = step["keys"]

    for record in records:
        new = dict(record)
        for key in keys:
            parts = key.split(".")
            d = new
            try:
                for part in parts[:-1]:
                    next_d = d.get(part)
                    if not isinstance(next_d, dict):
                        raise KeyError
                    d = next_d
                d.pop(parts[-1], None)  # silently ignore missing keys
            except Exception:
                continue  # skip malformed paths
        yield new


def apply_flatten(records: RecordStream, step: dict) -> RecordStream:
    """
    Flatten nested dictionaries into a single-level dictionary using dot notation.

    Args:
        records: A stream of records to flatten.
        step: Configuration dict containing the keys to flatten.

    Returns:
        A stream of flattened records.
    """
    for r in records:
        yield flatten_dict(r)


def apply_distinct(records: RecordStream, step: dict) -> RecordStream:
    keys = step.get("keys")
    keep = step.get("keep", "first")

    if keep == "first":
        return _distinct_first(records, keys)
    elif keep == "last":
        return _distinct_last(records, keys)
    else:
        raise ValueError("distinct keep must be 'first' or 'last'")


def _record_identity(record, keys):
    if keys:
        return tuple(get_field(record, k.split(".")) for k in keys)
    return tuple(sorted(record.items()))


def _distinct_first(records, keys):
    seen = set()
    for record in records:
        identity = _record_identity(record, keys)
        if identity in seen:
            continue
        seen.add(identity)
        yield record


def _distinct_last(records, keys):
    seen = OrderedDict()
    for record in records:
        identity = _record_identity(record, keys)
        seen[identity] = record  # later value overwrites

    yield from seen.values()


def apply_dropna(records: RecordStream, step: dict) -> RecordStream:
    """
    Drop records with missing values.

    Args:
        records: A stream of records to drop missing values from.
        step: Configuration dict containing the fields to drop missing values from.

    Returns:
        A stream of records with dropped records.
    """
    fields = step.get("fields")
    compiled = step.get("_compiled_fields")

    if fields:
        if not compiled:
            compiled = [f.split(".") for f in fields]
            step["_compiled_fields"] = compiled

        for record in records:
            if any(get_field(record, path) is None for path in compiled):
                continue
            yield record
    else:
        inferred_keys = None

        for record in records:
            if inferred_keys is None:
                inferred_keys = list(record.keys())

            if any(record.get(k) is None for k in inferred_keys):
                continue

            yield record


def apply_explode(records: RecordStream, step: dict) -> RecordStream:
    """
    Explode records based on a list of fields.

    Args:
        records: A stream of records to explode.
        step: Configuration dict containing the fields to explode.

    Returns:
        A stream of exploded records.
    """
    fields = step["fields"]
    compiled = step.get("_compiled_fields")

    if not compiled:
        compiled = [f.split(".") for f in fields]
        step["_compiled_fields"] = compiled

    for record in records:
        values = [get_field(record, f) for f in compiled]

        if all(isinstance(v, list) for v in values):
            lengths = [len(v) for v in values]
            if len(set(lengths)) != 1:
                raise ValueError(
                    f"Cannot explode fields with mismatched lengths: {lengths}"
                )

            if lengths[0] == 0:
                yield copy.deepcopy(record)
            else:
                for i in range(lengths[0]):
                    new_record = copy.deepcopy(record)
                    for f, v in zip(compiled, values):
                        set_nested_field(new_record, ".".join(f), v[i])
                    yield new_record
        else:
            yield copy.deepcopy(record)


def apply_join(records: RecordStream, step: dict) -> RecordStream:
    """
    Join records based on a list of keys. Dispatcher function that selects the appropriate join strategy.

    Args:
        records: A stream of records to join.
        step: Configuration dict containing the keys to join by.

    Returns:
        A stream of joined records.
    """
    # Future logic to select strategy will go here.
    # For now, always use hash join.
    return _apply_hash_join(records, step)


def _apply_sort_merge_join(records: RecordStream, step: dict) -> RecordStream:
    """
    Sort-merge join implementation for memory-efficient joins on pre-sorted data.

    Args:
        records: A stream of records to join (assumed to be sorted on join keys).
        step: Configuration dict containing the keys to join by.

    Returns:
        A stream of joined records.
    """
    from ..executor import FlowExecutor

    # Extract parameters
    on = step.get("on")
    left_on = step.get("left_on")
    right_on = step.get("right_on")
    lsuffix = step.get("lsuffix", "")
    rsuffix = step.get("rsuffix", "_right")
    how = step.get("how", "left")
    type_coercion = step.get("type_coercion", "strict")

    # Determine join keys
    left_keys: Union[List[str], None] = None
    right_keys: Union[List[str], None] = None

    if on is not None:
        left_keys = right_keys = on
    else:
        left_keys = left_on
        right_keys = right_on

    # Compile join keys
    compiled_left = step.get("_compiled_left")
    compiled_right = step.get("_compiled_right")

    if not compiled_left and left_keys is not None:
        compiled_left = [k.split(".") for k in left_keys]
        step["_compiled_left"] = compiled_left

    if not compiled_right:
        compiled_right = [k.split(".") for k in right_keys] if right_keys else []
        step["_compiled_right"] = compiled_right

    # Get left and right iterators
    left_iter = records
    right_iter = FlowExecutor(step["right_plan"]).execute()

    # Key extraction functions
    def left_key(record):
        return tuple(
            _coerce_join_key(get_field(record, k), type_coercion)
            for k in (compiled_left or [])
        )

    def right_key(record):
        return tuple(
            _coerce_join_key(get_field(record, k), type_coercion)
            for k in (compiled_right or [])
        )

    # Helper function to merge records with suffix handling
    def merge_records(left_rec, right_rec, is_left_primary=True):
        if is_left_primary:
            merged = dict(left_rec)
            for rk, rv in right_rec.items():
                if right_keys is not None and rk in right_keys:
                    continue
                if rk in merged:
                    if rsuffix:
                        merged[rk + rsuffix] = rv
                else:
                    merged[rk] = rv

            # Apply lsuffix if needed
            if lsuffix:
                for lk in list(merged.keys()):
                    if (
                        lk in left_rec
                        and left_keys is not None
                        and lk not in left_keys
                        and lk in right_rec
                    ):
                        if lk + lsuffix not in merged:
                            merged[lk + lsuffix] = merged.pop(lk)
        else:
            # Right is primary (for right joins)
            merged = dict(right_rec)
            for lk, lv in left_rec.items():
                if left_keys is not None and lk in left_keys:
                    continue
                if lk in merged:
                    if lsuffix:
                        merged[lk + lsuffix] = lv
                else:
                    merged[lk] = lv

        return merged

    # Create null record for unmatched sides
    def create_null_left(right_rec, sample_left=None):
        result = dict(right_rec)
        if sample_left:
            for lk in sample_left.keys():
                if right_keys is not None and lk not in right_keys:
                    field_name = lk + lsuffix if (lsuffix and lk in result) else lk
                    if field_name not in result:
                        result[field_name] = None
        return result

    def create_null_right(left_rec, sample_right=None):
        result = dict(left_rec)
        if sample_right:
            for rk in sample_right.keys():
                if left_keys is not None and rk not in left_keys:
                    field_name = rk + rsuffix if (rsuffix and rk in result) else rk
                    if field_name not in result:
                        result[field_name] = None
        return result

    # Group by key using itertools.groupby
    left_grouped = itertools.groupby(left_iter, key=left_key)
    right_grouped = itertools.groupby(right_iter, key=right_key)

    # Convert to iterators we can peek at
    try:
        left_key_val, left_group = next(left_grouped)
        left_group = list(left_group)  # Materialize group
        left_has_data = True
    except StopIteration:
        left_has_data = False
        left_key_val = None
        left_group = []

    try:
        right_key_val, right_group = next(right_grouped)
        right_group = list(right_group)  # Materialize group
        right_has_data = True
    except StopIteration:
        right_has_data = False
        right_key_val = None
        right_group = []

    sample_left = left_group[0] if left_group else None
    sample_right = right_group[0] if right_group else None

    # Main sort-merge loop
    while left_has_data or right_has_data:
        if not right_has_data or (
            left_has_data
            and left_key_val is not None
            and right_key_val is not None
            and left_key_val < right_key_val
        ):
            # Left key is smaller or no more right data
            if how in ["left", "outer"]:
                for left_rec in left_group:
                    yield create_null_right(left_rec, sample_right)
            elif how == "anti":
                for left_rec in left_group:
                    yield dict(left_rec)

            # Advance left
            try:
                left_key_val, left_group = next(left_grouped)
                left_group = list(left_group)
            except StopIteration:
                left_has_data = False

        elif not left_has_data or (
            right_has_data
            and left_key_val is not None
            and right_key_val is not None
            and right_key_val < left_key_val
        ):
            # Right key is smaller or no more left data
            if how in ["right", "outer"]:
                for right_rec in right_group:
                    yield create_null_left(right_rec, sample_left)

            # Advance right
            try:
                right_key_val, right_group = next(right_grouped)
                right_group = list(right_group)
            except StopIteration:
                right_has_data = False

        else:
            # Keys match
            if how != "anti":
                # Create cartesian product of matching groups
                for left_rec in left_group:
                    for right_rec in right_group:
                        if how == "right":
                            yield merge_records(
                                left_rec, right_rec, is_left_primary=False
                            )
                        else:
                            yield merge_records(
                                left_rec, right_rec, is_left_primary=True
                            )

            # Advance both
            try:
                left_key_val, left_group = next(left_grouped)
                left_group = list(left_group)
            except StopIteration:
                left_has_data = False

            try:
                right_key_val, right_group = next(right_grouped)
                right_group = list(right_group)
            except StopIteration:
                right_has_data = False


def _apply_hash_join(records: RecordStream, step: dict) -> RecordStream:
    """
    Hash join implementation for joining records.

    Args:
        records: A stream of records to join.
        step: Configuration dict containing the keys to join by.

    Returns:
        A stream of joined records.
    """
    from ..executor import FlowExecutor

    # Extract parameters
    on = step.get("on")
    left_on = step.get("left_on")
    right_on = step.get("right_on")
    lsuffix = step.get("lsuffix", "")
    rsuffix = step.get("rsuffix", "_right")
    how = step.get("how", "left")
    type_coercion = step.get("type_coercion", "strict")

    # Determine join keys
    left_keys: Union[List[str], None] = None
    right_keys: Union[List[str], None] = None

    if on is not None:
        left_keys = right_keys = on
    else:
        left_keys = left_on
        right_keys = right_on

    # Compile join keys
    compiled_left = step.get("_compiled_left")
    compiled_right = step.get("_compiled_right")

    if not compiled_left and left_keys is not None:
        compiled_left = [k.split(".") for k in left_keys]
        step["_compiled_left"] = compiled_left

    if not compiled_right:
        compiled_right = [k.split(".") for k in right_keys] if right_keys else []
        step["_compiled_right"] = compiled_right

    # Execute right plan and build index
    right_records: List[Record] = list(FlowExecutor(step["right_plan"]).execute())
    right_index: dict[tuple[Any, ...], list[dict]] = {}

    for r in right_records:
        key = tuple(
            _coerce_join_key(get_field(r, k), type_coercion)
            for k in (compiled_right or [])
        )
        right_index.setdefault(key, []).append(r)

    # Handle right join by swapping
    if how == "right":
        # Swap left and right, then do a left join
        def right_join_generator():
            # Build left index
            left_records: List[Record] = list(records)
            left_index: dict[tuple[Any, ...], list[dict]] = {}

            for l in left_records:
                key = tuple(
                    _coerce_join_key(get_field(l, k), type_coercion)
                    for k in (compiled_left or [])
                )
                left_index.setdefault(key, []).append(l)

            # Process right records as primary
            for right in right_records:
                key = tuple(
                    _coerce_join_key(get_field(right, k), type_coercion)
                    for k in (compiled_right or [])
                )
                matches = left_index.get(key)

                if not matches:
                    # No left match - yield right with nulls for left fields
                    joined = dict(right)

                    # Add null values for left-only fields
                    for left_rec in (
                        left_records[:1] if left_records else [{}]
                    ):  # Use first left record as template
                        for lk in left_rec.keys():
                            if right_keys is not None and lk not in right_keys:
                                left_name = (
                                    lk + lsuffix if (lsuffix and lk in joined) else lk
                                )
                                if left_name not in joined:
                                    joined[left_name] = None
                        break
                    yield joined
                else:
                    for left in matches:
                        joined = dict(right)
                        for lk, lv in left.items():
                            if right_keys is not None and lk in right_keys:
                                continue
                            if lk in joined:
                                joined[lk + lsuffix] = lv
                            else:
                                joined[lk] = lv
                        yield joined

        yield from right_join_generator()
        return

    # Track matched right keys for outer join
    matched_right_keys = set() if how == "outer" else None

    # Process left records
    for left in records:
        key = tuple(
            _coerce_join_key(get_field(left, k), type_coercion)
            for k in (compiled_left or [])
        )
        matches = right_index.get(key)

        if not matches:
            # No right match
            if how in ["left", "outer"]:
                yield dict(left)
            elif how == "anti":
                yield dict(left)
            # For inner join, skip unmatched left records
            continue

        # Has matches
        if how == "anti":
            # Anti join - skip records that have matches
            continue

        # Mark this key as matched for outer join
        if matched_right_keys is not None:
            matched_right_keys.add(key)

        # Join matched records
        for right in matches:
            joined = dict(left)
            for rk, rv in right.items():
                if right_keys is not None and rk in right_keys:
                    continue
                if rk in joined:
                    # Handle suffix collision
                    if rsuffix:
                        joined[rk + rsuffix] = rv
                    else:
                        # If no rsuffix, left value takes precedence
                        pass
                else:
                    joined[rk] = rv

            # Apply lsuffix to overlapping left fields if needed
            if lsuffix:
                for lk in list(joined.keys()):
                    if lk in left and left_keys is not None and lk not in left_keys:
                        # Check if this left field conflicts with a right field
                        if any(lk in r for r in matches):
                            # Rename left field with lsuffix
                            if lk + lsuffix not in joined:
                                joined[lk + lsuffix] = joined.pop(lk)

            yield joined

    # Handle outer join - emit unmatched right records
    if how == "outer":
        for key, right_group in right_index.items():
            if matched_right_keys is not None and key not in matched_right_keys:
                for right in right_group:
                    # Create record with right data and null left fields
                    joined = dict(right)

                    # Add null values for left-only fields
                    left_sample = (
                        next(iter(records), None)
                        if hasattr(records, "__iter__")
                        else None
                    )
                    if left_sample:
                        for lk in left_sample.keys():
                            if left_keys is not None and lk not in left_keys:
                                left_name = (
                                    lk + lsuffix if (lsuffix and lk in joined) else lk
                                )
                                if left_name not in joined:
                                    joined[left_name] = None

                    yield joined


def apply_split_array(records: RecordStream, step: dict) -> RecordStream:
    """
    Split an array into multiple records.

    Args:
        records: A stream of records to split arrays from.
        step: Configuration dict containing the field to split.

    Returns:
        A stream of split records.
    """
    field = step["field"]
    into = step["into"]

    for record in records:
        if field not in record and "." not in field:
            # Simple field missing entirely
            yield record
            continue

        value = get_field(record, field)

        # Skip if field is missing or explicitly None
        if value is None:
            yield record
            continue

        new_record = dict(record)

        if isinstance(value, (list, tuple)):
            for i, key in enumerate(into):
                new_record[key] = value[i] if i < len(value) else None
            yield new_record
        else:
            # Field exists but isn't a list → treat as error or pass through unchanged
            yield record


def apply_pivot(records: RecordStream, step: dict) -> RecordStream:
    """
    Pivot records based on a list of index fields.

    Args:
        records: A stream of records to pivot.
        step: Configuration dict containing the index fields to pivot by.

    Returns:
        A stream of pivoted records.
    """
    index_fields = step["index"]
    col_field = step["columns"]
    val_field = step["values"]

    compiled_index = step.get("_compiled_index")
    if not compiled_index:
        compiled_index = [f.split(".") for f in index_fields]
        step["_compiled_index"] = compiled_index

    compiled_col = step.get("_compiled_col") or col_field.split(".")
    compiled_val = step.get("_compiled_val") or val_field.split(".")
    step["_compiled_col"] = compiled_col
    step["_compiled_val"] = compiled_val

    grouped = defaultdict(list)
    for r in records:
        key = tuple(get_field(r, f) for f in compiled_index)
        grouped[key].append(r)

    for key, rows in grouped.items():
        result = {f: k for f, k in zip(index_fields, key)}
        for row in rows:
            col = get_field(row, compiled_col)
            val = get_field(row, compiled_val)
            if col is not None:
                result[col] = val
        yield result


def apply_summary(records: RecordStream, step: dict) -> RecordStream:
    """
    Apply a summary function to the records.

    Args:
        records: A stream of records to apply the summary function to.
        step: Configuration dict containing the summary function to apply.

    Returns:
        A stream of summary results.
    """
    agg_func = step["agg"]
    rows: List[Record] = list(records)
    result = agg_func(rows)

    if not isinstance(result, dict):
        raise ValueError("summary function must return a dict")

    yield result


def apply_sample_fraction(records: RecordStream, step: dict) -> RecordStream:
    """
    Sample a fraction of the records.

    Args:
        records: A stream of records to sample a fraction from.
        step: Configuration dict containing the fraction to sample.

    Returns:
        A stream of sampled records.
    """
    p = step["p"]
    seed = step.get("seed")

    rng = random.Random(seed)

    for r in records:
        if rng.random() < p:
            yield r


def apply_sample_n(records: RecordStream, step: dict) -> RecordStream:
    """
    Sample a fixed number of records.

    Args:
        records: A stream of records to sample from.
        step: Configuration dict with 'n' and optional 'seed'.

    Returns:
        A stream of sampled records.
    """
    n = step["n"]
    seed = step.get("seed")

    for r in reservoir_sample(records, n, seed):
        yield r


def apply_map(records: RecordStream, step: dict) -> RecordStream:
    """
    Apply a function to each record.

    Args:
        records: A stream of records to apply the function to.
        step: Configuration dict containing the function to apply.

    Returns:
        A stream of mapped records.
    """
    func = step["func"]
    for r in records:
        result = func(r)
        if result is None:
            continue
        if not isinstance(result, dict):
            raise TypeError("map function must return a dict")
        yield result


def apply_fused(records: RecordStream, step: dict) -> RecordStream:
    """
    Apply a fused sequence of map/assign/filter operations.

    Args:
        records: A stream of records to apply fused operations to.
        step: Configuration dict with an 'ops' list and potentially embedded steps.

    Returns:
        A stream of records with the fused operations applied.
    """
    # Extract embedded steps
    embedded_steps = step.get("steps", [])

    # Sanity fallback: reconstruct from original plan if needed
    if not embedded_steps:
        raise ValueError("Fused step missing original embedded steps")

    # Apply them sequentially
    for sub_step in embedded_steps:
        op = sub_step["op"]
        if op == "map":
            records = apply_map(records, sub_step)
        elif op == "assign":
            records = apply_assign(records, sub_step)
        elif op == "filter":
            records = apply_filter(records, sub_step)
        else:
            raise ValueError(f"Unsupported op in fused step: {op}")
    return records
