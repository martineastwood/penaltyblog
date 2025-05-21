import copy
import random
from collections import OrderedDict, defaultdict

from .utils import (
    fast_get_field,
    flatten_dict,
    get_field,
    reservoir_sample,
    set_nested_field,
)


def apply_filter(records, step):
    pred = step["predicate"]
    for r in records:
        if pred(r):
            yield r


def apply_assign(records, step):
    fields = step["fields"]
    for r in records:
        new = dict(r)
        for k, func in fields.items():
            new[k] = func(r)
        yield new


def apply_select(records, step):
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
            out = {}
            for path in compiled_paths:
                value = fast_get_field(record, path)
                set_nested_field(out, ".".join(path), value)
            yield out


def _del_nested(d, parts):
    """Delete a nested key from a dict given a list of parts."""
    for part in parts[:-1]:
        d = d.get(part, {})
    d.pop(parts[-1], None)


def _set_nested(d, parts, value):
    """Set a nested key in a dict given a list of parts."""
    for part in parts[:-1]:
        if part not in d or not isinstance(d[part], dict):
            d[part] = {}
        d = d[part]
    d[parts[-1]] = value


def apply_rename(records, step):
    mapping = step["mapping"]
    compiled_mapping = step.get("_compiled_mapping")
    if not compiled_mapping:
        compiled_mapping = [
            (old, old.split("."), new_key, new_key.split("."))
            for old, new_key in mapping.items()
        ]
        step["_compiled_mapping"] = compiled_mapping

    needs_deep = any("." in old or "." in new for old, new in mapping.items())

    for r in records:
        new = copy.deepcopy(r) if needs_deep else r.copy()
        for old, old_parts, new_key, new_parts in compiled_mapping:
            val = get_field(new, old_parts)
            if val is not None:
                _del_nested(new, old_parts)
                _set_nested(new, new_parts, val)
        yield new


def apply_sort(records, step):
    keys = step["keys"]
    ascending = step.get("ascending", [True] * len(keys))
    records = list(records)

    def sort_key(r):
        key_parts = []
        for k, asc in zip(keys, ascending):
            v = r.get(k)
            # Reverse sort by negating numeric or inverting sortable value
            key_parts.append(v if asc else _invert(v))
        return tuple(key_parts)

    return iter(sorted(records, key=sort_key))


def _invert(value):
    if isinstance(value, (int, float)):
        return -value
    elif isinstance(value, str):
        return "".join(chr(255 - ord(c)) for c in value)  # crude inverse for strings
    else:
        return value  # fallback: not reversed


def apply_limit(records, step):
    count = step["count"]
    for i, record in enumerate(records):
        if i >= count:
            break
        yield record


def apply_drop(records, step):
    keys = step["keys"]

    for record in records:
        new = dict(record)
        for key in keys:
            parts = key.split(".")
            d = new
            try:
                for part in parts[:-1]:
                    d = d.get(part)
                    if not isinstance(d, dict):
                        raise KeyError
                d.pop(parts[-1], None)  # silently ignore missing keys
            except Exception:
                continue  # skip malformed paths
        yield new


def apply_flatten(records, step):
    for r in records:
        yield flatten_dict(r)


def apply_distinct(records, step):
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


def apply_dropna(records, step):
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


def apply_explode(records, step):
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


def apply_join(records, step):
    from ..executor import FlowExecutor

    on = step["on"]
    suffix = step.get("suffix", "_right")
    how = step.get("how", "left")

    compiled = step.get("_compiled_on")
    if not compiled:
        compiled = [k.split(".") for k in on]
        step["_compiled_on"] = compiled

    right_records = list(FlowExecutor(step["right_plan"]).execute())

    right_index = {}
    for r in right_records:
        key = tuple(get_field(r, k) for k in compiled)
        right_index.setdefault(key, []).append(r)

    for left in records:
        key = tuple(get_field(left, k) for k in compiled)
        matches = right_index.get(key)

        if not matches:
            if how == "left":
                yield dict(left)
            continue

        for right in matches:
            joined = dict(left)
            for rk, rv in right.items():
                if rk in on:
                    continue
                if rk in joined:
                    joined[rk + suffix] = rv
                else:
                    joined[rk] = rv
            yield joined


def apply_split_array(records, step):
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


def apply_pivot(records, step):
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


def apply_summary(records, step):
    agg_func = step["agg"]
    rows = list(records)
    result = agg_func(rows)

    if not isinstance(result, dict):
        raise ValueError("summary function must return a dict")

    yield result


def apply_sample_fraction(records, step):
    p = step["p"]
    seed = step.get("seed")

    rng = random.Random(seed)

    for r in records:
        if rng.random() < p:
            yield r


def apply_sample_n(records, step):
    n = step["n"]
    seed = step.get("seed")

    for r in reservoir_sample(records, n, seed):
        yield r


def apply_map(records, step):
    func = step["func"]
    for r in records:
        result = func(r)
        if result is None:
            continue
        if not isinstance(result, dict):
            raise TypeError("map function must return a dict")
        yield result
