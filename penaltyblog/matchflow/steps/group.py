from collections import defaultdict
from typing import Iterator

from .utils import get_field


def apply_group_by(records, step) -> Iterator[dict]:
    """
    Group records by one or more fields.

    Args:
        records (list[dict]): The records to group.
        step (dict): The step to apply.

    Returns:
        list[dict]: The grouped records.
    """
    keys = step["keys"]
    compiled = step.get("_compiled_keys")

    if not compiled:
        compiled = [k.split(".") for k in keys]
        step["_compiled_keys"] = compiled

    grouped = defaultdict(list)
    for record in records:
        key = tuple(get_field(record, k) for k in compiled)
        grouped[key].append(record)

    for key, group_records in grouped.items():
        yield {"__group_key__": key, "__group_records__": group_records}


def apply_group_summary(records, step) -> Iterator[dict]:
    """
    Apply a summary function to each group of records.

    Args:
        records (list[dict]): The records to group.
        step (dict): The step to apply.

    Returns:
        list[dict]: The grouped records.
    """
    agg_func = step["agg"]
    group_keys = step.get("group_keys")  # get actual group key names if available

    for group in records:
        key = group["__group_key__"]
        rows = group["__group_records__"]

        result = agg_func(rows)
        if not isinstance(result, dict):
            raise ValueError("group_summary function must return a dict")

        if group_keys:
            output = {k: v for k, v in zip(group_keys, key)}
        else:
            output = {f"group_{i}": v for i, v in enumerate(key)}

        output.update(result)
        yield output


def apply_group_cumulative(records, step) -> Iterator[dict]:
    """
    Apply a cumulative function to each group of records.

    Args:
        records (list[dict]): The records to group.
        step (dict): The step to apply.

    Returns:
        list[dict]: The grouped records.
    """
    field = step["field"]
    alias = step["alias"]

    for group in records:
        key = group["__group_key__"]
        rows = group["__group_records__"]
        total = 0
        for r in rows:
            total += r.get(field, 0)
            new_r = dict(r)
            new_r[alias] = total
            yield new_r
