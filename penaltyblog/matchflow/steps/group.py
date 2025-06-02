from collections import defaultdict, deque
from typing import Iterator

from .utils import get_field


def apply_group_rolling_summary(records, step) -> Iterator[dict]:
    """
    Streaming rolling aggregation per group — supports row or time-based windows.
    """
    window = step["window"]
    min_periods = step.get("min_periods", 1)
    time_field = step.get("time_field")
    aggregators = step["aggregators"]

    # Parse time-based window string, e.g. "30s", "5m"
    def parse_time_window(w: str) -> int:
        units = {"s": 1, "m": 60}
        if not isinstance(w, str) or len(w) < 2:
            raise ValueError(
                f"Invalid window string: {w}, expected format: '30s' or '5m'"
            )
        unit = w[-1]
        if unit not in units:
            raise ValueError(f"Unsupported time unit in window: {w}")
        return int(w[:-1]) * units[unit]

    time_window_s = None
    if isinstance(window, str):
        if not time_field:
            raise ValueError("`time_field` is required for time-based rolling windows")
        time_window_s = parse_time_window(window)

    for group in records:
        rows = group["__group_records__"]
        buffer = deque()

        for row in rows:
            current_time = row.get(time_field) if time_field else None
            buffer.append(row)

            if time_window_s is not None:
                # Time-based window
                while (
                    buffer
                    and (current_time - buffer[0][time_field]).total_seconds()
                    > time_window_s
                ):
                    buffer.popleft()
            else:
                # Row-count window
                if len(buffer) > window:
                    buffer.popleft()

            result = dict(row)

            if len(buffer) >= min_periods:
                for alias, fn in aggregators.items():
                    result[alias] = fn(list(buffer))
            else:
                for alias in aggregators:
                    result[alias] = None

            yield result


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
