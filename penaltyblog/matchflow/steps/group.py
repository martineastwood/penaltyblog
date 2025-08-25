from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Iterator, Optional, Tuple, Union

from ..aggs_registry import resolve_aggregator
from .utils import get_field


def get_time_window_details(
    window: Union[int, float, str], time_field: Optional[str]
) -> Tuple[bool, Optional[int], Optional[float], Optional[datetime], bool]:
    """
    Determine the mode (count or time) and parse the window size.

    Parameters
    ----------
    window : int, float, or str
        Window size as integer/float (row count) or string (e.g. '5m', '1h').
    time_field : str or None
        Name of the time field, required for time-based windows.

    Returns
    -------
    tuple
        (count_mode: bool, count_window: Optional[int], time_window_seconds: Optional[float], origin: Optional[datetime], is_datetime: bool)
    """
    if isinstance(window, (int, float)):
        return True, int(window), float(window), None, False
    elif isinstance(window, str) and window[-1].lower() in {"s", "m", "h", "d"}:
        time_window_seconds = parse_window_size(window)
        if time_field is None:
            raise ValueError("String window requires a time_field")
        return False, None, time_window_seconds, None, False
    else:
        raise ValueError(
            f"Invalid window {window!r}: use int for row-count or str ending in s/m/h/d for time."
        )


def apply_group_rolling_summary(
    records: Iterator[dict[str, Any]], step: dict[str, Any]
) -> Iterator[dict[str, Any]]:
    """
    Lazily apply a rolling summary within each group.

    Two modes:
      - Count mode: `window` is int → last N rows
      - Time mode:  `window` is str ending in s/m/h/d → last T seconds

    In time mode, `time_field` must be datetime or timedelta.

    Parameters
    ----------
    records : Iterator[dict]
        Iterator of records, each a dict.
    step : dict
        Step configuration dict, must include 'window', 'aggregators', and optionally 'time_field', 'min_periods', 'step', '__group_keys'.

    Returns
    -------
    Iterator[dict]
        Iterator of records with rolling summary fields attached.
    """
    window = step["window"]
    aggregators = step["aggregators"]
    time_field = step.get("time_field")
    min_periods = step.get("min_periods", 1)
    raw_step = step.get("step")
    step_size = raw_step if (isinstance(raw_step, int) and raw_step > 0) else 1
    group_keys = step.get("__group_keys") or []

    count_mode, count_window, time_window_seconds, _, _ = get_time_window_details(
        window, time_field
    )

    def process_one_group(
        group_key_tuple: tuple[Any, ...], group_records: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        # sort by time_field if time mode, else leave original order

        if not count_mode and time_field is not None:
            # validate time_field type
            sample = get_field(group_records[0], time_field)
            if isinstance(sample, datetime):
                local_origin = sample
            elif isinstance(sample, timedelta):
                local_origin = None  # For timedelta, we don't need a datetime origin
            else:
                raise ValueError(
                    f"Rolling-summary: time_field '{time_field}' is {type(sample).__name__}; "
                    "for a string window you must supply datetime or timedelta values."
                )
            group_records = sorted(
                group_records, key=lambda r: get_field(r, time_field)
            )

        # Initialize window_deque regardless of mode
        window_deque: deque[dict[str, Any]] = deque()
        results = []

        for idx, row in enumerate(group_records):
            window_deque.append(row)

            # drop old items
            if count_mode and count_window is not None:
                while len(window_deque) > count_window:
                    window_deque.popleft()
                current_window = list(window_deque)
            else:
                # time‐based eviction
                if time_field is not None:
                    t = get_field(row, time_field)
                    now_s: float = 0.0
                    if isinstance(t, datetime) and local_origin is not None:
                        now_s = (t - local_origin).total_seconds()
                    elif isinstance(t, timedelta):
                        now_s = t.total_seconds()
                    else:
                        now_s = float(t) if t is not None else 0.0

                    while window_deque:
                        oldest = window_deque[0]
                        old_t = get_field(oldest, time_field)
                        old_s: float = 0.0
                        if isinstance(old_t, datetime) and local_origin is not None:
                            old_s = (old_t - local_origin).total_seconds()
                        elif isinstance(old_t, timedelta):
                            old_s = old_t.total_seconds()
                        else:
                            old_s = float(old_t) if old_t is not None else 0.0

                        if (
                            time_window_seconds is not None
                            and now_s - old_s > time_window_seconds
                        ):
                            window_deque.popleft()
                        else:
                            break
                current_window = list(window_deque)

            # emit if enough and on step
            if len(current_window) >= min_periods and (idx % step_size == 0):
                out = dict(row)
                # reattach group keys
                for key_name, key_val in zip(group_keys, group_key_tuple):
                    out[key_name] = key_val

                # compute aggregations
                for out_field, (fn, in_f) in aggregators.items():
                    agg_fn = resolve_aggregator((fn, in_f), out_field)
                    out[out_field] = agg_fn(current_window)
                results.append(out)

        return results

    def runner(records_iter):
        for group_dict in records_iter:
            key = group_dict["__group_key__"]
            recs = group_dict.get("__group_records__", [])
            yield from process_one_group(key, recs)

    return runner(records)


def parse_window_size(window_str: str) -> float:
    """
    Parse a window size string like '5m', '10m', '1h', '30s', '1d' to seconds (float).

    Parameters
    ----------
    window_str : str
        Window size string, must end with 's', 'm', 'h', or 'd'.

    Returns
    -------
    float
        Window size in seconds.

    Raises
    ------
    ValueError
        If the string cannot be parsed or has an unrecognized unit.
    """
    if not isinstance(window_str, str):
        raise ValueError(f"Expected string for freq, got {type(window_str).__name__}")
    unit = window_str[-1].lower()
    try:
        val = float(window_str[:-1])
    except:
        raise ValueError(f"Could not parse window size from '{window_str}'")
    if unit == "s":
        return val
    if unit == "m":
        return val * 60
    if unit == "h":
        return val * 3600
    if unit == "d":
        return val * 86400
    raise ValueError(f"Unrecognized unit '{unit}' in window '{window_str}'")


def apply_group_time_bucket(
    records: Iterator[dict[str, Any]], step: dict[str, Any]
) -> Iterator[dict[str, Any]]:
    """
    Assign each record in a group to a fixed, non-overlapping time bin.

    Two modes:
      - String freq with suffix (e.g. '5m', '1h'): requires datetime or timedelta time_field.
      - Numeric freq (int/float): buckets numeric values directly.

    Parameters
    ----------
    records : Iterator[dict]
        Iterator of group dicts, each with '__group_key__' and '__group_records__'.
    step : dict
        Step configuration dict, must include 'freq', 'aggregators', 'time_field', and optionally 'label', 'bucket_name', '__group_keys'.

    Returns
    -------
    Iterator[dict]
        Iterator of records with bucket assignments and aggregated fields.
    """
    freq = step["freq"]
    aggregators = step["aggregators"]
    time_field = step["time_field"]
    label_side = step.get("label", "left")
    bucket_name = step.get("bucket_name", "bucket")
    group_keys = step.get("__group_keys", [])

    numeric_mode, _, bucket_size, _, _ = get_time_window_details(freq, time_field)

    def process_one_group(
        group_key_tuple: tuple[Any, ...], group_records: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        # Extract non-null values
        def _get_time(r: dict[str, Any]) -> Any:
            if time_field is not None:
                return get_field(r, time_field)
            return None

        rows = [r for r in group_records if _get_time(r) is not None]
        if not rows:
            return []

        # Sample one to inspect type
        sample = _get_time(rows[0])

        # Initialize variables with proper types

        # Time-based mode: must be datetime or timedelta
        if not numeric_mode:
            if isinstance(sample, datetime):
                local_origin = sample
            elif isinstance(sample, timedelta):
                local_origin = timedelta(0)
            else:
                raise ValueError(
                    f"time_bucket: field '{time_field}' has type {type(sample).__name__}; "
                    "when freq has a time suffix you must provide datetime or timedelta values."
                )
            # sort by timestamp/timedelta
            rows.sort(key=_get_time)
        else:
            # numeric mode: we treat values as floats, no origin needed
            # Sort by numeric time field
            rows.sort(key=_get_time)

        # Partition into buckets
        buckets: dict[int, list[dict]] = {}
        labels: dict[int, Union[datetime, timedelta, float]] = {}

        for r in rows:
            t = _get_time(r)
            total: float = 0.0

            if (
                not numeric_mode
                and isinstance(t, datetime)
                and isinstance(local_origin, datetime)
            ):
                total = (t - local_origin).total_seconds()
            elif not numeric_mode and isinstance(t, timedelta):
                total = t.total_seconds()
            else:
                total = float(t) if t is not None else 0.0

            if bucket_size is not None:
                idx = int(total // bucket_size)
                buckets.setdefault(idx, []).append(r)

                if idx not in labels:
                    edge = (idx + (1 if label_side == "right" else 0)) * bucket_size

                    if not numeric_mode:
                        # datetime label
                        if isinstance(local_origin, datetime):
                            labels[idx] = local_origin + timedelta(seconds=edge)
                        else:
                            labels[idx] = timedelta(seconds=edge)
                    else:
                        labels[idx] = float(edge)  # Ensure numeric labels are float

        # Build output
        out = []
        for idx, group in buckets.items():
            row_out = {k: v for k, v in zip(group_keys, group_key_tuple)}
            row_out[bucket_name] = labels[idx]
            for out_field, (fn, in_f) in aggregators.items():
                agg = resolve_aggregator((fn, in_f), out_field)
                row_out[out_field] = agg(group)
            out.append(row_out)
        return out

    def runner(all_groups: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        for g in all_groups:
            key = g["__group_key__"]
            recs = g.get("__group_records__", [])
            result = process_one_group(key, recs)
            yield from result

    return runner(records)


def apply_group_by(
    records: Iterator[dict[str, Any]], step: dict[str, Any]
) -> Iterator[dict[str, Any]]:
    """
    Group records by one or more fields.

    Parameters
    ----------
    records : Iterator[dict]
        Iterator of records to group.
    step : dict
        Step configuration dict, must include 'keys'.

    Returns
    -------
    Iterator[dict]
        Iterator of group dicts, each with '__group_key__' and '__group_records__'.
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


def apply_group_summary(
    records: Iterator[dict[str, Any]], step: dict[str, Any]
) -> Iterator[dict[str, Any]]:
    """
    Apply a summary function to each group of records.

    Parameters
    ----------
    records : Iterator[dict]
        Iterator of group dicts, each with '__group_key__' and '__group_records__'.
    step : dict
        Step configuration dict, must include 'agg' and optionally 'group_keys'.

    Returns
    -------
    Iterator[dict]
        Iterator of summary dicts for each group.
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


def apply_group_cumulative(
    records: Iterator[dict[str, Any]], step: dict[str, Any]
) -> Iterator[dict[str, Any]]:
    """
    Apply a cumulative sum to a field for each group of records.

    Parameters
    ----------
    records : Iterator[dict]
        Iterator of group dicts, each with '__group_key__' and '__group_records__'.
    step : dict
        Step configuration dict, must include 'field' and 'alias'.

    Returns
    -------
    Iterator[dict]
        Iterator of records with cumulative field attached.
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
