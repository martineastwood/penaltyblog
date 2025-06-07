import datetime
from collections import defaultdict, deque
from typing import Iterator, Literal

from ..aggs_registry import resolve_aggregator
from .utils import get_field


def parse_window_size(window_str: str) -> float:
    """
    Parse a window size string like '5m', '2h', '30s', '1d' into a number of seconds.
    Supported suffixes: s=seconds, m=minutes, h=hours, d=days.
    """
    if not isinstance(window_str, str):
        raise ValueError(
            f"Expected a string for window, got {type(window_str).__name__}"
        )
    unit = window_str[-1].lower()
    try:
        val = float(window_str[:-1])
    except Exception:
        raise ValueError(f"Could not parse window size from '{window_str}'")
    if unit == "s":
        return val
    if unit == "m":
        return val * 60
    if unit == "h":
        return val * 3600
    if unit == "d":
        return val * 86400
    raise ValueError(f"Unrecognized time unit '{unit}' in window '{window_str}'")


def apply_group_rolling_summary(records: Iterator[dict], step: dict) -> Iterator[dict]:
    """
    Lazily apply a rolling summary within each group in a FlowGroup.
    Expects each incoming 'record' to be a dict of:
        {
          "__group_key__":     (<value1>, <value2>, …),
          "__group_records__": [list of raw rows belonging to that group]
        }
    Emits one “rolled‐up” row per original row in each group (whenever (i % step) == 0
    and window has at least min_periods), attaching the group‐by key fields as real columns.
    """
    window = step["window"]
    aggregators = step["aggregators"]
    time_field = step.get("time_field")
    min_periods = step.get("min_periods", 1)
    raw_step = step.get("step")
    step_size = raw_step if (isinstance(raw_step, int) and raw_step > 0) else 1
    group_keys = step.get("__group_keys") or []

    def process_one_group(group_key_tuple, group_records):
        """
        Given a single group’s key (tuple) and its list of original rows, yield
        one rolling‐summary row per input row (subject to min_periods/step_size).
        """
        # If time_field is set, sort that group by timestamp first:
        if time_field:
            group_records = sorted(
                group_records, key=lambda r: get_field(r, time_field)
            )

        window_deque = deque()
        results = []

        for idx, row in enumerate(group_records):
            # Append the new row into the sliding window
            window_deque.append(row)
            current_time = get_field(row, time_field) if time_field else None

            # If using a string window (e.g. "5m"), pop off any rows older than that Δ:
            if time_field and isinstance(window, str):
                max_seconds = parse_window_size(window)
                while window_deque:
                    oldest = window_deque[0]
                    oldest_time = get_field(oldest, time_field)
                    delta = current_time - oldest_time
                    if not isinstance(delta, datetime.timedelta):
                        # If these aren’t datetime objects, bail out
                        break
                    if delta.total_seconds() > max_seconds:
                        window_deque.popleft()
                    else:
                        break
            else:
                # Fixed‐size window by count
                while len(window_deque) > window:
                    window_deque.popleft()

            # Only emit when we have at least min_periods and (idx % step_size) == 0
            if len(window_deque) >= min_periods and (idx % step_size == 0):
                out = dict(row)  # start from the original row’s fields

                # Re‐attach each group_by key as a real column, if requested:
                for key_name, key_val in zip(group_keys, group_key_tuple):
                    out[key_name] = key_val

                # Now compute each aggregator on window_deque
                for out_field, (
                    agg_fn_name_or_callable,
                    in_field,
                ) in aggregators.items():
                    # `resolve_aggregator` can take either a callable or ("name", field) tuple
                    agg_callable = resolve_aggregator(
                        (agg_fn_name_or_callable, in_field), out_field
                    )
                    out[out_field] = agg_callable(list(window_deque))

                results.append(out)

        return results

    def runner(records):
        # First, “records” is an iterator over group‐dicts,
        # each of which has "__group_key__" and "__group_records__".
        for group_dict in records:
            grp_key = group_dict.get("__group_key__")
            grp_rows = group_dict.get("__group_records__", [])
            # We expect grp_key to be a tuple (even if it's length 1).
            # If group_keys was a single field, grp_key might be (val,) rather than val.
            for outrow in process_one_group(grp_key, grp_rows):
                yield outrow

    return runner(records)


def parse_window_size(window_str: str) -> float:
    """
    Parse a window size string like '5m','10m','1h','30s','1d' → seconds (float).
    Supported suffixes: 's', 'm', 'h', 'd'.
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


def apply_group_time_bucket(records, step):
    """
    For each group (provided as a dict with "__group_key__" & "__group_records__"),
    assign each record into a fixed, non-overlapping time bin of size `freq` (e.g. 5 minutes)
    and compute one aggregated output per bucket.

    Expects each incoming `record` to be:
        {
          "__group_key__": (<val1>, <val2>, …),
          "__group_records__": [ list of raw rows in that group ]
        }
    Emits exactly one row per non-empty bucket per group, with:
      - group-by key fields (as normal columns)
      - a bucket-label field (either bucket start or bucket end)
      - each aggregator’s output

    We assume `time_field` in each raw row is either a `datetime.datetime` or a `datetime.timedelta`.
    """
    freq_str = step["freq"]
    aggregators = step["aggregators"]
    time_field = step["time_field"]
    label_side = step.get("label", "left")
    group_keys = step.get("__group_keys", [])

    # Convert freq_str → bucket_size_seconds (float)
    bucket_size_seconds = parse_window_size(freq_str)

    def process_one_group(group_key_tuple, group_records):
        # 1) Sort group_records by time_field
        def _get_time(r):
            return get_field(r, time_field)

        # Filter out records without the time_field
        sorted_rows = sorted(
            (r for r in group_records if _get_time(r) is not None), key=_get_time
        )

        # 2) Determine, for each row, which bucket index it belongs to.
        #    We treat the "zero" origin as the *start of the match*, i.e. time=0.
        #    In other words, if row_time is a timedelta or datetime, we compute:
        #
        #      seconds_since_origin =
        #         row_time.total_seconds()           (if it's a timedelta)
        #         OR
        #         (row_time - min_datetime).total_seconds()  (if it's a datetime)
        #
        #    Then bucket_index = floor(seconds_since_origin / bucket_size_seconds).
        #
        #    We accumulate all rows with the same bucket_index into one bucket.

        # Detect if time_field values are timedelta or datetime
        sample_time = _get_time(sorted_rows[0])
        is_timedelta = isinstance(sample_time, datetime.timedelta)
        is_datetime = isinstance(sample_time, datetime.datetime)

        if t is None:
            continue  # Skip records without a valid time_field
        elif is_datetime:
            # Anchor the start (origin) at the earliest event’s full datetime
            origin = sample_time
            # Then for any row, seconds_since_origin = (row_time – origin).total_seconds()
        else:
            # Assume a timedelta (e.g. "00:01:17" → timedelta(minutes=1, seconds=17))
            origin = None
            # seconds_since_origin = row_time.total_seconds()

        buckets: dict[int, list] = defaultdict(list)
        bucket_labels: dict[int, float] = {}

        for row in sorted_rows:
            t = _get_time(row)
            if t is None:
                continue  # Skip records without a valid time_field
            elif is_datetime:
                delta = t - origin
                total_s = delta.total_seconds()
            else:
                # Attempt to treat as timedelta if not already
                try:
                    total_s = t.total_seconds()
                except AttributeError:
                    total_s = float(t)  # Assume it's a numeric type representing seconds

            bucket_index = int(total_s // bucket_size_seconds)
            buckets[bucket_index].append(row)

            # Compute bucket label (either left‐edge or right‐edge)
            if bucket_index not in bucket_labels:
                if label_side == "left":
                    label_s = bucket_index * bucket_size_seconds
                else:  # "right"
                    label_s = (bucket_index + 1) * bucket_size_seconds

                # Convert that numeric label_s back into same type as t
                if is_datetime:
                    bucket_labels[bucket_index] = origin + datetime.timedelta(
                        seconds=label_s
                    )
                else:
                    bucket_labels[bucket_index] = datetime.timedelta(seconds=label_s)

        # 3) Now that we have grouped rows into buckets[bucket_index],
        #    produce exactly one output row per bucket_index.
        results = []
        for bidx, rows_in_bucket in buckets.items():
            out = {}

            # reattach each group_by key as real columns
            for key_name, key_val in zip(group_keys, group_key_tuple):
                out[key_name] = key_val

            # attach the bucket label (e.g. "5m bucket starting at 00:00" or ending at 00:05)
            # We’ll call this field "bucket" (you can rename as you wish)
            out["bucket"] = bucket_labels[bidx]

            # For each user-supplied aggregator, resolve and apply over rows_in_bucket
            for out_field, (agg_fn_name_or_callable, in_field) in aggregators.items():
                agg_callable = resolve_aggregator(
                    (agg_fn_name_or_callable, in_field), out_field
                )
                out[out_field] = agg_callable(rows_in_bucket)

            results.append(out)

        return results

    def runner(all_group_dicts):
        for group_dict in all_group_dicts:
            grp_key = group_dict.get("__group_key__")
            grp_rows = group_dict.get("__group_records__", [])
            for outrow in process_one_group(grp_key, grp_rows):
                yield outrow

    return runner(records)


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
