from typing import Any, Callable

import numpy as np

_AGGS: dict[str, Callable] = {
    "sum": np.sum,
    "mean": np.mean,
    "min": np.min,
    "max": np.max,
    "count": len,
    "median": np.median,
    "std": np.std,
    "var": np.var,
    "nunique": lambda vals: len(set(vals)),
    "first": lambda vals: vals[0] if vals else None,
    "last": lambda vals: vals[-1] if vals else None,
    "any": lambda vals: any(vals),
    "all": lambda vals: all(vals),
}


def _resolve_agg(records: list[dict], spec) -> Any:
    if isinstance(spec, str):
        try:
            func = _AGGS[spec]
        except KeyError:
            raise ValueError(f"Unknown aggregate {spec!r}")
        return func(records)

    if isinstance(spec, tuple) and len(spec) == 2:
        field, agg = spec
        try:
            func = _AGGS[agg]
        except KeyError:
            raise ValueError(f"Unknown aggregate {agg!r}")
        vals = [r[field] for r in records if field in r and r[field] is not None]
        return func(vals)

    if callable(spec):
        return spec(records)

    raise ValueError(f"Bad aggregate spec {spec!r}")


def sanitize_filename(value: Any) -> str:
    return str(value).replace(" ", "_").replace("/", "-")
