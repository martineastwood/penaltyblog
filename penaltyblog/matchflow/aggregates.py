from collections import Counter

import numpy as np

from .aggs_registry import AGGS
from .steps.utils import get_field


@AGGS.register()
def count(field=None):
    return lambda rows: len(rows)


@AGGS.register()
def count_nonnull(field):
    return lambda rows: sum(1 for r in rows if get_field(r, field) is not None)


@AGGS.register("sum")
@AGGS.register("sum_")
def sum_(field):
    return lambda rows: np.nansum([_safe_get(r, field) for r in rows])


@AGGS.register("mean")
@AGGS.register("mean_")
def mean_(field):
    return lambda rows: (
        np.nanmean([_safe_get(r, field) for r in rows]) if rows else None
    )


@AGGS.register("min")
@AGGS.register("min_")
def min_(field):
    return lambda rows: np.nanmin([_safe_get(r, field) for r in rows]) if rows else None


@AGGS.register("max")
@AGGS.register("max_")
def max_(field):
    return lambda rows: np.nanmax([_safe_get(r, field) for r in rows]) if rows else None


@AGGS.register("std")
@AGGS.register("std_")
def std_(field):
    return lambda rows: np.nanstd([_safe_get(r, field) for r in rows]) if rows else None


@AGGS.register("median")
@AGGS.register("median_")
def median_(field):
    return lambda rows: (
        np.nanmedian([_safe_get(r, field) for r in rows]) if rows else None
    )


@AGGS.register("range")
@AGGS.register("range_")
def range_(field):
    return lambda rows: (
        (
            np.nanmax([_safe_get(r, field) for r in rows])
            - np.nanmin([_safe_get(r, field) for r in rows])
        )
        if rows
        else None
    )


@AGGS.register("iqr")
@AGGS.register("iqr_")
def iqr_(field):
    return lambda rows: (
        (
            np.nanpercentile([_safe_get(r, field) for r in rows], 75)
            - np.nanpercentile([_safe_get(r, field) for r in rows], 25)
        )
        if rows
        else None
    )


@AGGS.register("percentile")
def percentile_(field, q):
    return lambda rows: (
        np.nanpercentile([_safe_get(r, field) for r in rows], q) if rows else None
    )


@AGGS.register("mode")
def mode_(field):
    def _mode(rows):
        values = [get_field(r, field) for r in rows if get_field(r, field) is not None]
        if not values:
            return None
        return Counter(values).most_common(1)[0][0]

    return _mode


@AGGS.register("first")
@AGGS.register("first_")
def first_(field):
    return lambda rows: next(
        (get_field(r, field) for r in rows if get_field(r, field) is not None), None
    )


@AGGS.register("last")
@AGGS.register("last_")
def last_(field):
    return lambda rows: next(
        (
            get_field(r, field)
            for r in reversed(rows)
            if get_field(r, field) is not None
        ),
        None,
    )


@AGGS.register("unique")
def unique(field):
    return lambda rows: list({get_field(r, field) for r in rows})


@AGGS.register("list")
def list_(field):
    return lambda rows: [get_field(r, field) for r in rows]


@AGGS.register("all")
def all_(field):
    return lambda rows: all(get_field(r, field) for r in rows)


@AGGS.register("any")
def any_(field):
    return lambda rows: any(get_field(r, field) for r in rows)


def _safe_get(record, field):
    val = get_field(record, field)
    return np.nan if val is None else val
