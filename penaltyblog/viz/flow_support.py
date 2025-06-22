import re
from typing import Any, List, Optional, Union

import pandas as pd

from ..matchflow.flow import Flow


def normalize_path(path: str) -> str:
    # Convert location[0] → location.0
    return re.sub(r"\[(\d+)\]", r".\1", path)


def resolve_path(record: Any, path: str, default=None):
    """
    Safely follow a dot-separated path, auto-indexing into lists when
    a segment is all digits.
    """
    current = record
    for segment in path.split("."):
        if current is None:
            return default

        # dict lookup
        if isinstance(current, dict):
            current = current.get(segment, default)

        # list/tuple index
        elif isinstance(current, (list, tuple)) and segment.isdigit():
            idx = int(segment)
            if 0 <= idx < len(current):
                current = current[idx]
            else:
                return default

        else:
            return default

    return current if current is not None else default


def to_records(
    data: Union[Flow, pd.DataFrame, List[dict]], fields: Optional[List[str]] = None
) -> List[dict]:
    if isinstance(data, pd.DataFrame):
        # materialize rows
        records = data.to_dict(orient="records")
        if not fields:
            return records
        return [
            {field: resolve_path(rec, field) for field in fields} for rec in records
        ]
    elif isinstance(data, Flow):
        records = data.collect()
    elif isinstance(data, list):
        records = data
    else:
        raise TypeError(f"Unsupported input type: {type(data)}")

    if not fields:
        return records

    return [
        {field: resolve_path(record, field) for field in fields} for record in records
    ]
