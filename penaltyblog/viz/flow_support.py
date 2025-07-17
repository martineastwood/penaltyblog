import re
from typing import Any, List, Optional, Union

import pandas as pd

from ..matchflow.flow import Flow


def normalize_path(path: str) -> str:
    """
    Convert path expressions of the form location[0] into location.0, which
    is the syntax used in Flow's selectors.
    """
    return re.sub(r"\[(\d+)\]", r".\1", path)


def resolve_path(record: Any, path: str, default=None):
    """
    Resolve a dot-separated path in a record (which can be a dict, list, or tuple).
    If any part of the path is invalid, the function returns the default value.

    Args:
        record: The record to resolve the path from.
        path: The path to resolve, as a dot-separated string.
        default: The value to return if the path is invalid. Defaults to None.

    Returns:
        The resolved value or the default value if the path is invalid.
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
    """
    Convert a Flow, DataFrame, or list of dictionaries to a list of records.

    Args:
        data: The data to convert. Can be a Flow, DataFrame, or list of dictionaries.
        fields: A list of field names to include in the records. Defaults to None.

    Returns:
        A list of records, where each record is a dictionary containing the specified fields.
    """
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
