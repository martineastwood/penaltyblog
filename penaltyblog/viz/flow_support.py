import re
from typing import List, Optional, Union

import pandas as pd

from ..matchflow.flow import Flow
from ..matchflow.helpers import resolve_path


def normalize_path(path: str) -> str:
    # Convert location[0] → location.0
    return re.sub(r"\[(\d+)\]", r".\1", path)


def to_records(
    data: Union[Flow, pd.DataFrame, List[dict]], fields: Optional[List[str]] = None
) -> List[dict]:
    if isinstance(data, pd.DataFrame):
        return (
            data[fields].to_dict(orient="records")
            if fields
            else data.to_dict(orient="records")
        )
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
