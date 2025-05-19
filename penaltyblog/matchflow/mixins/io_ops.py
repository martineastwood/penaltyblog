# File: io_ops.py

import glob
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Union

import pandas as pd

from ..core import sanitize_filename

# ── Pick JSON backend ───────────────────────────────────────────────────────────
try:
    import orjson  # type: ignore

    def _dumps(obj: Any) -> str:
        # orjson.dumps returns bytes
        return orjson.dumps(obj).decode("utf-8")

    def _loads(s: Union[str, bytes]) -> Any:
        return orjson.loads(s)

except ImportError:
    import json as _json  # type: ignore

    _dumps = _json.dumps
    _loads = _json.loads
# ────────────────────────────────────────────────────────────────────────────────


class IOOpsMixin:
    def to_json(self) -> str:
        """
        Serialize the flow to a (compact) JSON string.

        Consumes the stream (materializes all records).

        Returns:
            str: The JSON string.
        """
        return _dumps(self.collect())

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert the Flow into a pandas DataFrame.

        Consumes the stream (materializes all records).

        Returns:
            pd.DataFrame: The DataFrame.
        """
        return pd.DataFrame(self.collect())

    def describe(
        self,
        percentiles: Optional[tuple[float, ...]] = (0.25, 0.5, 0.75),
        include: Optional[list[Any]] = None,
        exclude: Optional[list[Any]] = None,
    ) -> pd.DataFrame:
        """
        Generate descriptive statistics.

        Consumes (materializes) the stream to build a DataFrame.

        Args:
            percentiles (tuple of float): Percentiles to include between 0 and 1.
            include (list of dtypes or None): Which dtypes to include (as in pandas.describe).
            exclude (list of dtypes or None): Which dtypes to exclude.

        Returns:
            DataFrame: the same as pandas.DataFrame.describe().
        """
        df = pd.DataFrame(self.collect())
        return df.describe(percentiles=percentiles, include=include, exclude=exclude)

    def to_json_files(self, folder: Union[str, Path], by: Optional[str] = None) -> None:
        """
        Write each record to its own .json file under `folder`.

        Args:
            folder (Union[str, Path]): The directory to write files to.
            by (Optional[str]): Key to use for file names (default: 'id').

        Returns:
            None
        """
        folder_p = Path(folder)
        folder_p.mkdir(parents=True, exist_ok=True)

        for i, rec in enumerate(self.collect(), start=1):
            name = (
                sanitize_filename(rec.get(by, f"record_{i}")) if by else f"record_{i}"
            )
            path = folder_p / f"{name}.json"
            path.write_text(_dumps(rec), encoding="utf-8")

    def to_jsonl(self, path: Union[str, Path]) -> None:
        """
        Write out as JSON Lines (.jsonl), one record per line.

        Args:
            path (Union[str, Path]): The path to write the file to.

        Returns:
            None
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for rec in self.collect():
                f.write(_dumps(rec))
                f.write("\n")

    def to_json_single(self, path: Union[str, Path]) -> None:
        """
        Write all records as a single JSON array.

        Args:
            path (Union[str, Path]): The path to write the file to.

        Returns:
            None
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_dumps(self.collect()), encoding="utf-8")

    @classmethod
    def from_generator(cls, gen: Iterator[dict[Any, Any]]) -> "Flow":
        """
        Create a Flow from an iterator of records.

        Args:
            gen (Iterator[dict[Any, Any]]): An iterator of records.

        Returns:
            Flow: A Flow containing the records from the iterator.
        """
        return cls(gen)

    @classmethod
    def from_jsonl(cls, path: Union[str, Path]) -> "Flow":
        """
        Stream a .jsonl file (one JSON object per line).

        Args:
            path (Union[str, Path]): The path to the file to read.

        Returns:
            Flow: A Flow containing the records from the file.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)

        def gen():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line:
                    yield _loads(line)

        return cls.from_generator(gen())

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Flow":
        """
        Load a single .json file (object or array).

        Args:
            path (Union[str, Path]): The path to the file to read.

        Returns:
            Flow: A Flow containing the records from the file.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)

        text = p.read_text(encoding="utf-8")
        data = _loads(text)

        # if it’s a list, stream each element; otherwise wrap in a list
        iterable = data if isinstance(data, list) else [data]
        return cls.from_generator(iter(iterable))

    @classmethod
    def from_folder(cls, folder: Union[str, Path]) -> "Flow":
        """
        Stream every .json and .jsonl in a directory.

        Args:
            folder (Union[str, Path]): The directory to read from.

        Returns:
            Flow: A Flow containing the records from the directory.
        """
        folder_p = Path(folder)
        if not folder_p.is_dir():
            raise NotADirectoryError(folder)

        def gen():
            for p in folder_p.iterdir():
                if p.suffix.lower() == ".json":
                    yield from cls.from_file(p)
                elif p.suffix.lower() == ".jsonl":
                    yield from cls.from_jsonl(p)

        return cls.from_generator(gen())

    @classmethod
    def from_glob(cls, pattern: Union[str, Path]) -> "Flow":
        """
        Stream all JSON files matching a glob.

        Args:
            pattern (Union[str, Path]): The glob pattern.

        Returns:
            Flow: A Flow containing the records from the glob.
        """

        def gen():
            for fp in glob.glob(str(pattern), recursive=True):
                p = Path(fp)
                if p.suffix.lower() == ".json":
                    yield from cls.from_file(p)
                elif p.suffix.lower() == ".jsonl":
                    yield from cls.from_jsonl(p)

        return cls.from_generator(gen())

    @classmethod
    def from_records(
        cls, data: Union[dict[Any, Any], list[dict[Any, Any]], Iterable[dict[Any, Any]]]
    ) -> "Flow":
        """
        Create a Flow from one or more dict-like records.

        Args:
            data (dict | list[dict] | Iterable[dict]): The data to create the flow from.

        Returns:
            Flow: The created flow.
        """
        return cls(data)
