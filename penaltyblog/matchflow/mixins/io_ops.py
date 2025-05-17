"""
IO operations for handling a streaming data pipeline, specifically the Flow class.
"""

import glob
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import pandas as pd

from ..consumption_guard import guard_consumption
from ..core import sanitize_filename


class IOOpsMixin:

    @guard_consumption
    def to_json(self, indent: Optional[int] = None) -> str:
        """
        Serialize the flow to a JSON string.

        Consumes the stream (materializes all records).

        Args:
            indent (int or None): The number of spaces to use for indentation.
                - If None (default), the JSON string is compact.
                - If an integer n, the JSON string is formatted with n spaces per indentation level.

        Returns:
            str: The JSON string.
        """
        return json.dumps(self.collect(), indent=indent)

    @guard_consumption
    def to_pandas(self) -> pd.DataFrame:
        """
        Convert the Flow to a pandas DataFrame.

        Consumes (materializes) the stream to build a DataFrame.

        Returns:
            DataFrame: A pandas DataFrame containing the records.
        """
        self._consumed = self._is_consumable()
        return pd.DataFrame(self._records)

    @guard_consumption
    def describe(
        self,
        percentiles: tuple[float, ...] = (0.25, 0.5, 0.75),
        include: list | None = None,
        exclude: list | None = None,
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
        self._consumed = self._is_consumable()
        df = pd.DataFrame(self.collect())
        return df.describe(percentiles=percentiles, include=include, exclude=exclude)

    @guard_consumption
    def to_json_files(
        self, folder: Union[str, Path], by: Union[str, None] = None
    ) -> None:
        """
        Write each record to a separate JSON file in the given folder.

        Consumes (materializes) the stream and serializes every record to disk.

        Args:
            folder (str or Path): Output folder path. Will be created if needed.
            by (str, optional): Field to name the files by. Defaults to numbered files.

        Returns:
            None
        """
        self._consumed = self._is_consumable()
        folder_p = Path(folder)
        folder_p.mkdir(parents=True, exist_ok=True)

        data = self.collect()
        for i, record in enumerate(data, start=1):
            if by:
                name = sanitize_filename(record.get(by, f"record_{i}"))
            else:
                name = f"record_{i}"
            path = folder_p / f"{name}.json"
            path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    @guard_consumption
    def to_jsonl(self, path: Union[str, Path], encoding: str = "utf-8") -> None:
        """
        Save all records to a single JSON Lines (.jsonl) file.
        Each record is written as one line of JSON.

        Consumes (materializes) the stream and serializes every record to disk.

        Args:
            path (str or Path): Output file path.
            encoding (str, optional): File encoding. Defaults to "utf-8".

        Returns:
            None
        """
        self._consumed = self._is_consumable()
        p = Path(path)
        # ensure parent folder exists
        if p.parent:
            p.parent.mkdir(parents=True, exist_ok=True)

        data = self.collect()
        with p.open("w", encoding=encoding) as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False))
                f.write("\n")

    @guard_consumption
    def to_json_single(
        self, path: str | Path, encoding: str = "utf-8", indent: int | None = 2
    ) -> None:
        """
        Save all records to a single JSON file as an array.

        Consumes (materializes) the stream and serializes every record to disk.

        Args:
            path (str or Path): Output file path.
            encoding (str, optional): File encoding. Defaults to "utf-8".
            indent (int or None): Indentation level. Defaults to 2.

        Returns:
            None
        """
        self._consumed = self._is_consumable()
        p = Path(path)
        if p.parent:
            p.parent.mkdir(parents=True, exist_ok=True)

        data = self.collect()
        p.write_text(
            json.dumps(data, ensure_ascii=False, indent=indent),
            encoding=encoding,
        )

    @classmethod
    def from_generator(cls, generator_instance: Iterator[dict[Any, Any]]) -> "Flow":
        """
        Create a Flow from a generator function.

        Does not consume the stream.

        Args:
            generator_instance (Iterator[dict[Any, Any]]): A generator function.

        Returns:
            Flow: A Flow object.
        """
        return cls(generator_instance)

    @classmethod
    def from_jsonl(cls, path: str | Path, encoding: str = "utf-8") -> "Flow":
        """
        Load a .jsonl (JSON Lines) file into a Flow.
        Each line must be a valid JSON object.

        Consumes the file stream; the resulting Flow is a stream of records.

        Args:
            path (str or Path): Input file path.
            encoding (str, optional): File encoding. Defaults to "utf-8".

        Returns:
            Flow: A Flow object.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

        def generator():
            with p.open("r", encoding=encoding) as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)

        return cls.from_generator(generator())

    @classmethod
    def from_file(cls, path: str | Path, encoding: str = "utf-8") -> "Flow":
        """
        Load a local JSON file (list or single dict) into a Flow.
        Generic — no provider-specific assumptions.

        Consumes the file stream; the resulting Flow is a stream of records.

        Args:
            path (str or Path): Input file path.
            encoding (str, optional): File encoding. Defaults to "utf-8".

        Returns:
            Flow: A Flow object.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

        text = p.read_text(encoding=encoding)
        data: dict[Any, Any] | list[dict[Any, Any]]
        data = json.loads(text)

        if isinstance(data, list):
            return cls.from_generator(iter(data))
        else:
            return cls.from_generator(iter([data]))

    @classmethod
    def from_folder(cls, folder: str | Path, encoding: str = "utf-8") -> "Flow":
        """
        Load and stream all JSON records from a folder.
        - Flattens each file (list or single dict).
        - Skips non-JSON files.

        Consumes the file streams; the resulting Flow is a stream of records.

        Args:
            folder (str or Path): The path to the folder.
            encoding (str, optional): File encoding. Defaults to "utf-8".

        Returns:
            Flow: A Flow object.
        """
        folder_p = Path(folder)
        if not folder_p.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder_p}")

        def gen() -> Iterator[dict]:
            for p in folder_p.iterdir():
                if p.suffix.lower() != ".json":
                    continue
                text = p.read_text(encoding=encoding)
                data = json.loads(text)
                if isinstance(data, list):
                    yield from data
                elif isinstance(data, dict):
                    yield data
                # else: skip

        return cls.from_generator(gen())

    @classmethod
    def from_glob(cls, pattern: str | Path) -> "Flow":
        """
        Load and stream all JSON records matching a glob path.
        E.g. '*.json', 'data/events/*378*.json', '**/*.json'

        Consumes the file streams; the resulting Flow is a stream of records.

        Args:
            pattern (str or Path): The glob pattern.

        Returns:
            Flow: A Flow object.
        """

        def gen():
            for fp in glob.glob(str(pattern), recursive=True):
                p = Path(fp)
                if not p.is_file():
                    continue
                text = p.read_text(encoding="utf-8")
                data = json.loads(text)
                if isinstance(data, list):
                    yield from data
                elif isinstance(data, dict):
                    yield data
                # else skip

        return cls.from_generator(gen())

    @classmethod
    def from_records(
        cls, data: dict[Any, Any] | list[dict[Any, Any]] | Iterable[dict[Any, Any]]
    ) -> "Flow":
        """
        Create a Flow from one or more dict-like records.
        Accepts:
        - list of dicts
        - single dict
        - iterable of dicts

        Does not consume the stream.

        Args:
            data (dict | list[dict] | Iterable[dict]): The data to create the flow from.

        Returns:
            Flow: The created flow.
        """
        return cls(data)
