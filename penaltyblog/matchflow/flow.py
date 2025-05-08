import glob
import json
import random
from collections import defaultdict
from itertools import chain, islice, tee, zip_longest
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Union

import pandas as pd
import requests

from .core import _resolve_agg, sanitize_filename

if TYPE_CHECKING:
    # only for type‐checking; no runtime import
    from .flowgroup import FlowGroup


class Flow:
    """
    A class representing a flexible, lazy-evaluated data processing pipeline for working with record streams.
    """

    def __init__(self, records: Iterable[dict]):
        """
        Args:
            records (Iterable[dict]): An iterable of dictionaries representing the records to be processed.
        """
        """
        Args:
            records: Either a single dict, or any iterable of dicts.
        """
        flow: Flow = Flow.from_records(records)
        self._records: Iterator[dict[str, Any]] = flow._records

    def __len__(self) -> int:
        if isinstance(self._records, list):
            return len(self._records)

        it1, it2 = tee(self._records, 2)
        self._records = it2
        return sum(1 for _ in it1)

    def __iter__(self) -> Iterator[dict]:
        return iter(self._records)

    def __repr__(self) -> str:
        it1, it2 = tee(self._records, 2)
        self._records = it2

        sample = list(islice(it1, 3))
        total: Union[int, str]
        try:
            total = len(self)
        except Exception:
            total = "?"

        return f"<Penaltyblog Flow | n≈{total} | sample={sample}>"

    def __eq__(self, other: object) -> bool:
        """
        Compare a Flow to another Flow or to a list of records by materializing both.
        """
        if isinstance(other, Flow):
            # clone both sides
            it1a, it1b = tee(self._records, 2)
            it2a, it2b = tee(other._records, 2)
            # replace their streams with the second clones
            self._records = it1b
            other._records = it2b
            return list(it1a) == list(it2a)

        if isinstance(other, list):
            it, it_copy = tee(self._records, 2)
            self._records = it_copy
            return list(it) == other

        return NotImplemented

    def cache(self) -> "Flow":
        """
        Materialize all records seen so far into memory, and return a new Flow
        backed by that list. Subsequent operations are non-destructive and
        can be re-run arbitrarily.
        """
        # pull everything out of the current (possibly streaming) flow
        data = list(self._records)
        # build a new Flow that holds the data in a plain list
        new = object.__new__(Flow)
        # shallow‐copy each dict so that further transformations don't
        # mutate the user's original records
        new._records = [dict(r) for r in data]
        return new

    def filter(self, fn: Callable) -> "Flow":
        """
        Filter the records using the given function.

        Args:
            fn (Callable): A function that takes a record and
            returns a boolean.
        """
        return Flow(r for r in self._records if fn(r))

    def assign(self, **kwargs) -> "Flow":
        """
        Assign new fields to each record using the given functions.

        Args:
            **kwargs: Keyword arguments where the key is the name of the new
            field and the value is a function that takes a record and returns
            the value for the new field.
        """

        def mutate_record(record: dict) -> dict:
            for key, func in kwargs.items():
                record[key] = func(record)
            return record

        return Flow(mutate_record(r) for r in self._records)

    def select(self, *fields: str) -> "Flow":
        """
        Select the given (possibly nested) fields from each record.

        If a record has a key exactly matching the field name, that wins.
        Otherwise, if the name contains dots, we interpret it as nested lookup.

        Args:
            *fields (str): The names of the fields to select.
        """

        def select_fields(record: dict) -> dict:
            out = {}
            for field in fields:
                if field in record:
                    # exact key present → take it
                    out[field] = record[field]
                elif "." in field:
                    # try nested = record[a][b][c]...
                    parts = field.split(".")
                    val = record
                    for p in parts:
                        if isinstance(val, dict):
                            val = val.get(p)
                        else:
                            val = None
                            break
                    out[p] = val
                else:
                    # simple missing key
                    out[field] = None
            return out

        return Flow(select_fields(r) for r in self._records)

    def drop(self, *fields: str) -> "Flow":
        """
        Remove the given fields from each record.

        Args:
            *fields (str): The names of the fields to drop.
        """

        def remover(record: dict) -> dict:
            # shallow copy so we don’t mutate the original
            rec = dict(record)
            for f in fields:
                rec.pop(f, None)
            return rec

        return Flow(remover(r) for r in self._records)

    def sort(self, by: str, reverse: bool = False) -> "Flow":
        """
        Sort the records by the given field, always sending any records
        where the field is None to the very end.

        Note:
            This method exhausts the underlying stream by materializing
            all records into memory to perform the sort. After calling
            this, the original Flow cannot be iterated again.

        Args:
            by (str): The name of the field to sort by.
            reverse (bool, optional): Whether to sort in descending order. Defaults to False.
        """
        recs = list(self._records)
        non_null = [r for r in recs if r.get(by) is not None]
        nulls = [r for r in recs if r.get(by) is None]
        sorted_non_null = sorted(non_null, key=lambda r: r[by], reverse=reverse)
        return Flow.from_generator(iter(sorted_non_null + nulls))

    def limit(self, n: int) -> "Flow":
        """
        Limit the number of records to the given number.

        Args:
            n (int): The maximum number of records to return.

        Returns:
            Flow: New Flow with at most n records.
        """

        # tee the underlying iterator so we can consume one clone
        # but leave the other clone intact on `self`
        first_clone, second_clone = tee(self._records, 2)
        # replace our own _records with the second clone (so `self` is untouched)
        self._records = second_clone
        # return a brand‐new Flow over just the first n items
        return Flow.from_generator(islice(first_clone, n))

    def split_array(self, key: str, into: list[str]) -> "Flow":
        """
        Split the given array field into multiple fields.

        Args:
            key (str): The name of the array field to split.
            into (list[str]): The names of the fields to split into.
        """

        def splitter(record: dict) -> dict:
            raw = record.get(key, None)
            # only accept real lists, otherwise treat as if empty
            if isinstance(raw, list):
                values = raw
            else:
                values = [None] * len(into)
            # populate each new field
            for i, name in enumerate(into):
                record[name] = values[i] if i < len(values) else None
            return record

        return Flow(splitter(r) for r in self._records)

    def group_by(self, *keys: str) -> "FlowGroup":
        """
        Group records by the specified keys and return a FlowGroup object

        Args:
            *keys (str): The names of the fields to group by.
        """
        from .flowgroup import FlowGroup

        groups = defaultdict(list)
        for record in self._records:
            group_key = tuple(record.get(k) for k in keys)
            groups[group_key].append(record)
        return FlowGroup(keys, groups)

    def summary(self, **aggregates: str | tuple[str, str] | Callable) -> "Flow":
        """
        Summarize the stream by computing the given aggregates over each group.

        Note: This materializes the entire stream to compute aggregates; the Flow is consumed.

        Args:
            **aggregates (str | tuple[str, str] | callable): The aggregates to compute. Note that
            this causes the stream of data to be materialized.
        """
        data = list(self._records)
        row = {col: _resolve_agg(data, spec) for col, spec in aggregates.items()}
        return Flow([row])

    def concat(self, *others: "Flow") -> "Flow":
        """
        Concatenate this Flow with one or more other Flows.

        Args:
            *others (Flow): The Flows to concatenate.
        """
        return Flow.from_generator(chain(self._records, *(o._records for o in others)))

    def row_number(
        self, by: str, new_field: str = "row_number", reverse: bool = False
    ) -> "Flow":
        """
        Assigns a row number based on sorting by `by`.

        Note: This reads all records into memory to assign row numbers; the Flow is consumed.

        Args:
            by (str): The name of the field to sort by.
            new_field (str, optional): The name of the new field to add. Defaults to "row_number".
            reverse (bool, optional): Whether to sort in descending order. Defaults to False.
        """
        records = list(self._records)
        non_null = [r for r in records if r.get(by) is not None]
        nulls = [r for r in records if r.get(by) is None]
        sorted_non_null = sorted(non_null, key=lambda r: r[by], reverse=reverse)
        # assign ranks to the non‐nulls
        for idx, rec in enumerate(sorted_non_null, start=1):
            rec[new_field] = idx
        # keep the null‐valued ones at the end with rank=None
        for rec in nulls:
            rec[new_field] = None
        return Flow.from_generator(iter(sorted_non_null + nulls))

    def drop_duplicates(self, *fields: str, keep: str = "first") -> "Flow":
        """
        Drop duplicate records.
        If no fields given, consider the whole record.
        Otherwise consider only the given fields.
        keep: 'first', 'last', or False (drop all duplicates).
        """

        def gen() -> Iterator[dict[str, Any]]:
            seen: dict[Any, dict[str, Any]] = {}
            for record in self._records:
                if fields:
                    key = tuple(record.get(f) for f in fields)
                else:
                    # use the entire record (sorted items) as key
                    key = tuple(sorted(record.items()))
                if key in seen:
                    if keep == "last":
                        # replace previous with this one
                        seen[key] = record
                    elif keep is False:
                        # mark for removal
                        seen[key] = None
                    # else keep == "first": do nothing
                else:
                    seen[key] = record

            # yield in original insertion order, skipping any None
            for rec in seen.values():
                if rec is not None:
                    yield rec

        return Flow.from_generator(gen())

    def take_last(self, n: int) -> "Flow":
        """
        Take the last `n` records.

        Note: This reads the entire stream in order to return the last records;
        the Flow is consumed.

        Args:
            n (int): The number of records to take.
        """
        if n < 0:
            raise ValueError("n must be >= 0")

        if n == 0:
            return Flow.from_generator(iter([]))

        all_recs = list(self._records)
        return Flow.from_generator(iter(all_recs[-n:]))

    def unique(self, *fields: str) -> "Flow":
        """
        Return unique values of one or more fields.
        If one field: yields {field: value} for each distinct value.
        If multiple: yields dicts of those field combos.

        Args:
            *fields (str): The fields to return unique values for.
        """
        seen = set()

        def gen() -> Iterator[dict[str, Any]]:
            for record in self._records:
                if fields:
                    key = tuple(record.get(f) for f in fields)
                    if key not in seen:
                        seen.add(key)
                        if len(fields) == 1:
                            yield {fields[0]: key[0]}
                        else:
                            yield dict(zip(fields, key))
                else:
                    # no fields => drop duplicate records
                    key = tuple(sorted(record.items()))
                    if key not in seen:
                        seen.add(key)
                        yield record

        return Flow.from_generator(gen())

    def rename(self, **mapping: str) -> "Flow":
        """
        Rename keys: old_name=new_name, …

        Args:
            **mapping (str): The keys to rename.
        """

        def gen() -> Iterator[dict[str, Any]]:
            for record in self._records:
                rec = dict(record)
                for old, new in mapping.items():
                    if old in rec:
                        rec[new] = rec.pop(old)
                yield rec

        return Flow.from_generator(gen())

    def join(
        self,
        other: "Flow|list[dict]",
        left_on: str,
        right_on: str | None = None,
        fields: list[str] | None = None,
        how: str = "left",
    ) -> "Flow":
        right_on = right_on or left_on

        # pull the RHS into memory once
        if isinstance(other, Flow):
            right_data = list(other._records)
        elif isinstance(other, list):
            right_data = other
        else:
            raise TypeError("Join target must be a Flow or list of dicts.")

        if how not in ("left", "inner"):
            raise ValueError(f"Unknown join type {how!r}; expected 'left' or 'inner'.")

        # build lookup: key → row
        lookup: dict[Any, dict] = {r[right_on]: r for r in right_data if right_on in r}

        def gen():
            for left_rec in self._records:
                key = left_rec.get(left_on, None)
                right_rec = lookup.get(key)

                # no match
                if right_rec is None:
                    if how == "left":
                        yield dict(left_rec)  # keep the LHS
                    # if how == "inner": drop it
                    continue

                # match! merge
                out = dict(left_rec)  # shallow copy of the LHS
                if fields:
                    for f in fields:
                        out[f] = right_rec.get(f)
                else:
                    for k, v in right_rec.items():
                        if k != right_on:
                            out[k] = v
                yield out

        return Flow.from_generator(gen())

    def collect(self) -> list[dict]:
        """
        Materialize the flow into a list of dicts.

        Note: This consumes the stream and returns a list of all records.

        Returns:
            list[dict]: The records in the flow.
        """
        return list(self._records)

    def head(self, n: int = 5) -> "Flow":
        """
        Return the first n records of the flow.

        Args:
            n (int): The number of records to return.
        """
        return self.limit(n)

    def pipe(self, func: Callable, *args, **kwargs) -> Union["Flow", Any]:
        """
        Pipe the flow into a function.

        Args:
            func (callable): The function to pipe the flow into.
            *args: The arguments to pass to the function.
            **kwargs: The keyword arguments to pass to the function.

        Returns:
            Flow: The result of the function.
        """
        return func(self, *args, **kwargs)

    def to_json(self, indent: int | None = None) -> str:
        """
        Serialize the flow to a JSON string.

        Args:
            indent (int or None): The number of spaces to use for indentation.
                - If None (default), the JSON string is compact.
                - If an integer n, the JSON string is formatted with n spaces per indentation level.

        Returns:
            str: The JSON string.
        """
        return json.dumps(self.collect(), indent=indent)

    def first(self) -> dict | None:
        """
        Peek at the first record without consuming it.

        Args:
            n (int): The number of records to return.

        Returns:
            dict | None: The first record in the flow or None if empty.
        """
        it = iter(self._records)
        try:
            first = next(it)
        except StopIteration:
            return None

        self._records = chain([first], it)
        return first

    def last(self) -> dict | None:
        """
        Return the last record in the flow or None if empty.

        Note: This scans every record to find the last one;
        the Flow is consumed.

        Args:
            n (int): The number of records to return.

        Returns:
            dict | None: The last record in the flow or None if empty.
        """
        it1, it2 = tee(self._records, 2)
        self._records = it2
        last_rec = None
        for rec in it1:
            last_rec = rec
        return last_rec

    def is_empty(self) -> bool:
        """
        Return True if the flow has no records, without losing any data
        and without buffering the entire stream.

        Args:
            n (int): The number of records to return.

        Returns:
            bool: True if the flow is empty, False otherwise.
        """
        it = iter(self._records)
        try:
            first = next(it)
        except StopIteration:
            return True

        self._records = chain([first], it)
        return False

    def keys(self, limit: int | None = None) -> set[str]:
        """
        Return the union of keys across up to `limit` records.

        Args:
            limit (int or None): number of records to inspect.
                - If None (default), inspects all records.
                - If an integer n, only the first n records are checked.

        Does not consume the underlying stream.
        """
        # duplicate the iterator
        it1, it2 = tee(self._records, 2)
        self._records = it2

        keyset: set[str] = set()
        if limit is None:
            for r in it1:
                keyset.update(r.keys())
        else:
            for r in islice(it1, limit):
                keyset.update(r.keys())
        return keyset

    def explode(self, key: str) -> "Flow":
        """
        Explode a list-field into multiple records.

        Args:
            key (str): The name of the field to explode.

        Returns:
            Flow: A new Flow of the exploded records.
        """

        def generator():
            for record in self._records:
                values = record.get(key)
                if isinstance(values, list):
                    if values:
                        # one row per element
                        for item in values:
                            new_rec = dict(record)
                            new_rec[key] = item
                            yield new_rec
                    else:
                        # empty list → keep the record unchanged
                        yield record
                else:
                    # non-list or missing → keep as is
                    yield record

        return Flow.from_generator(generator())

    def explode_multi(self, keys: list[str], fillvalue=None) -> "Flow":
        """
        Explode multiple list-fields together (zip with fillvalue).

        Args:
            keys (list[str]): The names of the fields to explode.
            fillvalue (any, optional): The value to use for missing values. Defaults to None.

        Returns:
            Flow: A new Flow of the exploded records.
        """

    def explode_multi(self, keys: list[str], fillvalue=None) -> "Flow":
        if not keys:
            raise ValueError("keys must not be empty")

        def gen() -> Iterator[dict[str, Any]]:
            for rec in self._records:
                # pull out each field as a list (or empty list if missing / not a list)
                arrays = []
                for k in keys:
                    v = rec.get(k)
                    if isinstance(v, list):
                        arrays.append(v)
                    else:
                        arrays.append([])

                # if every list is empty, just yield the record as-is
                if all(len(arr) == 0 for arr in arrays):
                    yield rec
                    continue

                # otherwise explode
                for items in zip_longest(*arrays, fillvalue=fillvalue):
                    out = dict(rec)
                    for key, val in zip(keys, items):
                        out[key] = val
                    yield out

        return Flow.from_generator(gen())

    def sample(self, n: int, seed: int | None = None) -> "Flow":
        """
        Uniformly sample exactly `n` records from the stream (reservoir sampling).
        Returns a new Flow of length n (or fewer, if the stream has < n items).

        Note: This consumes the stream to build a reservoir of size n;
        the Flow is consumed.

        Args:
            n (int): The number of records to sample.
            seed (int, optional): The random seed. Defaults to None.

        Returns:
            Flow: A new Flow of the sampled records.
        """
        rnd = random.Random(seed)
        reservoir = []
        for i, record in enumerate(self._records, start=1):
            if i <= n:
                reservoir.append(record)
            else:
                j = rnd.randint(1, i)
                if j <= n:
                    reservoir[j - 1] = record
        return Flow.from_generator(iter(reservoir))

    def sample_frac(self, frac: float, seed: int | None = None) -> "Flow":
        """
        Bernoulli sample: include each record with probability `frac` (0.0–1.0).
        This yields an *approximate* fraction of the stream.

        Args:
            frac (float): The fraction of records to include.
            seed (int, optional): The random seed. Defaults to None.

        Returns:
            Flow: A new Flow of the sampled records.
        """
        rnd = random.Random(seed)
        return Flow.from_generator(r for r in self._records if rnd.random() < frac)

    def describe(
        self,
        percentiles: tuple[float, ...] = (0.25, 0.5, 0.75),
        include: list | None = None,
        exclude: list | None = None,
    ) -> pd.DataFrame:
        """
        Generate descriptive statistics.

        Args:
            percentiles (tuple of float): Percentiles to include between 0 and 1.
            include (list of dtypes or None): Which dtypes to include (as in pandas.describe).
            exclude (list of dtypes or None): Which dtypes to exclude.

        Returns:
            DataFrame: the same as pandas.DataFrame.describe().
        """
        df = pd.DataFrame(self.collect())
        return df.describe(percentiles=percentiles, include=include, exclude=exclude)

    def flatten(self, sep: str = ".") -> "Flow":
        """
        Recursively flatten nested dicts in each record.

        Nested keys are joined with `sep` to form the new field name.

        Args:
            sep (str, optional): Separator between parent and child keys. Defaults to ".".

        Returns:
            Flow: a new Flow whose records have no nested dicts.
        """

        def gen() -> Iterator[dict[str, Any]]:
            for record in self._records:
                flat = {}

                def _flatten(obj: dict, parent_key: str = ""):
                    for k, v in obj.items():
                        new_key = f"{parent_key}{sep}{k}" if parent_key else k
                        if isinstance(v, dict):
                            _flatten(v, new_key)
                        else:
                            flat[new_key] = v

                _flatten(record)
                yield flat

        return Flow.from_generator(gen())

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert the Flow to a pandas DataFrame.

        Note: This reads all records into a pandas DataFrame;
        the Flow is consumed.

        Returns:
            DataFrame: A pandas DataFrame containing the records.
        """
        return pd.DataFrame(self._records)

    def to_json_files(self, folder: str | Path, by: str | None = None) -> "Flow":
        """
        Write each record to a separate JSON file in the given folder.

        Note:
            This serializes every record to disk; the Flow is consumed.

        Args:
            folder (str or Path): Output folder path. Will be created if needed.
            by (str, optional): Field to name the files by. Defaults to numbered files.

        Returns:
            Flow: self
        """
        folder_p = Path(folder)
        folder_p.mkdir(parents=True, exist_ok=True)

        for i, record in enumerate(self._records, start=1):
            if by:
                name = sanitize_filename(record.get(by, f"record_{i}"))
            else:
                name = f"record_{i}"
            path = folder_p / f"{name}.json"
            path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        return self

    def to_jsonl(self, path: str | Path, encoding: str = "utf-8") -> "Flow":
        """
        Save all records to a single JSON Lines (.jsonl) file.
        Each record is written as one line of JSON.

        Note:
            This serializes every record to disk; the Flow is consumed.

        Args:
            path (str or Path): Output file path.
            encoding (str, optional): File encoding. Defaults to "utf-8".

        Returns:
            Flow: self
        """
        p = Path(path)
        # ensure parent folder exists
        if p.parent:
            p.parent.mkdir(parents=True, exist_ok=True)

        with p.open("w", encoding=encoding) as f:
            for record in self._records:
                f.write(json.dumps(record, ensure_ascii=False))
                f.write("\n")

        return self

    def to_json_single(
        self, path: str | Path, encoding: str = "utf-8", indent: int | None = 2
    ) -> "Flow":
        """
        Save all records to a single JSON file as an array.

        Note:
            This serializes every record to disk; the Flow is consumed.

        Args:
            path (str or Path): Output file path.
            encoding (str, optional): File encoding. Defaults to "utf-8".
            indent (int or None): Indentation level. Defaults to 2.

        Returns:
            Flow: self
        """
        p = Path(path)
        if p.parent:
            p.parent.mkdir(parents=True, exist_ok=True)

        p.write_text(
            json.dumps(list(self._records), ensure_ascii=False, indent=indent),
            encoding=encoding,
        )
        return self

    @classmethod
    def from_generator(cls, generator_instance: Iterator[dict]) -> "Flow":
        """
        Create a Flow from a generator function.
        """
        return cls(generator_instance)

    @classmethod
    def from_jsonl(cls, path: str | Path, encoding: str = "utf-8") -> "Flow":
        """
        Load a .jsonl (JSON Lines) file into a Flow.
        Each line must be a valid JSON object.

        Note:
            This reads the file line by line and consumes the stream.

        Args:
            path (str or Path): Input file path.
            encoding (str, optional): File encoding. Defaults to "utf-8".
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

        Note:
            This reads the entire file into memory; the Flow is consumed.

        Args:
            path (str or Path): Input file path.
            encoding (str, optional): File encoding. Defaults to "utf-8".
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

        text = p.read_text(encoding=encoding)
        data = json.loads(text)

        if isinstance(data, list):
            return cls.from_generator(record for record in data)
        else:
            return cls.from_generator([data])

    @classmethod
    def from_folder(cls, folder: str | Path, encoding: str = "utf-8") -> "Flow":
        """
        Load and stream all JSON records from a folder.
        - Flattens each file (list or single dict).
        - Skips non-JSON files.

        Note:
            This reads every matching file; the Flow is consumed.

        Args:
            folder (str or Path): The path to the folder.
            encoding (str, optional): File encoding. Defaults to "utf-8".
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

        Note:
            This reads every matching file; the Flow is consumed.

        Args:
            pattern (str or Path): The glob pattern.
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
    def from_records(cls, data: dict | list[dict] | Iterable[dict]) -> "Flow":
        """
        Create a Flow from one or more dict-like records.
        Accepts:
        - list of dicts
        - single dict
        - iterable of dicts

        Args:
            data (dict | list[dict] | Iterable[dict]): The data to create the flow from.

        Returns:
            Flow: The created flow.
        """
        inst = object.__new__(cls)

        # normalize into a generator of shallow‐copied dicts
        if isinstance(data, dict):
            iterable = [dict(data)]
        elif isinstance(data, list):
            iterable = [dict(r) for r in data]
        elif isinstance(data, str):
            raise TypeError("Expected dict, list[dict], or iterable of dicts")
        elif hasattr(data, "__iter__"):
            iterable = (dict(r) for r in data)
        else:
            raise TypeError("Expected dict, list[dict], or iterable of dicts")

        inst._records = iter(iterable)
        return inst

    class statsbomb:
        @staticmethod
        def from_github_file(match_id: int, type: str = "events"):
            """
            Load a StatsBomb event data file from GitHub.

            Args:
                match_id (int): The StatsBomb match ID.
                type (str, optional): The type of data to load. Defaults to "events".
            """
            url = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/{type}/{match_id}.json"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                return Flow.from_generator(r for r in data)
            else:
                return Flow.from_generator([data])
