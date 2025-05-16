import glob
import json
import os
import random
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable
from itertools import chain, islice, tee, zip_longest
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional, Union

import pandas as pd
import requests

from .core import _resolve_agg, sanitize_filename

try:
    import statsbombpy
except ImportError:
    statsbombpy = None

if TYPE_CHECKING:
    # only for type‐checking; no runtime import
    from .flowgroup import FlowGroup


class Flow:
    """
    A class representing a flow of data records.

    Args:
        records (Iterable[dict]): An iterable of dictionaries representing the records to be processed.

    Notes:
        Many methods in this class operate on a stream of records. Methods that materialize or exhaust the stream will be explicitly documented.
    """

    def __init__(self, records: Iterable[dict[Any, Any]]):
        """
        Initialize a Flow instance from an iterable of records.

        Args:
            records (Iterable[dict]): An iterable of dictionaries representing the records to be processed.

        Does not consume the data stream.
        """
        self._records: Union[Iterable[dict[Any, Any]], list[dict[Any, Any]]]
        if isinstance(records, Flow):
            self._records = records._records
        elif isinstance(records, dict):
            self._records = [dict(records)]
        elif isinstance(records, list):
            self._records = [dict(r) for r in records]
        elif isinstance(records, (str, bytes)):
            raise TypeError(
                "Cannot build Flow from text; expected dict or iterable of dicts"
            )
        elif isinstance(records, Iterable):
            self._records = (dict(r) for r in records)
        else:
            raise TypeError("Expected dict, list of dicts, or iterable of dicts")

    def __len__(self) -> int:
        """
        Return the number of records in the Flow.

        Consumes the stream (materializes all records).

        Returns:
            int: The number of records in the Flow.
        """
        return len(self.collect())

    def __iter__(self) -> Iterator[dict]:
        """
        Return an iterator over the records in the Flow.

        May consume the stream if iterated fully.

        Returns:
            Iterator[dict]: An iterator over the records in the Flow.
        """
        return iter(self._records)

    def __repr__(self) -> str:
        """
        Return a string representation of the Flow.

        Does not consume the stream.
        """
        if isinstance(self._records, list):
            sample = self._records[:3]
            return f"<Penaltyblog Flow | n={len(self._records)} | sample={sample}>"
        else:
            return "<Penaltyblog Flow (streaming)>"

    def __eq__(self, other: object) -> bool:
        """
        Compare this Flow to another Flow or to a list of dicts.

        Consumes (materializes) both Flows' streams.

        Returns:
            bool: True if the sequences of records are equal.
        """
        # collect self into a list and replace
        self_list = list(self._records)
        self._records = self_list

        # Flow vs Flow
        if isinstance(other, Flow):
            other_list = list(other._records)
            other._records = other_list
            return self_list == other_list

        # Flow vs list
        if isinstance(other, list):
            return self_list == other

        return NotImplemented

    def materialize(self) -> "Flow":
        """
        Materializes the stream into a list of dicts. Note that this consumes the stream
        of data and the flow will now be empty.

        Returns:
            Flow: A new Flow instance that is a fully materialized copy of the
            current stream, backed by a list of records. This allows for safe
            re-scanning and manipulation without affecting the original stream.
        """
        return Flow(self.collect())

    def fork(self) -> tuple["Flow", "Flow"]:
        """
        Fork this Flow into two independent streams using itertools.tee.

        Returns:
            tuple[Flow, Flow]: A tuple of two new Flow instances,
            each backed by an independent iterator of the original stream.
            These flows are one-shot and cannot be iterated more than once.
        """
        it1, it2 = tee(self._records, 2)
        return Flow(it1), Flow(it2)

    def filter(self, fn: Callable) -> "Flow":
        """
        Filter the records using the given function.

        Does not consume the stream.

        Args:
            fn (Callable): The function to use for filtering.

        Returns:
            Flow: A new Flow with the filtered records.
        """

        return Flow(r for r in self._records if fn(r))

    def assign(self, **kwargs) -> "Flow":
        """
        Assign new fields to each record using the given functions.

        Does not consume the stream.

        Args:
            **kwargs (dict): The functions to use for assigning new fields.

        Returns:
            Flow: A new Flow with the assigned fields.
        """

        def mutate_record(record: dict) -> dict:
            rec = dict(record)
            for key, func in kwargs.items():
                rec[key] = func(rec)
            return rec

        return Flow(mutate_record(r) for r in self._records)

    def select(self, *fields: str, leaf_names: bool = False) -> "Flow":
        """
        Select the given (possibly nested) fields from each record.

        Does not consume the stream.

        Args:
            *fields (str): The fields to select.
            leaf_names (bool, optional): Whether to use leaf names for nested fields. Defaults to False.

        Returns:
            Flow: A new Flow with the selected fields.
        """

        if leaf_names:
            # compute the would‐be output keys
            leaf_keys = [f.split(".")[-1] for f in fields]
            # find duplicates
            dup_counts = Counter(leaf_keys)
            dupes = [k for k, cnt in dup_counts.items() if cnt > 1]
            if dupes:
                warnings.warn(
                    f"select(..., leaf_names=True) will produce duplicate keys: {dupes}. "
                    "Later ones will overwrite earlier ones.",
                    UserWarning,
                    stacklevel=2,
                )

        def select_fields(record: dict[Any, Any]) -> dict[Any, Any]:
            out = {}
            for field in fields:
                if field in record:
                    key = field if not leaf_names else field.split(".")[-1]
                    out[key] = record[field]
                elif "." in field:
                    parts = field.split(".")
                    val: Any = record
                    for p in parts:
                        if isinstance(val, dict):
                            val = val.get(p)
                        else:
                            val = None
                            break
                    key = parts[-1] if leaf_names else field
                    out[key] = val
                else:
                    key = field if not leaf_names else field.split(".")[-1]
                    out[key] = None
            return out

        return Flow(select_fields(r) for r in self._records)

    def drop(self, *fields: str) -> "Flow":
        """
        Remove the given fields from each record.

        Does not consume the stream.

        Args:
            *fields (str): The fields to remove.

        Returns:
            Flow: A new Flow with the removed fields.
        """

        def remover(record: dict) -> dict:
            # shallow copy so we don’t mutate the original
            rec = dict(record)
            for f in fields:
                rec.pop(f, None)
            return rec

        return Flow(remover(r) for r in self._records)

    def sort(
        self, by: Union[str, list[str], tuple[str, ...]], reverse: bool = False
    ) -> "Flow":
        """
        Sort the records by one or more fields, always sending any records
        where any of the sort fields are None to the very end.

        Consumes (materializes) the stream to perform the sort.

        Args:
            by (str or list/tuple of str): The field name, or list/tuple of field names, to sort by.
            reverse (bool, optional): Whether to sort in descending order. Defaults to False.

        Returns:
            Flow: A new Flow with the records sorted by the given field(s).
        """

        def _get_key_func(fields):
            if isinstance(fields, str):
                return lambda r: r.get(fields)
            return lambda r: tuple(r.get(f) for f in fields)

        def _is_null_record(record, fields):
            if isinstance(fields, str):
                return record.get(fields) is None
            return any(record.get(f) is None for f in fields)

        def _lazy_sort():
            recs = list(self._records)
            fields = by
            key_func = _get_key_func(fields)
            non_null = [r for r in recs if not _is_null_record(r, fields)]
            nulls = [r for r in recs if _is_null_record(r, fields)]
            for r in sorted(non_null, key=key_func, reverse=reverse):
                yield r
            yield from nulls

        return Flow(_lazy_sort())

    def limit(self, n: int) -> "Flow":
        """
        Limit the number of records to the given number.

        Does not consume the stream.

        Args:
            n (int): The number of records to limit to.

        Returns:
            Flow: A new Flow with the limited number of records.
        """
        if n < 0:
            raise ValueError("n must be non-negative")

        return Flow(islice(self._records, n))

    def split_array(self, key: str, into: list[str]) -> "Flow":
        """
        Split the given array field into multiple fields (keeping the original array).
        Does not consume the stream.

        Args:
            key (str): The name of the array field to split.
            into (list[str]): The names of the new fields to create.

        Returns:
            Flow: A new Flow with the split array.
        """

        def splitter(record: dict) -> dict:
            rec = dict(record)
            raw = rec.get(key)
            if isinstance(raw, list):
                if len(raw) < len(into):
                    warnings.warn(
                        f"{key!r} has only {len(raw)} elements but expected {len(into)}",
                        UserWarning,
                    )
                for i, name in enumerate(into):
                    rec[name] = raw[i] if i < len(raw) else None
            return rec

        return Flow(splitter(r) for r in self._records)

    def group_by(self, *keys: str) -> "FlowGroup":
        """
        Group records by the specified keys and return a FlowGroup object.

        Consumes (materializes) the stream to group records.

        Args:
            *keys (str): The names of the fields to group by.

        Returns:
            FlowGroup: A FlowGroup object
        """
        from .flowgroup import FlowGroup

        groups = defaultdict(list)
        for record in self._records:
            group_key = tuple(record.get(k) for k in keys)
            groups[group_key].append(dict(record))
        return FlowGroup(keys, groups)

    def summary(self, **aggregates: Union[str, tuple[str, str], Callable]) -> "Flow":
        """
        Summarize the stream by computing the given aggregates over each group.

        Consumes (materializes) the stream to compute aggregates.

        Args:
            **aggregates (Union[str, tuple[str, str], Callable]): The aggregates to compute.

        Returns:
            Flow: A new Flow with the summary rows.
        """

        def gen():
            data = list(self._records)  # only executed at iteration time
            row = {col: _resolve_agg(data, spec) for col, spec in aggregates.items()}
            yield row

        return Flow(gen())

    def concat(self, *others: "Flow") -> "Flow":
        """
        Concatenate this Flow with one or more other Flows.

        Does not consume the stream.

        Args:
            *others (Flow): The other Flows to concatenate.

        Returns:
            Flow: A new Flow with the concatenated records.
        """

        return Flow(chain(self._records, *(o._records for o in others)))

    def row_number(
        self, by: str, new_field: str = "row_number", reverse: bool = False
    ) -> "Flow":
        """
        Assigns a row number based on sorting by `by`.

        Consumes (materializes) the stream to assign row numbers.

        Args:
            by (str): The name of the field to sort by.
            new_field (str, optional): The name of the new field to add. Defaults to "row_number".
            reverse (bool, optional): Whether to sort in descending order. Defaults to False.

        Returns:
            Flow: A new Flow with the row numbers assigned.
        """

        def gen():
            recs = list(self._records)  # only runs when someone iterates
            non_null = [dict(r) for r in recs if r.get(by) is not None]  # shallow copy
            nulls = [dict(r) for r in recs if r.get(by) is None]  # shallow copy
            sorted_non_null = sorted(non_null, key=lambda r: r[by], reverse=reverse)
            for idx, rec in enumerate(sorted_non_null, start=1):
                rec[new_field] = idx
            for rec in nulls:
                rec[new_field] = None
            yield from (sorted_non_null + nulls)

        return Flow(gen())

    def drop_duplicates(self, *fields: str, keep: str = "first") -> "Flow":
        """
        Drop duplicate records.

        Consumes (materializes) the stream to identify duplicates.

        If no fields are specified, it considers the entire record for duplication.
        If fields are provided, only those fields are used to identify duplicates.

        Args:
            *fields (str): The fields to use for deduplication.
            keep (str, optional): How to handle duplicates. Defaults to "first".
                - "first": Keep the first occurrence of a duplicate set.
                - "last": Keep the last occurrence.
                - False: Drop all records that are part of any duplicate set.

        Returns:
            Flow: A new Flow with the duplicate records dropped.
        """

        def gen() -> Iterator[dict[Any, Any]]:
            seen: dict[Any, dict] = {}
            for record in self._records:
                if fields:
                    key = tuple(record.get(f) for f in fields)
                else:
                    key = tuple(sorted(record.items()))
                if key in seen:
                    if keep == "last":
                        seen[key] = record
                    elif keep is False:
                        seen[key] = None
                    # else keep == "first": ignore
                else:
                    seen[key] = record
            for rec in seen.values():
                if rec is not None:
                    yield rec

        return Flow(gen())

    def take_last(self, n: int) -> "Flow":
        """
        Take the last `n` records.

        Consumes (materializes) the stream to return the last records.

        Args:
            n (int): The number of records to take.

        Returns:
            Flow: A new Flow with the last `n` records.
        """

        def gen():
            if n < 0:
                raise ValueError("n must be >= 0")
            records = list(self._records)
            if n == 0:
                return  # yields nothing
            for rec in records[-n:]:
                yield rec

        return Flow(gen())

    def unique(self, *fields: str) -> "Flow":
        """
        Return unique values of one or more fields.
        Supports dot notation for nested keys.

        If one field: yields {field: value} for each distinct value.
        If multiple: yields dicts of those field combos.

        Consumes (materializes) the stream to determine uniqueness.

        Args:
            *fields (str): The fields to return unique values for.

        Returns:
            Flow: A new Flow with unique values of one or more fields.
        """

        def get_nested(record: Optional[dict[Any, Any]], path: str):
            parts = path.split(".")
            val = record
            for part in parts:
                if isinstance(val, dict):
                    val = val.get(part)
                else:
                    return None
            return val

        def gen():
            seen = set()
            for record in self._records:
                if fields:
                    key = tuple(get_nested(record, f) for f in fields)
                    if key not in seen:
                        seen.add(key)
                        if len(fields) == 1:
                            yield {fields[0]: key[0]}
                        else:
                            yield dict(zip(fields, key))
                else:
                    key = tuple(sorted(record.items()))
                    if key not in seen:
                        seen.add(key)
                        yield record

        return Flow(gen())

    def rename(self, **mapping: str) -> "Flow":
        """
        Rename keys: old_name=new_name, …

        Does not consume the stream.

        Args:
            **mapping (str): The keys to rename.

        Returns:
            Flow: A new Flow with renamed keys.
        """

        def gen() -> Iterator[dict[str, Any]]:
            for record in self._records:
                rec = record.copy()  # shallow copy
                for old, new in mapping.items():
                    if old in rec:
                        rec[new] = rec.pop(old)
                yield rec

        return Flow(gen())

    def join(
        self,
        other: Union["Flow", list[dict]],
        left_on: str,
        right_on: Union[str, None] = None,
        fields: Union[list[str], None] = None,
        how: str = "left",
    ) -> "Flow":
        """
        Join a Flow with another Flow or a list of dicts.
        The right side of the join is fully materialized into memory.
        The left stream is not fully materialized, but is iterated.

        Args:
            other: The Flow or list of dicts to join with.
            left_on: The field name in this Flow to join on.
            right_on: The field name in the other Flow/list to join on. If None, defaults to left_on.
            fields: An optional list of field names to include from the other Flow/list. If None, all fields are included.
            how: The type of join to perform. If "left", all records from this Flow are kept, with matching records from the other Flow/list added. If "inner", only records with a match in the other Flow/list are kept.

        Returns:
            Flow: A new Flow with the joined records.
        """
        right_on = right_on or left_on

        # pull the RHS into memory once
        if isinstance(other, Flow):
            right_data = [
                dict(r) for r in other._records
            ]  # shallow copy of each record
        elif isinstance(other, list):
            right_data = [dict(r) for r in other]  # shallow copy of each record
        else:
            raise TypeError("Join target must be a Flow or list of dicts.")

        if how not in ("left", "inner"):
            raise ValueError(f"Unknown join type {how!r}; expected 'left' or 'inner'.")

        # build lookup: key → row
        lookup: dict[Any, dict] = {
            r[right_on]: dict(r) for r in right_data if right_on in r
        }  # shallow copy

        def gen():
            for left_rec in self._records:
                key = left_rec.get(left_on, None)
                right_rec = lookup.get(key)

                # no match
                if right_rec is None:
                    if how == "left":
                        yield dict(left_rec)  # shallow copy of the LHS
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

        return Flow(gen())

    def collect(self) -> list[dict]:
        """
        Materializes the stream into a list of dicts. Note that this consumes the stream
        of data and the flow will now be empty.

        Returns:
            list[dict]: The records in the flow.
        """
        return list(self._records)

    def head(self, n: int = 5) -> "Flow":
        """
        Return the first n records of the flow.

        Consumes the stream up to n records (materializes those records).

        Args:
            n (int): The number of records to return.

        Returns:
            Flow: A new Flow with the first n records.
        """
        if n < 0:
            raise ValueError("n must be >= 0")
        return self.limit(n)

    def pipe(self, func: Callable, *args, **kwargs) -> Union["Flow", Any]:
        """
        Pipe the flow into a function.

        Consumes the stream if the piped function does so.

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

        Consumes the stream (materializes all records).

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
        Returns the first record in the flow or None if empty.

        Consumes the stream (materializes all records).

        Returns:
            dict | None: The first record in the flow or None if empty.
        """
        lst = self.collect()
        return lst[0] if lst else None

    def last(self) -> dict | None:
        """
        Return the last record in the flow or None if empty.

        Consumes (materializes) the stream.

        Returns:
            dict | None: The last record in the flow or None if empty.
        """
        lst = self.collect()
        return lst[-1] if lst else None

    def is_empty(self) -> bool:
        """
        Return True if the flow has no records, without losing any data
        and without buffering the entire stream.

        Consumes the stream (materializes all records).

        Returns:
            bool: True if the flow is empty, False otherwise.
        """
        lst = self.collect()
        return not lst

    def keys(self, limit: int | None = None) -> set[str]:
        """
        Return the union of keys across up to `limit` records.

        Consumes the stream (materializes all records, or up to `limit` if specified).

        Args:
            limit (int or None): number of records to inspect.
                - If None (default), inspects all records.
                - If an integer n, only the first n records are checked.

        Returns:
            set[str]: The union of keys across up to `limit` records.
        """
        data = self.collect()
        keyset: set[str] = set()
        if limit is None:
            for record in data:
                keyset.update(record.keys())
        else:
            for record in data[:limit]:
                keyset.update(record.keys())
        return keyset

    def explode(self, key: str) -> "Flow":
        """
        Explode a list-field into multiple records.

        Does not consume the stream.

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

        return Flow(generator())

    def explode_multi(self, keys: list[str], fillvalue=None) -> "Flow":
        """
        Explode multiple list-fields together (zip with fillvalue).

        Does not consume the stream.

        Args:
            keys (list[str]): The names of the fields to explode.
            fillvalue (any, optional): The value to use for missing values. Defaults to None.

        Returns:
            Flow: A new Flow of the exploded records.
        """
        if not keys:
            raise ValueError("keys must not be empty")

        if not isinstance(keys, list):
            raise TypeError("keys must be a list")

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

        return Flow(gen())

    def sample(self, n: int, seed: int | None = None) -> "Flow":
        """
        Uniformly sample exactly `n` records from the stream (reservoir sampling).
        Returns a new Flow of length n (or fewer, if the stream has < n items).

        Consumes (materializes) the stream to build a reservoir of size n.

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
        return Flow(iter(reservoir))

    def sample_frac(self, frac: float, seed: int | None = None) -> "Flow":
        """
        Bernoulli sample: include each record with probability `frac` (0.0–1.0).
        This yields an *approximate* fraction of the stream.

        Does not consume the stream.

        Args:
            frac (float): The fraction of records to include.
            seed (int, optional): The random seed. Defaults to None.

        Returns:
            Flow: A new Flow of the sampled records.
        """
        # check frac is between 0 and 1
        rnd = random.Random(seed)
        return Flow(r for r in self._records if rnd.random() < frac)

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
        df = pd.DataFrame(self.collect())
        return df.describe(percentiles=percentiles, include=include, exclude=exclude)

    def flatten(self, sep: str = ".") -> "Flow":
        """
        Recursively flatten nested dicts in each record.

        Does not consume the stream.

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

        return Flow(gen())

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert the Flow to a pandas DataFrame.

        Consumes (materializes) the stream to build a DataFrame.

        Returns:
            DataFrame: A pandas DataFrame containing the records.
        """
        return pd.DataFrame(self._records)

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
        p = Path(path)
        # ensure parent folder exists
        if p.parent:
            p.parent.mkdir(parents=True, exist_ok=True)

        data = self.collect()
        with p.open("w", encoding=encoding) as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False))
                f.write("\n")

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

    class statsbomb:

        DEFAULT_CREDS = {
            "user": os.environ.get("SB_USERNAME"),
            "passwd": os.environ.get("SB_PASSWORD"),
        }

        @staticmethod
        def _require_statsbombpy():
            if statsbombpy is None:
                raise ImportError(
                    "statsbombpy is required. Install with `pip install statsbombpy`"
                )

        @classmethod
        def competitions(cls) -> "Flow":
            """
            Get Flow of all available competitions.

            Returns:
                Flow: A Flow of competition records.
            """
            cls._require_statsbombpy()
            from statsbombpy import sb

            data = list(sb.competitions(fmt="dict").values())
            return Flow(data)

        @classmethod
        def matches(
            cls, competition_id: int, season_id: int, creds: Optional[dict] = None
        ) -> "Flow":
            """
            Get Flow of matches for a given competition + season.

            Args:
                competition_id (int): The competition ID.
                season_id (int): The season ID.

            Returns:
                Flow: A Flow of match records.
            """
            cls._require_statsbombpy()
            from statsbombpy import sb

            data = list(
                sb.matches(
                    competition_id=competition_id,
                    season_id=season_id,
                    fmt="dict",
                    creds=creds or cls.DEFAULT_CREDS,
                ).values()
            )
            return Flow(data)

        @classmethod
        def lineups(cls, match_id: int, creds: Optional[dict] = None) -> "Flow":
            """
            Get Flow of lineups for a given match.

            Args:
                match_id (int): The match ID.
                creds (dict, optional): StatsBomb credentials. Defaults to DEFAULT_CREDS.

            Returns:
                Flow: A Flow of match records.
            """
            cls._require_statsbombpy()
            from statsbombpy import sb

            data = list(
                sb.lineups(
                    match_id=match_id, fmt="dict", creds=creds or cls.DEFAULT_CREDS
                ).values()
            )
            return Flow(data)

        @classmethod
        def events(
            cls,
            match_id: int,
            include_360_metrics: bool = False,
            creds: Optional[dict] = None,
        ) -> "Flow":
            """
            Get Flow of events for a given match.

            Args:
                match_id (int): The match ID.
                creds (dict, optional): StatsBomb credentials. Defaults to DEFAULT_CREDS.

            Returns:
                Flow: A Flow of match records.
            """
            cls._require_statsbombpy()
            from statsbombpy import sb

            data = list(
                sb.events(
                    match_id=match_id,
                    fmt="dict",
                    creds=creds or cls.DEFAULT_CREDS,
                    include_360_metrics=include_360_metrics,
                ).values()
            )
            return Flow(data)

        @staticmethod
        def from_github_file(file_id: int, type: str = "events") -> "Flow":
            """
            Load a StatsBomb event data file from GitHub.

            Consumes the HTTP response; the resulting Flow is a stream of records.

            Args:
                file_id (int): The StatsBomb file ID.
                type (str, optional): The type of data to load. Defaults to "events". Can be one of:
                    - "events"
                    - "lineups"
                    - "three-sixty"
                    - "matches"

            Returns:
                Flow: A Flow object.
            """
            url = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/{type}/{file_id}.json"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                return Flow(r for r in data)
            else:
                return Flow(iter([data]))
