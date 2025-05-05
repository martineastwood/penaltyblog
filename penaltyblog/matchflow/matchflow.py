import glob
import json
import os
from collections import defaultdict
from itertools import chain

import numpy as np
import pandas as pd
import requests

from .helpers import sanitize_filename

_AGGS = {
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


class Flow:
    def __init__(self, records):
        # records: iterable of dicts
        self._records = records

    @classmethod
    def from_generator(cls, generator):
        return cls(generator)

    def filter(self, fn):
        return Flow(r for r in self._records if fn(r))

    def assign(self, **kwargs):
        def mutate_record(record):
            for key, func in kwargs.items():
                record[key] = func(record)
            return record

        return Flow(mutate_record(r) for r in self._records)

    def select(self, *fields):
        def select_fields(record):
            return {field: record.get(field, None) for field in fields}

        return Flow(select_fields(r) for r in self._records)

    def drop(self, *fields):
        """
        Remove the given fields from each record.
        Usage:
        flow.drop("team", "position")
        """

        def remover(record):
            # shallow copy so we don’t mutate the original
            rec = dict(record)
            for f in fields:
                rec.pop(f, None)
            return rec

        return Flow(remover(r) for r in self._records)

    def sort(self, by, reverse=False):
        # forces materialization
        return Flow(
            sorted(self._records, key=lambda r: r.get(by, None), reverse=reverse)
        )

    def limit(self, n):
        def limited():
            count = 0
            for record in self._records:
                if count >= n:
                    break
                yield record
                count += 1

        return Flow(limited())

    def split_array(self, key, into):
        def splitter(record):
            values = record.get(key, [None] * len(into))
            for i, name in enumerate(into):
                record[name] = values[i] if i < len(values) else None
            return record

        return Flow(splitter(r) for r in self._records)

    def group_by(self, *keys):
        # Group records by the specified keys and return a Group object
        groups = defaultdict(list)
        for record in self._records:
            group_key = tuple(record.get(k) for k in keys)
            groups[group_key].append(record)
        return Group(keys, groups)

    def summary(self, **aggregates):
        data = list(self._records)
        row = {}
        for col, spec in aggregates.items():
            if isinstance(spec, str):
                func = _AGGS.get(spec)
                if not func:
                    raise ValueError(f"Unknown aggregate '{spec}'")
                row[col] = func(data) if spec != "count" else func(data)

            elif isinstance(spec, tuple) and len(spec) == 2:
                field, func_name = spec
                func = _AGGS.get(func_name)
                if not func:
                    raise ValueError(f"Unknown aggregate '{func_name}'")
                values = [r.get(field) for r in data if r.get(field) is not None]
                row[col] = func(values) if values or func_name == "count" else None

            elif callable(spec):
                row[col] = spec(data)

            else:
                raise ValueError(f"Bad summarize spec for '{col}': {spec}")

        return Flow([row])

    def row_number(self, by, new_field="row_number", reverse=False):
        """
        Assigns a row number based on sorting by `by`.
        """
        records = list(self._records)
        sorted_recs = sorted(records, key=lambda r: r.get(by, None), reverse=reverse)
        for idx, rec in enumerate(sorted_recs, start=1):
            rec[new_field] = idx
        return Flow(sorted_recs)

    def drop_duplicates(self, *fields, keep="first"):
        """
        Drop duplicate records.
        If no fields given, consider the whole record.
        Otherwise consider only the given fields.
        keep: 'first', 'last', or False (drop all duplicates).
        """

        def gen():
            seen = {}
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

    def take_last(self, n):
        """
        Take the last `n` records.
        """
        all_recs = list(self._records)
        return Flow.from_generator(iter(all_recs[-n:]))

    def unique(self, *fields):
        """
        Return unique values of one or more fields.
        If one field: yields {field: value} for each distinct value.
        If multiple: yields dicts of those field combos.
        """
        seen = set()

        def gen():
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

    def rename(self, **mapping):
        """
        Rename keys: old_name=new_name, …
        """

        def gen():
            for record in self._records:
                rec = dict(record)
                for old, new in mapping.items():
                    if old in rec:
                        rec[new] = rec.pop(old)
                yield rec

        return Flow.from_generator(gen())

    def join(self, other, left_on, right_on=None, fields=None, how="left"):
        """
        Join this Flow with another (lookup) Flow or list of dicts.

        Args:
            other (Flow or list): right-hand records to join from
            left_on (str): field in this Flow to match on
            right_on (str): field in other to match against (default = left_on)
            fields (list): fields to include from right side (default = all)
            how (str): "left" (default) or "inner"

        Returns:
            A new Flow with joined records.
        """
        right_on = right_on or left_on

        # Materialize right-hand side into lookup dict
        if isinstance(other, Flow):
            right_data = list(other._records)
        elif isinstance(other, list):
            right_data = other
        else:
            raise TypeError("Join target must be a Flow or list of dicts.")

        lookup = {r[right_on]: r for r in right_data if right_on in r}

        def generator():
            for record in self._records:
                key = record.get(left_on)
                match = lookup.get(key)

                if match:
                    joined = dict(record)
                    if fields:
                        for f in fields:
                            joined[f] = match.get(f)
                    else:
                        for k, v in match.items():
                            if k != right_on:
                                joined[k] = v
                    yield joined

                elif how == "left":
                    yield record  # keep unmatched record

                # else (how == "inner"): skip unmatched

        return Flow.from_generator(generator())

    def collect(self):
        return list(self._records)

    def head(self, n=5):
        return self.limit(n).collect()

    def pipe(self, func, *args, **kwargs):
        return func(self, *args, **kwargs)

    def to_json(self, indent=None):
        return json.dumps(self.collect(), indent=indent)

    def first(self):
        """
        Return the first record in the flow or None if empty.
        """
        for r in self._records:
            return r
        return None

    def last(self):
        """
        Return the last record in the flow or None if empty.
        Materializes the flow.
        """
        data = list(self._records)
        return data[-1] if data else None

    def is_empty(self):
        """
        Return True if the flow has no records.
        """
        try:
            next(iter(self._records))
            return False
        except StopIteration:
            return True

    def keys(self):
        """
        Return a set of all keys across all records.
        Useful for schema inspection.
        """
        keyset = set()
        for r in self._records:
            keyset.update(r.keys())
        return keyset

    def explode(self, key):
        def generator():
            for record in self._records:
                values = record.get(key)
                if isinstance(values, list):
                    for item in values:
                        new_rec = dict(record)
                        new_rec[key] = item
                        yield new_rec
                else:
                    yield record  # keep as-is if not a list

        return Flow.from_generator(generator())

    def to_pandas(self):
        return pd.DataFrame(self._records)

    def to_json_files(self, folder, by=None):
        """
        Write each record to a separate JSON file in the given folder.

        Args:
            folder (str): Output folder path. Will be created if needed.
            by (str, optional): Field to name the files by (e.g., "id", "match_id").
                                If None, defaults to record_1.json, record_2.json, ...
        """
        os.makedirs(folder, exist_ok=True)

        for i, record in enumerate(self._records, start=1):
            if by:
                name = sanitize_filename(record.get(by, f"record_{i}"))
            else:
                name = f"record_{i}"
            path = os.path.join(folder, f"{name}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

        return self

    def to_jsonl(self, path, encoding="utf-8"):
        """
        Save all records to a single JSON Lines (.jsonl) file.
        Each record is written as one line of JSON.
        """
        with open(path, "w", encoding=encoding) as f:
            for record in self._records:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")

        return self

    def to_json_single(self, path, encoding="utf-8", indent=2):
        """
        Save all records to a single JSON file as an array.
        """
        with open(path, "w", encoding=encoding) as f:
            json.dump(list(self._records), f, ensure_ascii=False, indent=indent)

        return self

    @classmethod
    def from_jsonl(cls, path, encoding="utf-8"):
        """
        Load a .jsonl (JSON Lines) file into a Flow.
        Each line must be a valid JSON object.
        """

        def generator():
            with open(path, "r", encoding=encoding) as f:
                for line in f:
                    if line.strip():  # skip empty lines
                        yield json.loads(line)

        return cls.from_generator(generator())

    @classmethod
    def from_file(cls, path: str, encoding="utf-8"):
        """
        Load a local JSON file (list or single dict) into a Flow.
        Generic — no provider-specific assumptions.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r", encoding=encoding) as f:
            data = json.load(f)

        if isinstance(data, list):
            return cls.from_generator(record for record in data)
        else:
            return cls.from_generator([data])

    @classmethod
    def from_folder(cls, folder_path: str, encoding="utf-8"):
        """
        Load and stream all JSON records from a folder.
        - Flattens each file (list or single dict).
        - Skips non-JSON files.
        """

        def generator():
            for filename in os.listdir(folder_path):
                if filename.endswith(".json"):
                    full_path = os.path.join(folder_path, filename)
                    with open(full_path, "r", encoding=encoding) as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                yield item
                        elif isinstance(data, dict):
                            yield data
                        else:
                            continue  # unsupported type

        return cls.from_generator(generator())

    @classmethod
    def from_glob(cls, pattern):
        """
        Load and stream all JSON records matching a glob path.
        E.g. '*.json', 'data/events/*378*.json', 'data/**/*.json'
        """

        def generator():
            for path in glob.glob(pattern, recursive=True):
                if os.path.isdir(path):
                    continue
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            yield item
                    elif isinstance(data, dict):
                        yield data

        return cls.from_generator(generator())

    @classmethod
    def from_records(cls, data):
        """
        Create a Flow from one or more dict-like records.
        Accepts:
        - list of dicts
        - single dict
        - iterable of dicts
        """
        if isinstance(data, dict):
            # Single record
            return cls.from_generator([data])
        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                return cls.from_generator(r for r in data)
        elif hasattr(data, "__iter__"):
            # Iterable of dicts
            return cls.from_generator(data)

        raise TypeError(
            "Expected a list of dicts, a single dict, or iterable of dicts."
        )

    class statsbomb:
        @staticmethod
        def from_github_file(match_id: int, type: str = "events"):
            url = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/{type}/{match_id}.json"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                return Flow.from_generator(r for r in data)
            else:
                return Flow.from_generator([data])


class Group:
    def __init__(self, keys, groups):
        # keys: tuple of field names
        # groups: dict mapping key tuples to list of records
        self.group_keys = keys
        self.groups = groups

    def ungroup(self):
        """
        Flatten the filtered/grouped records back into a Flow.
        Returns a Flow over the original records of each remaining group.
        """
        # chain all the lists of records together
        flat_iter = chain.from_iterable(self.groups.values())
        return Flow.from_generator(flat_iter)

    def drop_duplicates(self, *fields, keep="first"):
        """
        Within each group, drop duplicate records by `fields`.
        keep: 'first', 'last', or False (drop all duplicates).
        """
        new_groups = {}
        for key, recs in self.groups.items():
            seen = {}
            for rec in recs:
                if fields:
                    k = tuple(rec.get(f) for f in fields)
                else:
                    k = tuple(sorted(rec.items()))
                if k in seen:
                    if keep == "last":
                        seen[k] = rec
                    elif keep is False:
                        seen[k] = None
                else:
                    seen[k] = rec
            # preserve order, drop None
            new_groups[key] = [r for r in seen.values() if r is not None]
        return Group(self.group_keys, new_groups)

    def tail(self, n):
        """
        Within each group, keep only the last `n` records.
        """
        new_groups = {
            key: recs[-n:] if len(recs) > n else recs[:]
            for key, recs in self.groups.items()
        }
        return Group(self.group_keys, new_groups)

    def unique(self, *fields):
        """
        Within each group, keep only unique records by `fields`.
        If no fields, dedups whole record.
        """
        new_groups = {}
        for key, recs in self.groups.items():
            seen = set()
            unique_recs = []
            for rec in recs:
                if fields:
                    k = tuple(rec.get(f) for f in fields)
                else:
                    k = tuple(sorted(rec.items()))
                if k not in seen:
                    seen.add(k)
                    unique_recs.append(rec)
            new_groups[key] = unique_recs
        return Group(self.group_keys, new_groups)

    def rename(self, **mapping):
        """
        Within each group, rename record keys via old=new mapping.
        """
        new_groups = {}
        for key, recs in self.groups.items():
            renamed = []
            for rec in recs:
                r = dict(rec)
                for old, new in mapping.items():
                    if old in r:
                        r[new] = r.pop(old)
                renamed.append(r)
            new_groups[key] = renamed
        return Group(self.group_keys, new_groups)

    def filter(self, fn):
        # Keep only groups where fn(records) is True
        new_groups = {k: v for k, v in self.groups.items() if fn(v)}
        return Group(self.group_keys, new_groups)

    def sort(self, by, reverse=False):
        # Sort records within each group
        new_groups = {}
        for k, records in self.groups.items():
            new_groups[k] = sorted(
                records, key=lambda r: r.get(by, None), reverse=reverse
            )
        return Group(self.group_keys, new_groups)

    def head(self, n=5):
        # Take first n records in each group
        new_groups = {k: v[:n] for k, v in self.groups.items()}
        return Group(self.group_keys, new_groups)

    def summary(self, **aggregates):
        summary_rows = []
        for key_tuple, records in self.groups.items():
            row = dict(zip(self.group_keys, key_tuple))
            for col, spec in aggregates.items():
                if isinstance(spec, str):
                    func = _AGGS.get(spec)
                    if not func:
                        raise ValueError(f"Unknown aggregate '{spec}'")
                    row[col] = func(records)

                elif isinstance(spec, tuple) and len(spec) == 2:
                    field, func_name = spec
                    func = _AGGS.get(func_name)
                    if not func:
                        raise ValueError(f"Unknown aggregate '{func_name}'")
                    values = [r.get(field) for r in records if r.get(field) is not None]
                    row[col] = func(values) if values or func_name == "count" else None

                elif callable(spec):
                    row[col] = spec(records)

                else:
                    raise ValueError(f"Bad summarize spec for '{col}': {spec}")

            summary_rows.append(row)

        return Flow(summary_rows)

    def row_number(self, by, new_field="row_number", reverse=False):
        """
        Assigns a row number within each group.
        """
        new_groups = {}
        for key, recs in self.groups.items():
            sorted_recs = sorted(recs, key=lambda r: r.get(by, None), reverse=reverse)
            for idx, rec in enumerate(sorted_recs, start=1):
                rec[new_field] = idx
            new_groups[key] = sorted_recs
        return Group(self.group_keys, new_groups)

    def first(self):
        """
        Return a Group with only the first group (by insertion order).
        Useful for debugging.
        """
        for key in self.groups:
            return Group(self.group_keys, {key: self.groups[key]})
        return Group(self.group_keys, {})  # empty group

    def last(self):
        """
        Return a Group with only the last group.
        """
        if not self.groups:
            return Group(self.group_keys, {})
        last_key = list(self.groups.keys())[-1]
        return Group(self.group_keys, {last_key: self.groups[last_key]})

    def pipe(self, func, *args, **kwargs):
        return func(self, *args, **kwargs)

    def collect(self):
        # Return the raw groups dict
        return self.groups

    def to_pandas(self, agg_funcs=None):
        # Shortcut to convert a summarize directly to pandas
        if agg_funcs:
            return self.summarize(**agg_funcs).to_pandas()
        else:
            # Flatten groups into DataFrame
            rows = []
            for key, recs in self.groups.items():
                for rec in recs:
                    row = dict(zip(self.group_keys, key))
                    row.update(rec)
                    rows.append(row)
            return pd.DataFrame(rows)
