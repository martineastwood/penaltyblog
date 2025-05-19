"""
Transform operations for handling a streaming data pipeline, specifically the Flow class.
"""

import warnings
from collections import Counter
from itertools import chain, islice, zip_longest
from typing import Any, Callable, Iterator, Optional, Union

from ..helpers import delete_path, get_field, resolve_path, set_path


class TransformOpsMixin:
    def filter(self, fn: Callable) -> "Flow":
        """
        Filter the records using the given function.

        Does not consume the stream.

        Args:
            fn (Callable): The function to use for filtering.

        Returns:
            Flow: A new Flow with the filtered records.
        """
        return self.__class__(r for r in self._records if fn(r))

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
            for key, func in kwargs.items():
                record[key] = func(record)
            return record

        return self.__class__(mutate_record(r) for r in self._records)

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

        # Pre‐compute accessors
        accessors = [get_field(f) for f in fields]

        if leaf_names:
            # Warn if leaf names collide
            leaf_keys = [f.rsplit(".", 1)[-1] for f in fields]
            dupes = [k for k, cnt in Counter(leaf_keys).items() if cnt > 1]
            if dupes:
                warnings.warn(
                    f"select(..., leaf_names=True) will produce duplicate keys: {dupes}. "
                    "Later ones will overwrite earlier ones.",
                    UserWarning,
                    stacklevel=2,
                )

        def select_fields(record: dict[str, Any]) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for field, accessor in zip(fields, accessors):
                # if record literally has this exact key (even if it contains dots), use it
                if field in record:
                    val = record[field]
                else:
                    # otherwise fall back to nested-path lookup
                    val = accessor(record)
                # choose output key name
                key_name = field.rsplit(".", 1)[-1] if leaf_names else field
                out[key_name] = val
            return out

        return self.__class__(select_fields(r) for r in self._records)

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
            for f in fields:
                if f in record:
                    del record[f]
                else:
                    delete_path(record, f)
            return record

        return self.__class__(remover(r) for r in self._records)

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
        if isinstance(by, (list, tuple)):
            getters = [get_field(f) for f in by]

            def get_sort_key(record):
                return tuple(g(record) for g in getters)

            def is_null_record(record):
                return any(g(record) is None for g in getters)

        else:
            getter = get_field(by)

            def get_sort_key(record):
                return getter(record)

            def is_null_record(record):
                return getter(record) is None

        def _lazy_sort():
            recs = self.collect()

            non_null = [r for r in recs if not is_null_record(r)]
            nulls = [r for r in recs if is_null_record(r)]

            for r in sorted(non_null, key=get_sort_key, reverse=reverse):
                yield r
            yield from nulls

        return self.__class__(_lazy_sort())

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
            raw = record[key] if key in record else resolve_path(record, key)
            if isinstance(raw, list):
                if len(raw) < len(into):
                    warnings.warn(
                        f"{key!r} has only {len(raw)} elements but expected {len(into)}",
                        UserWarning,
                    )
                for i, name in enumerate(into):
                    record[name] = raw[i] if i < len(raw) else None
            return record

        return self.__class__(splitter(r) for r in self._records)

    def concat(self, *others: "Flow") -> "Flow":
        """
        Concatenate this Flow with one or more other Flows.

        Does not consume the stream.

        Args:
            *others (Flow): The other Flows to concatenate.

        Returns:
            Flow: A new Flow with the concatenated records.
        """

        return self.__class__(chain(self._records, *(o._records for o in others)))

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
        getter = get_field(by)

        def gen():
            recs = self.collect()

            non_null = [r for r in recs if getter(r) is not None]
            nulls = [r for r in recs if getter(r) is None]

            sorted_non_null = sorted(non_null, key=getter, reverse=reverse)

            for idx, rec in enumerate(sorted_non_null, start=1):
                rec[new_field] = idx
                yield rec

            for rec in nulls:
                rec[new_field] = None
                yield rec

        return self.__class__(gen())

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

        def gen():
            seen = set()
            accessors = [get_field(f) for f in fields]
            for record in self.collect():
                if fields:
                    key = tuple(accessor(record) for accessor in accessors)
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

        return self.__class__(gen())

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

            for record in self.collect():
                for old, new in mapping.items():
                    # Prefer flat key rename if present
                    if old in record:
                        record[new] = record.pop(old)
                    else:
                        val = resolve_path(record, old)
                        if val is not None:
                            delete_path(record, old)
                            set_path(record, new, val)
                yield record

        return self.__class__(gen())

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

        return self.__class__(gen())

    def first(self) -> Optional[dict]:
        """
        Returns the first record in the flow or None if empty.

        Consumes the stream (materializes all records).

        Returns:
            dict | None: The first record in the flow or None if empty.
        """
        self._consumed = self._is_consumable()
        lst = self.collect()
        return lst[0] if lst else None

    def last(self) -> Optional[dict]:
        """
        Return the last record in the flow or None if empty.

        Consumes (materializes) the stream.

        Returns:
            dict | None: The last record in the flow or None if empty.
        """
        self._consumed = self._is_consumable()
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
        self._consumed = self._is_consumable()
        lst = self.collect()
        return not lst

    def keys(self, limit: Optional[int] = None) -> set[str]:
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
        self._consumed = self._is_consumable()
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
                            record[key] = item
                            yield record
                    else:
                        # empty list → keep the record unchanged
                        yield record
                else:
                    # non-list or missing → keep as is
                    yield record

        return self.__class__(generator())

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
                    for key, val in zip(keys, items):
                        rec[key] = val
                    yield rec

        return self.__class__(gen())

    def take_last(self, n: int) -> "Flow":
        """
        Take the last `n` records.

        Consumes (materializes) the stream to return the last records.

        Args:
            n (int): The number of records to take.

        Returns:
            Flow: A new Flow with the last `n` records.
        """
        self._consumed = self._is_consumable()

        def gen():
            if n < 0:
                raise ValueError("n must be >= 0")
            records = self.collect()
            if n == 0:
                return  # yields nothing
            for rec in records[-n:]:
                yield rec

        return self.__class__(gen())

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
        from ..flow import Flow

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
                if fields:
                    for f in fields:
                        left_rec[f] = right_rec.get(f)
                else:
                    for k, v in right_rec.items():
                        if k != right_on:
                            left_rec[k] = v
                yield left_rec

        return self.__class__(gen())

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

        return self.__class__(islice(self._records, n))

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
