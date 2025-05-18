"""
FlowGroup class for handling a streaming data pipeline, specifically the Flow class.
"""

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import pandas as pd

from .core import _resolve_agg

if TYPE_CHECKING:
    # only for type‐checking; no runtime import
    from .flow import Flow


class FlowGroup:
    """
    A group of records sharing the same key tuple.
    """

    def __init__(
        self, keys: Tuple[str, ...], groups: dict[tuple[Any, ...], List[dict[str, Any]]]
    ) -> None:
        """
        A group of records sharing the same key tuple.

        Args:
            keys (tuple[str]): The group key tuple.
            groups (dict[tuple, list]): The group records.
        """
        self.group_keys = keys
        self.groups = groups

    def __iter__(self) -> Iterator[Tuple[tuple[Any, ...], List[dict[str, Any]]]]:
        """
        Iterate over (group_key_tuple, records_list) pairs.

        Yields:
            Iterator[Tuple[tuple[Any, ...], List[dict[str, Any]]]]: An iterator over tuples where each tuple
            contains a group key and the corresponding list of records.
        """
        return iter(self.groups.items())

    def __len__(self) -> int:
        """
        Return the number of groups.

        Returns:
            int: The number of groups.
        """
        return len(self.groups)

    def __repr__(self) -> str:
        """
        Return a brief summary: number of groups and a sample of the first few keys.
        """
        n = len(self)
        sample_keys = list(self.groups.keys())[:3]
        return f"<Penaltyblog Flow Group | n_groups={n} | sample_keys={sample_keys}>"

    def keys(self) -> List[tuple[Any, ...]]:
        """
        Return a list of the current group key tuples.

        Returns:
            List[tuple[Any, ...]]: A list of the current group key tuples.
        """
        return list(self.groups.keys())

    def ungroup(self) -> "Flow":
        """
        Flatten the filtered/grouped records back into a Flow.
        Returns a Flow over the original records of each remaining group.
        """
        from .flow import Flow

        def gen():
            for key_tuple, recs in self.groups.items():
                # build a dict of the grouping columns, e.g. {"team": "A", "period": 1}
                prefix = dict(zip(self.group_keys, key_tuple))
                for rec in recs:
                    # shallow‐merge so we don't mutate the originals
                    out = {**prefix, **rec}
                    yield out

        return Flow.from_generator(gen())

    def drop_duplicates(self, *fields: str, keep: str = "first") -> "FlowGroup":
        """
        Within each group, drop duplicate records by `fields`.
        keep: 'first', 'last', or False (drop all duplicates).

        Args:
            fields (tuple[str]): The fields to use for deduplication.
            keep (str, optional): How to handle duplicates. Defaults to "first".

        Returns:
            FlowGroup: A new FlowGroup with duplicates dropped.
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
        return FlowGroup(self.group_keys, new_groups)

    def tail(self, n: int) -> "FlowGroup":
        """
        Within each group, keep only the last `n` records.

        Args:
            n (int): The number of records to keep.

        Returns:
            FlowGroup: A new FlowGroup with the last `n` records in each group.
        """
        new_groups = {
            key: recs[-n:] if len(recs) > n else recs[:]
            for key, recs in self.groups.items()
        }
        return FlowGroup(self.group_keys, new_groups)

    def unique(self, *fields: str) -> "FlowGroup":
        """
        Within each group, keep only unique records by `fields`.
        If no fields, dedups whole record.

        Args:
            fields (tuple[str]): The fields to use for deduplication.

        Returns:
            FlowGroup: A new FlowGroup with unique records in each group.
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
        return FlowGroup(self.group_keys, new_groups)

    def rename(self, **mapping: str) -> "FlowGroup":
        """
        Within each group, rename record keys via old=new mapping.

        Args:
            mapping (dict[str, str]): The mapping of old keys to new keys.

        Returns:
            FlowGroup: A new FlowGroup with renamed keys.
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
        return FlowGroup(self.group_keys, new_groups)

    def filter(self, fn: Callable[[list[dict]], bool]) -> "FlowGroup":
        """
        Within each group, keep only records where fn(records) is True.

        Args:
            fn (Callable[[list[dict]], bool]): The function to use for filtering.

        Returns:
            FlowGroup: A new FlowGroup with filtered records
        """
        new_groups = {k: v for k, v in self.groups.items() if fn(v)}
        return FlowGroup(self.group_keys, new_groups)

    def sort(self, by: str, reverse: bool = False) -> "FlowGroup":
        """
        Sort records within each group by the given field, always putting
        records where the field is None at the end.

        Args:
            by (str): The field to sort by.
            reverse (bool, optional): Whether to reverse the sort order. Defaults to False.

        Returns:
            FlowGroup: A new FlowGroup with sorted records
        """
        new_groups = {}
        for k, records in self.groups.items():
            non_null = [r for r in records if r.get(by) is not None]
            nulls = [r for r in records if r.get(by) is None]
            sorted_non_null = sorted(non_null, key=lambda r: r[by], reverse=reverse)
            new_groups[k] = sorted_non_null + nulls
        return FlowGroup(self.group_keys, new_groups)

    def head(self, n: int = 5) -> "FlowGroup":
        """
        Take first n records in each group

        Args:
            n (int, optional): The number of records to take. Defaults to 5.

        Returns:
            FlowGroup: A new FlowGroup with the first n records in each group.
        """
        new_groups = {k: v[:n] for k, v in self.groups.items()}
        return FlowGroup(self.group_keys, new_groups)

    def summary(self, **aggregates: Union[str, Tuple[str, str], Callable]) -> "Flow":
        """
        Compute aggregates for each group and return a new Flow with the results.

        Args:
            **aggregates: The aggregates to compute.

        Returns:
            Flow: New Flow with summary rows
        """
        from .flow import Flow

        summary_rows = []
        for key_tuple, records in self.groups.items():
            row = dict(zip(self.group_keys, key_tuple))
            for col, spec in aggregates.items():
                value = _resolve_agg(records, spec)
                # Check if value is a non-scalar (but allow str/bytes)
                if isinstance(value, (list, tuple, dict, set)):
                    raise ValueError(
                        f"Aggregate '{col}' returned a non-scalar value for group {key_tuple}. Aggregates must return a single value per group."
                    )
                row[col] = value
            summary_rows.append(row)
        return Flow(iter(summary_rows))

    def row_number(
        self, by: str, new_field: str = "row_number", reverse: bool = False
    ) -> "FlowGroup":
        """
        Assigns a row number within each group.

        Args:
            by (str): The field to use for sorting.
            new_field (str, optional): The name of the new field. Defaults to "row_number".
            reverse (bool, optional): Whether to reverse the sort order. Defaults to False.

        Returns:
            FlowGroup: A new FlowGroup with row numbers
        """
        new_groups = {}
        for key, recs in self.groups.items():
            sorted_recs = sorted(recs, key=lambda r: r.get(by, None), reverse=reverse)
            for idx, rec in enumerate(sorted_recs, start=1):
                rec[new_field] = idx
            new_groups[key] = sorted_recs
        return FlowGroup(self.group_keys, new_groups)

    def cumulative(
        self, field: str, new_field: str = "cumulative", sort_by: Optional[str] = None
    ) -> "FlowGroup":
        """
        Compute a cumulative sum of `field` within each group.

        Args:
            field: The numeric field to accumulate.
            new_field: Name of the new cumulative field.
            sort_by: Field to sort records by before accumulating.

        Returns:
            FlowGroup: Groups with cumulative values added.
        """
        new_groups = {}
        for key, recs in self.groups.items():
            # Sort records; place None values at the end
            if sort_by:

                def _key(r):
                    k = r.get(sort_by)
                    return (k is None, k)

                records = sorted(recs, key=_key)
            else:
                records = list(recs)

            total = 0
            cum_recs = []
            for rec in records:
                # Safely add numeric field values
                val = rec.get(field, 0) or 0
                total += val
                r = dict(rec)
                r[new_field] = total
                cum_recs.append(r)
            new_groups[key] = cum_recs
        return FlowGroup(self.group_keys, new_groups)

    def first(self) -> "FlowGroup":
        """
        Return a Group with only the first group (by insertion order).
        Useful for debugging.

        Returns:
            FlowGroup: A new FlowGroup with row numbers
        """
        for key in self.groups:
            return FlowGroup(self.group_keys, {key: self.groups[key]})
        return FlowGroup(self.group_keys, {})  # empty group

    def last(self) -> "FlowGroup":
        """
        Return a Group with only the last group.

        Returns:
            FlowGroup: A new FlowGroup with row numbers
        """
        if not self.groups:
            return FlowGroup(self.group_keys, {})
        last_key = list(self.groups.keys())[-1]
        return FlowGroup(self.group_keys, {last_key: self.groups[last_key]})

    def is_empty(self) -> bool:
        """
        Return True if the group is empty.

        Returns:
            bool: True if the group is empty, False otherwise.
        """
        return not bool(self.groups)

    def pipe(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Apply a function to the group.

        Args:
            func (Callable): The function to apply.
            *args: Additional arguments to pass to the function.
            **kwargs: Additional keyword arguments to pass to the function.

        Returns:
            Any: The result of the function.
        """
        return func(self, *args, **kwargs)

    def collect(self) -> dict[Any, List[dict[str, Any]]]:
        """
        Return the raw groups dict.

        Returns:
            dict[Any, List[dict[str, Any]]]: The raw groups dict.
        """
        return self.groups

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert the group to a pandas DataFrame.

        Returns:
            pd.DataFrame: The group as a pandas DataFrame.
        """
        return self.ungroup().to_pandas()
