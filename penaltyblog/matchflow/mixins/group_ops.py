"""
Group operations for handling a streaming data pipeline, specifically the Flow class.
"""

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Iterator, Union

if TYPE_CHECKING:
    from ..flowgroup import FlowGroup
    from ..flow import Flow

from ..consumption_guard import guard_consumption
from ..core import _resolve_agg


class GroupOpsMixin:

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

        return self.__class__(gen())

    @guard_consumption
    def group_by(self, *keys: str) -> "FlowGroup":
        """
        Group records by the specified keys and return a FlowGroup object.

        Consumes (materializes) the stream to group records.

        Args:
            *keys (str): The names of the fields to group by.

        Returns:
            FlowGroup: A FlowGroup object
        """
        self._consumed = self._is_consumable()
        from ..flowgroup import FlowGroup

        groups = defaultdict(list)
        for record in self._records:
            group_key = tuple(record.get(k) for k in keys)
            groups[group_key].append(dict(record))
        return FlowGroup(keys, groups)

    @guard_consumption
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

        return self.__class__(gen())
