"""
Core operations for handling a streaming data pipeline, specifically the Flow class.
"""

from itertools import tee
from typing import Iterator

from ..consumption_guard import guard_consumption


class CoreOpsMixin:

    def __len__(self) -> int:
        """
        Return the number of records in the Flow.

        Consumes the stream (materializes all records).

        Returns:
            int: The number of records in the Flow.
        """
        return len(self.collect())

    @guard_consumption
    def __iter__(self) -> Iterator[dict]:
        """
        Return an iterator over the records in the Flow.

        May consume the stream if iterated fully.

        Returns:
            Iterator[dict]: An iterator over the records in the Flow.
        """
        self._consumed = self._is_consumable()
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
        from ..flow import Flow

        self_list = self.collect()
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
        Materializes the stream and returns a new Flow instance. Note that this consumes the stream
        of data and the original flow will now be empty.

        Returns:
            Flow: A new Flow instance that is a fully materialized copy of the
            current stream, backed by a list of records. This allows for safe
            re-scanning and manipulation without affecting the original stream.
        """
        return self.__class__(self.collect())

    @guard_consumption
    def collect(self) -> list[dict]:
        """
        Materializes the stream into a list of dicts. Note that this consumes the stream
        of data and the flow will now be empty.

        Returns:
            list[dict]: The records in the flow.
        """
        self._consumed = self._is_consumable()
        return list(self._records)

    def fork(self) -> tuple["Flow", "Flow"]:
        """
        Fork this Flow into two independent streams using itertools.tee.

        Returns:
            tuple[Flow, Flow]: A tuple of two new Flow instances,
            each backed by an independent iterator of the original stream.
            These flows are one-shot and cannot be iterated more than once.
        """
        it1, it2 = tee(self._records, 2)
        return self.__class__(it1), self.__class__(it2)
