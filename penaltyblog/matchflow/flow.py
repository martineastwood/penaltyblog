"""
Flow class for handling a streaming data pipeline.
"""

import collections.abc
from collections.abc import Iterable
from typing import Any, Union

from .mixins.core_ops import CoreOpsMixin
from .mixins.group_ops import GroupOpsMixin
from .mixins.io_ops import IOOpsMixin
from .mixins.sample_ops import SampleOpsMixin
from .mixins.transform_ops import TransformOpsMixin
from .statsbomb.statsbombflow import statsbomb


class Flow(GroupOpsMixin, CoreOpsMixin, IOOpsMixin, TransformOpsMixin, SampleOpsMixin):
    """
    A class representing a flow of data records.

    Args:
        records (Iterable[dict[Any, Any]]): An iterable of dictionaries representing the records to be processed.

    Notes:
        Many methods in this class operate on a stream of records. Methods that materialize or exhaust the stream will be explicitly documented.
    """

    def __init__(self, records: Iterable[dict[Any, Any]]):
        self._consumed = False
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

    def _is_consumable(self):
        return isinstance(self._records, collections.abc.Iterator)


Flow.statsbomb = statsbomb
