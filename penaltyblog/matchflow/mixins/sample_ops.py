"""
Sample operations for handling a streaming data pipeline, specifically the Flow class.
"""

import random
from typing import Optional


class SampleOpsMixin:
    def sample(self, n: int, seed: Optional[int] = None) -> "Flow":
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
        for i, record in enumerate(self.collect(), start=1):
            if i <= n:
                reservoir.append(record)
            else:
                j = rnd.randint(1, i)
                if j <= n:
                    reservoir[j - 1] = record
        return self.__class__(iter(reservoir))

    def sample_frac(self, frac: float, seed: Optional[int] = None) -> "Flow":
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
        return self.__class__(r for r in self._records if rnd.random() < frac)
