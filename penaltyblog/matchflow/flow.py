"""
Flow class for handling a streaming data pipeline.
"""

import json
from pprint import pprint
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

if TYPE_CHECKING:
    from .flowgroup import FlowGroup

import pandas as pd

from .aggs_registry import resolve_aggregator
from .executor import FlowExecutor, is_materializing_op
from .predicates_helpers import and_
from .steps.utils import flatten_dict, get_field, schema

PlanNode = Dict[str, Any]


class Flow:
    def __init__(self, plan: Optional[List[PlanNode]] = None):
        self.plan = plan or []

    def __eq__(self, other):
        return isinstance(other, Flow) and self.plan == other.plan

    def __iter__(self):
        return iter(self.collect())

    def __len__(self):
        return sum(1 for _ in self.collect())

    def __repr__(self):
        return f"<Flow steps={len(self.plan)}>"

    @staticmethod
    def from_folder(path: str) -> "Flow":
        """
        Create a Flow from a folder of records.
        Args:
            path (str): The path to the folder containing the records.
        Returns:
            Flow: A new Flow pointing to the records.
        """
        return Flow(plan=[{"op": "from_folder", "path": path}])

    @staticmethod
    def from_json(path: str) -> "Flow":
        """Lazily load a list of records from a JSON file.

        Args:
            path (str): The path to the JSON file.
        Returns:
            Flow: A new Flow pointing to the records.
        """
        return Flow(plan=[{"op": "from_json", "path": path}])

    @staticmethod
    def from_jsonl(path: str) -> "Flow":
        """Lazily load records from a JSONL file.

        Args:
            path (str): The path to the JSONL file.
        Returns:
            Flow: A new Flow pointing to the records.
        """
        return Flow(plan=[{"op": "from_jsonl", "path": path}])

    @staticmethod
    def from_glob(pattern: str) -> "Flow":
        """
        Create a Flow from a glob pattern.

        Args:
            pattern (str): Glob pattern (e.g., "data/**/*.json").

        Returns:
            Flow: A new Flow streaming matching files.
        """
        return Flow(plan=[{"op": "from_glob", "pattern": pattern}])

    @staticmethod
    def from_records(records: List[dict]) -> "Flow":
        """
        Create a Flow from a list of records.
        Args:
            records (List[dict]): The list of records to create a Flow from.
        Returns:
            Flow: A new Flow with the records.
        """
        if isinstance(records, dict):
            raise TypeError(
                "`from_records()` expects a list of dicts, not a single dict"
            )

        return Flow(plan=[{"op": "from_materialized", "records": records}])

    def to_json(self, path: str, indent=4):
        """
        Write the flow to a JSON file (as a list of records).

        Args:
            path (str): The path to the JSON file.
            indent (int, optional): The number of spaces to use for indentation.
        """
        records = list(self.collect())
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=indent)

    def to_jsonl(self, path: str):
        """
        Write the flow to a JSONL file (one record per line).

        Args:
            path (str): The path to the JSONL file.
        """
        with open(path, "w", encoding="utf-8") as f:
            for row in self.collect():
                f.write(json.dumps(row) + "\n")

    def to_pandas(self) -> pd.DataFrame:
        """
        Collect the flow into a pandas DataFrame.

        Returns:
            pd.DataFrame: The collected records as a DataFrame.
        """
        return pd.DataFrame(self.collect())

    def count(self) -> int:
        """
        Count the number of records in the flow.

        Returns:
            int: The number of records.
        """
        return sum(1 for _ in self.collect())

    def is_empty(self) -> bool:
        """
        Check if the flow yields any records without fully collecting it.

        Returns:
            bool: True if there are no records, False otherwise.
        """
        it = FlowExecutor(self.plan).execute()
        try:
            next(it)
            return False
        except StopIteration:
            return True

    def filter(self, *predicates: Callable[[dict], bool]) -> "Flow":
        if not predicates:
            return self

        if len(predicates) == 1:
            predicate = predicates[0]
        else:
            predicate = and_(*predicates)

        return Flow(self.plan + [{"op": "filter", "predicate": predicate}])

    def assign(self, **fields: Callable[[dict], Any]) -> "Flow":
        """
        Assign new fields to each record.
        Args:
            **fields (dict[str, Callable[[dict], Any]]): The fields to assign.
        Returns:
            Flow: A new Flow with assigned fields.
        """
        return Flow(self.plan + [{"op": "assign", "fields": fields}])

    def select(self, *fields: str) -> "Flow":
        """
        Select specific fields from each record.
        Args:
            *fields (str): The fields to select.
        Returns:
            Flow: A new Flow with selected fields.
        """
        return Flow(self.plan + [{"op": "select", "fields": list(fields)}])

    def flatten(self) -> "Flow":
        """
        Flatten nested dictionaries into a single-level dictionary using dot notation.

        Returns:
            Flow: A new Flow with flattened records.
        """
        return Flow(self.plan + [{"op": "flatten"}])

    def distinct(self, *keys: str, keep: str = "first") -> "Flow":
        """
        Remove duplicate records.

        Args:
            *keys (str): Optional field names to determine uniqueness.
            keep (str): 'first' (default) or 'last' to control which duplicate is retained.

        Returns:
            Flow: A new Flow with duplicates removed.
        """
        if keep not in {"first", "last"}:
            raise ValueError("keep must be 'first' or 'last'")

        return Flow(
            self.plan
            + [
                {
                    "op": "distinct",
                    "keys": list(keys) if keys else None,
                    "keep": keep,
                }
            ]
        )

    def rename(self, **mapping: str):
        """
        Rename keys in each record according to mapping of old=new.
        Args:
            mapping (dict[str, str]): The mapping of old keys to new keys.
        Returns:
            Flow: A new Flow with renamed keys.
        """
        return Flow(self.plan + [{"op": "rename", "mapping": mapping}])

    def group_by(self, *keys: str) -> "FlowGroup":
        """
        Group records by one or more fields.

        Args:
            *keys (str): Field names to group by.

        Returns:
            FlowGroup: A new FlowGroup with grouped records.
        """
        from .group import FlowGroup

        return FlowGroup(self.plan + [{"op": "group_by", "keys": list(keys)}])

    def grouped(self, key: str):
        """
        Group records by a single field.

        Args:
            key (str): The field to group by.

        Returns:
            Iterator[Tuple[Any, List[Dict[str, Any]]]]: The groups.
        """
        from itertools import groupby
        from operator import itemgetter

        records = sorted(self.collect(), key=itemgetter(key))
        for k, group in groupby(records, key=itemgetter(key)):
            yield k, list(group)

    def summary(self, aggregators: Union[Callable, dict[str, Any]]) -> "Flow":
        """
        Supports:
        - Callable (e.g. lambda rows: {...})
        - Dict of {alias: callable}
        - Dict of {alias: "name"} or (name/callable, field)
        """
        if callable(aggregators):
            agg_func = aggregators

        elif isinstance(aggregators, dict):

            def agg_func(rows):
                return {
                    alias: resolve_aggregator(value, alias)(rows)
                    for alias, value in aggregators.items()
                }

        else:
            raise TypeError("summary() requires a callable or dict")

        return Flow(self.plan + [{"op": "summary", "agg": agg_func}])

    def sort_by(self, *keys: str, ascending: bool | list[bool] = True) -> "Flow":
        """
        Sort records by one or more fields.

        Args:
            *keys (str): Field names to sort by.
            ascending (bool or list[bool], optional): Sort order(s). Either a single bool
                applied to all keys or a list of bools (one per key).

        Returns:
            Flow: A new Flow with sorted records.
        """
        keys = list(keys)

        # Normalize ascending to a list of same length as keys
        if isinstance(ascending, bool):
            ascending_list = [ascending] * len(keys)
        elif isinstance(ascending, list):
            if len(ascending) != len(keys):
                raise ValueError("Length of 'ascending' must match number of keys.")
            ascending_list = ascending
        else:
            raise TypeError("'ascending' must be a bool or list of bools.")

        return Flow(
            self.plan + [{"op": "sort", "keys": keys, "ascending": ascending_list}]
        )

    def limit(self, n: int) -> "Flow":
        """
        Limit the number of records returned.

        Args:
            n (int): The maximum number of records.

        Returns:
            Flow: A new Flow that yields up to n records.
        """
        return Flow(self.plan + [{"op": "limit", "count": n}])

    def head(self, n=5) -> list:
        """
        Runs the flow and returns the first n records.

        Args:
            n (int): The number of records to return.

        Returns:
            list: A list of the first n records.
        """
        return self.limit(n).collect()

    def show(self, n: int = 5):
        """
        Print the first `n` records in a pretty format.

        Args:
            n (int): Number of records to show.
        """
        for i, row in enumerate(self.collect()):
            if i >= n:
                break
            pprint(row)

    def keys(self, limit: int = 100) -> set[str]:
        """
        Infer the schema of the flow.

        Args:
            limit (int): Number of records to sample for schema inference.

        Returns:
            set[str]: The set of keys.
        """
        sample = self.limit(limit).collect()
        all_keys: set[str] = set()
        for record in sample:
            flat = flatten_dict(record)
            all_keys.update(flat.keys())
        return all_keys

    def drop(self, *keys: str) -> "Flow":
        """
        Drop one or more fields from each record. Supports dot notation for nested fields.

        Args:
            *keys (str): Field names to remove.

        Returns:
            Flow: A new Flow with fields removed.
        """
        return Flow(self.plan + [{"op": "drop", "keys": list(keys)}])

    def dropna(self, *fields: str) -> "Flow":
        """
        Drop records where any of the specified fields are None or missing.
        If no fields are given, drops records where any top-level value is None.

        Args:
            *fields (str): Optional field paths (dot notation) to check for None.

        Returns:
            Flow: A new Flow with records containing nulls removed.
        """
        return Flow(
            self.plan + [{"op": "dropna", "fields": list(fields) if fields else None}]
        )

    def concat(self, *others: "Flow") -> "Flow":
        """
        Concatenate this flow with one or more other flows.

        Args:
            *others (Flow): One or more flows to concatenate.

        Returns:
            Flow: A new Flow representing the concatenated sequence.
        """
        return Flow(
            [{"op": "from_concat", "plans": [self.plan] + [f.plan for f in others]}]
        )

    def explode(self, *fields: str) -> "Flow":
        """
        Explode one or more list fields into multiple records (in sync).
        All fields must be lists of the same length in each record.

        Args:
            *fields (str): One or more field names (dot notation allowed).

        Returns:
            Flow: A new Flow with records exploded along the given fields.
        """
        return Flow(self.plan + [{"op": "explode", "fields": list(fields)}])

    def join(
        self,
        other: "Flow",
        on: str | list[str],
        how: str = "left",
        suffix: str = "_right",
    ) -> "Flow":
        """
        Join with another Flow.

        Args:
            other (Flow): The right-hand flow to join with.
            on (str or list[str]): Field(s) to join on.
            how (str): 'left' (default) or 'inner'.
            suffix (str): Suffix for conflicting right-side keys.

        Returns:
            Flow: A new Flow representing the join.
        """
        how = how.lower()
        if how not in {"left", "inner"}:
            raise ValueError("Only 'left' and 'inner' joins are supported currently.")

        return Flow(
            self.plan
            + [
                {
                    "op": "join",
                    "on": [on] if isinstance(on, str) else on,
                    "right_plan": other.plan,
                    "how": how,
                    "suffix": suffix,
                }
            ]
        )

    def split_array(self, field: str, into: list[str]) -> "Flow":
        """
        Split an array field into separate fields by index.

        Args:
            field (str): The field containing a list/array.
            into (list[str]): The names of the output fields.

        Returns:
            Flow: A new Flow with the array split into separate fields.
        """
        return Flow(
            self.plan
            + [
                {
                    "op": "split_array",
                    "field": field,
                    "into": into,
                }
            ]
        )

    def pivot(self, index: str | list[str], columns: str, values: str) -> "Flow":
        """
        Pivot records: turn row values into columns.

        Args:
            index (str or list[str]): Fields to group by.
            columns (str): Field whose values become column names.
            values (str): Field whose values fill the new columns.

        Returns:
            Flow: A new Flow with records pivoted into wide format.
        """
        return Flow(
            self.plan
            + [
                {
                    "op": "pivot",
                    "index": [index] if isinstance(index, str) else index,
                    "columns": columns,
                    "values": values,
                }
            ]
        )

    def cache(self) -> "Flow":
        """
        Cache the records in memory.

        Returns:
            Flow: A new Flow with the records cached.
        """
        records = list(self.collect())
        return Flow([{"op": "from_materialized", "records": records}])

    def explain(self):
        i = 0
        while i < len(self.plan):
            step = self.plan[i]
            op = step["op"]

            # Detect fusion block
            if op in {"map", "assign", "filter"}:
                fused = []
                j = i
                while j < len(self.plan) and self.plan[j]["op"] in {
                    "map",
                    "assign",
                    "filter",
                }:
                    fused.append(self.plan[j]["op"])
                    j += 1
                if len(fused) > 1:
                    print(f"{i+1:>2}. fused: {fused}")
                    i = j
                    continue
                # If only one, treat it normally
            # Standard step display
            details = {k: v for k, v in step.items() if k != "op"}
            if op == "from_materialized" and "records" in details:
                details["records"] = f"<{len(details['records'])} records>"
            if op in {"map", "pipe", "filter"} and "func" in details:
                func = details["func"]
                details["func"] = getattr(func, "__name__", repr(func))

            materializing = "⚠️ materializes data" if is_materializing_op(op) else ""
            print(f"{i+1:>2}. {op}: {details} {materializing}")
            i += 1

    def collect(self) -> list:
        """
        Collect all records from the flow.

        Returns:
            list: A list of records.
        """
        return list(FlowExecutor(self.plan).execute())

    def schema(self, n=100) -> dict[str, type]:
        """
        Infer the schema of the flow.

        Args:
            n (int): Number of records to sample for schema inference.

        Returns:
            dict[str, type]: The inferred schema.
        """
        sample = list(self.limit(n).collect())
        return schema(sample)

    def cast(self, **casts: Union[type, Callable[[Any], Any]]) -> "Flow":
        """
        Cast fields to specified types or functions.

        Args:
            **casts (type or Callable[[Any], Any]): The types or functions to cast to.

        Returns:
            Flow: A new Flow with the fields cast.
        """

        def make_caster(path, func):
            def caster(record):
                value = get_field(record, path)
                try:
                    return func(value)
                except Exception:
                    return value  # fallback: leave original if cast fails

            return path, caster

        assignments = {path: make_caster(path, func)[1] for path, func in casts.items()}
        return self.assign(**assignments)

    def sample_fraction(self, p: float, seed: Optional[int] = None) -> "Flow":
        """
        Lazily sample a fraction of records.

        Args:
            p (float): Sampling probability (0 < p < 1).
            seed (int, optional): Random seed.

        Returns:
            Flow: A new Flow with sampling applied.
        """
        return Flow(self.plan + [{"op": "sample_fraction", "p": p, "seed": seed}])

    def sample_n(self, n: int, seed: Optional[int] = None) -> "Flow":
        """
        Lazily sample n records using reservoir sampling.

        Args:
            n (int): Number of records to sample.
            seed (int, optional): Random seed.

        Returns:
            Flow: A new Flow with sampling applied.
        """
        return Flow(self.plan + [{"op": "sample_n", "n": n, "seed": seed}])

    def map(self, func: Callable[[dict], dict]) -> "Flow":
        """
        Apply a function to each record. Should return a full record. If the function returns None,
        the record is dropped.

        Args:
            func (Callable[[dict], dict]): A function that takes a record and returns a modified one.

        Returns:
            Flow: A new Flow with the transformed records.
        """
        return Flow(self.plan + [{"op": "map", "func": func}])

    def pipe(self, func: Callable[["Flow"], "Flow"]) -> "Flow":
        """
        Lazily apply a function to this Flow and return the resulting Flow.
        The function will be executed at collect-time, not immediately.

        The function should return a new Flow, typically using this one as input.
        """
        return Flow(self.plan + [{"op": "pipe", "func": func}])


from .contrib.statsbomb import statsbomb as statsbomb_module

Flow.statsbomb = statsbomb_module
