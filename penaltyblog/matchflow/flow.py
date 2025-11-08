"""
Flow class for handling a streaming data pipeline.
"""

import inspect
import itertools
import json
import urllib.parse
from pprint import pprint
from time import perf_counter
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    cast,
)

from tabulate import tabulate
from typing_extensions import Literal

if TYPE_CHECKING:
    from .flowgroup import FlowGroup

import fsspec
import pandas as pd
from tqdm.auto import tqdm

from .aggs_registry import resolve_aggregator
from .executor import FlowExecutor
from .helpers import explain_plan, set_path, show_tabular
from .optimizer import FlowOptimizer
from .plotting import plot_flow_plan
from .predicates_helpers import and_
from .query import parse_query_expr
from .steps.utils import flatten_dict, get_field, schema


def _handle_missing_dependency(path: str) -> None:
    """
    Check if required cloud storage dependencies are installed and provide helpful error messages.

    Args:
        path (str): The path being accessed

    Raises:
        ImportError: If required dependency is missing
    """
    protocol_mapping = {
        "s3://": "s3fs",
        "gs://": "gcsfs",
        "gcs://": "gcsfs",
        "azure://": "adlfs",
        "abfs://": "adlfs",
        "abfss://": "adlfs",
    }

    for protocol, package in protocol_mapping.items():
        if path.startswith(protocol):
            try:
                __import__(package)
            except ImportError:
                raise ImportError(
                    f"To access {protocol} paths, install {package}: pip install {package}"
                ) from None
            break


PlanNode = Dict[str, Any]


class Flow:
    def __init__(self, plan: Optional[List[PlanNode]] = None, optimize: bool = False):
        self.plan = plan or []
        self.optimize = optimize

    def __eq__(self, other):
        return isinstance(other, Flow) and self.plan == other.plan

    def __iter__(self):
        return iter(self.collect())

    def __len__(self):
        return sum(1 for _ in self.collect())

    def __repr__(self):
        return f"<Flow steps={len(self.plan)}>"

    def _next(self, op: dict) -> "Flow":
        return Flow(self.plan + [op], optimize=self.optimize)

    @staticmethod
    def from_folder(
        path: str,
        optimize: bool = False,
        storage_options: Optional[Dict[str, Any]] = None,
    ) -> "Flow":
        """
        Create a Flow from a folder of records.
        Args:
            path (str): The path to the folder containing the records.
            optimize (bool): Whether to optimize the flow.
            storage_options (dict, optional): Additional options for cloud storage backends.
                For S3: {"key": "access_key", "secret": "secret_key", "endpoint_url": "url"}
                For GCS: {"token": "path/to/token.json"}
                For Azure: {"account_name": "name", "account_key": "key"}
        Returns:
            Flow: A new Flow pointing to the records.
        """
        plan_step = {"op": "from_folder", "path": path}
        if storage_options:
            plan_step["storage_options"] = storage_options
        return Flow(plan=[plan_step], optimize=optimize)

    @staticmethod
    def from_json(
        path: str,
        optimize: bool = False,
        storage_options: Optional[Dict[str, Any]] = None,
    ) -> "Flow":
        """Lazily load a list of records from a JSON file.

        Args:
            path (str): The path to the JSON file.
            optimize (bool): Whether to optimize the flow.
            storage_options (dict, optional): Additional options for cloud storage backends.
                For S3: {"key": "access_key", "secret": "secret_key", "endpoint_url": "url"}
                For GCS: {"token": "path/to/token.json"}
                For Azure: {"account_name": "name", "account_key": "key"}
        Returns:
            Flow: A new Flow pointing to the records.
        """
        plan_step = {"op": "from_json", "path": path}
        if storage_options:
            plan_step["storage_options"] = storage_options
        return Flow(plan=[plan_step], optimize=optimize)

    @staticmethod
    def from_jsonl(
        path: str,
        optimize: bool = False,
        storage_options: Optional[Dict[str, Any]] = None,
    ) -> "Flow":
        """Lazily load records from a JSONL file.

        Args:
            path (str): The path to the JSONL file.
            optimize (bool): Whether to optimize the flow.
            storage_options (dict, optional): Additional options for cloud storage backends.
                For S3: {"key": "access_key", "secret": "secret_key", "endpoint_url": "url"}
                For GCS: {"token": "path/to/token.json"}
                For Azure: {"account_name": "name", "account_key": "key"}
        Returns:
            Flow: A new Flow pointing to the records.
        """
        plan_step = {"op": "from_jsonl", "path": path}
        if storage_options:
            plan_step["storage_options"] = storage_options
        return Flow(plan=[plan_step], optimize=optimize)

    @staticmethod
    def from_glob(
        pattern: str,
        optimize: bool = False,
        storage_options: Optional[Dict[str, Any]] = None,
    ) -> "Flow":
        """
        Create a Flow from a glob pattern.

        Args:
            pattern (str): Glob pattern (e.g., "data/**/*.json").
            optimize (bool): Whether to optimize the flow.
            storage_options (dict, optional): Additional options for cloud storage backends.
                For S3: {"key": "access_key", "secret": "secret_key", "endpoint_url": "url"}
                For GCS: {"token": "path/to/token.json"}
                For Azure: {"account_name": "name", "account_key": "key"}

        Returns:
            Flow: A new Flow streaming matching files.
        """
        plan_step = {"op": "from_glob", "pattern": pattern}
        if storage_options:
            plan_step["storage_options"] = storage_options
        return Flow(plan=[plan_step], optimize=optimize)

    @staticmethod
    def from_records(records: List[dict], optimize: bool = False) -> "Flow":
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

        return Flow(
            plan=[{"op": "from_materialized", "records": records}], optimize=optimize
        )

    def to_json(
        self, path: str, indent=4, storage_options: Optional[Dict[str, Any]] = None
    ):
        """
        Write the flow to a JSON file (as a list of records).

        Args:
            path (str): The path to the JSON file.
            indent (int, optional): The number of spaces to use for indentation.
            storage_options (dict, optional): Additional options for cloud storage backends.
                For S3: {"key": "access_key", "secret": "secret_key", "endpoint_url": "url"}
                For GCS: {"token": "path/to/token.json"}
                For Azure: {"account_name": "name", "account_key": "key"}
        """
        storage_options = storage_options or {}

        # Check dependencies for cloud storage
        _handle_missing_dependency(path)

        records = list(self.collect())
        with fsspec.open(path, "w", encoding="utf-8", **storage_options) as f:
            json.dump(records, f, indent=indent)

    def to_jsonl(self, path: str, storage_options: Optional[Dict[str, Any]] = None):
        """
        Write the flow to a JSONL file (one record per line).

        Args:
            path (str): The path to the JSONL file.
            storage_options (dict, optional): Additional options for cloud storage backends.
                For S3: {"key": "access_key", "secret": "secret_key", "endpoint_url": "url"}
                For GCS: {"token": "path/to/token.json"}
                For Azure: {"account_name": "name", "account_key": "key"}
        """
        storage_options = storage_options or {}

        # Check dependencies for cloud storage
        _handle_missing_dependency(path)

        with fsspec.open(path, "w", encoding="utf-8", **storage_options) as f:
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
        plan = FlowOptimizer(self.plan).optimize() if self.optimize else self.plan
        it = FlowExecutor(plan).execute()
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

        return self._next({"op": "filter", "predicate": predicate})

    def assign(self, **fields: Callable[[dict], Any]) -> "Flow":
        """
        Assign new fields to each record.
        Args:
            **fields (dict[str, Callable[[dict], Any]]): The fields to assign.
        Returns:
            Flow: A new Flow with assigned fields.
        """
        return self._next({"op": "assign", "fields": fields})

    def select(self, *fields: str) -> "Flow":
        """
        Select specific fields from each record.
        Args:
            *fields (str): The fields to select.
        Returns:
            Flow: A new Flow with selected fields.
        """
        return self._next({"op": "select", "fields": list(fields)})

    def flatten(self) -> "Flow":
        """
        Flatten nested dictionaries into a single-level dictionary using dot notation.

        Returns:
            Flow: A new Flow with flattened records.
        """
        return self._next({"op": "flatten"})

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

        return self._next(
            {
                "op": "distinct",
                "keys": list(keys) if keys else None,
                "keep": keep,
            }
        )

    def rename(self, **mapping: str):
        """
        Rename keys in each record according to mapping of old=new.
        Args:
            mapping (dict[str, str]): The mapping of old keys to new keys.
        Returns:
            Flow: A new Flow with renamed keys.
        """
        return self._next({"op": "rename", "mapping": mapping})

    def group_by(self, *keys: str) -> "FlowGroup":
        """
        Group records by one or more fields.

        Args:
            *keys (str): Field names to group by.

        Returns:
            FlowGroup: A new FlowGroup with grouped records.
        """
        from .group import FlowGroup

        return FlowGroup(
            self.plan + [{"op": "group_by", "keys": list(keys)}], optimize=self.optimize
        )

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

        return self._next({"op": "summary", "agg": agg_func})

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
        keys_list: list[str] = list(keys)

        # Normalize ascending to a list of same length as keys
        if isinstance(ascending, bool):
            ascending_list = [ascending] * len(keys)
        elif isinstance(ascending, list):
            if len(ascending) != len(keys):
                raise ValueError("Length of 'ascending' must match number of keys.")
            ascending_list = ascending
        else:
            raise TypeError("'ascending' must be a bool or list of bools.")

        return self._next(
            {"op": "sort", "keys": keys_list, "ascending": ascending_list}
        )

    def limit(self, n: int) -> "Flow":
        """
        Limit the number of records returned.

        Args:
            n (int): The maximum number of records.

        Returns:
            Flow: A new Flow that yields up to n records.
        """
        return self._next({"op": "limit", "count": n})

    def head(self, n=5) -> list:
        """
        Runs the flow and returns the first n records.

        Args:
            n (int): The number of records to return.

        Returns:
            list: A list of the first n records.
        """
        return self.limit(n).collect()

    def show(self, n: int = 5, format: Literal["table", "record"] = "table"):
        """
        Print the first `n` records in a pretty format.

        Args:
            n (int): Number of records to show.
            format (Literal["table", "record"]): Format to use for display.
        """
        rows = list(itertools.islice(self.collect(), n))

        if format == "record":
            for r in rows:
                pprint(r)
            return

        if format == "table":
            show_tabular(rows)
            return

        raise ValueError(f"Unknown format: {format}")

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
        return self._next({"op": "drop", "keys": list(keys)})

    def dropna(self, *fields: str) -> "Flow":
        """
        Drop records where any of the specified fields are None or missing.
        If no fields are given, drops records where any top-level value is None.

        Args:
            *fields (str): Optional field paths (dot notation) to check for None.

        Returns:
            Flow: A new Flow with records containing nulls removed.
        """
        return self._next({"op": "dropna", "fields": list(fields) if fields else None})

    def concat(self, *others: "Flow") -> "Flow":
        """
        Concatenate this flow with one or more other flows.

        Args:
            *others (Flow): One or more flows to concatenate.

        Returns:
            Flow: A new Flow representing the concatenated sequence.
        """
        return self._next(
            {"op": "from_concat", "plans": [self.plan] + [f.plan for f in others]}
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
        return self._next({"op": "explode", "fields": list(fields)})

    def join(
        self,
        other: "Flow",
        on: Union[str, List[str], None] = None,
        left_on: Union[str, List[str], None] = None,
        right_on: Union[str, List[str], None] = None,
        how: Literal["left", "right", "outer", "inner", "anti"] = "left",
        lsuffix: str = "",
        rsuffix: str = "_right",
        type_coercion: Literal["strict", "auto", "string"] = "strict",
    ) -> "Flow":
        """
        Join with another Flow.

        Args:
            other (Flow): The right-hand flow to join with.
            on (str, list[str], or None): Field(s) to join on when key names are the same.
            left_on (str, list[str], or None): Left-side field(s) to join on.
            right_on (str, list[str], or None): Right-side field(s) to join on.
            how (str): Type of join - 'left', 'right', 'outer', 'inner', or 'anti'.
            lsuffix (str): Suffix for conflicting left-side keys.
            rsuffix (str): Suffix for conflicting right-side keys.
            type_coercion (str): How to handle type differences in join keys:
                - 'strict': Exact type matching (default, preserves current behavior)
                - 'auto': Smart coercion (1 matches '1' matches 1.0)
                - 'string': Convert all join keys to strings for comparison

        Returns:
            Flow: A new Flow representing the join.
        """
        # Validation logic
        if on is not None and (left_on is not None or right_on is not None):
            raise ValueError(
                "Cannot specify both 'on' and 'left_on'/'right_on' parameters"
            )

        if on is None and (left_on is None or right_on is None):
            raise ValueError(
                "Either 'on' must be provided, or both 'left_on' and 'right_on' must be provided"
            )

        if left_on is not None and right_on is not None:
            # Normalize to lists for comparison
            left_keys = [left_on] if isinstance(left_on, str) else left_on
            right_keys = [right_on] if isinstance(right_on, str) else right_on
            if len(left_keys) != len(right_keys):
                raise ValueError(
                    "'left_on' and 'right_on' must have the same number of keys"
                )

        how = cast(Literal["left", "right", "outer", "inner", "anti"], how.lower())
        if how not in {"left", "right", "outer", "inner", "anti"}:
            raise ValueError(
                f"Unsupported join type: {how}. Must be one of 'left', 'right', 'outer', 'inner', 'anti'"
            )

        # Prepare step parameters
        step_params = {
            "op": "join",
            "right_plan": other.plan,
            "how": how,
            "lsuffix": lsuffix,
            "rsuffix": rsuffix,
            "type_coercion": type_coercion,
        }

        if on is not None:
            step_params["on"] = [on] if isinstance(on, str) else on
        else:
            step_params["left_on"] = [left_on] if isinstance(left_on, str) else left_on
            step_params["right_on"] = (
                [right_on] if isinstance(right_on, str) else right_on
            )

        return self._next(step_params)

    def split_array(self, field: str, into: list[str]) -> "Flow":
        """
        Split an array field into separate fields by index.

        Args:
            field (str): The field containing a list/array.
            into (list[str]): The names of the output fields.

        Returns:
            Flow: A new Flow with the array split into separate fields.
        """
        return self._next(
            {
                "op": "split_array",
                "field": field,
                "into": into,
            }
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
        return self._next(
            {
                "op": "pivot",
                "index": [index] if isinstance(index, str) else index,
                "columns": columns,
                "values": values,
            }
        )

    def cache(self) -> "Flow":
        """
        Cache the records in memory.

        Returns:
            Flow: A new Flow with the records cached.
        """
        records = list(self.collect())
        return self._next({"op": "from_materialized", "records": records})

    def explain(self, optimize: Optional[bool] = None, compare: bool = False):
        """
        Print a readable version of the plan.

        Args:
            optimize (bool): Whether to show the optimized plan (default True).
            compare (bool): If True, show both pre- and post-optimization plans.
        """
        effective_opt = self.optimize if optimize is None else optimize
        raw = self.plan

        if effective_opt or compare:
            optimizer = FlowOptimizer(raw)
            opt_plan = optimizer.optimize()
            validated_plan = optimizer._validate_rolling_has_sort(opt_plan)
        else:
            validated_plan = None

        explain_plan(raw, optimized_plan=validated_plan, compare=compare)

    def collect(
        self,
        optimize: Optional[bool] = None,
        progress: Optional[Literal["output", "input"]] = None,
        total_records: Optional[int] = None,
    ) -> list:
        """
        Collect all records from the flow.

        Args:
            optimize (bool): Whether to optimize the plan before execution.
            progress (Optional[Literal["output", "input"]]): Whether to show progress.
            total_records (Optional[int]): Total number of rows to expect.

        Returns:
            list: A list of records.
        """

        # figure out whether to optimize
        effective_optimize = self.optimize if optimize is None else optimize
        plan = FlowOptimizer(self.plan).optimize() if effective_optimize else self.plan

        # === INPUT‐PROGRESS MODE ===
        if progress == "input":
            # unwrap the very first step
            source_step, *rest = plan
            # dispatch it & wrap in tqdm
            raw_iter = FlowExecutor([source_step]).execute()
            wrapped = tqdm(
                raw_iter,
                total=total_records,
                desc="input rows",
                unit="it",
            )
            # build a new plan: from_materialized over our wrapped iterator
            wrapped_plan = [
                {"op": "from_materialized", "records": wrapped},
            ] + rest
            return list(FlowExecutor(wrapped_plan).execute())

        # === OUTPUT‐PROGRESS MODE ===
        if progress == "output":
            gen = FlowExecutor(plan).execute()
            wrapped = tqdm(gen, total=total_records, desc="output rows", unit="it")
            return list(wrapped)

        # === NO PROGRESS BAR ===
        return list(FlowExecutor(plan).execute())

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

    def with_schema(
        self,
        schema: dict[str, Union[type, Callable[[Any], Any]]],
        strict: bool = False,
        drop_extra: bool = False,
    ) -> "Flow":
        """
        Cast fields to specified types/functions and optionally validate or prune fields.

        Args:
            schema (dict): Mapping of field names (dot paths allowed) to types or casting functions.
            strict (bool): If True, raise an error on cast failure. Otherwise, fallback to original value.
            drop_extra (bool): If True, only retain fields explicitly listed in the schema.

        Returns:
            Flow: A new Flow with schema enforcement applied.
        """

        def cast_and_set(record):
            rec = record.copy()
            for path, func in schema.items():
                value = get_field(rec, path)
                try:
                    casted = func(value)
                except Exception:
                    if strict:
                        raise ValueError(f"Failed to cast field '{path}' to {func}")
                    casted = value
                set_path(rec, path, casted)
            if drop_extra:
                new_rec = {}
                for path in schema.keys():
                    val = get_field(rec, path)
                    set_path(new_rec, path, val)
                return new_rec
            return rec

        return self.map(cast_and_set)

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
        return self._next({"op": "sample_fraction", "p": p, "seed": seed})

    def sample_n(self, n: int, seed: Optional[int] = None) -> "Flow":
        """
        Lazily sample n records using reservoir sampling.

        Args:
            n (int): Number of records to sample.
            seed (int, optional): Random seed.

        Returns:
            Flow: A new Flow with sampling applied.
        """
        return self._next({"op": "sample_n", "n": n, "seed": seed})

    def map(self, func: Callable[[dict], dict]) -> "Flow":
        """
        Apply a function to each record. Should return a full record. If the function returns None,
        the record is dropped.

        Args:
            func (Callable[[dict], dict]): A function that takes a record and returns a modified one.

        Returns:
            Flow: A new Flow with the transformed records.
        """
        return self._next({"op": "map", "func": func})

    def pipe(self, func: Callable[["Flow"], "Flow"]) -> "Flow":
        """
        Lazily apply a function to this Flow and return the resulting Flow.
        The function will be executed at collect-time, not immediately.

        The function should return a new Flow, typically using this one as input.
        """
        return self._next({"op": "pipe", "func": func})

    def query(self, expr: str):
        """
        Filter rows using query string

        Args:
            expr (str): Query string

        Returns:
            Flow: A new Flow with the filtered records.

        Examples:
            # Basic comparisons
            flow.query("age > 30 and name == 'Phil Foden'")

            # Using variables with @
            player = "Mohamed Salah"
            flow.query("type.name == 'Shot' and player.name == @player")

            # Date filtering
            flow.query("date > datetime(2024, 1, 1)")
            cutoff_date = datetime(2024, 6, 15)
            flow.query("match_date >= @cutoff_date")

            # String operations
            flow.query("name.contains('son') and status == 'active'")

            # Regular expression matching
            flow.query("name.regex('^[A-Z][a-z]+$')")  # Names starting with capital letter
            flow.query("name.match('\\d{4}', 0)")  # Contains 4 digits in a row
        """
        current_frame = inspect.currentframe()
        frame = current_frame.f_back if current_frame else None
        local_vars = frame.f_locals if frame else {}

        predicate = parse_query_expr(expr, local_vars=local_vars)
        return self.filter(predicate)

    def plot_plan(self, compare: bool = False):
        """
        Visualize the flow plan.

        Args:
            compare (bool):
                - True: show two subplots (raw vs. optimized).
                - False: show a single subplot. If this Flow was constructed
                  with optimize=True, show the optimized plan; otherwise the raw.
        """
        plot_flow_plan(
            self.plan,
            optimize=self.optimize,
            compare=compare,
        )

    def profile(
        self,
        optimize: bool | None = None,
        fmt: Literal["table", "records"] = "table",
    ):
        """
        Profile each step in the plan. Returns a report of
        (step_index, op_name, time_s, rows_emitted).

        Args:
          optimize: whether to optimize the plan (default = self.optimize)
          fmt: 'table' to print a table, 'records' to return the raw list of dicts.
        """
        # 1. pick effective plan
        effective_opt = self.optimize if optimize is None else optimize
        plan = FlowOptimizer(self.plan).optimize() if effective_opt else self.plan

        stats = []
        # 2. for each prefix of the plan, execute just that prefix
        for idx in range(1, len(plan) + 1):
            prefix = plan[:idx]
            op = prefix[-1]["op"]

            start = perf_counter()
            rows = 0
            # fully materialize prefix so we can count rows
            for _ in FlowExecutor(prefix).execute():
                rows += 1
            elapsed = perf_counter() - start

            stats.append(
                {
                    "step": idx,
                    "op": op,
                    "time_s": round(elapsed, 3),
                    "rows": rows,
                }
            )

        if fmt == "json":
            return stats

        # pretty‐print via tabulate
        print(
            tabulate(
                [[s["step"], s["op"], s["time_s"], s["rows"]] for s in stats],
                headers=["#", "op", "sec", "rows"],
                tablefmt="github",
            )
        )

    def get_url(self) -> str:
        """
        For a Flow created from an API source (e.g. Opta),
        construct and return the URL that will be called.
        This is a debugging utility and may not work for all flows.
        """
        if not self.plan:
            raise ValueError("Cannot get URL for an empty plan.")

        step = self.plan[0]
        op = step.get("op")

        if op == "from_opta":
            from .steps.opta.endpoints import OptaEndpointBuilder

            source = step.get("source")
            args = step.get("args", {})
            creds = args.get("creds", {})

            auth_key = creds.get("auth_key", "YOUR_AUTH_KEY")
            rt_mode = creds.get("rt_mode", "b")

            endpoint_builder = OptaEndpointBuilder(
                base_url=step["base_url"],
                asset_type=step["asset_type"],
                auth_key=auth_key,
            )

            url, params = endpoint_builder.build_request_details(source, args)
            params["_rt"] = rt_mode

            query_string = urllib.parse.urlencode(params)
            return f"{url}?{query_string}"

        elif op == "from_statsbomb":
            raise NotImplementedError("get_url() is not yet implemented for Statsbomb.")
        else:
            raise ValueError(f"get_url() is not supported for flows with op '{op}'.")


from .contrib.opta import opta as opta_module
from .contrib.statsbomb import statsbomb as statsbomb_module

Flow.statsbomb = statsbomb_module  # type: ignore[attr-defined]
Flow.opta = opta_module  # type: ignore[attr-defined]
