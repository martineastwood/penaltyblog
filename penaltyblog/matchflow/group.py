from typing import Any, Callable, Dict, List, Optional, Union

from .aggs_registry import resolve_aggregator
from .executor import FlowExecutor
from .flow import Flow

PlanNode = Dict[str, Any]


class FlowGroup:
    def __init__(self, plan: List[PlanNode]):
        self.plan = plan

    def __iter__(self):
        from .executor import FlowExecutor

        for group in FlowExecutor(self.plan).execute():
            key = group["__group_key__"]
            records = group["__group_records__"]
            yield key if len(key) > 1 else key[0], records

    def summary(self, aggregators: Union[Callable, dict[str, Any]]):
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

        group_keys = None
        for step in reversed(self.plan):
            if step["op"] == "group_by":
                group_keys = step["keys"]
                break

        return Flow(
            self.plan
            + [{"op": "group_summary", "agg": agg_func, "group_keys": group_keys}]
        )

    def sort_by(self, *keys: str, ascending: bool = True):
        return FlowGroup(
            self.plan
            + [
                {
                    "op": "sort",
                    "keys": list(keys),
                    "ascending": (
                        [ascending] * len(keys)
                        if isinstance(ascending, bool)
                        else ascending
                    ),
                }
            ]
        )

    def cumulative(self, field: str, alias: Optional[str] = None):
        return Flow(
            self.plan
            + [
                {
                    "op": "group_cumulative",
                    "field": field,
                    "alias": alias or f"cumulative_{field}",
                }
            ]
        )

    def select(self, *fields: str):
        return FlowGroup(self.plan + [{"op": "select", "fields": fields}])

    def to_flow(self):
        return Flow(self.plan)

    def collect(self):
        return list(FlowExecutor(self.plan).execute())

    def explain(self):
        for step in self.plan:
            print(f"• {step['op']}: { {k: v for k, v in step.items() if k != 'op'} }")
