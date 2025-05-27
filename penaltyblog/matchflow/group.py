from typing import Any, Callable, Dict, List, Optional, Union

from .aggs_registry import resolve_aggregator
from .executor import FlowExecutor
from .flow import Flow
from .optimizer import FlowOptimizer

PlanNode = Dict[str, Any]


class FlowGroup:
    def __init__(self, plan: List[PlanNode], optimize: bool = False):
        self.plan = plan
        self.optimize = optimize

    def _get_plan(self) -> List[PlanNode]:
        if self.optimize:
            return FlowOptimizer(self.plan).optimize()
        return self.plan

    def __iter__(self):
        from .executor import FlowExecutor

        for group in FlowExecutor(self._get_plan()).execute():
            key = group["__group_key__"]
            records = group["__group_records__"]
            yield key if len(key) > 1 else key[0], records

    def summary(self, aggregators: Union[Callable, dict[str, Any]]) -> "FlowGroup":
        """
        Supports:
        - Callable (e.g. lambda rows: {...})
        - Dict of {alias: callable}
        - Dict of {alias: "name"} or (name/callable, field)

        Args:
            aggregators (Union[Callable, dict[str, Any]]): The aggregators to apply.

        Returns:
            FlowGroup: A new FlowGroup with the summary applied.
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
        for step in reversed(self._get_plan()):
            if step["op"] == "group_by":
                group_keys = step["keys"]
                break

        return Flow(
            self._get_plan()
            + [{"op": "group_summary", "agg": agg_func, "group_keys": group_keys}]
        )

    def sort_by(self, *keys: str, ascending: bool = True):
        return FlowGroup(
            self._get_plan()
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
        """
        Args:
            field (str): The field to cumulative.
            alias (Optional[str], optional): The alias for the cumulative field. Defaults to None.

        Returns:
            Flow: A new Flow with the cumulative applied.
        """
        return Flow(
            self._get_plan()
            + [
                {
                    "op": "group_cumulative",
                    "field": field,
                    "alias": alias or f"cumulative_{field}",
                }
            ]
        )

    def select(self, *fields: str):
        """
        Args:
            *fields (str): The fields to select.

        Returns:
            FlowGroup: A new FlowGroup with the select applied.
        """
        return FlowGroup(self._get_plan() + [{"op": "select", "fields": fields}])

    def to_flow(self):
        """
        Convert the FlowGroup to a Flow.

        Returns:
            Flow: A new Flow with the group applied.
        """
        return Flow(self._get_plan())

    def collect(self):
        """
        Collect the FlowGroup into a list of records.

        Returns:
            list: The collected records.
        """
        return list(FlowExecutor(self._get_plan()).execute())

    def explain(self):
        """
        Print the plan for the FlowGroup.
        """
        for step in self._get_plan():
            print(f"• {step['op']}: { {k: v for k, v in step.items() if k != 'op'} }")
