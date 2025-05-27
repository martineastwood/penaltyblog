from typing import Any, Callable, Dict, List, Optional, Union

from .aggs_registry import resolve_aggregator
from .executor import FlowExecutor
from .flow import Flow
from .helpers import explain_plan
from .optimizer import FlowOptimizer
from .plotting import plot_flow_plan

PlanNode = Dict[str, Any]


class FlowGroup:
    """
    A FlowGroup is a collection of records grouped by one or more keys.
    """

    def __init__(self, plan: List[PlanNode], optimize: bool = False):
        """
        Args:
            plan (List[PlanNode]): The plan to execute.
            optimize (bool, optional): Whether to optimize the plan. Defaults to False.
        """
        self.plan = plan
        self.optimize = optimize

    def __iter__(self):
        """
        Iterate over the groups.
        """
        plan_to_execute = (
            FlowOptimizer(self.plan).optimize() if self.optimize else self.plan
        )
        for group in FlowExecutor(plan_to_execute).execute():
            key = group.get("__group_key__")
            records = group.get("__group_records__")
            yield (key if isinstance(key, tuple) and len(key) == 1 else key), records

    def summary(self, aggregators: Union[Callable, Dict[str, Any]]) -> Flow:
        """
        Apply group-summary without eagerly optimizing previous steps.

        Args:
            aggregators (Union[Callable, Dict[str, Any]]): The aggregators to apply.

        Returns:
            Flow: A new Flow with the group-summary applied.
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

        # Find original group_by keys
        group_keys = None
        for step in reversed(self.plan):
            if step.get("op") == "group_by":
                group_keys = step.get("keys")
                break

        # Build new Flow with raw plan + summary, carry optimize flag
        return Flow(
            plan=self.plan
            + [
                {
                    "op": "group_summary",
                    "agg": agg_func,
                    "group_keys": group_keys,
                }
            ],
            optimize=self.optimize,
        )

    def sort_by(self, *keys: str, ascending: bool = True) -> "FlowGroup":
        """
        Sort the groups by one or more fields.

        Args:
            *keys (str): Field names to sort by.
            ascending (bool or list[bool], optional): Sort order(s). Either a single bool
                applied to all keys or a list of bools (one per key).

        Returns:
            FlowGroup: A new FlowGroup with sorted groups.
        """
        return FlowGroup(
            plan=self.plan
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
            ],
            optimize=self.optimize,
        )

    def cumulative(self, field: str, alias: Optional[str] = None) -> Flow:
        """
        Apply group-cumulative without eagerly optimizing previous steps.

        Args:
            field (str): The field to cumulative.
            alias (Optional[str], optional): The alias for the cumulative field. Defaults to None.

        Returns:
            Flow: A new Flow with the group-cumulative applied.
        """
        return Flow(
            plan=self.plan
            + [
                {
                    "op": "group_cumulative",
                    "field": field,
                    "alias": alias or f"cumulative_{field}",
                }
            ],
            optimize=self.optimize,
        )

    def select(self, *fields: str) -> "FlowGroup":
        """
        Select specific fields from each record.

        Args:
            *fields (str): The fields to select.

        Returns:
            FlowGroup: A new FlowGroup with selected fields.
        """
        return FlowGroup(
            plan=self.plan + [{"op": "select", "fields": list(fields)}],
            optimize=self.optimize,
        )

    def to_flow(self) -> Flow:
        """
        Convert the FlowGroup to a Flow.

        Returns:
            Flow: A new Flow with the same plan and optimize flag.
        """
        return Flow(plan=self.plan, optimize=self.optimize)

    def collect(self) -> List[Dict[str, Any]]:
        """
        Collect the FlowGroup into a list of records.

        Returns:
            List[Dict[str, Any]]: The collected records.
        """
        return list(
            FlowExecutor(Flow(self.plan, optimize=self.optimize).plan).execute()
        )

    def plot_plan(self, compare: bool = False):
        """
        Visualize the flow group plan.

        Args:
            compare (bool):
                - True: show two subplots (raw vs. optimized).
                - False: show a single subplot. If this FlowGroup was constructed
                  with optimize=True, show the optimized plan; otherwise the raw.
        """
        plot_flow_plan(
            self.plan,
            optimize=self.optimize,
            compare=compare,
            title_prefix="Group: ",
        )

    def explain(self, optimize: Optional[bool] = None, compare: bool = False):
        """
        Explain the plan.

        Returns:
            None
        """
        effective_opt = self.optimize if optimize is None else optimize
        raw = self.plan
        opt_plan = FlowOptimizer(raw).optimize() if effective_opt or compare else None
        explain_plan(raw, optimized_plan=opt_plan, compare=compare)
