from typing import Any, Dict, List

from .steps import group, source, transform

PlanNode = Dict[str, Any]


def is_materializing_op(op_name: str) -> bool:
    """
    Returns True if the operation is expected to materialize or buffer records
    in memory (i.e., requires full dataset to proceed).
    """
    return op_name in {
        "sort",
        "limit",
        "dropna",
        "distinct",
        "cache",
        "summary",
        "group_by",
        "group_summary",
        "group_cumulative",
        "pivot",
        "schema",
    }


class FlowExecutor:
    def __init__(self, plan: List[PlanNode]):
        self.plan = plan

    def execute(self):
        from .flow import Flow

        gen = source.dispatch(self.plan[0])
        i = 1
        while i < len(self.plan):
            step = self.plan[i]
            op = step["op"]

            if op == "map":
                gen = transform.apply_map(gen, step)
            elif op == "assign":
                gen = transform.apply_assign(gen, step)
            elif op == "filter":
                gen = transform.apply_filter(gen, step)
            elif op == "select":
                gen = transform.apply_select(gen, step)
            elif op == "rename":
                gen = transform.apply_rename(gen, step)
            elif op == "group_by":
                gen = group.apply_group_by(gen, step)
            elif op == "group_summary":
                gen = group.apply_group_summary(gen, step)
            elif op == "group_cumulative":
                gen = group.apply_group_cumulative(gen, step)
            elif op == "group_rolling_summary":
                gen = group.apply_group_rolling_summary(gen, step)
            elif op == "group_time_bucket":
                gen = group.apply_group_time_bucket(gen, step)
            elif op == "summary":
                gen = transform.apply_summary(gen, step)
            elif op == "sort":
                gen = transform.apply_sort(gen, step)
            elif op == "limit":
                gen = transform.apply_limit(gen, step)
            elif op == "drop":
                gen = transform.apply_drop(gen, step)
            elif op == "dropna":
                gen = transform.apply_dropna(gen, step)
            elif op == "explode":
                gen = transform.apply_explode(gen, step)
            elif op == "flatten":
                gen = transform.apply_flatten(gen, step)
            elif op == "distinct":
                gen = transform.apply_distinct(gen, step)
            elif op == "join":
                gen = transform.apply_join(gen, step)
            elif op == "split_array":
                gen = transform.apply_split_array(gen, step)
            elif op == "pivot":
                gen = transform.apply_pivot(gen, step)
            elif op == "sample_fraction":
                gen = transform.apply_sample_fraction(gen, step)
            elif op == "sample_n":
                gen = transform.apply_sample_n(gen, step)
            elif op == "pipe":
                func = step["func"]
                flow = func(Flow(self.plan[:i]))
                return FlowExecutor(flow.plan).execute()
            elif op == "fused":
                gen = transform.apply_fused(gen, step)
            elif op == "from_materialized":
                gen = iter(step["records"])
            elif op == "from_concat":
                gen = source.from_concat(step)
            else:
                raise ValueError(f"Unknown plan op: {op}")

            i += 1

        return gen
