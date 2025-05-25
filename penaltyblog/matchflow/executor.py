from typing import Any, Dict, List

from .steps import group, source, transform

PlanNode = Dict[str, Any]

FUSABLE_OPS = {"map", "filter", "assign"}


def is_materializing_op(op_name: str) -> bool:
    """
    Returns True if the operation is expected to materialize or buffer records
    in memory (i.e., requires full dataset to proceed).
    """
    return op_name in {
        "sort",
        "limit",  # materializes to truncate
        "dropna",  # inspects multiple keys
        "distinct",  # needs to track seen keys
        "cache",  # explicitly buffers all data
        "summary",  # aggregates entire dataset
        "group_by",  # collects into group buckets
        "group_summary",  # aggregates grouped records
        "group_cumulative",  # runs cumulative logic per group
        "pivot",  # restructures data after full pass
        "schema",  # implicit materialization via limit+inspect
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

            # --- FUSION of adjacent map/assign/filter ---
            if op in FUSABLE_OPS:
                fused_steps = []
                while i < len(self.plan) and self.plan[i]["op"] in FUSABLE_OPS:
                    fused_steps.append(self.plan[i])
                    i += 1
                gen = self._apply_fused_transform(gen, fused_steps)
                continue

            # --- Non-fusible operators ---
            if op == "select":
                gen = transform.apply_select(gen, step)
            elif op == "rename":
                gen = transform.apply_rename(gen, step)
            elif op == "group_by":
                gen = group.apply_group_by(gen, step)
            elif op == "group_summary":
                gen = group.apply_group_summary(gen, step)
            elif op == "group_cumulative":
                gen = group.apply_group_cumulative(gen, step)
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
            else:
                raise ValueError(f"Unknown plan op: {op}")

            i += 1

        return gen

    def _apply_fused_transform(self, records, steps):
        def fused():
            for r in records:
                for step in steps:
                    op = step["op"]
                    if op == "map":
                        r = step["func"](r)
                        if r is None:
                            break
                        if not isinstance(r, dict):
                            raise TypeError("map function must return a dict")
                    elif op == "assign":
                        new_r = dict(r)
                        for k, func in step["fields"].items():
                            new_r[k] = func(r)
                        r = new_r
                    elif op == "filter":
                        if not step["predicate"](r):
                            r = None
                            break
                if r is not None:
                    yield r

        return fused()
