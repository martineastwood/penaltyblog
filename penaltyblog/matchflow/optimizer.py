class FlowOptimizer:
    def __init__(self, plan):
        self.plan = plan

    def optimize(self):
        plan = self.plan
        plan = self._fuse_map_assign_filter(plan)
        plan = self._pushdown_filters(plan)
        plan = self._pushdown_limit(plan)
        plan = self._eliminate_redundant_steps(plan)
        return plan

    def _eliminate_redundant_steps(self, plan):
        new_plan = []
        i = 0
        while i < len(plan):
            step = plan[i]
            if i > 0 and step["op"] == plan[i - 1]["op"]:
                if step["op"] in {"drop", "dropna"} and step == plan[i - 1]:
                    # Remove exact duplicates silently
                    i += 1
                    continue
            new_step = dict(step)
            new_plan.append(new_step)
            i += 1
        return new_plan

    def _pushdown_filters(self, plan):
        new_plan = []
        filters = []

        for step in plan:
            if step["op"] == "filter":
                filters.append(dict(step))  # clone for safety
            elif step["op"].startswith("from_") and filters:
                new_plan.append(dict(step))  # clone source
                for f in filters:
                    f.setdefault("_notes", []).append("pushed down from later step")
                new_plan.extend(filters)
                filters = []
            else:
                new_plan.append(dict(step))

        for f in filters:
            f.setdefault("_notes", []).append("could not push further")
        new_plan.extend(filters)
        return new_plan

    def _pushdown_limit(self, plan):
        limit_step = None
        new_plan = []
        limit_was_moved = False

        for step in reversed(plan):
            if step["op"] == "limit":
                limit_step = dict(step)
            elif limit_step and step["op"] in {
                "map",
                "assign",
                "select",
                "drop",
                "rename",
            }:
                limit_was_moved = True
                new_plan.insert(0, dict(step))
            else:
                if limit_step:
                    if limit_was_moved:
                        limit_step.setdefault("_notes", []).append(
                            "pushed down from later step"
                        )
                    new_plan.insert(0, limit_step)
                    limit_step = None
                    limit_was_moved = False
                new_plan.insert(0, dict(step))

        if limit_step:
            if limit_was_moved:
                limit_step.setdefault("_notes", []).append(
                    "pushed down to earliest safe point"
                )
            new_plan.insert(0, limit_step)

        return new_plan

    def _fuse_map_assign_filter(self, plan):
        new_plan = []
        i = 0
        fusables = {"map", "assign", "filter"}

        while i < len(plan):
            if plan[i]["op"] in fusables:
                fused_ops = []
                j = i
                while j < len(plan) and plan[j]["op"] in fusables:
                    fused_ops.append(plan[j])
                    j += 1

                if len(fused_ops) > 1:
                    fused_op = {
                        "op": "fused",
                        "ops": [step["op"] for step in fused_ops],
                        "steps": [dict(step) for step in fused_ops],
                        "_notes": [
                            f"fused: {', '.join(step['op'] for step in fused_ops)}"
                        ],
                    }
                    new_plan.append(fused_op)
                else:
                    new_plan.append(dict(fused_ops[0]))
                i = j
            else:
                new_plan.append(dict(plan[i]))
                i += 1

        return new_plan
