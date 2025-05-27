class FlowOptimizer:
    """
    Optimizer for a flow plan.
    """

    def __init__(self, plan):
        """
        Initialize the optimizer with a plan.

        Args:
            plan (list[dict]): The plan to optimize.
        """
        self.plan = plan

    def optimize(self):
        """
        Optimize the plan.

        Returns:
            list[dict]: The optimized plan.
        """
        plan = self.plan
        plan = self._fuse_map_assign_filter(plan)
        plan = self._pushdown_filters(plan)
        plan = self._pushdown_limit(plan)
        plan = self._pushdown_select_drop(plan)
        plan = self._eliminate_redundant_steps(plan)
        return plan

    def _get_fields_used(self, step: dict) -> set[str]:
        """
        Get the fields used by a step.

        Args:
            step (dict): The step to analyze.

        Returns:
            set[str]: The fields used by the step.
        """
        op = step["op"]
        if op in {"select", "drop", "dropna"}:
            return set(step.get("fields") or [])
        elif op == "assign":
            # Too dynamic to analyze safely
            return set()
        elif op == "cast":
            return set(step["casts"].keys()) if "casts" in step else set()
        elif op == "filter":
            # Could try to introspect later, for now be conservative
            return set()
        elif op == "summary":
            return set()
        elif op == "rename":
            return set(step["mapping"].keys())
        elif op == "join":
            return set(step["on"])
        elif op == "sort":
            return set(step["keys"])
        elif op == "group_by":
            return set(step["keys"])
        else:
            return set()

    def _compute_required_fields(self, plan: list[dict]) -> list[set[str]]:
        required = set()
        required_by_step = []

        for step in reversed(plan):
            required_by_step.append(required.copy())
            required |= self._get_fields_used(step)

        return list(reversed(required_by_step))

    def _is_already_early_enough(self, plan: list[dict], index: int) -> bool:
        """
        Returns True if the step at `index` is already directly after a source op.
        """
        return index > 0 and plan[index - 1]["op"].startswith("from_")

    def _pushdown_select_drop(self, plan: list[dict]) -> list[dict]:
        """
        Pushdown select and drop operations to earlier points in the plan.

        Args:
            plan (list[dict]): The plan to optimize.

        Returns:
            list[dict]: The optimized plan.
        """
        required_fields_list = self._compute_required_fields(plan)
        new_plan = []
        pending_push = []

        # These ops block pushdown beyond them
        blocked_ops = {"assign", "map", "pipe", "summary", "join", "pivot", "explode"}

        for i, step in enumerate(plan):
            op = step["op"]

            # Safe pushdown of select
            if op == "select":
                selected = set(step.get("fields") or [])
                required_after = required_fields_list[i]

                if required_after.issubset(selected):
                    if self._is_already_early_enough(plan, i):
                        new_plan.append(step)  # already well-placed
                    else:
                        step = dict(step)
                        step.setdefault("_notes", []).append(
                            "pushed down to earlier point"
                        )
                        pending_push.append(step)
                    continue

            # Safe pushdown of drop
            elif op == "drop":
                dropped = set(step.get("fields") or [])
                required_after = required_fields_list[i]

                if required_after.isdisjoint(dropped):
                    if self._is_already_early_enough(plan, i):
                        new_plan.append(step)
                    else:
                        step = dict(step)
                        step.setdefault("_notes", []).append(
                            "pushed down to earlier point"
                        )
                        pending_push.append(step)
                    continue

            # Stop pushdown at any dynamic/op-opaque steps
            if op in blocked_ops:
                new_plan.extend(pending_push)
                pending_push = []

            new_plan.append(step)

            if step["op"].startswith("from_") and pending_push:
                new_plan.extend(pending_push)
                pending_push = []

        if pending_push:
            new_plan.extend(pending_push)

        return new_plan

    def _eliminate_redundant_steps(self, plan):
        """
        Eliminate redundant steps from the plan.

        Args:
            plan (list[dict]): The plan to optimize.

        Returns:
            list[dict]: The optimized plan.
        """
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
        """
        Pushdown filters to earlier points in the plan.

        Args:
            plan (list[dict]): The plan to optimize.

        Returns:
            list[dict]: The optimized plan.
        """
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
        """
        Pushdown limit to earlier points in the plan.

        Args:
            plan (list[dict]): The plan to optimize.

        Returns:
            list[dict]: The optimized plan.
        """
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
        """
        Fuse map, assign, and filter operations.

        Args:
            plan (list[dict]): The plan to optimize.

        Returns:
            list[dict]: The optimized plan.
        """
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
