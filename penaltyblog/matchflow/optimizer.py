class FlowOptimizer:
    """
    Optimizer for a flow plan.

    Performs conservative optimizations: it fuses simple operations,
    pushes down filters, limits, and select/drop operations only when
    provably safe, and eliminates redundant steps.
    """

    MAX_PASSES = 5

    def __init__(self, plan):
        """
        Initialize the optimizer with a plan.

        Args:
            plan (list[dict]): The plan to optimize.
        """
        self.plan = plan

    def optimize(self):
        plan = self.plan
        for _ in range(self.MAX_PASSES):
            new_plan = self._optimize_once(plan)
            if new_plan == plan:
                break
            plan = new_plan
        return plan

    def _optimize_once(self, plan):
        plan = self._fuse_map_assign_filter(plan)
        plan = self._pushdown_filters(plan)
        plan = self._pushdown_limit(plan)
        plan = self._pushdown_select_drop(plan)
        plan = self._eliminate_redundant_steps(plan)
        return plan

    def _get_fields_used(self, step):
        op = step.get("op")
        if op == "select":
            return set(step.get("fields", []))
        elif op == "drop":
            return set(step.get("keys", []))
        elif op == "dropna":
            return set(step.get("fields") or [])
        elif op == "rename":
            mapping = step.get("mapping", {})
            # Consider both old and new names as used
            return set(mapping.keys()) | set(mapping.values())
        elif op == "assign":
            # Fields created by assign must be preserved
            return set(step.get("fields", {}).keys())
        elif op == "cast":
            return set(step.get("casts", {}).keys())
        elif op == "filter":
            # Arbitrary predicate: block select/drop above filter instead
            return set()
        elif op == "join":
            return set(step.get("on", []))
        elif op == "sort":
            return set(step.get("keys", []))
        elif op == "group_by":
            return set(step.get("keys", []))
        else:
            return set()

    def _compute_required_fields(self, plan):
        required = set()
        required_by_step = []
        for step in reversed(plan):
            required_by_step.append(required.copy())
            required |= self._get_fields_used(step)
        return list(reversed(required_by_step))

    def _is_already_early_enough(self, plan, index):
        return index > 0 and plan[index - 1].get("op", "").startswith("from_")

    def _pushdown_select_drop(self, plan):
        required_fields_list = self._compute_required_fields(plan)
        new_plan = []
        pending_push = []
        # Block select/drop pushdown across any of these ops
        blocked_ops = {
            "assign",
            "filter",
            "map",
            "pipe",
            "join",
            "pivot",
            "concat",
            "explode",
            "group_by",
            "summary",
            "group_summary",
            "fused",
        }

        for i, step in enumerate(plan):
            op = step.get("op")

            # Candidate for pushdown
            if op in {"select", "drop"}:
                if op == "select":
                    fields = set(step.get("fields", []))
                    cond = required_fields_list[i].issubset(fields)
                else:
                    fields = set(step.get("keys", []))
                    cond = required_fields_list[i].isdisjoint(fields)

                if cond:
                    # Already at earliest safe point
                    if self._is_already_early_enough(plan, i):
                        new_plan.append(step)
                    else:
                        moved = dict(step)
                        moved["_original_index"] = i
                        pending_push.append(moved)
                    continue

            # Flush pending when hitting a blocker
            if op in blocked_ops:
                new_plan.extend(pending_push)
                pending_push = []

            new_plan.append(step)

            # Flush immediately after any source
            if op and op.startswith("from_") and pending_push:
                new_plan.extend(pending_push)
                pending_push = []

        return self._annotate_moves(new_plan)

    def _annotate_moves(self, plan):
        result = []
        for idx, step in enumerate(plan):
            if "_original_index" in step:
                orig = step.pop("_original_index")
                if idx < orig:
                    note = (
                        "moved earlier in plan"
                        if orig - idx > 1
                        else "reordered (same logical position)"
                    )
                    step.setdefault("_notes", []).append(note)
            result.append(step)
        return result

    def _eliminate_redundant_steps(self, plan):
        new_plan = []
        prev_op = None
        for step in plan:
            op = step.get("op")
            if op in {"drop", "dropna"} and op == prev_op:
                continue
            new_plan.append(dict(step))
            prev_op = op
        return new_plan

    def _pushdown_filters(self, plan):
        new_plan = []
        pending = []
        pending_orig_idx = []
        # Block pushing filters across these ops
        blocking = {
            "select",
            "drop",
            "dropna",
            "rename",
            "flatten",
            "map",
            "assign",
            "pipe",
            "join",
            "group_by",
            "group_summary",
            "group_cumulative",
            "sort",
            "limit",
            "explode",
            "pivot",
            "summary",
        }

        for idx, step in enumerate(plan):
            op = step.get("op")

            # Stash filters
            if op in {"filter"}:
                pending.append(step.copy())
                pending_orig_idx.append(idx)
                continue

            # Flush before any blocking op
            if op in blocking and pending:
                for filt, orig in zip(pending, pending_orig_idx):
                    tagged = filt.copy()
                    new_idx = len(new_plan)
                    if new_idx < orig:
                        tagged.setdefault("_notes", []).append(
                            "pushed down from later step"
                        )
                    new_plan.append(tagged)
                pending = []
                pending_orig_idx = []

            # Flush immediately after source
            if op and op.startswith("from_") and pending:
                for filt, orig in zip(pending, pending_orig_idx):
                    tagged = filt.copy()
                    new_idx = len(new_plan) + 1
                    if new_idx < orig:
                        tagged.setdefault("_notes", []).append(
                            "pushed down from later step"
                        )
                    new_plan.append(tagged)
                pending = []
                pending_orig_idx = []

            new_plan.append(step.copy())

        # Append any remaining filters at end
        for filt, orig in zip(pending, pending_orig_idx):
            new_plan.append(filt)

        return new_plan

    def _pushdown_limit(self, plan):
        limit_step = None
        new_plan = []
        moved = False
        for step in reversed(plan):
            if step.get("op") == "limit":
                limit_step = dict(step)
            elif limit_step and step.get("op") in {
                "assign",
                "select",
                "drop",
                "rename",
            }:
                # Do not push limit past map, since map may drop records
                moved = True
                new_plan.insert(0, dict(step))
            else:
                if limit_step:
                    if moved:
                        limit_step.setdefault("_notes", []).append(
                            "pushed down from later step"
                        )
                    new_plan.insert(0, limit_step)
                    limit_step, moved = None, False
                new_plan.insert(0, dict(step))
        if limit_step:
            if moved:
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
            if plan[i].get("op") in fusables:
                j = i
                group = []
                while j < len(plan) and plan[j].get("op") in fusables:
                    group.append(plan[j])
                    j += 1
                if len(group) > 1:
                    fused = {
                        "op": "fused",
                        "ops": [s["op"] for s in group],
                        "steps": [dict(s) for s in group],
                        "_notes": [f"fused: {', '.join(s['op'] for s in group)}"],
                    }
                    new_plan.append(fused)
                else:
                    new_plan.append(dict(group[0]))
                i = j
            else:
                new_plan.append(dict(plan[i]))
                i += 1
        return new_plan
