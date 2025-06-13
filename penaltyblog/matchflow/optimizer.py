import copy


class FlowOptimizer:
    """
    Optimizer for a flow plan.

    Performs conservative optimizations: it fuses simple operations,
    pushes down filters, limits, and select/drop operations only when
    provably safe, and eliminates redundant steps.
    """

    MAX_PASSES = 5

    def __init__(self, plan):
        self.plan = plan

    FIELD_USAGE_HANDLERS = {
        "select": lambda step: set(step.get("fields", [])),
        "drop": lambda step: set(step.get("keys", [])),
        "dropna": lambda step: set(step.get("fields") or []),
        "rename": lambda step: set(step.get("mapping", {}).keys())
        | set(step.get("mapping", {}).values()),
        "assign": lambda step: set(step.get("fields", {}).keys()),
        "cast": lambda step: set(step.get("casts", {}).keys()),
        "filter": lambda step: set(),
        "join": lambda step: set(step.get("on", [])),
        "sort": lambda step: set(step.get("keys", [])),
        "group_by": lambda step: set(step.get("keys", [])),
        "group_rolling_summary": lambda step: (
            ({step.get("time_field")} if step.get("time_field") else set())
            | {
                agg[1]
                for agg in step.get("aggregators", {}).values()
                if isinstance(agg, tuple)
            }
        ),
    }

    def _is_order_sensitive(self, op: str) -> bool:
        return op in {
            "sort",
            "group_summary",
            "group_cumulative",
            "group_rolling_summary",
            "pivot",
        }

    def _blocks_filter_pushdown(self, op: str) -> bool:
        return self._is_order_sensitive(op) or op in {
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
            "summary",
            "limit",
            "explode",
        }

    def optimize(self):
        plan = copy.deepcopy(self.plan)
        for _ in range(self.MAX_PASSES):
            new_plan = self._optimize_once(plan)
            if new_plan == plan:
                break
            plan = new_plan
        plan = self._validate_rolling_has_sort(plan)
        return plan

    def _optimize_once(self, plan):
        plan = self._fuse_map_assign_filter(plan)
        plan = self._pushdown_filters(plan)
        plan = self._pushdown_limit(plan)
        plan = self._pushdown_select_drop(plan)
        plan = self._eliminate_redundant_steps(plan)
        return plan

    def _get_fields_used(self, step):
        return self.FIELD_USAGE_HANDLERS.get(step.get("op"), lambda s: set())(step)

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

        for i, step in enumerate(plan):
            op = step.get("op")

            if op in {"select", "drop"}:
                if op == "select":
                    fields = set(step.get("fields", []))
                    cond = required_fields_list[i].issubset(fields)
                else:
                    fields = set(step.get("keys", []))
                    cond = required_fields_list[i].isdisjoint(fields)

                if cond:
                    if self._is_already_early_enough(plan, i):
                        new_plan.append(step)
                    else:
                        moved = dict(step)
                        moved["_original_index"] = i
                        pending_push.append(moved)
                    continue

            if self._is_order_sensitive(op):
                new_plan.extend(pending_push)
                pending_push = []

            new_plan.append(step)

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

        for idx, step in enumerate(plan):
            op = step.get("op")

            if op == "filter":
                pending.append(step.copy())
                pending_orig_idx.append(idx)
                continue

            if self._blocks_filter_pushdown(op) and pending:
                for filt, orig in zip(pending, pending_orig_idx):
                    tagged = filt.copy()
                    if len(new_plan) < orig:
                        tagged.setdefault("_notes", []).append(
                            "pushed down from later step"
                        )
                    new_plan.append(tagged)
                pending.clear()
                pending_orig_idx.clear()

            if op and op.startswith("from_") and pending:
                for filt, orig in zip(pending, pending_orig_idx):
                    tagged = filt.copy()
                    if len(new_plan) + 1 < orig:
                        tagged.setdefault("_notes", []).append(
                            "pushed down from later step"
                        )
                    new_plan.append(tagged)
                pending.clear()
                pending_orig_idx.clear()

            new_plan.append(step.copy())

        for filt in pending:
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

    def _validate_rolling_has_sort(self, plan):
        validated_plan = []
        last_group_by_idx = -1

        for idx, step in enumerate(plan):
            op = step.get("op")

            if op == "group_by":
                last_group_by_idx = idx

            if op == "group_rolling_summary":
                sorted_before = any(
                    p.get("op") == "sort" for p in plan[last_group_by_idx + 1 : idx]
                )
                if not sorted_before:
                    step = dict(step)
                    step.setdefault("_notes", []).append(
                        "⚠️  group_rolling_summary used without prior sort — results may be unstable"
                    )
            validated_plan.append(step)
        return validated_plan
