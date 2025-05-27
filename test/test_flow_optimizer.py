from penaltyblog.matchflow.flow import Flow
from penaltyblog.matchflow.optimizer import FlowOptimizer


def test_fuse_map_filter():
    plan = [
        {"op": "from_jsonl", "path": "data.jsonl"},
        {"op": "map", "func": lambda x: x},
        {"op": "filter", "predicate": lambda x: True},
    ]
    optimized = FlowOptimizer(plan).optimize()

    assert len(optimized) == 2
    assert optimized[1]["op"] == "fused"
    assert optimized[1]["ops"] == ["map", "filter"]


def test_single_filter_not_fused():
    plan = [
        {"op": "from_json", "path": "data.json"},
        {"op": "filter", "predicate": lambda x: True},
    ]
    optimized = FlowOptimizer(plan).optimize()

    assert len(optimized) == 2
    assert optimized[1]["op"] == "filter"


def test_non_fusible_ops_untouched():
    plan = [
        {"op": "from_jsonl", "path": "file.jsonl"},
        {"op": "select", "fields": ["a", "b"]},
    ]
    optimized = FlowOptimizer(plan).optimize()

    assert optimized == plan


def test_multiple_assigns_and_filter_fused():
    plan = [
        {"op": "from_jsonl", "path": "file.jsonl"},
        {"op": "assign", "fields": {"x": lambda r: 1}},
        {"op": "assign", "fields": {"y": lambda r: 2}},
        {"op": "filter", "predicate": lambda r: True},
    ]
    optimized = FlowOptimizer(plan).optimize()

    assert len(optimized) == 2
    assert optimized[1]["op"] == "fused"
    assert optimized[1]["ops"] == ["assign", "assign", "filter"]


def apply_limit_pass(plan):
    """Utility to run only _pushdown_limit on the plan."""
    opt = FlowOptimizer(plan)
    return opt._pushdown_limit(plan)


def test_limit_pushes_over_stateless_ops():
    plan = [
        {"op": "from_jsonl", "path": "data.jsonl"},
        {"op": "filter", "predicate": lambda r: True},
        {"op": "assign", "fields": {"x": lambda r: 1}},
        {"op": "limit", "count": 10},
    ]
    new_plan = apply_limit_pass(plan)
    ops = [step["op"] for step in new_plan]
    assert ops == ["from_jsonl", "filter", "limit", "assign"]


def test_limit_not_pushed_over_join():
    plan = [
        {"op": "from_jsonl", "path": "data.jsonl"},
        {"op": "join", "on": ["id"], "right_plan": [], "how": "left"},
        {"op": "limit", "count": 5},
    ]
    new_plan = apply_limit_pass(plan)
    ops = [step["op"] for step in new_plan]
    # limit must stay last
    assert ops == ["from_jsonl", "join", "limit"]


def test_limit_not_pushed_over_filter():
    plan = [
        {"op": "from_jsonl", "path": "data.jsonl"},
        {"op": "filter", "predicate": lambda r: True},
        {"op": "limit", "count": 5},
    ]
    new_plan = apply_limit_pass(plan)
    ops = [step["op"] for step in new_plan]
    # limit must stay last
    assert ops == ["from_jsonl", "filter", "limit"]


def test_limit_not_pushed_if_not_possible():
    plan = [
        {"op": "from_json", "path": "data.json"},
        {"op": "limit", "count": 2},
    ]
    new_plan = apply_limit_pass(plan)
    assert new_plan == plan  # no change expected


def test_limit_inserted_once():
    plan = [
        {"op": "from_folder", "path": "data/"},
        {"op": "filter", "predicate": lambda r: True},
        {"op": "limit", "count": 10},
        {"op": "sort", "keys": ["foo"], "ascending": [True]},
    ]
    new_plan = apply_limit_pass(plan)
    # Should push before filter only once
    assert [s["op"] for s in new_plan].count("limit") == 1


def test_multiple_limits_only_outer_remains():
    plan = [
        {"op": "from_json", "path": "foo.json"},
        {"op": "limit", "count": 100},
        {"op": "filter", "predicate": lambda r: True},
        {"op": "limit", "count": 10},
    ]
    new_plan = apply_limit_pass(plan)
    # outermost limit takes precedence
    ops = [step["op"] for step in new_plan]
    assert ops.count("limit") == 2
    assert new_plan[1]["count"] == 100
    assert new_plan[3]["count"] == 10


def test_optimizer_pushdown_select():
    flow = Flow.from_jsonl("data.jsonl").select("x", "y").assign(z=lambda r: r["y"] + 1)
    plan = flow.plan
    opt_plan = FlowOptimizer(plan).optimize()

    # Ensure select came before assign
    select_index = next(i for i, step in enumerate(opt_plan) if step["op"] == "select")
    assign_index = next(i for i, step in enumerate(opt_plan) if step["op"] == "assign")
    assert select_index < assign_index
