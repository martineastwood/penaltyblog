import copy

import networkx as nx

from .optimizer import FlowOptimizer

NODE_COLOURS = {
    "from_json": "#AED6F1",
    "from_jsonl": "#AED6F1",
    "from_glob": "#AED6F1",
    "from_folder": "#AED6F1",
    "from_statsbomb": "#AED6F1",
    "from_materialized": "#AED6F1",
    "from_concat": "#AED6F1",
    "select": "#ABEBC6",
    "drop": "#ABEBC6",
    "dropna": "#ABEBC6",
    "rename": "#ABEBC6",
    "flatten": "#ABEBC6",
    "assign": "#F9E79F",
    "cast": "#F9E79F",
    "map": "#F9E79F",
    "pipe": "#F9E79F",
    "explode": "#F9E79F",
    "split_array": "#F9E79F",
    "filter": "#F5B7B1",
    "distinct": "#F5B7B1",
    "group_by": "#D2B4DE",
    "group_summary": "#D2B4DE",
    "summary": "#D2B4DE",
    "cumulative": "#D2B4DE",
    "sort": "#FAD7A0",
    "limit": "#FAD7A0",
    "head": "#FAD7A0",
    "join": "#A3E4D7",
    "concat": "#A3E4D7",
    "sample_n": "#EDBB99",
    "sample_fraction": "#EDBB99",
    "pivot": "#C39BD3",
    "cache": "#D5DBDB",
    "schema": "#D5DBDB",
    "fused": "#D5DBDB",
    "pipe": "#D5DBDB",
}


def _to_nx_graph(plan):
    G = nx.DiGraph()
    for i, step in enumerate(plan):
        label = f"{i + 1}. {step['op']}"
        G.add_node(i, label=label, op=step["op"], step=step)
        if i > 0:
            G.add_edge(i - 1, i)
    return G


def _vertical_layout(G, spacing=2.5):
    return {node: (0.5, -i * spacing) for i, node in enumerate(G.nodes())}


def plot_plan(plan, ax, title=""):
    G = _to_nx_graph(plan)
    pos = _vertical_layout(G)

    colors = [NODE_COLOURS.get(data["op"], "#D5DBDB") for _, data in G.nodes(data=True)]
    labels = nx.get_node_attributes(G, "label")

    nx.draw_networkx_nodes(
        G, pos, node_color=colors, node_size=500, edgecolors="black", ax=ax
    )
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", ax=ax)

    for node, (x, y) in pos.items():
        label = labels[node]
        notes = G.nodes[node]["step"].get("_notes")
        if notes:
            label += f"\n// " + "; ".join(notes)
        ax.text(x + 0.05, y, label, ha="left", va="center", fontsize=9, wrap=True)

    ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=12)
    ax.axis("off")


def plot_flow_plan(plan, optimize=False, compare=False, title_prefix=""):
    """
    Helper to visualize a plan (single or compare mode), optionally optimizing.
    optimizer_cls: class to use for optimization (must have .optimize()).
    """
    import matplotlib.pyplot as plt

    if compare:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(10, max(2, len(plan) * 0.75)),
            sharey=True,
        )
        plot_plan(plan, axes[0], f"{title_prefix}Original Plan")
        optimized = FlowOptimizer(copy.deepcopy(plan)).optimize()
        plot_plan(optimized, axes[1], f"{title_prefix}Optimized Plan")
        plt.tight_layout()
        plt.show()
    else:
        if optimize:
            plan_to_plot = FlowOptimizer(plan).optimize()
            title = f"{title_prefix}Optimized Plan"
        else:
            plan_to_plot = plan
            title = f"{title_prefix}Original Plan"
        fig, ax = plt.subplots(figsize=(5, max(2, len(plan_to_plot) * 0.75)))
        plot_plan(plan_to_plot, ax, title)
        # plt.tight_layout()
        plt.show()
