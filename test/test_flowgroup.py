import pandas as pd
import pytest

from penaltyblog.matchflow.flow import Flow
from penaltyblog.matchflow.flowgroup import FlowGroup


@pytest.fixture
def sample_records():
    return [
        {"id": 1, "grp": "A", "v": 5},
        {"id": 2, "grp": "A", "v": 7},
        {"id": 3, "grp": "B", "v": 3},
    ]


def test_group_by_and_ungroup(sample_records):
    fg: FlowGroup = Flow(sample_records).group_by("grp")
    groups = fg.groups
    assert set(groups.keys()) == {("A",), ("B",)}

    flat = fg.ungroup().collect()
    assert sorted([r["id"] for r in flat]) == [1, 2, 3]

    assert Flow(sample_records).group_by("grp").ungroup() == Flow(sample_records)
    assert isinstance(Flow(sample_records).group_by("grp").ungroup(), Flow)


def test_summary_row_number(sample_records):
    fg = Flow(sample_records).group_by("grp")
    summary = fg.summary(count=("id", "count")).collect()
    assert any(r["count"] == 2 for r in summary)

    rn = fg.row_number("v").groups[("A",)][0]
    assert rn["row_number"] == 1


def test_first_last_is_empty(sample_records):
    fg = Flow(sample_records).group_by("grp")
    first = fg.first().groups
    assert list(first.keys())[0] == ("A",)
    last = fg.last().groups
    assert list(last.keys())[-1] == ("B",)
    assert Flow([]).group_by("x").is_empty()


def test_to_pandas(sample_records):
    fg = Flow(sample_records).group_by("grp")
    df_flat = fg.to_pandas()
    assert isinstance(df_flat, pd.DataFrame)
    df_sum = fg.to_pandas(agg_funcs={"sum_v": ("v", "sum")})
    assert df_sum.loc[df_sum["grp"] == "A", "sum_v"].iloc[0] == 12


def test_pipe(sample_records):
    fg = Flow(sample_records).group_by("grp")
    # pipe to Flow again (example)
    f2 = fg.pipe(lambda g: g.ungroup().filter(lambda r: r["v"] > 5))
    assert isinstance(f2, Flow)
    assert all(r["v"] > 5 for r in f2.collect())


def test_iter_len_repr_collect(sample_records):
    fg = Flow(sample_records).group_by("grp")
    # __iter__
    items = list(iter(fg))
    assert set(k for k, _ in items) == {("A",), ("B",)}
    # __len__
    assert len(fg) == 2
    # __repr__
    rep = repr(fg)
    assert "Penaltyblog Flow Group" in rep and "n_groups=2" in rep
    # collect
    groups = fg.collect()
    assert isinstance(groups, dict)
    assert {k for k in groups} == {("A",), ("B",)}


def test_sort_tail_unique_rename(sample_records):
    fg = Flow(sample_records).group_by("grp")
    # Add duplicate and None for sort/unique
    fg2 = Flow(
        sample_records
        + [{"id": 4, "grp": "A", "v": None}, {"id": 1, "grp": "A", "v": 5}]
    ).group_by("grp")
    # sort
    sorted_fg = fg2.sort(by="v")
    a_group = sorted_fg.groups[("A",)]
    assert a_group[-1]["v"] is None
    # tail
    tailed_fg = fg2.tail(1)
    assert all(len(recs) == 1 for recs in tailed_fg.groups.values())
    # unique
    unique_fg = fg2.unique()
    assert len(unique_fg.groups[("A",)]) < len(fg2.groups[("A",)])
    # rename
    renamed_fg = fg.rename(grp="group")
    for recs in renamed_fg.groups.values():
        for rec in recs:
            assert "group" in rec


def test_keys(sample_records):
    fg = Flow(sample_records).group_by("grp")
    keys = fg.keys()
    assert set(keys) == {("A",), ("B",)}


def test_drop_duplicates(sample_records):
    recs = sample_records + [{"id": 1, "grp": "A", "v": 5}]
    fg = Flow(recs).group_by("grp")
    # keep first (default)
    dd_first = fg.drop_duplicates("id", keep="first")
    assert len(dd_first.groups[("A",)]) == 2
    # keep last
    dd_last = fg.drop_duplicates("id", keep="last")
    # Check that, for each id, the record matches the last occurrence in the input
    last_rec_by_id = {rec["id"]: rec for rec in recs[::-1]}
    for rec in dd_last.groups[("A",)]:
        assert rec == last_rec_by_id[rec["id"]]
    # keep False (drop all dups)
    dd_none = fg.drop_duplicates("id", keep=False)
    assert all(rec["id"] != 1 for rec in dd_none.groups[("A",)])


def test_filter(sample_records):
    fg = Flow(sample_records).group_by("grp")
    # Only keep groups with sum of 'v' > 10
    filtered = fg.filter(lambda recs: sum(r["v"] for r in recs) > 10)
    assert list(filtered.groups.keys()) == [("A",)]


def test_head(sample_records):
    recs = sample_records + [{"id": 4, "grp": "A", "v": 9}]
    fg = Flow(recs).group_by("grp")
    head_fg = fg.head(1)
    assert all(len(recs) == 1 for recs in head_fg.groups.values())


def test_rename_edge_cases(sample_records):
    fg = Flow(sample_records).group_by("grp")
    # Rename to an existing key
    recs = [{"id": 1, "grp": "A", "foo": 10}]
    fg2 = Flow(recs).group_by("grp")
    renamed = fg2.rename(id="foo")
    for recs in renamed.groups.values():
        for rec in recs:
            assert "foo" in rec
    # Rename non-existent key (should not fail)
    fg3 = fg.rename(not_a_key="new")
    for recs in fg3.groups.values():
        for rec in recs:
            assert "new" not in rec


def test_sort_edge_cases(sample_records):
    recs = sample_records + [{"id": 4, "grp": "A"}]  # missing 'v'
    fg = Flow(recs).group_by("grp")
    sorted_fg = fg.sort(by="v")
    # For groups with a record missing 'v', that record should be last
    for group in sorted_fg.groups.values():
        if any("v" not in rec or rec["v"] is None for rec in group):
            assert group[-1].get("v", None) is None
        else:
            # For groups with all valid 'v', check order is ascending
            vs = [rec["v"] for rec in group]
            assert vs == sorted(vs)


def test_row_number_edge_cases(sample_records):
    fg = Flow(sample_records).group_by("grp")
    # reverse order
    rn_rev = fg.row_number("v", reverse=True)
    for recs in rn_rev.groups.values():
        assert recs[0]["row_number"] == 1
    # custom field name
    rn_custom = fg.row_number("v", new_field="pos")
    for recs in rn_custom.groups.values():
        assert "pos" in recs[0]
