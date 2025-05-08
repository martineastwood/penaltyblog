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
