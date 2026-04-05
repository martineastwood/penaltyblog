import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from penaltyblog.matchflow import Flow
from penaltyblog.viz.pitch import Pitch
from penaltyblog.xt import XTData, XTModel, load_pretrained_xt
from penaltyblog.xt.model import (
    _clip_coords,
    _coerce_bool,
    _coords_to_cell,
    _safe_cell_index,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_xtdata(df, **kwargs):
    """Create an XTData from a DataFrame with canonical column names."""
    defaults = dict(
        events=df,
        x="x",
        y="y",
        event_type="event_type",
        end_x="end_x",
        end_y="end_y",
        is_success="is_success",
    )
    defaults.update(kwargs)
    return XTData(**defaults)


def simple_pass_shot_df():
    """Minimal dataset: 1 successful pass + 1 missed shot + 1 goal."""
    return pd.DataFrame(
        {
            "x": [10, 10, 90],
            "y": [10, 10, 10],
            "end_x": [90, np.nan, np.nan],
            "end_y": [10, np.nan, np.nan],
            "event_type": ["pass", "shot", "shot"],
            "is_success": [True, False, True],
        }
    )


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def test_clip_coords():
    vals = np.array([-5.0, 0.0, 50.0, 120.0])
    clipped = _clip_coords(vals)
    assert np.allclose(clipped, [0.0, 0.0, 50.0, 100.0])


def test_clip_coords_nan():
    vals = np.array([np.nan, 50.0])
    clipped = _clip_coords(vals)
    assert np.isnan(clipped[0])
    assert clipped[1] == 50.0


def test_coords_to_cell_and_flatten():
    l, w = 4, 2
    x = np.array([0.0, 99.9, 100.0])
    y = np.array([0.0, 50.0, 99.9])
    cells = _coords_to_cell(x, y, l, w)
    assert cells[0] == 0  # (gx=0, gy=0) -> 0*4+0
    assert cells[1] == 7  # (gx=3, gy=1) -> 1*4+3
    assert cells[2] == 7  # clipped to (3,1) -> 7


def test_safe_cell_index_nan():
    x = np.array([10.0, np.nan, 50.0])
    y = np.array([10.0, 50.0, np.nan])
    cells = _safe_cell_index(x, y, 4, 2)
    assert cells[0] >= 0
    assert cells[1] == -1
    assert cells[2] == -1


# ---------------------------------------------------------------------------
# XTData
# ---------------------------------------------------------------------------


def test_xtdata_validates_required_columns():
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError, match="Missing required columns"):
        XTData(events=df, x="x", y="y", event_type="event_type")


def test_xtdata_validates_paired_end_coords():
    df = pd.DataFrame({"x": [1], "y": [2], "event_type": ["pass"], "ex": [3]})
    with pytest.raises(ValueError, match="Both or neither"):
        XTData(events=df, x="x", y="y", event_type="event_type", end_x="ex")


def test_xtdata_normalized_df_adds_missing_columns():
    df = pd.DataFrame({"x": [10], "y": [20], "event_type": ["pass"]})
    data = XTData(events=df, x="x", y="y", event_type="event_type")
    ndf = data.df
    for col in [
        "end_x",
        "end_y",
        "is_success",
    ]:
        assert col in ndf.columns


def test_xtdata_accepts_matchflow_flow_events():
    flow = Flow.from_records(
        [{"x": 10, "y": 20, "event_type": "pass", "end_x": 30, "end_y": 40}]
    )
    data = XTData(events=flow, x="x", y="y", event_type="event_type")
    assert isinstance(data.events, pd.DataFrame)
    ndf = data.df
    assert ndf["event_type"].iloc[0] == "pass"


def test_xtdata_normalizes_custom_coordinate_ranges():
    df = pd.DataFrame(
        {
            "x": [0.0, 60.0, 120.0],
            "y": [0.0, 40.0, 80.0],
            "end_x": [120.0, np.nan, 0.0],
            "end_y": [80.0, np.nan, 0.0],
            "event_type": ["pass", "shot", "pass"],
        }
    )
    data = XTData(
        events=df,
        x="x",
        y="y",
        event_type="event_type",
        end_x="end_x",
        end_y="end_y",
        x_range=(0.0, 120.0),
        y_range=(0.0, 80.0),
    )
    ndf = data.df
    assert np.allclose(ndf["x"].to_numpy(), [0.0, 50.0, 100.0])
    assert np.allclose(ndf["y"].to_numpy(), [0.0, 50.0, 100.0])
    assert np.isclose(ndf["end_x"].iloc[0], 100.0)
    assert np.isclose(ndf["end_y"].iloc[0], 100.0)
    assert np.isnan(ndf["end_x"].iloc[1])
    assert np.isnan(ndf["end_y"].iloc[1])


def test_xtdata_rejects_invalid_coordinate_range():
    df = pd.DataFrame({"x": [0], "y": [0], "event_type": ["pass"]})
    with pytest.raises(ValueError, match="x_range"):
        XTData(
            events=df,
            x="x",
            y="y",
            event_type="event_type",
            x_range=(100.0, 0.0),
        )


def test_xtdata_map_events():
    df = pd.DataFrame(
        {
            "x": [10],
            "y": [20],
            "raw_type": ["Pass"],
            "ex": [30],
            "ey": [40],
            "success": [True],
        }
    )
    data = XTData(
        events=df,
        x="x",
        y="y",
        event_type="raw_type",
        end_x="ex",
        end_y="ey",
        is_success="success",
    ).map_events(event_map={"Pass": "pass"})
    assert data.df["event_type"].iloc[0] == "pass"


# ---------------------------------------------------------------------------
# Event classification
# ---------------------------------------------------------------------------


def test_classification_pass_is_move():
    df = simple_pass_shot_df()
    data = make_xtdata(df)
    model = XTModel(l=2, w=1)
    ndf = data.df
    role = model._classify_events(ndf)
    assert role.iloc[0] == "move"


def test_classification_shot_is_shot():
    df = simple_pass_shot_df()
    data = make_xtdata(df)
    model = XTModel(l=2, w=1)
    role = model._classify_events(data.df)
    assert role.iloc[1] == "shot"
    assert role.iloc[2] == "shot"


def test_classification_carry():
    df = pd.DataFrame(
        {
            "x": [10],
            "y": [10],
            "end_x": [80.0],
            "end_y": [50.0],
            "event_type": ["carry"],
            "is_success": [True],
        }
    )
    data = make_xtdata(df)
    model = XTModel(l=2, w=1, include_carries=True)
    role = model._classify_events(data.df)
    assert role.iloc[0] == "move"


def test_classification_carry_excluded():
    df = pd.DataFrame(
        {
            "x": [10],
            "y": [10],
            "end_x": [np.nan],
            "end_y": [np.nan],
            "event_type": ["carry"],
            "is_success": [True],
        }
    )
    data = make_xtdata(df)
    model = XTModel(l=2, w=1, include_carries=False)
    role = model._classify_events(data.df)
    assert role.iloc[0] == "ignore"


def test_classification_throw_in():
    df = pd.DataFrame(
        {
            "x": [30],
            "y": [0],
            "end_x": [50],
            "end_y": [30],
            "event_type": ["throw_in"],
            "is_success": [True],
        }
    )
    data = make_xtdata(df)
    model = XTModel(l=2, w=1, include_throw_ins=True)
    role = model._classify_events(data.df)
    assert role.iloc[0] == "move"


def test_classification_throw_in_excluded():
    df = pd.DataFrame(
        {
            "x": [30],
            "y": [0],
            "end_x": [50],
            "end_y": [30],
            "event_type": ["throw_in"],
            "is_success": [True],
        }
    )
    data = make_xtdata(df)
    model = XTModel(l=2, w=1, include_throw_ins=False)
    role = model._classify_events(data.df)
    assert role.iloc[0] == "ignore"


def test_classification_free_kick_pass_vs_shot():
    df = pd.DataFrame(
        {
            "x": [30, 30],
            "y": [50, 50],
            "end_x": [60, np.nan],
            "end_y": [50, np.nan],
            "event_type": ["free_kick", "free_kick_shot"],
            "is_success": [True, False],
        }
    )
    data = make_xtdata(df)
    model = XTModel(l=2, w=1, include_free_kicks=True)
    role = model._classify_events(data.df)
    assert role.iloc[0] == "move"
    assert role.iloc[1] == "shot"


def test_classification_corner_is_move():
    df = pd.DataFrame(
        {
            "x": [0],
            "y": [0],
            "end_x": [80],
            "end_y": [50],
            "event_type": ["corner"],
            "is_success": [True],
        }
    )
    data = make_xtdata(df)
    model = XTModel(l=2, w=1, include_corners=True)
    role = model._classify_events(data.df)
    assert role.iloc[0] == "move"


def test_classification_corner_excluded():
    df = pd.DataFrame(
        {
            "x": [0],
            "y": [0],
            "end_x": [80],
            "end_y": [50],
            "event_type": ["corner"],
            "is_success": [True],
        }
    )
    data = make_xtdata(df)
    model = XTModel(l=2, w=1, include_corners=False)
    role = model._classify_events(data.df)
    assert role.iloc[0] == "ignore"


def test_classification_penalty_ignored():
    df = pd.DataFrame(
        {
            "x": [88],
            "y": [50],
            "end_x": [np.nan],
            "end_y": [np.nan],
            "event_type": ["penalty"],
            "is_success": [True],
        }
    )
    data = make_xtdata(df)
    model = XTModel(l=2, w=1)
    role = model._classify_events(data.df)
    assert role.iloc[0] == "ignore"


# ---------------------------------------------------------------------------
# Inclusion/exclusion flags
# ---------------------------------------------------------------------------


def test_inclusion_flags_affect_fit():
    df = pd.DataFrame(
        {
            "x": [10, 10, 10, 90],
            "y": [10, 10, 10, 10],
            "end_x": [90, 90, np.nan, np.nan],
            "end_y": [10, 10, np.nan, np.nan],
            "event_type": ["pass", "throw_in", "shot", "shot"],
            "is_success": [True, True, False, True],
        }
    )
    data = make_xtdata(df)

    # With throw-ins
    m1 = XTModel(l=2, w=1, include_throw_ins=True).fit(data)
    # Without throw-ins
    m2 = XTModel(l=2, w=1, include_throw_ins=False).fit(data)

    # Move count should differ
    assert m1.move_probability_.sum() != m2.move_probability_.sum()


# ---------------------------------------------------------------------------
# Count construction & probabilities
# ---------------------------------------------------------------------------


def test_goal_probability_smoothing():
    df = pd.DataFrame(
        {
            "x": [10],
            "y": [10],
            "end_x": [np.nan],
            "end_y": [np.nan],
            "event_type": ["shot"],
            "is_success": [False],
        }
    )
    data = make_xtdata(df)
    model = XTModel(l=2, w=1).fit(data)
    cell = _coords_to_cell(np.array([10.0]), np.array([10.0]), 2, 1)[0]
    gp = model.goal_probability_.reshape(-1)[cell]
    # (0 + 1) / (1 + 2) = 1/3
    assert np.isclose(gp, 1 / 3)


def test_goal_probability_with_goal():
    df = pd.DataFrame(
        {
            "x": [90, 90],
            "y": [50, 50],
            "end_x": [np.nan, np.nan],
            "end_y": [np.nan, np.nan],
            "event_type": ["shot", "shot"],
            "is_success": [False, True],
        }
    )
    data = make_xtdata(df)
    model = XTModel(l=2, w=1).fit(data)
    cell = _coords_to_cell(np.array([90.0]), np.array([50.0]), 2, 1)[0]
    gp = model.goal_probability_.reshape(-1)[cell]
    # (1 + 1) / (2 + 2) = 0.5
    assert np.isclose(gp, 0.5)


# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------


def test_transition_matrix_row_normalization():
    df = pd.DataFrame(
        {
            "x": [10, 10],
            "y": [10, 10],
            "end_x": [90, 90],
            "end_y": [10, 10],
            "event_type": ["pass", "pass"],
            "is_success": [True, True],
        }
    )
    data = make_xtdata(df)
    model = XTModel(l=2, w=1).fit(data)
    start = _coords_to_cell(np.array([10.0]), np.array([10.0]), 2, 1)[0]
    end = _coords_to_cell(np.array([90.0]), np.array([10.0]), 2, 1)[0]
    row = model.transition_matrix_[start]
    assert np.isclose(row.sum(), 1.0)
    assert np.isclose(row[end], 1.0)


def test_transition_matrix_zero_row():
    """A cell with no moves should have an all-zero transition row."""
    df = pd.DataFrame(
        {
            "x": [90],
            "y": [10],
            "end_x": [np.nan],
            "end_y": [np.nan],
            "event_type": ["shot"],
            "is_success": [True],
        }
    )
    data = make_xtdata(df)
    model = XTModel(l=2, w=1).fit(data)
    # Cell with no moves
    cell = _coords_to_cell(np.array([10.0]), np.array([10.0]), 2, 1)[0]
    assert model.transition_matrix_[cell].sum() == 0.0


def test_per_family_transitions():
    """Different move families from the same cell should use their own
    transition distributions, not a single pooled one."""
    # 3-cell grid (l=3, w=1).  From cell 0:
    #   - passes go to cell 2 (far)
    #   - throw-ins go to cell 1 (near)
    # Shot + goal in cell 2 so it has xT value.
    df = pd.DataFrame(
        {
            "x": [10, 10, 90],
            "y": [50, 50, 50],
            "end_x": [90, 40, np.nan],
            "end_y": [50, 50, np.nan],
            "event_type": ["pass", "throw_in", "shot"],
            "is_success": [True, True, True],
        }
    )
    data = make_xtdata(df)
    model = XTModel(l=3, w=1, include_throw_ins=True).fit(data)

    cell0 = _coords_to_cell(np.array([10.0]), np.array([50.0]), 3, 1)[0]
    cell1 = _coords_to_cell(np.array([40.0]), np.array([50.0]), 3, 1)[0]
    cell2 = _coords_to_cell(np.array([90.0]), np.array([50.0]), 3, 1)[0]

    row = model.transition_matrix_[cell0]
    # Both destinations should have weight
    assert row[cell1] > 0, "throw-in destination should have weight"
    assert row[cell2] > 0, "pass destination should have weight"


def test_sparse_family_shrinks_toward_pooled():
    """A sparse family's transition should be pulled toward the pooled
    distribution rather than being 100% to one noisy destination."""
    # 3-cell grid.  From cell 0:
    #   - 20 passes go to cell 2 (dominant pattern)
    #   - 1 throw-in goes to cell 1 (sparse)
    # Without shrinkage, throw-in T would be 100% to cell 1.
    # With shrinkage (k=5), it should be pulled toward the pooled
    # pattern which is heavily weighted to cell 2.
    n_passes = 20
    rows = []
    for _ in range(n_passes):
        rows.append(
            {
                "x": 10,
                "y": 50,
                "end_x": 90,
                "end_y": 50,
                "event_type": "pass",
                "is_success": True,
            }
        )
    # One sparse throw-in to a different destination
    rows.append(
        {
            "x": 10,
            "y": 50,
            "end_x": 40,
            "end_y": 50,
            "event_type": "throw_in",
            "is_success": True,
        }
    )
    # Shot in cell 2 so there's xT value (goal)
    rows.append(
        {
            "x": 90,
            "y": 50,
            "end_x": np.nan,
            "end_y": np.nan,
            "event_type": "shot",
            "is_success": True,
        }
    )

    df = pd.DataFrame(rows)
    data = make_xtdata(df)
    model = XTModel(l=3, w=1, include_throw_ins=True).fit(data)

    cell1 = _coords_to_cell(np.array([40.0]), np.array([50.0]), 3, 1)[0]
    cell2 = _coords_to_cell(np.array([90.0]), np.array([50.0]), 3, 1)[0]

    # The throw-in family has 1 event to cell 1.
    # The pooled transition from cell 0 is ~95% to cell 2 (20 passes
    # + 1 throw-in, so 20/21 to cell 2, 1/21 to cell 1).
    # After shrinkage with k=5:
    #   T_throwin_smoothed[cell0, cell1] = (1 + 5 * 1/21) / (1 + 5)
    #   T_throwin_smoothed[cell0, cell2] = (0 + 5 * 20/21) / (1 + 5)
    # So the throw-in family should still send some weight to cell 2
    # even though the raw throw-in data only went to cell 1.
    #
    # We test this indirectly: if we exclude throw-ins and only have
    # passes, cell 0's transition is 100% to cell 2.  With throw-ins
    # included, cell 1 gets some weight but cell 2 should still dominate.
    row = model.transition_matrix_
    cell0 = _coords_to_cell(np.array([10.0]), np.array([50.0]), 3, 1)[0]
    t_to_cell1 = row[cell0, cell1]
    t_to_cell2 = row[cell0, cell2]

    # Cell 2 should still be the dominant destination
    assert t_to_cell2 > t_to_cell1
    # But cell 1 should have some weight (throw-in destination)
    assert t_to_cell1 > 0


# ---------------------------------------------------------------------------
# Direct solve
# ---------------------------------------------------------------------------


def test_direct_solve_shape_and_finite():
    df = simple_pass_shot_df()
    data = make_xtdata(df)
    model = XTModel(l=2, w=1).fit(data)
    assert model.surface_.shape == (1, 2)
    assert np.isfinite(model.surface_).all()


def test_trivial_one_cell_case():
    df = pd.DataFrame(
        {
            "x": [10, 20],
            "y": [10, 20],
            "end_x": [10, np.nan],
            "end_y": [10, np.nan],
            "event_type": ["pass", "shot"],
            "is_success": [True, True],
        }
    )
    data = make_xtdata(df)
    model = XTModel(l=1, w=1).fit(data)
    # shot_prob = 1/2, move_prob = 1/2, goal_prob = (1+1)/(1+2) = 2/3
    # S = 0.5 * 2/3 = 1/3, M = 0.5, T = [[1.0]]
    # (1 - 0.5*1) X = 1/3 => X = 2/3
    expected = (0.5 * (2 / 3)) / (1 - 0.5)
    assert np.isclose(model.surface_[0, 0], expected)


def test_two_cell_transition():
    """Cell with shots should have higher xT than cell with only passes
    when the passing cell also has shots (splitting its action budget)."""
    df = pd.DataFrame(
        {
            "x": [10, 10, 10, 90, 90],
            "y": [10, 10, 10, 10, 10],
            "end_x": [90, 90, np.nan, np.nan, np.nan],
            "end_y": [10, 10, np.nan, np.nan, np.nan],
            "event_type": ["pass", "pass", "shot", "shot", "shot"],
            "is_success": [True, True, False, False, True],
        }
    )
    data = make_xtdata(df)
    model = XTModel(l=2, w=1).fit(data)
    flat = model.surface_.reshape(-1)
    cell0 = _coords_to_cell(np.array([10.0]), np.array([10.0]), 2, 1)[0]
    cell1 = _coords_to_cell(np.array([90.0]), np.array([10.0]), 2, 1)[0]
    assert flat[cell1] > 0
    assert flat[cell0] > 0


# ---------------------------------------------------------------------------
# value_at
# ---------------------------------------------------------------------------


def test_value_at():
    df = simple_pass_shot_df()
    data = make_xtdata(df)
    model = XTModel(l=2, w=1).fit(data)
    v0 = model.value_at(10, 10)
    v1 = model.value_at(90, 10)
    assert v1 > v0


def test_value_at_clips():
    df = simple_pass_shot_df()
    data = make_xtdata(df)
    model = XTModel(l=2, w=1).fit(data)
    # Out-of-bounds should be clipped and not raise
    v = model.value_at(-5, 200)
    assert np.isfinite(v)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_score_output_columns():
    df = simple_pass_shot_df()
    data = make_xtdata(df)
    model = XTModel(l=2, w=1).fit(data)
    scored = model.score(data)
    assert {"xt_start", "xt_end", "xt_added"}.issubset(scored.columns)


def test_score_preserves_all_rows():
    df = simple_pass_shot_df()
    data = make_xtdata(df)
    model = XTModel(l=2, w=1).fit(data)
    scored = model.score(data)
    assert len(scored) == len(df)


def test_score_nan_for_non_moves():
    df = simple_pass_shot_df()
    data = make_xtdata(df)
    model = XTModel(l=2, w=1).fit(data)
    scored = model.score(data)
    # Shots should have NaN
    shot_rows = scored[scored["event_type"] == "shot"]
    assert shot_rows["xt_added"].isna().all()


def test_score_delta_matches_value_at():
    df = simple_pass_shot_df()
    data = make_xtdata(df)
    model = XTModel(l=2, w=1).fit(data)
    scored = model.score(data)

    pass_row = scored[scored["event_type"] == "pass"].iloc[0]
    v_start = model.value_at(10, 10)
    v_end = model.value_at(90, 10)
    assert np.isclose(pass_row["xt_added"], v_end - v_start)


def test_score_mixed_event_families():
    """Scoring works across multiple move families."""
    df = pd.DataFrame(
        {
            "x": [10, 10, 90],
            "y": [10, 10, 10],
            "end_x": [90, 90, np.nan],
            "end_y": [10, 10, np.nan],
            "event_type": ["pass", "throw_in", "shot"],
            "is_success": [True, True, True],
        }
    )
    data = make_xtdata(df)
    model = XTModel(l=2, w=1, include_throw_ins=True).fit(data)
    scored = model.score(data)
    # Both pass and throw_in should be scored
    assert pd.notna(scored.iloc[0]["xt_added"])
    assert pd.notna(scored.iloc[1]["xt_added"])


def test_score_returns_original_columns():
    """Scored DataFrame should have original column names, not normalized ones."""
    df = pd.DataFrame(
        {
            "loc_x": [10, 90],
            "loc_y": [10, 10],
            "pass_dest_x": [90, np.nan],
            "pass_dest_y": [10, np.nan],
            "type": ["pass", "shot"],
            "success": [True, True],
        }
    )
    data = XTData(
        events=df,
        x="loc_x",
        y="loc_y",
        event_type="type",
        end_x="pass_dest_x",
        end_y="pass_dest_y",
        is_success="success",
    )
    model = XTModel(l=2, w=1).fit(data)
    scored = model.score(data)
    assert "loc_x" in scored.columns
    assert "type" in scored.columns


def test_fit_accepts_raw_dataframe():
    df = simple_pass_shot_df()
    model = XTModel(l=2, w=1).fit(df)
    assert model.fitted_ is True
    assert model.surface_.shape == (1, 2)


def test_fit_accepts_matchflow_flow():
    flow = Flow.from_records(simple_pass_shot_df().to_dict(orient="records"))
    model = XTModel(l=2, w=1).fit(flow)
    assert model.fitted_ is True
    assert model.surface_.shape == (1, 2)


def test_fit_accepts_raw_dataframe_with_column_mapping_and_ranges():
    df = pd.DataFrame(
        {
            "loc_x": [10, 10, 90],
            "loc_y": [10, 10, 10],
            "dest_x": [90, np.nan, np.nan],
            "dest_y": [10, np.nan, np.nan],
            "etype": ["Pass", "Shot", "Shot"],
            "outcome": ["Complete", "Saved", "Goal"],
        }
    )
    model = XTModel(l=2, w=1).fit(
        df,
        x="loc_x",
        y="loc_y",
        event_type="etype",
        end_x="dest_x",
        end_y="dest_y",
        is_success="outcome",
        x_range=(0, 120),
        y_range=(0, 80),
        event_map={"Pass": "pass", "Shot": "shot"},
        success_map={"Complete": True, "Saved": False, "Goal": True},
    )
    assert model.fitted_ is True


def test_score_accepts_raw_dataframe_with_column_mapping():
    df = pd.DataFrame(
        {
            "loc_x": [10, 90],
            "loc_y": [10, 10],
            "dest_x": [90, np.nan],
            "dest_y": [10, np.nan],
            "etype": ["Pass", "Shot"],
            "outcome": ["Complete", "Goal"],
        }
    )
    model = XTModel(l=2, w=1).fit(
        df,
        x="loc_x",
        y="loc_y",
        event_type="etype",
        end_x="dest_x",
        end_y="dest_y",
        is_success="outcome",
        event_map={"Pass": "pass", "Shot": "shot"},
        success_map={"Complete": True, "Goal": True},
    )
    scored = model.score(
        df,
        x="loc_x",
        y="loc_y",
        event_type="etype",
        end_x="dest_x",
        end_y="dest_y",
        is_success="outcome",
        event_map={"Pass": "pass", "Shot": "shot"},
        success_map={"Complete": True, "Goal": True},
    )
    assert "xt_added" in scored.columns
    assert scored["xt_added"].notna().sum() == 1


def test_score_reuses_fit_schema_by_default():
    df = pd.DataFrame(
        {
            "loc_x": [10, 90],
            "loc_y": [10, 10],
            "dest_x": [90, np.nan],
            "dest_y": [10, np.nan],
            "etype": ["Pass", "Shot"],
            "outcome": ["Complete", "Goal"],
        }
    )
    model = XTModel(l=2, w=1).fit(
        df,
        x="loc_x",
        y="loc_y",
        event_type="etype",
        end_x="dest_x",
        end_y="dest_y",
        is_success="outcome",
        event_map={"Pass": "pass", "Shot": "shot"},
        success_map={"Complete": True, "Goal": True},
    )
    scored = model.score(df)
    assert "xt_added" in scored.columns
    assert scored["xt_added"].notna().sum() == 1


def test_score_accepts_matchflow_flow():
    df = simple_pass_shot_df()
    model = XTModel(l=2, w=1).fit(df)
    flow = Flow.from_records(df.to_dict(orient="records"))
    scored = model.score(flow)
    assert "xt_added" in scored.columns
    assert scored["xt_added"].notna().sum() == 1


# ---------------------------------------------------------------------------
# Save/load round trip
# ---------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path):
    df = simple_pass_shot_df()
    data = make_xtdata(df)
    model = XTModel(l=2, w=1).fit(data)

    path = tmp_path / "xt_model.npz"
    model.save(str(path))
    loaded = XTModel.load(str(path))

    assert loaded.fitted_
    assert loaded.l == model.l
    assert loaded.w == model.w
    assert loaded.include_carries == model.include_carries
    assert loaded.include_throw_ins == model.include_throw_ins
    assert np.allclose(loaded.surface_, model.surface_)
    assert np.allclose(loaded.transition_matrix_, model.transition_matrix_)


def test_save_load_preserves_metadata(tmp_path):
    df = simple_pass_shot_df()
    data = make_xtdata(df)
    model = XTModel(l=4, w=3, include_carries=False, coord_policy="clip").fit(data)

    path = tmp_path / "xt.npz"
    model.save(str(path))
    loaded = XTModel.load(str(path))

    assert loaded.include_carries is False
    assert loaded.coord_policy == "clip"
    assert loaded.metadata_["grid"] == [4, 3]


# ---------------------------------------------------------------------------
# Pretrained model
# ---------------------------------------------------------------------------


def test_load_pretrained_xt():
    model = load_pretrained_xt(name="default")
    assert model.fitted_
    assert model.surface_.shape == (12, 16)


def test_load_pretrained_xt_invalid():
    with pytest.raises(ValueError, match="No pretrained"):
        load_pretrained_xt(name="nonexistent")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def test_plotting():
    model = load_pretrained_xt(name="default")
    pitch = model.plot()
    assert isinstance(pitch, Pitch)
    assert any(isinstance(t, go.Heatmap) for t in pitch.fig.data)


def test_plot_custom_pitch():
    model = load_pretrained_xt(name="default")
    pitch = Pitch(provider="opta")
    result = model.plot(pitch=pitch)
    assert result is pitch


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_grid_size():
    with pytest.raises(ValueError, match="positive"):
        XTModel(l=0, w=12)


def test_invalid_coord_policy():
    with pytest.raises(ValueError, match="coord_policy"):
        XTModel(coord_policy="invalid")


def test_score_before_fit_raises():
    df = simple_pass_shot_df()
    data = make_xtdata(df)
    model = XTModel(l=2, w=1)
    with pytest.raises(ValueError, match="not fitted"):
        model.score(data)


def test_value_at_before_fit_raises():
    model = XTModel(l=2, w=1)
    with pytest.raises(ValueError, match="Call \\.fit\\(\\) first"):
        model.value_at(50, 50)


def test_fit_empty_dataset_raises():
    df = pd.DataFrame(
        {
            "x": [10],
            "y": [10],
            "end_x": [np.nan],
            "end_y": [np.nan],
            "event_type": ["penalty"],
            "is_success": [False],
        }
    )
    data = make_xtdata(df)
    with pytest.raises(ValueError, match="Check that event_map maps"):
        XTModel(l=2, w=1).fit(data)


def test_load_rejects_invalid_grid_metadata(tmp_path):
    path = tmp_path / "bad_xt.npz"
    np.savez(
        path,
        surface=np.zeros((1, 1)),
        shot_probability=np.zeros((1, 1)),
        goal_probability=np.zeros((1, 1)),
        move_probability=np.zeros((1, 1)),
        transition_matrix=np.zeros((1, 1)),
        meta_json=np.array(['{"model_type":"xt","grid":[null,null]}']),
    )

    with pytest.raises(ValueError, match="metadata.grid"):
        XTModel.load(str(path))


def test_fit_missing_is_success_raises():
    df = pd.DataFrame(
        {
            "x": [10, 90],
            "y": [10, 10],
            "end_x": [90, np.nan],
            "end_y": [10, np.nan],
            "event_type": ["pass", "shot"],
        }
    )
    with pytest.raises(ValueError, match="Missing success information"):
        XTModel(l=2, w=1).fit(df)


def test_fit_warns_on_out_of_bounds_coords():
    df = pd.DataFrame(
        {
            "x": [10, 120, 90],
            "y": [10, 10, 10],
            "end_x": [90, 90, np.nan],
            "end_y": [10, 10, np.nan],
            "event_type": ["pass", "pass", "shot"],
            "is_success": [True, True, True],
        }
    )
    data = make_xtdata(df)
    model = XTModel(l=2, w=1, coord_policy="warn")
    with pytest.warns(UserWarning, match="outside expected 0..100"):
        model.fit(data)


def test_fit_errors_on_out_of_bounds_coords():
    df = pd.DataFrame(
        {
            "x": [10, 120, 90],
            "y": [10, 10, 10],
            "end_x": [90, 90, np.nan],
            "end_y": [10, 10, np.nan],
            "event_type": ["pass", "pass", "shot"],
            "is_success": [True, True, True],
        }
    )
    data = make_xtdata(df)
    model = XTModel(l=2, w=1, coord_policy="error")
    with pytest.raises(ValueError, match="outside expected 0..100"):
        model.fit(data)


def test_fit_clip_policy_no_warning():
    df = pd.DataFrame(
        {
            "x": [10, 120, 90],
            "y": [10, 10, 10],
            "end_x": [90, 90, np.nan],
            "end_y": [10, 10, np.nan],
            "event_type": ["pass", "pass", "shot"],
            "is_success": [True, True, True],
        }
    )
    data = make_xtdata(df)
    model = XTModel(l=2, w=1, coord_policy="clip")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model.fit(data)
    assert len(record) == 0


def test_score_errors_on_out_of_bounds_coords():
    train_df = simple_pass_shot_df()
    model = XTModel(l=2, w=1, coord_policy="error").fit(make_xtdata(train_df))
    score_df = train_df.copy()
    score_df.loc[0, "end_x"] = 150.0
    with pytest.raises(ValueError, match="outside expected 0..100"):
        model.score(make_xtdata(score_df))


def test_value_at_error_policy_raises_on_out_of_bounds():
    df = simple_pass_shot_df()
    model = XTModel(l=2, w=1, coord_policy="error").fit(df)
    with pytest.raises(ValueError, match="outside expected 0..100"):
        model.value_at(-1, 50)


def test_value_at_warn_policy_warns_on_out_of_bounds():
    df = simple_pass_shot_df()
    model = XTModel(l=2, w=1, coord_policy="warn").fit(df)
    with pytest.warns(UserWarning, match="outside expected 0..100"):
        model.value_at(-1, 50)


def test_value_at_rejects_non_finite_inputs():
    df = simple_pass_shot_df()
    model = XTModel(l=2, w=1).fit(df)
    with pytest.raises(ValueError, match="finite numbers"):
        model.value_at(np.inf, 50)
    with pytest.raises(ValueError, match="finite numbers"):
        model.value_at(50, -np.inf)


# ---------------------------------------------------------------------------
# Fitted attributes
# ---------------------------------------------------------------------------


def test_fitted_attributes_present():
    df = simple_pass_shot_df()
    data = make_xtdata(df)
    model = XTModel(l=2, w=1).fit(data)

    assert hasattr(model, "surface_")
    assert hasattr(model, "shot_probability_")
    assert hasattr(model, "goal_probability_")
    assert hasattr(model, "move_probability_")
    assert hasattr(model, "transition_matrix_")
    assert hasattr(model, "metadata_")
    assert hasattr(model, "included_move_families_")
    assert hasattr(model, "included_shot_families_")
    assert model.fitted_ is True


def test_xtdata_df_is_cached():
    df = simple_pass_shot_df()
    data = make_xtdata(df)
    assert data.df is data.df


def test_included_families_tracked():
    df = pd.DataFrame(
        {
            "x": [10, 10, 90],
            "y": [10, 10, 10],
            "end_x": [90, 90, np.nan],
            "end_y": [10, 10, np.nan],
            "event_type": ["pass", "throw_in", "shot"],
            "is_success": [True, True, True],
        }
    )
    data = make_xtdata(df)
    model = XTModel(l=2, w=1, include_throw_ins=True).fit(data)
    assert "pass" in model.included_move_families_
    assert "throw_in" in model.included_move_families_
    assert "shot" in model.included_shot_families_


# ---------------------------------------------------------------------------
# Boolean coercion (_coerce_bool)
# ---------------------------------------------------------------------------


def test_coerce_bool_native_bools():
    s = pd.Series([True, False, True])
    assert _coerce_bool(s, default=True).tolist() == [True, False, True]


def test_coerce_bool_numeric():
    s = pd.Series([1, 0, 1.0, 0.0])
    assert _coerce_bool(s, default=False).tolist() == [True, False, True, False]


def test_coerce_bool_rejects_strings():
    s = pd.Series(["True", "False", "Goal"])
    with pytest.raises(ValueError, match="Invalid values found in is_success"):
        _coerce_bool(s, default=True)


def test_coerce_bool_nan_uses_default():
    s = pd.Series([True, None, np.nan])
    assert _coerce_bool(s, default=False).tolist() == [True, False, False]
    assert _coerce_bool(s, default=True).tolist() == [True, True, True]


def test_coerce_bool_rejects_non_binary_numeric():
    s = pd.Series([1, 2, 0])
    with pytest.raises(ValueError, match="Expected booleans or numeric 0/1"):
        _coerce_bool(s, default=False)


def test_fit_with_string_success_column_raises():
    df = pd.DataFrame(
        {
            "x": [10, 10, 90],
            "y": [10, 10, 10],
            "end_x": [90, 90, np.nan],
            "end_y": [10, 10, np.nan],
            "event_type": ["pass", "pass", "shot"],
            "is_success": ["True", "False", "True"],
        }
    )
    data = make_xtdata(df)
    with pytest.raises(ValueError, match="pass success_map"):
        XTModel(l=2, w=1).fit(data)


def test_fit_with_string_success_column_via_success_map():
    df = pd.DataFrame(
        {
            "x": [10, 10, 90],
            "y": [10, 10, 10],
            "end_x": [90, 90, np.nan],
            "end_y": [10, 10, np.nan],
            "event_type": ["pass", "pass", "shot"],
            "is_success": ["True", "False", "True"],
        }
    )
    model = XTModel(l=2, w=1).fit(
        df,
        success_map={"True": True, "False": False},
    )
    cell0 = _coords_to_cell(np.array([10.0]), np.array([10.0]), 2, 1)[0]
    mp = model.move_probability_.reshape(-1)[cell0]
    sp = model.shot_probability_.reshape(-1)[cell0]
    assert np.isclose(mp, 0.5)
    assert np.isclose(sp, 0.0)


# ---------------------------------------------------------------------------
# Metadata round-trip for family lists
# ---------------------------------------------------------------------------


def test_save_load_preserves_families(tmp_path):
    df = pd.DataFrame(
        {
            "x": [10, 10, 90],
            "y": [10, 10, 10],
            "end_x": [90, 90, np.nan],
            "end_y": [10, 10, np.nan],
            "event_type": ["pass", "throw_in", "shot"],
            "is_success": [True, True, True],
        }
    )
    data = make_xtdata(df)
    model = XTModel(l=2, w=1, include_throw_ins=True).fit(data)

    path = tmp_path / "xt_families.npz"
    model.save(str(path))
    loaded = XTModel.load(str(path))

    assert loaded.included_move_families_ == model.included_move_families_
    assert loaded.included_shot_families_ == model.included_shot_families_


def test_save_handles_numpy_scalars_in_fit_schema_maps(tmp_path):
    df = pd.DataFrame(
        {
            "x": [10, 90],
            "y": [10, 10],
            "end_x": [90, np.nan],
            "end_y": [10, np.nan],
            "event_type": ["Pass", "Shot"],
            "is_success": ["Complete", "Goal"],
        }
    )
    model = XTModel(l=2, w=1).fit(
        df,
        event_map={"Pass": "pass", "Shot": "shot"},
        success_map={"Complete": np.bool_(True), "Goal": np.bool_(True)},
    )

    path = tmp_path / "xt_json_scalars.npz"
    model.save(str(path))
    loaded = XTModel.load(str(path))
    assert loaded.fitted_


# ---------------------------------------------------------------------------
# Ignored event types
# ---------------------------------------------------------------------------


def test_postmatch_penalty_ignored():
    df = pd.DataFrame(
        {
            "x": [88],
            "y": [50],
            "end_x": [np.nan],
            "end_y": [np.nan],
            "event_type": ["postmatch_penalty"],
            "is_success": [True],
        }
    )
    data = make_xtdata(df)
    model = XTModel(l=2, w=1)
    role = model._classify_events(data.df)
    assert role.iloc[0] == "ignore"


# ---------------------------------------------------------------------------
# map_events with success_map
# ---------------------------------------------------------------------------


def test_map_events_success():
    df = pd.DataFrame(
        {
            "x": [10, 90],
            "y": [10, 10],
            "raw_type": ["Pass", "Shot"],
            "outcome": ["Complete", "Goal"],
            "ex": [90, np.nan],
            "ey": [10, np.nan],
        }
    )
    data = XTData(
        events=df,
        x="x",
        y="y",
        event_type="raw_type",
        end_x="ex",
        end_y="ey",
        is_success="outcome",
    ).map_events(
        event_map={"Pass": "pass", "Shot": "shot"},
        success_map={"Complete": True, "Incomplete": False, "Goal": True},
    )
    ndf = data.df
    assert ndf["is_success"].iloc[0] == True  # noqa: E712
    assert ndf["is_success"].iloc[1] == True  # noqa: E712
