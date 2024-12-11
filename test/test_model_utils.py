import numpy as np
import pandas as pd

import penaltyblog as pb


def test_rho_correction_vec():
    df = pd.DataFrame(
        {
            "goals_home": [0, 0, 1, 1],
            "goals_away": [0, 1, 0, 1],
            "home_exp": [0.5, 0.5, 0.5, 0.5],
            "away_exp": [0.5, 0.5, 0.5, 0.5],
            "rho": [0.2, 0.2, 0.2, 0.2],
        }
    )
    dc_adj = pb.models.utils.rho_correction_vec(df)
    assert np.allclose(dc_adj, [0.95, 1.1, 1.1, 0.8])


def test_rho_correction():
    assert pb.models.utils.rho_correction(0, 0, 0.5, 0.5, 0.2) == 0.95
    assert pb.models.utils.rho_correction(0, 1, 0.5, 0.5, 0.2) == 1.1
    assert pb.models.utils.rho_correction(1, 0, 0.5, 0.5, 0.2) == 1.1
    assert pb.models.utils.rho_correction(1, 1, 0.5, 0.5, 0.2) == 0.8
