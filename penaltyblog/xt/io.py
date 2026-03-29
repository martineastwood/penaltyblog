from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np


def save_xt_npz(
    path: str, arrays: Dict[str, np.ndarray], metadata: Dict[str, Any]
) -> None:
    meta_json = json.dumps(metadata, sort_keys=True)
    np.savez(
        path,
        surface=arrays["surface"],
        shot_probability=arrays["shot_probability"],
        goal_probability=arrays["goal_probability"],
        move_probability=arrays["move_probability"],
        transition_matrix=arrays["transition_matrix"],
        meta_json=np.array([meta_json]),
    )


def load_xt_npz(path: str) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    with np.load(path, allow_pickle=False) as npz:
        meta_json = str(npz["meta_json"][0])
        metadata = json.loads(meta_json)
        arrays = {
            "surface": npz["surface"],
            "shot_probability": npz["shot_probability"],
            "goal_probability": npz["goal_probability"],
            "move_probability": npz["move_probability"],
            "transition_matrix": npz["transition_matrix"],
        }
    return arrays, metadata
