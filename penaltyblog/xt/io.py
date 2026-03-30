from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np


def _normalize_json_key(key: Any) -> str | int | float | bool | None:
    if isinstance(key, np.generic):
        key = key.item()
    if isinstance(key, (str, int, float, bool)) or key is None:
        return key
    return str(key)


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {
            _normalize_json_key(k): _normalize_json_value(v) for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(v) for v in value]
    return value


def save_xt_npz(
    path: str, arrays: Dict[str, np.ndarray], metadata: Dict[str, Any]
) -> None:
    normalized_metadata = _normalize_json_value(metadata)
    meta_json = json.dumps(normalized_metadata, sort_keys=True)
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
