"""Serialisation helpers for xT model arrays and metadata."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

_TYPED_DICT_MARKER = "__penaltyblog_typed_dict__"


def _normalize_json_scalar(value: Any) -> str | int | float | bool | None:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _normalize_json_scalar(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        if all(isinstance(k, str) for k in value):
            return {k: _normalize_json_value(v) for k, v in value.items()}
        return {
            _TYPED_DICT_MARKER: [
                [_normalize_json_scalar(k), _normalize_json_value(v)]
                for k, v in value.items()
            ]
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(v) for v in value]
    return value


def _restore_json_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_restore_json_value(v) for v in value]
    if isinstance(value, dict):
        if set(value) == {_TYPED_DICT_MARKER}:
            return {
                _restore_json_value(k): _restore_json_value(v)
                for k, v in value[_TYPED_DICT_MARKER]
            }
        return {k: _restore_json_value(v) for k, v in value.items()}
    return value


def save_xt_npz(
    path: str, arrays: dict[str, np.ndarray], metadata: dict[str, Any]
) -> None:
    """Save xT model arrays and JSON-serializable metadata to ``.npz``."""
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


def load_xt_npz(path: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load xT model arrays and metadata from an ``.npz`` artifact."""
    try:
        with np.load(path, allow_pickle=False) as npz:
            meta_json = str(npz["meta_json"][0])
            metadata = _restore_json_value(json.loads(meta_json))
            arrays = {
                "surface": npz["surface"],
                "shot_probability": npz["shot_probability"],
                "goal_probability": npz["goal_probability"],
                "move_probability": npz["move_probability"],
                "transition_matrix": npz["transition_matrix"],
            }
    except KeyError as e:
        raise ValueError(
            f"Invalid xT model artifact: missing required array '{e.args[0]}'"
        ) from e
    return arrays, metadata
