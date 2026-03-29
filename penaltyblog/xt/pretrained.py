from __future__ import annotations

import importlib.resources as resources

from .model import XTModel

_PRETRAINED_FILES = {
    "default": "xt_default_v1.npz",
}


def load_pretrained_xt(name: str = "default") -> XTModel:
    """
    Load a bundled pretrained xT model.

    Parameters
    ----------
    name : str
        Name of the pretrained artifact. Currently ``"default"`` is
        the only supported value.
    """
    if name not in _PRETRAINED_FILES:
        available = ", ".join(sorted(_PRETRAINED_FILES))
        raise ValueError(
            f"No pretrained xT artifact named {name!r}. " f"Available: {available}"
        )

    filename = _PRETRAINED_FILES[name]
    data_path = resources.files("penaltyblog.xt").joinpath("data", filename)
    with resources.as_file(data_path) as path:
        return XTModel.load(str(path))
