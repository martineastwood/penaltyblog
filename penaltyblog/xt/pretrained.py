"""Pretrained xT model artifacts bundled with the package."""

from __future__ import annotations

import importlib.resources as resources

from .model import XTModel

_PRETRAINED_FILES = {
    "default": "xt_default_v1.npz",
}


def load_pretrained_xt(name: str = "default") -> XTModel:
    """
    Load a bundled pretrained xT model.

    This is the quickest way to get started — no training data required.

    .. note::
        The pretrained ``"default"`` model was fitted on Opta-format event data
        from multiple professional seasons across several European leagues.
        It uses a 16 × 12 pitch grid with passes, carries, throw-ins,
        goal kicks, corners, and free kicks included.

        For best results in a specific competition, era, or tactical context,
        train your own model with ``XTModel().fit(your_data)`` instead.

    Parameters
    ----------
    name : str
        Name of the pretrained artifact. Currently ``"default"`` is the only
        supported value.

    Returns
    -------
    XTModel
        A fitted :class:`XTModel` instance ready for scoring, querying, or
        plotting.  No call to :meth:`~XTModel.fit` is needed.

    Examples
    --------
    >>> import penaltyblog as pb
    >>> model = pb.xt.load_pretrained_xt()
    >>> model.value_at(85, 50)   # xT near the penalty spot (0–100 coordinates)
    >>> scored = model.score(df)
    >>> pitch = model.plot()
    >>> pitch.fig.show()
    """
    if name not in _PRETRAINED_FILES:
        available = ", ".join(sorted(_PRETRAINED_FILES))
        raise ValueError(
            f"No pretrained xT artifact named {name!r}. " f"Available: {available}"
        )

    filename = _PRETRAINED_FILES[name]
    data_path = resources.files("penaltyblog.xt").joinpath("data").joinpath(filename)
    with resources.as_file(data_path) as path:
        return XTModel.load(str(path))
