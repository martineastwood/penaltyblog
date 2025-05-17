"""
Rating systems for football teams.
"""

from .colley import Colley  # noqa
from .elo import Elo  # noqa
from .massey import Massey  # noqa
from .pi import PiRatingSystem  # noqa

__all__ = [
    "Colley",
    "Elo",
    "Massey",
    "PiRatingSystem",
]
