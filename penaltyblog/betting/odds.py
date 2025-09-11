"""
Odds Conversion Utilities
"""

from typing import List, Optional, Union

from ..implied.models import OddsFormat, OddsInput


def convert_odds(
    odds: List[Union[float, str]],
    odds_format: Union[str, OddsFormat],
    market_names: Optional[List[str]] = None,
) -> List[float]:
    """
    Converts odds from a specified format to decimal odds.

    This is a convenience function that wraps the functionality from the
    `penaltyblog.implied` submodule.

    Parameters
    ----------
    odds : List[Union[float, str]]
        The odds to convert.
    odds_format : str or OddsFormat
        The format of the provided odds.
    market_names : List[str], optional
        Names for each market outcome.

    Returns
    -------
    List[float]
        The odds converted to decimal format.

    Examples
    --------
    >>> from penaltyblog.betting import convert_odds
    >>> american_odds = [+170, +130, +340]
    >>> convert_odds(american_odds, "american")
    [2.7, 2.3, 4.4]
    >>> fractional_odds = ['7/4', '13/10', '7/2']
    >>> convert_odds(fractional_odds, "fractional")
    [2.75, 2.3, 4.5]
    """
    if isinstance(odds_format, str):
        try:
            odds_format = OddsFormat(odds_format)
        except ValueError:
            raise ValueError(f"Unknown odds format: {odds_format}")

    odds_input = OddsInput(values=odds, format=odds_format, market_names=market_names)
    return odds_input.to_decimal()
