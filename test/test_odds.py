import pytest

from penaltyblog.betting.odds import convert_odds
from penaltyblog.implied.models import OddsFormat


def test_convert_odds_american():
    """
    Test conversion from American odds
    """
    american_odds = [+170, +130, +340]
    decimal_odds = convert_odds(american_odds, "american")

    assert decimal_odds == pytest.approx([2.7, 2.3, 4.4])

    american_odds = [-110, -120]
    decimal_odds = convert_odds(american_odds, "american")
    assert decimal_odds == pytest.approx([1.9090909090909092, 1.8333333333333333])


def test_convert_odds_fractional():
    """
    Test conversion from fractional odds
    """
    fractional_odds = ["7/4", "13/10", "7/2"]
    decimal_odds = convert_odds(fractional_odds, "fractional")
    assert decimal_odds == pytest.approx([2.75, 2.3, 4.5])


def test_convert_odds_decimal():
    """
    Test conversion from decimal odds (should be idempotent)
    """
    odds = [2.5, 3.0, 1.8]
    decimal_odds = convert_odds(odds, "decimal")
    assert decimal_odds == pytest.approx([2.5, 3.0, 1.8])


def test_convert_odds_enum():
    """
    Test conversion using OddsFormat enum
    """
    american_odds = [+170, +130, +340]
    decimal_odds = convert_odds(american_odds, OddsFormat.AMERICAN)
    assert decimal_odds == pytest.approx([2.7, 2.3, 4.4])


def test_convert_odds_invalid_format():
    """
    Test that an invalid format raises a ValueError
    """
    with pytest.raises(ValueError, match="Unknown odds format: invalid"):
        convert_odds([2.5, 3.0], "invalid")
