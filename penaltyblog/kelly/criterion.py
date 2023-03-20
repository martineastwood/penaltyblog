def criterion(decimal_odds: float, true_prob: float, fraction: float = 1) -> float:
    """
    The Kelly Criterion is a formula that determines the
    optimal theoretical size for a bet.

    https://en.wikipedia.org/wiki/Kelly_criterion

    Parameters
    ----------
    decimal_odds : float
        The odds for the event in European decimal format, e.g. 1.50

    true_prob : float
        The true probability of the event e.g. 0-1

    fraction : float
        The fraction of the Kelly Criterion to use. A value of 1.0 gives the full
        Kelly, 0.5 is a half Kelly etc. Reducing the fraction reduces the amount
        recommended to wager while reducing volatility

    Returns
    -------
    float
        The recomended fraction of the bank roll to wager

    Examples
    --------
    >>> criterion(1.5, 0.7, 1/3)
    """
    crit = ((true_prob * decimal_odds) - 1) / (decimal_odds - 1)
    return crit * fraction
