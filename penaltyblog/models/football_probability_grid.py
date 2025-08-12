from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class FootballProbabilityGrid:
    """
    Probability grid over exact football (soccer) scores P(H=i, A=j).

    This class wraps a matrix of exact-score probabilities and provides
    fast, vectorised access to common betting/analytics markets (1X2,
    totals with push handling, Asian handicaps including quarter lines,
    BTTS, double chance, DNB, etc.), along with convenient distributions
    (team goals, total goals) and expected points.

    Backward compatibility:
    - Retains the original `total_goals(over_under, strike)` and
      `asian_handicap(home_away, strike)` methods/signatures/semantics.
    - Keeps attribute names used previously (`grid`, `home_goal_expectation`,
      `away_goal_expectation`).

    Parameters
    ----------
    goal_matrix : NDArray
        2D array where entry [h, a] is P(Home goals = h, Away goals = a).
        Rows index home goals; columns index away goals.
    home_goal_expectation : float
        Expected number of home goals (E[H]).
    away_goal_expectation : float
        Expected number of away goals (E[A]).
    normalize : bool, optional (default=False)
        If True, normalises `goal_matrix` to sum to 1. If False, assumes
        it already sums (approximately) to 1.

    Notes
    -----
    - All markets are derived directly from the same scoreline grid, so
      probabilities are internally consistent by construction.
    - Computations use NumPy boolean masks and lightweight caching for
      speed when properties are accessed repeatedly.
    """

    goal_matrix: NDArray
    home_goal_expectation: float
    away_goal_expectation: float
    normalize: bool = True

    # Internal numeric tolerance
    _tolerance: float = 1e-9

    def __post_init__(self) -> None:
        grid = np.asarray(self.goal_matrix, dtype=np.float64)
        if grid.ndim != 2 or grid.size == 0:
            raise ValueError("goal_matrix must be a non-empty 2D array")
        if np.any(grid < -self._tolerance):
            raise ValueError("goal_matrix contains negative probabilities")

        total = grid.sum()
        if self.normalize:
            if total <= 0:
                raise ValueError("goal_matrix sums to zero; cannot normalize")
            grid = grid / total
        else:
            # Allow slight deviations if user disabled normalization.
            if not np.isclose(total, 1.0, atol=1e-6):
                # Soft fail: keep as is to preserve backward behaviour.
                # In future, I may add a warning in here?
                pass

        self.grid = grid  # keep original attribute name for back-compat
        # Precompute index grids for fast vectorised masks
        self._I, self._J = np.indices(self.grid.shape)
        # Simple cache for frequently used sums
        self._cache: Dict[str, float] = {}

    # ---------------------------------------------------------------------
    # Representation
    # ---------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            "Module: Penaltyblog\n\n"
            "Class: FootballProbabilityGrid\n\n"
            f"Home Goal Expectation: {self.home_goal_expectation:.3f}\n"
            f"Away Goal Expectation: {self.away_goal_expectation:.3f}\n\n"
            f"Home Win: {self.home_win:.4f}\n"
            f"Draw: {self.draw:.4f}\n"
            f"Away Win: {self.away_win:.4f}\n"
        )

    # ---------------------------------------------------------------------
    # Core 1X2 markets
    # ---------------------------------------------------------------------
    @property
    def home_win(self) -> float:
        """Probability of a home win (H > A)."""
        return self._sum_mask("home_win", self._I > self._J)

    @property
    def draw(self) -> float:
        """Probability of a draw (H = A)."""
        return self._sum_mask("draw", self._I == self._J)

    @property
    def away_win(self) -> float:
        """Probability of an away win (A > H)."""
        return self._sum_mask("away_win", self._I < self._J)

    @property
    def home_draw_away(self) -> List[float]:
        """1X2 probabilities as [P(Home), P(Draw), P(Away)]."""
        return [self.home_win, self.draw, self.away_win]

    # ---------------------------------------------------------------------
    # Both Teams To Score and complements
    # ---------------------------------------------------------------------
    @property
    def btts_yes(self) -> float:
        """Probability that both teams score (H > 0 and A > 0)."""
        return self._sum_mask("btts", (self._I > 0) & (self._J > 0))

    @property
    def btts_no(self) -> float:
        """Probability that NOT both teams score (at least one team scores 0)."""
        return 1.0 - self.btts_yes

    # ---------------------------------------------------------------------
    # Convenience markets often used by analysts
    # ---------------------------------------------------------------------
    @property
    def double_chance_1x(self) -> float:
        """Double chance 1X: P(Home or Draw)."""
        return self.home_win + self.draw

    @property
    def double_chance_x2(self) -> float:
        """Double chance X2: P(Draw or Away)."""
        return self.draw + self.away_win

    @property
    def double_chance_12(self) -> float:
        """Double chance 12: P(Home or Away)."""
        return self.home_win + self.away_win

    @property
    def draw_no_bet_home(self) -> float:
        """
        Draw No Bet (Home) win probability, conditional on the bet not pushing.

        Interpreted as P(Home wins | not Draw) = P(Home) / (1 - P(Draw)).
        Useful for deriving fair DNB odds; Draw returns stake.
        """
        return self.home_win / (1.0 - self.draw) if self.draw < 1.0 else 0.0

    @property
    def draw_no_bet_away(self) -> float:
        """
        Draw No Bet (Away) win probability, conditional on the bet not pushing.

        Interpreted as P(Away wins | not Draw) = P(Away) / (1 - P(Draw)).
        """
        return self.away_win / (1.0 - self.draw) if self.draw < 1.0 else 0.0

    # ---------------------------------------------------------------------
    # Totals (Over/Under) with push handling
    # ---------------------------------------------------------------------
    def totals(self, line: float) -> Tuple[float, float, float]:
        """
        Compute Under/Push/Over probabilities for a totals line.

        Parameters
        ----------
        line : float
            Totals line (e.g., 2.0, 2.5, 3.0). Integer lines can push.

        Returns
        -------
        (under, push, over) : tuple of float
            Probabilities that total goals are < line, == line (if integer),
            or > line, respectively. For half-lines, push = 0.
        """
        s = self._I + self._J
        under = self.grid[s < line].sum()
        push = self.grid[s == line].sum() if float(line).is_integer() else 0.0
        over = self.grid[s > line].sum()
        return float(under), float(push), float(over)

    def total_goals(self, over_under: str, strike: float) -> float:
        """
        Backward-compatible Over/Under probability (without push).

        Parameters
        ----------
        over_under : {'over', 'under'}
            Side of the market.
        strike : float
            Totals line (e.g., 2.0, 2.5, 3.0).

        Returns
        -------
        float
            Probability of 'over' or 'under' at the given line (push excluded).

        Notes
        -----
        For integer lines (e.g., 2.0), this returns P(total > line) for 'over'
        and P(total < line) for 'under'. If you need the push probability as
        well, use `totals(strike)`.
        """
        under, push, over = self.totals(strike)
        if over_under == "over":
            return over
        elif over_under == "under":
            return under
        else:
            raise ValueError("over_under must be 'over' or 'under'")

    # ---------------------------------------------------------------------
    # Asian Handicap with proper settlement (win / push / lose)
    # ---------------------------------------------------------------------
    def asian_handicap_probs(self, side: str, line: float) -> Dict[str, float]:
        """
        Asian handicap settlement probabilities for Win/Push/Lose.

        Supports integer, half, and quarter lines (split stakes).

        Parameters
        ----------
        side : {'home', 'away'}
            Which team receives the handicap.
        line : float
            Handicap line, e.g., -0.5, +0.25, -1.0, +1.75, etc.
            Positive lines favour the chosen side (they receive goals).

        Returns
        -------
        dict
            Dictionary with keys {'win', 'push', 'lose'} giving the
            probabilities that the bet wins, pushes, or loses.

        Notes
        -----
        Quarter lines (±0.25, ±0.75, etc.) are treated as half-stakes on the
        adjacent half-lines (e.g., -0.25 = 50% at 0.0 and 50% at -0.5).
        """
        if side not in {"home", "away"}:
            raise ValueError("side must be 'home' or 'away'")

        # Goal difference from the perspective of the chosen side
        gd = (self._I - self._J) if side == "home" else (self._J - self._I)

        def single_line_probs(l: float) -> Tuple[float, float, float]:
            # Settlement against a single line (win/push/lose)
            win = self.grid[gd > l].sum()
            push = self.grid[gd == l].sum() if float(l).is_integer() else 0.0
            lose = self.grid[gd < l].sum()
            return float(win), float(push), float(lose)

        frac = line - np.floor(line)
        # Handle negative fractions consistently (e.g., -0.25)
        if frac < 0:
            frac += 1.0
            base = np.floor(line) - 1.0
        else:
            base = np.floor(line)

        # Quarter-line handling (split across neighbouring half-lines)
        if np.isclose(frac, 0.25) or np.isclose(frac, 0.75):
            lower = base + (0.0 if frac < 0.5 else 0.5)
            upper = lower + 0.5
            w1, p1, l1 = single_line_probs(lower)
            w2, p2, l2 = single_line_probs(upper)
            return {
                "win": 0.5 * (w1 + w2),
                "push": 0.5 * (p1 + p2),
                "lose": 0.5 * (l1 + l2),
            }
        else:
            w, p, l = single_line_probs(line)
            return {"win": w, "push": p, "lose": l}

    def asian_handicap(self, home_away: str, strike: float) -> float:
        """
        Backward-compatible Asian handicap 'win' probability (no push).

        Parameters
        ----------
        home_away : {'home', 'away'}
            Which side you are backing on the handicap.
        strike : float
            Handicap line.

        Returns
        -------
        float
            Probability the bet settles as a win (excludes push and lose).

        Notes
        -----
        For full Win/Push/Lose breakdown (especially for integer lines or
        quarter lines), prefer `asian_handicap_probs(...)`.
        """
        if home_away not in {"home", "away"}:
            raise ValueError("home_away must be 'home' or 'away'")
        return self.asian_handicap_probs(home_away, strike)["win"]

    # ---------------------------------------------------------------------
    # Distributions & exact scores
    # ---------------------------------------------------------------------
    def exact_score(self, h: int, a: int) -> float:
        """
        Probability of an exact scoreline.

        Parameters
        ----------
        h : int
            Home goals.
        a : int
            Away goals.

        Returns
        -------
        float
            P(H = h, A = a). Returns 0.0 if (h, a) lies outside the grid.
        """
        if h < 0 or a < 0:
            return 0.0
        if h >= self.grid.shape[0] or a >= self.grid.shape[1]:
            return 0.0
        return float(self.grid[h, a])

    def home_goal_distribution(self) -> NDArray:
        """
        Marginal distribution over home goals.

        Returns
        -------
        np.ndarray
            Vector `p[h] = P(H = h)`.
        """
        return self.grid.sum(axis=1)

    def away_goal_distribution(self) -> NDArray:
        """
        Marginal distribution over away goals.

        Returns
        -------
        np.ndarray
            Vector `p[a] = P(A = a)`.
        """
        return self.grid.sum(axis=0)

    def total_goals_distribution(self) -> NDArray:
        """
        Distribution over total goals T = H + A.

        Returns
        -------
        np.ndarray
            Vector `p[t] = P(T = t)` for t = 0..(Hmax + Amax).
        """
        rows, cols = self.grid.shape
        max_total = rows + cols - 2
        out = np.zeros(max_total + 1, dtype=np.float64)
        # Sum anti-diagonals where h + a = t
        for t in range(max_total + 1):
            h = np.arange(max(0, t - (cols - 1)), min(rows - 1, t) + 1)
            a = t - h
            out[t] = self.grid[h, a].sum()
        return out

    # ---------------------------------------------------------------------
    # Clean sheets, win to nil, expected points
    # ---------------------------------------------------------------------
    def win_to_nil_home(self) -> float:
        """
        Probability the home team wins to nil (A = 0 and H > 0).

        Returns
        -------
        float
        """
        # Sum over (h > 0, a = 0)
        return float(self.grid[1:, 0].sum())

    def win_to_nil_away(self) -> float:
        """
        Probability the away team wins to nil (H = 0 and A > 0).

        Returns
        -------
        float
        """
        # Sum over (h = 0, a > 0)
        return float(self.grid[0, 1:].sum())

    def expected_points_home(self) -> float:
        """
        Expected points for the home team under 3/1/0 scoring.

        Returns
        -------
        float
            3 * P(Home win) + 1 * P(Draw).
        """
        return 3.0 * self.home_win + 1.0 * self.draw

    def expected_points_away(self) -> float:
        """
        Expected points for the away team under 3/1/0 scoring.

        Returns
        -------
        float
            3 * P(Away win) + 1 * P(Draw).
        """
        return 3.0 * self.away_win + 1.0 * self.draw

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _sum_mask(self, key: str, mask: NDArray) -> float:
        """
        Sum grid entries under a boolean mask, with lightweight caching.

        Parameters
        ----------
        key : str
            Cache key for the computed sum.
        mask : np.ndarray of bool
            Boolean mask over `grid` of the same shape.

        Returns
        -------
        float
            Sum of `grid[mask]`.
        """
        if key not in self._cache:
            self._cache[key] = float(self.grid[mask].sum())
        return self._cache[key]
