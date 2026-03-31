---
name: penaltyblog
description: Use when helping users write code with the penaltyblog Python package — goal models, xT, betting math, MatchFlow pipelines, implied odds, scrapers, FPL, ratings, backtesting, visualizations, and scoring metrics. Focus on the public API and provide minimal runnable examples.
---

# Penaltyblog Agent Skill

## Scope

This skill is for coding assistants helping users with the `penaltyblog` Python package. Prioritize the public API and stable workflows for:

- Goal and betting models in `penaltyblog.models`
- Expected Threat (xT) in `penaltyblog.xt`
- Betting mathematics in `penaltyblog.betting`
- MatchFlow pipelines in `penaltyblog.matchflow`
- Implied odds calculations in `penaltyblog.implied`
- Data scrapers in `penaltyblog.scrapers`
- Fantasy Premier League in `penaltyblog.fpl`
- Team rating systems in `penaltyblog.ratings`
- Strategy backtesting in `penaltyblog.backtest`
- Pitch visualizations and diagnostics in `penaltyblog.viz`
- Scoring metrics in `penaltyblog.metrics`

Internal modules (`penaltyblog.bayes`, `penaltyblog.utils`) are not part of the public API. Only reference them if explicitly asked.

## Quick Orientation

- Import pattern: `import penaltyblog as pb`
- Public modules: `pb.models`, `pb.xt`, `pb.betting`, `pb.matchflow`, `pb.implied`, `pb.scrapers`, `pb.fpl`, `pb.ratings`, `pb.backtest`, `pb.viz`, `pb.metrics`
- Package root: `penaltyblog/`

## How To Respond

- Provide short, runnable Python snippets.
- If data is required, ask for the user's dataset or create a minimal placeholder and label it clearly.
- Default to the simplest correct model or method; explain how to swap to alternatives.
- Prefer public classes/functions in module `__init__` exports.

---

## Models

Use this when a user asks to fit or compare goal models, generate probabilities, or access betting markets from model outputs.

### Public Entry Points

- `pb.models.PoissonGoalsModel`
- `pb.models.DixonColesGoalModel`
- `pb.models.BivariatePoissonGoalModel`
- `pb.models.NegativeBinomialGoalModel`
- `pb.models.ZeroInflatedPoissonGoalsModel`
- `pb.models.WeibullCopulaGoalsModel`
- `pb.models.BayesianGoalModel`
- `pb.models.HierarchicalBayesianGoalModel`
- `pb.models.dixon_coles_weights`

### Common API

All goal models share these methods:

- `fit()`
- `predict(home_team, away_team, max_goals=10, normalize=True)`
- `predict_many(home_teams, away_teams, max_goals=10, normalize=True)`
- `get_params()`
- `save(path)` / `load(path)`

### Minimal Example

```python
import penaltyblog as pb

model = pb.models.PoissonGoalsModel(
    train["goals_home"],
    train["goals_away"],
    train["team_home"],
    train["team_away"],
)
model.fit()

prediction = model.predict("Arsenal", "Manchester City")
print(prediction.home_draw_away)
```

### Time Weighting

Use Dixon-Coles exponential decay weights when recent matches should matter more:

```python
weights = pb.models.dixon_coles_weights(train["date"], xi=0.001)
model = pb.models.PoissonGoalsModel(
    train["goals_home"],
    train["goals_away"],
    train["team_home"],
    train["team_away"],
    weights=weights,
)
model.fit()
```

### Probability Grid Outputs

`model.predict()` returns a `FootballProbabilityGrid` with ready-to-use markets:

- `home_draw_away`
- `both_teams_to_score`
- `total_goals("over", 2.5)`
- `asian_handicap("home", -0.5)`

---

## Betting

Use this when a user asks about Kelly Criterion, value bets, arbitrage, or odds conversion.

### Public Entry Points

- `pb.betting.kelly_criterion`
- `pb.betting.multiple_kelly_criterion`
- `pb.betting.identify_value_bet`
- `pb.betting.calculate_bet_value`
- `pb.betting.find_arbitrage_opportunities`
- `pb.betting.arbitrage_hedge`
- `pb.betting.convert_odds`

### Minimal Example

```python
import penaltyblog as pb

result = pb.betting.kelly_criterion(
    decimal_odds=2.5,
    true_prob=0.45,
    kelly_fraction=0.5,
)
print(result.stake, result.edge, result.expected_growth)
```

### Value Bets

```python
result = pb.betting.identify_value_bet(
    bookmaker_odds=2.5,
    estimated_probability=0.45,
    kelly_fraction=0.5,
)
print(result.is_value_bet, result.edge, result.expected_value)
```

### Arbitrage Detection

```python
# outcomes_odds_matrix: rows = outcomes, columns = bookmakers
odds_matrix = [
    [2.10, 2.05, 2.15],  # Home win odds from 3 bookmakers
    [3.40, 3.50, 3.30],  # Draw odds
    [3.60, 3.70, 3.50],  # Away win odds
]
result = pb.betting.find_arbitrage_opportunities(odds_matrix)
print(result.has_arbitrage, result.guaranteed_return)
```

### Odds Conversion

```python
# Convert American odds to decimal
decimal = pb.betting.convert_odds(["+170", "+130", "+340"], odds_format="american")
```

---

## MatchFlow

Use this when a user asks to process event data, build a pipeline, or stream from StatsBomb/Opta.

### Public Entry Points

- `pb.matchflow.Flow`
- `pb.matchflow.where_equals` and other predicates

### Core Ideas

- Flow is a lazy, stream-first query engine for nested JSON.
- Transformations are lazy until you call `.collect()` or `.to_pandas()`.
- You can filter, select, group, and summarize without flattening early.

### Minimal Example

```python
from penaltyblog.matchflow import Flow, where_equals

flow = (
    Flow.statsbomb.events(match_id=19716)
    .filter(where_equals("type.name", "Shot"))
    .select("player.name", "location", "shot.statsbomb_xg")
)

for shot in flow.head(5):
    print(shot)
```

### Common Methods

- `filter(...)`, `assign(...)`, `select(...)`
- `group_by(...).summary(...)`
- `collect()`, `to_pandas()`, `to_jsonl()`

### Predicates

- `where_equals`, `where_not_equals`, `where_in`
- `where_gt`, `where_gte`, `where_lt`, `where_lte`
- `where_exists`, `where_is_null`, `where_contains`
- Combine with `and_`, `or_`, `not_`

---

## xT (Expected Threat)

Use this when a user asks to fit or score position-based expected threat models from event data.

### Public Entry Points

- `pb.xt.XTModel`
- `pb.xt.XTData`
- `pb.xt.load_pretrained_xt`

### Input Types

- `XTModel.fit(...)` and `XTModel.score(...)` accept:
  - `XTData`
  - `pandas.DataFrame`
  - `penaltyblog.matchflow.Flow`
- `XTData(events=...)` accepts:
  - `pandas.DataFrame`
  - `penaltyblog.matchflow.Flow`

### Minimal Example

```python
import penaltyblog as pb

xt = pb.xt.XTModel(l=16, w=12, coord_policy="warn")
xt.fit(df)
scored = xt.score(df)
print(scored[["xt_start", "xt_end", "xt_added"]].head())
```

### With MatchFlow

```python
from penaltyblog.matchflow import Flow
from penaltyblog.xt import XTModel

flow = Flow.from_records(records)
xt = XTModel()
xt.fit(flow)
scored = xt.score(flow)
```

### XTData Mapping Example

```python
from penaltyblog.xt import XTData, XTModel

data = XTData(
    events=df,
    x="location_x",
    y="location_y",
    event_type="type",
    end_x="end_x",
    end_y="end_y",
    is_success="outcome",
).map_events(
    event_map={"Pass": "pass", "Shot": "shot"},
    success_map={"Complete": True, "Incomplete": False, "Goal": True},
)

model = XTModel().fit(data)
```

---

## Implied Odds

Use this when a user asks to convert bookmaker odds into implied probabilities.

### Public Entry Points

- `pb.implied.calculate_implied`
- `pb.implied.ImpliedMethod`
- `pb.implied.OddsFormat`

### Minimal Example

```python
import penaltyblog as pb

odds = [2.7, 2.3, 4.4]
result = pb.implied.calculate_implied(odds)
print(result.probabilities, result.margin)
```

### Methods

String or enum — `MULTIPLICATIVE` (default), `ADDITIVE`, `POWER`, `SHIN`, `DIFFERENTIAL_MARGIN_WEIGHTING`, `ODDS_RATIO`, `LOGARITHMIC`.

### Odds Formats

`DECIMAL` (default), `AMERICAN`, `FRACTIONAL`.

### Advanced Example

```python
from penaltyblog.implied.models import ImpliedMethod, OddsFormat

american_odds = ["+170", "+130", "+340"]
result = pb.implied.calculate_implied(
    odds=american_odds,
    method=ImpliedMethod.SHIN,
    odds_format=OddsFormat.AMERICAN,
    market_names=["Home", "Draw", "Away"],
)
print(result.probabilities, result.method_params)
```

---

## Scrapers

Use this when a user asks to fetch football data from external sources.

### Public Entry Points

- `pb.scrapers.FBRef(competition, season, team_mappings=None)`
- `pb.scrapers.Understat(competition, season, team_mappings=None)`
- `pb.scrapers.FootballData(competition, season, team_mappings=None)`
- `pb.scrapers.ClubElo(team_mappings=None)`
- `pb.scrapers.get_example_team_name_mappings()`

### Minimal Example

```python
import penaltyblog as pb

fbref = pb.scrapers.FBRef("EPL", "2024-2025")
fixtures = fbref.get_fixtures()
print(fixtures.head())
```

### Available Methods

**FBRef**: `get_fixtures()`, `get_stats(stat_type="standard")`, `list_stat_types()`

**Understat**: `get_fixtures()`, `get_shots(match_id)`, `get_fixture_info(match_id)`, `get_player_season(player_id)`, `get_player_shots(player_id)`

**FootballData**: `get_fixtures()`

**ClubElo**: `get_elo_by_date(date=None)`, `get_elo_by_team(team)`, `get_team_names()`

### Team Name Mappings

Standardize team names across data sources:

```python
mappings = pb.scrapers.get_example_team_name_mappings()
fbref = pb.scrapers.FBRef("EPL", "2024-2025", team_mappings=mappings)
understat = pb.scrapers.Understat("EPL", "2024", team_mappings=mappings)
```

---

## Fantasy Premier League (FPL)

Use this when a user asks about FPL data, player stats, or team optimization.

### Public Entry Points

All are module-level functions returning DataFrames:

- `pb.fpl.get_current_gameweek()`
- `pb.fpl.get_gameweek_info()`
- `pb.fpl.get_player_id_mappings()`
- `pb.fpl.get_player_data()`
- `pb.fpl.get_player_history(player_id)`
- `pb.fpl.get_rankings(page=1)`
- `pb.fpl.get_entry_picks_by_gameweek(entry_id, gameweek=1)`
- `pb.fpl.get_entry_transfers(entry_id)`
- `pb.fpl.optimise_team(formation="2-5-5-3", budget=100)`

### Minimal Example

```python
import penaltyblog as pb

players = pb.fpl.get_player_data()
print(players[["web_name", "total_points", "now_cost"]].head(10))
```

### Team Optimization

```python
solution, team = pb.fpl.optimise_team(formation="2-5-5-3", budget=100)
print(team)
```

---

## Ratings

Use this when a user asks about team rating systems (Elo, Colley, Massey, Pi).

### Public Entry Points

- `pb.ratings.Elo(k=20.0, home_field_advantage=100.0)`
- `pb.ratings.Colley(goals_home, goals_away, teams_home, teams_away, include_draws=True, draw_weight=0.5)`
- `pb.ratings.Massey(goals_home, goals_away, teams_home, teams_away)`
- `pb.ratings.PiRatingSystem(alpha=0.15, beta=0.10, k=0.75, sigma=1.0)`

### Minimal Example (Elo)

```python
import penaltyblog as pb

elo = pb.ratings.Elo(k=20.0, home_field_advantage=100.0)

# Update after each match (result: 0=home win, 1=draw, 2=away win)
elo.update_ratings("Arsenal", "Chelsea", 0)
elo.update_ratings("Chelsea", "Liverpool", 2)

print(elo.get_team_rating("Arsenal"))
print(elo.calculate_match_probabilities("Arsenal", "Chelsea"))
```

### Batch Ratings (Colley / Massey)

```python
colley = pb.ratings.Colley(
    goals_home=df["goals_home"],
    goals_away=df["goals_away"],
    teams_home=df["team_home"],
    teams_away=df["team_away"],
)
print(colley.get_ratings())

massey = pb.ratings.Massey(
    goals_home=df["goals_home"],
    goals_away=df["goals_away"],
    teams_home=df["team_home"],
    teams_away=df["team_away"],
)
print(massey.get_ratings())  # columns: team, rating, offence, defence
```

### Notes

- Elo and PiRatingSystem are incremental — update match by match.
- Colley and Massey are batch — pass full fixture lists at once.

---

## Backtest

Use this when a user asks to test a betting strategy on historical data.

### Public Entry Points

- `pb.backtest.Backtest(data, start_date, end_date, stop_at_negative=False)`
- `pb.backtest.Account(bankroll)`
- `pb.backtest.Context` — passed to logic/trainer functions

### Minimal Example

```python
import penaltyblog as pb

def logic(ctx):
    # ctx.fixture is the current match row
    # ctx.lookback is historical data up to this point
    # ctx.account lets you place bets
    if ctx.fixture["predicted_home_prob"] > 0.6:
        ctx.account.place_bet(
            odds=ctx.fixture["odds_home"],
            stake=10,
            outcome=1 if ctx.fixture["result"] == "H" else 0,
        )

bt = pb.backtest.Backtest(
    data=fixtures_df,  # must have a "date" column
    start_date="2024-01-01",
    end_date="2024-12-31",
)
bt.start(bankroll=1000, logic=logic)
print(bt.results())
```

### Results

`bt.results()` returns a dict with: Total Bets, Successful Bets, Successful Bet %, Max/Min Bankroll, Profit, ROI.

### Optional Trainer

Pass a `trainer` function to retrain a model at each date:

```python
def trainer(ctx):
    model = pb.models.PoissonGoalsModel(...)
    model.fit()
    ctx.model = model

bt.start(bankroll=1000, logic=logic, trainer=trainer)
```

---

## Visualization

Use this when a user asks to plot football pitches or MCMC diagnostics.

### Public Entry Points

- `pb.viz.Pitch(provider="statsbomb", width=600, height=500, theme="minimal", orientation="horizontal", view="full")`
- `pb.viz.Theme(preset_name)` — presets: `classic`, `night`, `retro`, `minimal`, `turf`
- `pb.viz.plot_trace(model)`, `pb.viz.plot_posterior(model)`, `pb.viz.plot_autocorr(model)`, `pb.viz.plot_convergence(model)`, `pb.viz.plot_diagnostics(model)`

### Pitch Example

```python
import penaltyblog as pb

pitch = pb.viz.Pitch(provider="statsbomb", theme="minimal")
# Add layers to the pitch, then display
fig = pitch.figure
fig.show()
```

### Bayesian Diagnostics

```python
model = pb.models.BayesianGoalModel(...)
model.fit()

fig = pb.viz.plot_diagnostics(model)
fig.show()
```

### Notes

- All visualizations return Plotly `Figure` objects.
- Pitch providers: `statsbomb`, `wyscout`, `opta`, etc.
- Views: `"full"`, `"left"`, `"right"`, or coordinate tuples.

---

## Metrics

Use this when a user asks to score or evaluate model predictions.

### Public Entry Points

- `pb.metrics.rps_array(predictions, observed)`
- `pb.metrics.rps_average(predictions, observed)`
- `pb.metrics.multiclass_brier_score(predictions, observed)`
- `pb.metrics.ignorance_score(predictions, observed)`

### Minimal Example

```python
import penaltyblog as pb

predictions = [
    [0.6, 0.3, 0.1],
    [0.2, 0.3, 0.5],
]
observed = [0, 2]  # 0=home win, 1=draw, 2=away win

print(pb.metrics.rps_average(predictions, observed))
print(pb.metrics.multiclass_brier_score(predictions, observed))
print(pb.metrics.ignorance_score(predictions, observed))
```
