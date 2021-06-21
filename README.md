# Penalty Blog

The `penaltyblog` package contains code from [http://pena.lt/y/blog](http://pena.lt/y/blog) for working with football (soccer) data.

## Requirements

    - python >=3.6
    - numpy >=1.19.2
    - pandas >=1.1.3
    - scipy >=1.5.0

## Installation

`pip install penaltyblog`


## Example

There are examples of all the functions available in the [Examples section](https://github.com/martineastwood/penaltyblog/tree/master/examples).

## Download Data from football-data.co.uk

`penaltyblog` contains some helper functions for downloading data from [football-data.co.uk](http://football-data.co.uk).


### List the countries available 

```python
import penaltyblog as pb
pd.footballdata.list_countries()
```

```
['belgium',
 'england',
 'france',
 'germany',
 'greece',
 'italy',
 'portugal',
 'scotland',
 'spain',
 'turkey']
```

### Fetch the data

The first parameter is the country of interest, the second is the starting year of the season and the third paramater is the level of the division of interest, where `0` is the highest division (e.g. England's Premier League), `1` is the second highest (e.g. England's Championship) etc.

```python
df = pb.footballdata.fetch_data("england", 2018, 0)
df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].head()
```

|    | Date                | HomeTeam     | AwayTeam       |   FTHG |   FTAG |
|---:|:--------------------|:-------------|:---------------|-------:|-------:|
|  0 | 2018-08-10 00:00:00 | Man United   | Leicester      |      2 |      1 |
|  1 | 2018-08-11 00:00:00 | Bournemouth  | Cardiff        |      2 |      0 |
|  2 | 2018-08-11 00:00:00 | Fulham       | Crystal Palace |      0 |      2 |
|  3 | 2018-08-11 00:00:00 | Huddersfield | Chelsea        |      0 |      3 |
|  4 | 2018-08-11 00:00:00 | Newcastle    | Tottenham      |      1 |      2 |

## Predicting Goals

`penaltyblog` contains models designed for predicting the number of goals scored in football (soccer) games. Although aimed at football (soccer), they may also be useful for other sports, such as hockey.

### The Basic Poisson Model

Let's start off by downloading some example scores from the awesome [football-data](http://football-data.co.uk) website.

```python
import penaltyblog as pb
df = pb.footballdata.fetch_data("England", 2018, 0)
df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].head()
```

|    | Date                | HomeTeam     | AwayTeam       |   FTHG |   FTAG |
|---:|:--------------------|:-------------|:---------------|-------:|-------:|
|  0 | 2018-08-10 00:00:00 | Man United   | Leicester      |      2 |      1 |
|  1 | 2018-08-11 00:00:00 | Bournemouth  | Cardiff        |      2 |      0 |
|  2 | 2018-08-11 00:00:00 | Fulham       | Crystal Palace |      0 |      2 |
|  3 | 2018-08-11 00:00:00 | Huddersfield | Chelsea        |      0 |      3 |
|  4 | 2018-08-11 00:00:00 | Newcastle    | Tottenham      |      1 |      2 |

Next, we create a basic Poisson model and fit it to the data.

```python
pois = pb.poisson.PoissonGoalsModel(
    df["FTHG"], df["FTAG"], df["HomeTeam"], df["AwayTeam"])
pois.fit()
```

Let's take a look at the fitted parameters.

```python
pois
```

```
Module: Penaltyblog

Model: Poisson

Number of parameters: 41
Log Likelihood: -1065.077
AIC: 2212.154

Team                 Attack               Defence             
------------------------------------------------------------
Arsenal              1.362                -0.856              
Bournemouth          1.115                -0.555              
Brighton             0.634                -0.731              
Burnley              0.894                -0.595              
Cardiff              0.614                -0.592              
Chelsea              1.202                -1.135              
Crystal Palace       1.004                -0.839              
Everton              1.055                -0.978              
Fulham               0.626                -0.431              
Huddersfield         0.184                -0.507              
Leicester            1.0                  -0.939              
Liverpool            1.532                -1.683              
Man City             1.598                -1.633              
Man United           1.249                -0.807              
Newcastle            0.805                -0.948              
Southampton          0.891                -0.641              
Tottenham            1.264                -1.131              
Watford              1.03                 -0.731              
West Ham             1.026                -0.801              
Wolves               0.916                -0.985              
------------------------------------------------------------
Home Advantage: 0.225
```

### The Dixon and Coles Adjustment

The basic Poisson model struggles somewhat with the probabilities for low scoring games. Dixon and Coles (1997) added in an adjustment factor (rho) that modifies the probabilities for 0-0, 1-0 and 0-1 scorelines to account for this.

```python
dc = pb.poisson.DixonColesGoalModel(
    df["FTHG"], df["FTAG"], df["HomeTeam"], df["AwayTeam"])
dc.fit()
dc
```

```
Module: Penaltyblog

Model: Dixon and Coles

Number of parameters: 42
Log Likelihood: -1064.943
AIC: 2213.886

Team                 Attack               Defence             
------------------------------------------------------------
Arsenal              1.36                 -0.858              
Bournemouth          1.115                -0.555              
Brighton             0.632                -0.733              
Burnley              0.897                -0.592              
Cardiff              0.615                -0.591              
Chelsea              1.205                -1.13               
Crystal Palace       1.007                -0.837              
Everton              1.054                -0.977              
Fulham               0.625                -0.433              
Huddersfield         0.18                 -0.507              
Leicester            0.996                -0.94               
Liverpool            1.534                -1.679              
Man City             1.599                -1.638              
Man United           1.251                -0.807              
Newcastle            0.806                -0.946              
Southampton          0.897                -0.636              
Tottenham            1.259                -1.137              
Watford              1.031                -0.729              
West Ham             1.023                -0.803              
Wolves               0.914                -0.988              
------------------------------------------------------------
Home Advantage: 0.225
Rho: -0.041
```


### The Rue and Salvesen Adjustment

Rue and Salvesen (1999) added in an additional psycological effect factor (gamma) where Team A will under-estimate Team B if Team A is stronger than team B. They also truncate scorelines to a maximum of five goals, e.g. a score of 7-3 becomes 5-3, stating that any goals above 5 are non-informative.

```python
rs = pb.poisson.RueSalvesenGoalModel(
    df["FTHG"], df["FTAG"], df["HomeTeam"], df["AwayTeam"])

rs.fit()
rs
```

```
Module: Penaltyblog

Model: Rue Salvesen

Number of parameters: 43
Log Likelihood: -1061.167
AIC: 2208.334

Team                 Attack               Defence             
------------------------------------------------------------
Arsenal              1.496                -0.988              
Bournemouth          1.27                 -0.705              
Brighton             0.559                -0.655              
Burnley              0.964                -0.654              
Cardiff              0.585                -0.556              
Chelsea              1.183                -1.129              
Crystal Palace       1.027                -0.852              
Everton              1.034                -0.974              
Fulham               0.653                -0.455              
Huddersfield         0.023                -0.359              
Leicester            0.978                -0.916              
Liverpool            1.445                -1.585              
Man City             1.503                -1.571              
Man United           1.367                -0.917              
Newcastle            0.72                 -0.854              
Southampton          0.942                -0.693              
Tottenham            1.243                -1.131              
Watford              1.097                -0.79               
West Ham             1.062                -0.836              
Wolves               0.85                 -0.919              
------------------------------------------------------------
Home Advantage: 0.222
Rho: -0.04
Gamma: 0.692
```


### Making Predictions

To make a prediction using any of the above models, just pass the name of the home and away teams to the `predict` function. This returns the `FootballProbabilityGrid` class that can convert the output from the model into probabilities for various betting markets.

```python
probs = rs.predict("Watford", "Wolves")
probs
```

```
Module: Penaltyblog

Class: FootballProbabilityGrid

Home Goal Expectation: 1.3094663275662697
Away Goal Expectation: 1.2096084394688094

Home Win: 0.3843462702759874
Draw: 0.2787256663458056
Away Win: 0.3369280633610962
```

### Home / Draw / Away

```python
# also known as 1x2
probs.home_draw_away
```

```python
[0.3843462702759874, 0.2787256663458056, 0.3369280633610962]
```

### Total Goals

```python
probs.total_goals("over", 2.5)
```

```python
0.4610704441088047
```

```python
probs.total_goals("under", 2.5)
```

```python
0.5389295558740843
```

### Asian Handicaps

```python
probs.asian_handicap("home", 1.5)
```

```python
0.17531437781532913
```

```python
probs.asian_handicap("away", -1.5)
```

```python
0.8246856221675609
```

### Model Parameters

You can access the model's parameters via the `get_params` function.

```python
from pprint import pprint
params = my_model.get_params()
pprint(params)
```

```python
{'attack_Arsenal': 1.4960574633781003,
 'attack_Bournemouth': 1.2701540413261327,
 'attack_Brighton': 0.559186251363228,
 'attack_Burnley': 0.9644520899122194,
 'attack_Cardiff': 0.5847648397569006,
 'attack_Chelsea': 1.1828466188120765,
 'attack_Crystal Palace': 1.0273361069287597,
 'attack_Everton': 1.0335248035400801,
 'attack_Fulham': 0.6531864264818924,
 'attack_Huddersfield': 0.023109559960240708,
 'attack_Leicester': 0.977933119588144,
 'attack_Liverpool': 1.4451581320799645,
 'attack_Man City': 1.5025454369883477,
 'attack_Man United': 1.366845541477835,
 'attack_Newcastle': 0.720009733703693,
 'attack_Southampton': 0.9416226570416543,
 'attack_Tottenham': 1.2427047093744437,
 'attack_Watford': 1.096790079793436,
 'attack_West Ham': 1.06163359275858,
 'attack_Wolves': 0.8501387957342722,
 'defence_Arsenal': -0.9879049844176601,
 'defence_Bournemouth': -0.704968272653022,
 'defence_Brighton': -0.6545658993274335,
 'defence_Burnley': -0.6541299575160815,
 'defence_Cardiff': -0.5555542344325824,
 'defence_Chelsea': -1.128898069212659,
 'defence_Crystal Palace': -0.8518829374985971,
 'defence_Everton': -0.9742632965054263,
 'defence_Fulham': -0.45545035895833286,
 'defence_Huddersfield': -0.3586836043107179,
 'defence_Leicester': -0.915653821531362,
 'defence_Liverpool': -1.5850200706445228,
 'defence_Man City': -1.5713140731733608,
 'defence_Man United': -0.9165982110339421,
 'defence_Newcastle': -0.8538889602642802,
 'defence_Southampton': -0.6925502345992922,
 'defence_Tottenham': -1.1307038809506598,
 'defence_Watford': -0.7898505955782175,
 'defence_West Ham': -0.8356435683761823,
 'defence_Wolves': -0.9188112323803922,
 'home_advantage': 0.22164932659641978,
 'rho': -0.04033232667301132,
 'rue_salvesen': 0.6922490800541602}
```

## Ratings

### Massey Ratings

Calculates the overall [Massey ratings](https://en.wikipedia.org/wiki/Kenneth_Massey), plus Massey attack and defence ratings too.

```python
import penaltyblog as pb

df = pb.footballdata.fetch_data("england", 2020, 0)
pb.ratings.massey(df["FTHG"], df["FTAG"], df["HomeTeam"], df["AwayTeam"])
```

|    | team             |   rating |    offence |   defence |
|---:|:-----------------|---------:|-----------:|----------:|
|  0 | Man City         |    1.275 |  1.48618   | -0.211184 |
|  1 | Man United       |    0.725 |  1.23896   | -0.513962 |
|  2 | Liverpool        |    0.65  |  1.10424   | -0.45424  |
|  3 | Tottenham        |    0.575 |  1.10841   | -0.533406 |
|  4 | Chelsea          |    0.55  |  0.832018  | -0.282018 |
|  5 | Leicester        |    0.45  |  1.11535   | -0.665351 |
|  6 | Arsenal          |    0.4   |  0.757018  | -0.357018 |
|  7 | West Ham         |    0.375 |  0.952851  | -0.577851 |
|  8 | Aston Villa      |    0.225 |  0.76674   | -0.54174  |
|  9 | Leeds            |    0.2   |  0.962573  | -0.762573 |
| 10 | Everton          |   -0.025 |  0.558406  | -0.583406 |
| 11 | Brighton         |   -0.15  |  0.370906  | -0.520906 |
| 12 | Wolves           |   -0.4   |  0.273684  | -0.673684 |
| 13 | Newcastle        |   -0.4   |  0.551462  | -0.951462 |
| 14 | Southampton      |   -0.525 |  0.586184  | -1.11118  |
| 15 | Burnley          |   -0.55  |  0.198684  | -0.748684 |
| 16 | Crystal Palace   |   -0.625 |  0.425073  | -1.05007  |
| 17 | Fulham           |   -0.65  |  0.0375731 | -0.687573 |
| 18 | West Brom        |   -1.025 |  0.280629  | -1.30563  |
| 19 | Sheffield United |   -1.075 | -0.13326   | -0.94174  |

### Colley Ratings

Calculates [Colley ratings](https://en.wikipedia.org/wiki/Colley_Matrix). Since Colley ratings don't explicitly define how to handle tied results, you can set whether to include them and how much weighting they should receive compared to a win.

```python
import penaltyblog as pb

df = pb.footballdata.fetch_data("england", 2020, 0)
pb.ratings.colley(df["FTHG"], df["FTAG"], df["HomeTeam"], df["AwayTeam"], include_draws=True, draw_weight=1/3)
```

|    | team             |   rating |
|---:|:-----------------|---------:|
|  0 | Man City         | 1.42857  |
|  1 | Man United       | 1.38095  |
|  2 | Liverpool        | 1.3254   |
|  3 | Chelsea          | 1.31746  |
|  4 | West Ham         | 1.28571  |
|  5 | Leicester        | 1.27778  |
|  6 | Tottenham        | 1.2619   |
|  7 | Arsenal          | 1.24603  |
|  8 | Everton          | 1.2381   |
|  9 | Leeds            | 1.21429  |
| 10 | Aston Villa      | 1.19841  |
| 11 | Brighton         | 1.14286  |
| 12 | Newcastle        | 1.13492  |
| 13 | Wolves           | 1.13492  |
| 14 | Crystal Palace   | 1.11905  |
| 15 | Southampton      | 1.10317  |
| 16 | Burnley          | 1.0873   |
| 17 | Fulham           | 1.03175  |
| 18 | West Brom        | 1        |
| 19 | Sheffield United | 0.904762 |


```python
import penaltyblog as pb

df = pb.footballdata.fetch_data("england", 2020, 0)
pb.ratings.colley(df["FTHG"], df["FTAG"], df["HomeTeam"], df["AwayTeam"], include_draws=False)
```

|    | team             |   rating |
|---:|:-----------------|---------:|
|  0 | Man City         | 0.75     |
|  1 | Man United       | 0.678571 |
|  2 | Liverpool        | 0.630952 |
|  3 | Chelsea          | 0.619048 |
|  4 | Leicester        | 0.595238 |
|  5 | West Ham         | 0.595238 |
|  6 | Tottenham        | 0.571429 |
|  7 | Arsenal          | 0.559524 |
|  8 | Everton          | 0.547619 |
|  9 | Leeds            | 0.535714 |
| 10 | Aston Villa      | 0.511905 |
| 11 | Newcastle        | 0.440476 |
| 12 | Wolves           | 0.440476 |
| 13 | Brighton         | 0.428571 |
| 14 | Crystal Palace   | 0.428571 |
| 15 | Southampton      | 0.416667 |
| 16 | Burnley          | 0.392857 |
| 17 | Fulham           | 0.321429 |
| 18 | West Brom        | 0.297619 |
| 19 | Sheffield United | 0.238095 |


## Implied Probabilities

Removes the overround and gets the implied probabilities from odds via a variety of methods

### Multiplicative

Normalizes the probabilites so they sum to 1.0 by dividing the inverse of the odds by the sum of the inverse of the odds

```python
import penaltyblog as pb

odds = [2.7, 2.3, 4.4]
pb.implied.multiplicative(odds)
```

```python
{'implied_probabilities': array([0.35873804, 0.42112726, 0.2201347 ]),
 'margin': 0.03242570633874986,
 'method': 'multiplicative'}
```

### Additive

Normalizes the probabilites so they sum to 1.0 by removing an equal amount from each

```python
import penaltyblog as pb

odds = [2.7, 2.3, 4.4]
pb.implied.additive(odds)
```

```python
{'implied_probabilities': array([0.3595618 , 0.42397404, 0.21646416]),
 'margin': 0.03242570633874986,
 'method': 'additive'}
```

### Power

Solves for the power coefficient that normalizes the inverse of the odds to sum to 1.0

```python
import penaltyblog as pb

odds = [2.7, 2.3, 4.4]
pb.implied.power(odds)
```

```python
{'implied_probabilities': array([0.3591711 , 0.42373075, 0.21709815]),
 'margin': 0.03242570633874986,
 'method': 'power',
 'k': 1.0309132393169356}
 ```

### Shin

Uses the Shin (1992, 1993) method to calculate the implied probabilities

```python
import penaltyblog as pb

odds = [2.7, 2.3, 4.4]
pb.implied.shin(odds)
```

```python
{'implied_probabilities': array([0.35934392, 0.42324385, 0.21741223]),
 'margin': 0.03242570633874986,
 'method': 'shin',
 'z': 0.016236442857291165}
 ```

### Differential Margin Weighting

Uses the differential margin approach described by Joesph Buchdahl in his `wisdom of the crowds` article

```python
import penaltyblog as pb

odds = [2.7, 2.3, 4.4]
pb.implied.differential_margin_weighting(odds)
```

```python
{'implied_probabilities': array([0.3595618 , 0.42397404, 0.21646416]),
 'margin': 0.03242570633874986,
 'method': 'differential_margin_weighting'}
 ```

### Odds Ratio

Uses Keith Cheung's odds ratio approach, as discussed by Joesph Buchdahl's in his `wisdom of the crowds` article, to calculate the implied probabilities

```python
import penaltyblog as pb

odds = [2.7, 2.3, 4.4]
pb.implied.odds_ratio(odds)
```

```python
{'implied_probabilities': array([0.35881036, 0.42256142, 0.21862822]),
 'margin': 0.03242570633874986,
 'method': 'odds_ratio',
 'c': 1.05116912729218}
 ```

## Rank Probability Scores

Based on Constantinou and Fenton (2021), `penaltyblog` contains a function for calculating Rank Probability Scores for assessing home, draw, away probability forecasts.

`predictions` is a list of home, draw, away probabilities and `observed` is the zero-based index for which outcome actually occurred.

```python
import penaltyblog as pb

predictions = [
    [1, 0, 0],
    [0.9, 0.1, 0],
    [0.8, 0.1, 0.1],
    [0.5, 0.25, 0.25],
    [0.35, 0.3, 0.35],
    [0.6, 0.3, 0.1],
    [0.6, 0.25, 0.15],
    [0.6, 0.15, 0.25],
    [0.57, 0.33, 0.1],
    [0.6, 0.2, 0.2],
]

observed = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]

for p, o in zip(predictions, observed):
    rps = pb.metrics.rps(p, o)
    print(round(rps, 4))
```

```python
0.0
0.005
0.025
0.1562
0.1225
0.185
0.0913
0.1113
0.0975
0.1
```

## Download ELO rating from clubelo.com

### Download ELO ratings for a given date

```python
import penaltyblog as pb
df = pb.clubelo.fetch_rankings_by_date(2010, 1, 1)
df.head()
```

|    |   Rank | Club        | Country   |   Level |     Elo | From                | To                  |
|---:|-------:|:------------|:----------|--------:|--------:|:--------------------|:--------------------|
|  0 |      1 | Barcelona   | ESP       |       1 | 1987.68 | 2009-12-18 00:00:00 | 2010-01-02 00:00:00 |
|  1 |      2 | Chelsea     | ENG       |       1 | 1945.54 | 2009-12-29 00:00:00 | 2010-01-16 00:00:00 |
|  2 |      3 | Man United  | ENG       |       1 | 1928.53 | 2009-12-31 00:00:00 | 2010-01-09 00:00:00 |
|  3 |      4 | Real Madrid | ESP       |       1 | 1902.72 | 2009-12-20 00:00:00 | 2010-01-03 00:00:00 |
|  4 |      5 | Inter       | ITA       |       1 | 1884.49 | 2009-12-21 00:00:00 | 2010-01-06 00:00:00 |

### List all teams with ratings available

```python
import penaltyblog as pb
teams = pb.clubelo.list_all_teams()
teams[:5]
```

```python
['Man City', 'Bayern', 'Liverpool', 'Real Madrid', 'Man United']
```

### Download Historical ELO ratings for a given team

```python
import penaltyblog as pb
df = pb.clubelo.fetch_rankings_by_team("barcelona")
df.head()
```

|    | Rank   | Club      | Country   |   Level |     Elo | From                | To                  |
|---:|:-------|:----------|:----------|--------:|--------:|:--------------------|:--------------------|
|  0 | None   | Barcelona | ESP       |       1 | 1636.7  | 1939-10-22 00:00:00 | 1939-12-03 00:00:00 |
|  1 | None   | Barcelona | ESP       |       1 | 1626.1  | 1939-12-04 00:00:00 | 1939-12-10 00:00:00 |
|  2 | None   | Barcelona | ESP       |       1 | 1636.73 | 1939-12-11 00:00:00 | 1939-12-17 00:00:00 |
|  3 | None   | Barcelona | ESP       |       1 | 1646.95 | 1939-12-18 00:00:00 | 1939-12-24 00:00:00 |
|  4 | None   | Barcelona | ESP       |       1 | 1637.42 | 1939-12-25 00:00:00 | 1939-12-31 00:00:00 |

## References

- Mark J. Dixon and Stuart G. Coles (1997) Modelling Association Football Scores and Inefficiencies in the Football Betting Market.
- Håvard Rue and Øyvind Salvesen (1999) Prediction and Retrospective Analysis of Soccer Matches in a League.
- Anthony C. Constantinou and Norman E. Fenton (2012) Solving the problem of inadequate scoring rules for assessing probabilistic football forecast models
- Hyun Song Shin (1992) Prices of State Contingent Claims with Insider Traders, and the Favourite-Longshot Bias
- Hyun Song Shin (1993) Measuring the Incidence of Insider Trading in a Market for State-Contingent Claims
- Joseph Buchdahl (2015) The Wisdom of the Crowd
