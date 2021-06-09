# Penalty Blog

The `penaltyblog` package contains code from [http://pena.lt/y/blog](http://pena.lt/y/blog) for working with football (soccer) data.

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
pois = pb.PoissonGoalsModel(
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

Number of parameters: 42
Log Likelihood: -1065.077
AIC: 2214.154

Team                 Attack               Defence             
------------------------------------------------------------
Arsenal              1.362                -1.062              
Bournemouth          1.115                -0.761              
Brighton             0.634                -0.937              
Burnley              0.894                -0.801              
Cardiff              0.614                -0.798              
Chelsea              1.202                -1.341              
Crystal Palace       1.004                -1.045              
Everton              1.055                -1.184              
Fulham               0.626                -0.637              
Huddersfield         0.184                -0.712              
Leicester            0.999                -1.145              
Liverpool            1.532                -1.889              
Man City             1.598                -1.839              
Man United           1.249                -1.013              
Newcastle            0.805                -1.153              
Southampton          0.891                -0.846              
Tottenham            1.264                -1.337              
Watford              1.03                 -0.937              
West Ham             1.026                -1.007              
Wolves               0.916                -1.191              
------------------------------------------------------------
Home Advantage: 0.225
Intercept: 0.206
```

### The Dixon and Coles Adjustment

The basic Poisson model struggles somewhat with the probabilities for low scoring games. Dixon and Coles (1997) added in an adjustment factor (rho) that modifies the probabilities for 0-0, 1-0 and 0-1 scorelines to account for this.

```python
dc = pb.DixonColesGoalModel(
    df["FTHG"], df["FTAG"], df["HomeTeam"], df["AwayTeam"])
dc.fit()
dc
```

```
Module: Penaltyblog

Model: Dixon and Coles

Number of parameters: 43
Log Likelihood: -1064.943
AIC: 2215.886

Team                 Attack               Defence             
------------------------------------------------------------
Arsenal              1.36                 -0.982              
Bournemouth          1.115                -0.679              
Brighton             0.632                -0.858              
Burnley              0.897                -0.717              
Cardiff              0.615                -0.715              
Chelsea              1.205                -1.254              
Crystal Palace       1.007                -0.961              
Everton              1.054                -1.102              
Fulham               0.625                -0.557              
Huddersfield         0.18                 -0.631              
Leicester            0.996                -1.064              
Liverpool            1.534                -1.803              
Man City             1.599                -1.762              
Man United           1.251                -0.931              
Newcastle            0.806                -1.07               
Southampton          0.897                -0.761              
Tottenham            1.259                -1.261              
Watford              1.031                -0.854              
West Ham             1.023                -0.927              
Wolves               0.914                -1.113              
------------------------------------------------------------
Home Advantage: 0.225
Intercept: 0.124
Rho: -0.041
```


### The Rue and Salvesen Adjustment

Rue and Salvesen (1999) added in an additional psycological effect factor (gamma) where Team A will under-estimate Team B if Team A is stronger than team B. They also truncate scorelines to a maximum of five goals, e.g. a score of 7-3 becomes 5-3, stating that any goals above 5 are non-informative.

```python
rs = pb.RueSalvesenGoalModel(
    df["FTHG"], df["FTAG"], df["HomeTeam"], df["AwayTeam"])

rs.fit()
rs
```

```
Module: Penaltyblog

Model: Rue Salvesen

Number of parameters: 44
Log Likelihood: -1061.167
AIC: 2210.334

Team                 Attack               Defence             
------------------------------------------------------------
Arsenal              1.435                -1.068              
Bournemouth          1.2                  -0.776              
Brighton             0.594                -0.831              
Burnley              0.935                -0.766              
Cardiff              0.6                  -0.712              
Chelsea              1.194                -1.281              
Crystal Palace       1.019                -0.985              
Everton              1.044                -1.126              
Fulham               0.641                -0.585              
Huddersfield         0.096                -0.573              
Leicester            0.988                -1.067              
Liverpool            1.487                -1.768              
Man City             1.533                -1.743              
Man United           1.315                -1.006              
Newcastle            0.761                -1.036              
Southampton          0.921                -0.814              
Tottenham            1.244                -1.274              
Watford              1.067                -0.902              
West Ham             1.045                -0.961              
Wolves               0.881                -1.091              
------------------------------------------------------------
Home Advantage: 0.222
Intercept: 0.141
Rho: -0.04
Gamma: 0.373
```


### Making Predictions

To make a prediction using any of the above models, just pass the name of the home and away teams to the `predict` function. This returns the `FootballProbabilityGrid` class that can convert the output from the model into probabilities for various betting markets.

```python
probs = my_model.predict("Liverpool", "Stoke")
```

### Home / Draw / Away

```python
# also known as 1x2
probs.home_draw_away
```

```
[0.5193995875820345, 0.3170596913687951, 0.1635407210315597]
```

### Total Goals

```python
probs.total_goals("over", 2.5)
```

```
0.31911650768322447
```

```python
probs.total_goals("under", 2.5)
```

```
0.680883492299145
```

### Asian Handicaps

```python
probs.asian_handicap("home", 1.5)
```

```
0.2602616248461783
```

```python
probs.asian_handicap("away", -1.5)
```

```
0.7397383751361912
```

### Model Parameters

You can access the model's parameters via the `get_params` function.

```python
from pprint import pprint
params = my_model.get_params()
pprint(params)
```

```
{'attack_Arsenal': 1.3650671020694474,
 'attack_Aston Villa': 0.6807140182913024,
 'attack_Blackburn': 0.971135574781119,
 'attack_Bolton': 0.9502712140456423,
 'attack_Chelsea': 1.235466344414206,
 'attack_Everton': 0.9257685468926837,
 'attack_Fulham': 0.9122902202053228,
 'attack_Liverpool': 0.8684673939949753,
 'attack_Man City': 1.543379586931267,
 'attack_Man United': 1.4968564161865994,
 'attack_Newcastle': 1.1095636706231062,
 'attack_Norwich': 1.0424304866584615,
 'attack_QPR': 0.827439335780754,
 'attack_Stoke': 0.6248927873330669,
 'attack_Sunderland': 0.8510292333101492,
 'attack_Swansea': 0.8471368133406263,
 'attack_Tottenham': 1.2496040004504756,
 'attack_West Brom': 0.8625207332372105,
 'attack_Wigan': 0.8177807129177644,
 'attack_Wolves': 0.8181858085358248,
 'defence_Arsenal': -1.2192247076852236,
 'defence_Aston Villa': -1.0566859588325535,
 'defence_Blackburn': -0.7430288162188969,
 'defence_Bolton': -0.7268011436918458,
 'defence_Chelsea': -1.2065700516830344,
 'defence_Everton': -1.3564763976122773,
 'defence_Fulham': -1.1159544166204092,
 'defence_Liverpool': -1.3293118049518535,
 'defence_Man City': -1.6549894606952225,
 'defence_Man United': -1.5728126940204685,
 'defence_Newcastle': -1.1186158411320268,
 'defence_Norwich': -0.8865413401238464,
 'defence_QPR': -0.9124617361500764,
 'defence_Stoke': -1.0766419199030601,
 'defence_Sunderland': -1.2049421203955355,
 'defence_Swansea': -1.1077243368907703,
 'defence_Tottenham': -1.3160823704397775,
 'defence_West Brom': -1.1014569193066301,
 'defence_Wigan': -0.932997180492951,
 'defence_Wolves': -0.6618461794219439,
 'home_advantage': 0.2655860528422758,
 'intercept': 0.23467961435272489,
 'rho': -0.1375912978446625,
 'rue_salvesen': 0.1401430558820631}
```

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
    rps = pb.rps(p, o)
    print(round(rps, 4))
```

```
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

```
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
