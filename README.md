# Penalty Blog

The `penaltyblog` package contains code from [http://pena.lt/y/blog](http://pena.lt/y/blog) for working with football (soccer) data.

## Installation

`pip install penaltyblog`

### Predicting Goals

`penaltyblog` contains models designed for predicting the number of goals scored in football (soccer) games. Although aimed at football (soccer), they may also be useful for other sports, such as hockey.

#### The Basic Poisson Model
Let's start off by downloading some example scores from the awesome [football-data](http://football-data.co.uk) website.

```python
import penaltyblog as pb
df = pb.get_example_data()
df.head()
```

|    | Date                | HomeTeam   | AwayTeam    |   FTHG |   FTAG |
|---:|:--------------------|:-----------|:------------|-------:|-------:|
|  0 | 2011-08-13 00:00:00 | Blackburn  | Wolves      |      1 |      2 |
|  1 | 2011-08-13 00:00:00 | Fulham     | Aston Villa |      0 |      0 |
|  2 | 2011-08-13 00:00:00 | Liverpool  | Sunderland  |      1 |      1 |
|  3 | 2011-08-13 00:00:00 | Newcastle  | Arsenal     |      0 |      0 |
|  4 | 2011-08-13 00:00:00 | QPR        | Bolton      |      0 |      4 |

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
Log Likelihood: -1088.991
AIC: 2261.982

Team                 Attack               Defence             
------------------------------------------------------------
Arsenal              1.362                -1.023              
Aston Villa          0.671                -0.981              
Blackburn            0.957                -0.583              
Bolton               0.913                -0.598              
Chelsea              1.229                -1.096              
Everton              0.96                 -1.251              
Fulham               0.93                 -1.009              
Liverpool            0.898                -1.254              
Man City             1.571                -1.53               
Man United           1.531                -1.405              
Newcastle            1.084                -1.001              
Norwich              1.025                -0.747              
QPR                  0.834                -0.756              
Stoke                0.643                -0.982              
Sunderland           0.86                 -1.115              
Swansea              0.843                -1.013              
Tottenham            1.239                -1.21               
West Brom            0.866                -0.993              
Wigan                0.807                -0.819              
Wolves               0.778                -0.541              
------------------------------------------------------------
Home Advantage: 0.268
Intercept: 0.12
```

#### The Dixon and Coles Adjustment
The basic Poisson model struggles somewhat with the probabilities for low scoring games. Dixon and Coles (1997) added in an adjustment factor (rho) that modifies the probabilities for 0-0, 1-0 and 0-1 scorelines to acocunt for this.

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
Log Likelihood: -1087.359
AIC: 2260.719

Team                 Attack               Defence             
------------------------------------------------------------
Arsenal              1.371                -0.997              
Aston Villa          0.692                -0.937              
Blackburn            0.943                -0.554              
Bolton               0.92                 -0.564              
Chelsea              1.235                -1.057              
Everton              0.941                -1.239              
Fulham               0.933                -0.982              
Liverpool            0.887                -1.215              
Man City             1.559                -1.514              
Man United           1.524                -1.397              
Newcastle            1.096                -0.972              
Norwich              1.018                -0.714              
QPR                  0.821                -0.74               
Stoke                0.642                -0.961              
Sunderland           0.861                -1.083              
Swansea              0.85                 -0.98               
Tottenham            1.24                 -1.174              
West Brom            0.865                -0.971              
Wigan                0.811                -0.794              
Wolves               0.792                -0.504              
------------------------------------------------------------
Home Advantage: 0.273
Intercept: 0.089
Rho: -0.134
```


#### The Rue and Salvesen Adjustment
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
Log Likelihood: -1077.987
AIC: 2243.975

Team                 Attack               Defence             
------------------------------------------------------------
Arsenal              1.371                -1.222              
Aston Villa          0.675                -1.048              
Blackburn            0.979                -0.748              
Bolton               0.958                -0.731              
Chelsea              1.239                -1.207              
Everton              0.919                -1.347              
Fulham               0.91                 -1.111              
Liverpool            0.861                -1.319              
Man City             1.543                -1.652              
Man United           1.498                -1.571              
Newcastle            1.112                -1.118              
Norwich              1.048                -0.89               
QPR                  0.828                -0.91               
Stoke                0.618                -1.066              
Sunderland           0.846                -1.197              
Swansea              0.844                -1.102              
Tottenham            1.251                -1.314              
West Brom            0.86                 -1.096              
Wigan                0.818                -0.93               
Wolves               0.824                -0.665              
------------------------------------------------------------
Home Advantage: 0.266
Intercept: 0.232
Rho: -0.138
Gamma: 0.184
```


### Making Predictions
To make a prediction using any of the above models, just pass the name of the home and away teams to the `predict` function. This returns the `FootballProbabilityGrid` class that can convert the output from the model into probabilities for various betting markets.

```python
probs = rs.predict("Liverpool", "Stoke")
```

#### Home / Draw / Away
```python
# also known as 1x2
probs.home_draw_away
```

```
[0.5193995875820345, 0.3170596913687951, 0.1635407210315597]
```

#### Total Goals
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

#### Asian Handicaps
```python
probs.asian_handicap("home", 1.5)
```

```
0.2602616248461783
```

```python
probs.total_goals("away", -1.5)
```

```
0.7397383751361912
```