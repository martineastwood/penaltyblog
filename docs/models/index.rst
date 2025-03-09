Predictive Models
========================

This section provides powerful statistical models for predicting the outcomes of football (soccer) matches.

Each model can generate accurate probability estimates across various betting markets, including match results, total goals, Asian handicaps, and over/under goals.

For users interested in understanding the underlying theory of these models, links to detailed explanations and foundational research papers are provided via the the `pena.lt/y/blog`_ website:

- `Poisson model`_
- `Dixon and Coles model`_
- `Goal Expectancy`_

.. toctree::
   :maxdepth: 1
   :caption: Examples:

   poisson
   bivariate_poisson
   zero_inflated_poisson
   dixon_coles
   negative_binomial
   weibull_copula
   goal_expectancy


.. _`pena.lt/y/blog`: http://www.pena.lt/y/blog.html
.. _`Poisson model`: http://www.pena.lt/y/2021/06/18/predicting-football-results-using-the-poisson-distribution/
.. _`Dixon and Coles model`: http://www.pena.lt/y/2021/06/24/predicting-football-results-using-python-and-dixon-and-coles/
.. _`Goal Expectancy`: https://pena.lt/y/2022/12/02/goal-expectancy-from-bookmakers-odds-using-python/
