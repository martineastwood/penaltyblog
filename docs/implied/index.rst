Implied Odds
==============

.. raw:: html

   <a href="https://colab.research.google.com/drive/1o-tOetyWmSY_1WczN8WhWsl62Uz5T65F?usp=sharing" target="_blank">
      <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
   </a>
   <br><br>

Calculating implied odds from bookmaker odds is essential because bookmakers include a profit margin (overround) in their published odds, which distorts the true probabilities of match outcomes.

By removing this margin, bettors and analysts obtain more accurate estimates of the actual probabilities assigned by bookmakers.

These "implied odds" provide a clearer basis for comparing betting opportunities, evaluating the fairness of offered odds, and developing informed betting strategies or predictive models.

`penaltyblog` contains many functions for accurately extracting implied probabilities from bookmaker odds, featuring several algorithms, including Shin's method, to adjust for bookmaker margins (overrounds).

.. toctree::
   :maxdepth: 1
   :caption: Examples:

   implied
