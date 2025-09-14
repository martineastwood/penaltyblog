=======================================
Implied Odds Probabilities (``implied``)
=======================================

This submodule provides a comprehensive toolkit for converting bookmaker odds into their underlying "true" probabilities. It achieves this by removing the bookmaker's margin (the overround) using various statistical methods.

The primary entry point is the ``calculate_implied()`` function, which handles all calculations and returns a structured data object.

Quick Start
===========

Here's a basic example of how to convert decimal odds into probabilities using the default ``multiplicative`` method.

.. code-block:: python

   import penaltyblog as pb

   # Odds for a Home Win, Draw, and Away Win
   odds = [2.7, 2.3, 4.4]

   result = pb.implied.calculate_implied(odds)

   print(f"Probabilities: {result.probabilities}")
   print(f"Margin: {result.margin:.4f}")
   print(f"Method Used: {result.method.value}")

.. code-block:: text

   Probabilities: [0.35873803615739097, 0.4211272598369373, 0.22013470400567173]
   Margin: 0.0324
   Method Used: multiplicative

Main Function
=============

All calculations are performed through this central function. It accepts various odds formats and allows you to specify the desired calculation method.

.. code-block:: python

   penaltyblog.implied.calculate_implied(
       odds: Union[List[float], List[str], OddsInput],
       method: Union[str, ImpliedMethod] = ImpliedMethod.MULTIPLICATIVE,
       odds_format: Union[str, OddsFormat] = OddsFormat.DECIMAL,
       market_names: Optional[List[str]] = None,
   ) -> ImpliedProbabilities

Description
-----------

Calculate implied probabilities from odds using the specified method.

**Parameters**

- **odds** (``List[float] | List[str] | OddsInput``): The odds to convert. Can be a list of values or an ``OddsInput`` object.
- **method** (``str | ImpliedMethod, optional``): The method to use. Defaults to ``"multiplicative"``.
- **odds_format** (``str | OddsFormat, optional``): The format of the provided odds. Defaults to ``"decimal"``.
- **market_names** (``List[str], optional``): Names for each market outcome.

**Returns**

``ImpliedProbabilities``: A type-safe container with the calculated probabilities and metadata.

Data Models
===========

The submodule uses type-safe dataclasses for handling odds input and probability output. This ensures your code is clear, predictable, and less prone to errors.

OddsInput
---------

The ``OddsInput`` class is used to standardize various odds formats (Decimal, American, Fractional) before calculation. While it's often used internally, you can instantiate it directly.

**Attributes**:

- **odds**: The list of original odds.
- **odds_format**: The format of the odds (e.g., ``OddsFormat.DECIMAL``).
- **market_names**: The names for each outcome.

**Methods**:

- ``to_decimal()``: Converts the stored odds to decimal format.

ImpliedProbabilities Output
---------------------------

The ``ImpliedProbabilities`` class is the return type for all calculations. It's a container holding the results and useful metadata.

**Attributes**:

- **probabilities**: A list of the calculated "true" probabilities.
- **margin**: The original margin (overround) from the bookmaker's odds.
- **method**: The method used for the calculation (e.g., ``ImpliedMethod.SHIN``).
- **market_names**: The names for each outcome.
- **method_params**: A dictionary containing any special parameters returned by the method (e.g., Shin's ``z`` or Power's ``k``).

**Properties**:

``probabilities_dict``: Returns the probabilities as a dictionary mapped to the market names.

Available Methods & Formats
============================

You can easily specify which calculation method to use and what format the input odds are in.

Calculation Methods
-------------------

The method parameter accepts a string or an ``ImpliedMethod`` enum member. Each method represents a different theory of how the bookmaker's margin is applied.

- ``MULTIPLICATIVE`` (default)
- ``ADDITIVE``
- ``POWER``
- ``SHIN``
- ``DIFFERENTIAL_MARGIN_WEIGHTING``
- ``ODDS_RATIO``
- ``LOGARITHMIC``

Odds Formats
------------

The odds_format parameter accepts a string or an ``OddsFormat`` enum member.

- ``DECIMAL`` (default)
- ``AMERICAN``
- ``FRACTIONAL``

Advanced Usage
==============

Using Different Methods and Formats
------------------------------------

Here's how to calculate probabilities from American odds using Shin's method. This example also shows how to access method-specific parameters from the result.

.. code-block:: python

   import penaltyblog as pb
   from penaltyblog.implied.models import ImpliedMethod, OddsFormat

   american_odds = ["+170", "+130", "+340"]
   market_names = ["Home", "Draw", "Away"]

   result = pb.implied.calculate_implied(
       odds=american_odds,
       method=ImpliedMethod.SHIN,
       odds_format=OddsFormat.AMERICAN,
       market_names=market_names,
   )

   print(f"Shin's Probabilities: {result.probabilities}")

   # Access method-specific parameters returned by some methods
   if result.method_params and "z" in result.method_params:
       print(f"Shin's z parameter: {result.method_params['z']:.4f}")

.. code-block:: text

   Shin's Probabilities: [0.35934391959159157, 0.42324384818283234, 0.21741223222458853]
   Shin's z parameter: 0.0162
