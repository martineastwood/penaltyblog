========================
Odds Conversion Utilities
========================

This submodule provides a simple and reliable way to convert betting odds from different common formats into a standardized Decimal format.

Since most analytical functions in penaltyblog expect odds to be in decimal format, this utility is a crucial first step for working with data from various sources.

Supported Odds Formats
======================

The converter understands the three most common odds formats used worldwide.

- **Decimal Odds (e.g., ``2.50``, ``1.80``)**: Represents the total amount returned for a 1-unit stake, including the original stake. A stake of £10 at 2.50 odds returns £25 (£15 profit + £10 stake). This is typically the standard format in Europe, Australia, and Canada.
- **American (Moneyline) Odds (e.g., ``+150``, ``-200``)**: A positive number shows how much profit you win for a 100-unit stake (e.g., ``+150`` means you win £150 profit for a £100 stake). A negative number shows how much you must stake to win 100 units of profit (e.g., ``-200`` means you must stake £200 to win £100 profit).
- **Fractional Odds (e.g., ``"5/2"``, ``"2/1"``)**: Shows the profit relative to the stake. The first number is the profit, the second is the stake. For example, ``5/2`` (read "five to two") means you win £5 profit for every £2 you stake. This format is common in the UK and Ireland.

The ``convert_odds`` Function
=============================

This is the primary, easy-to-use function for all conversions. It takes a list of odds in a specified format and returns them as a list of decimal odds.

.. code-block:: python

   penaltyblog.betting.convert_odds(
       odds: List[Union[float, str]],
       odds_format: Union[str, OddsFormat],
       market_names: List[str] = None,
   ) -> List[float]

Parameters
----------

- **odds** (``List[Union[float, str]]``): The list of odds you want to convert. This can be numbers for Decimal/American or strings for Fractional.
- **odds_format** (``str | OddsFormat``): The format of the odds you are providing. This can be a string (e.g., ``"american"``, ``"fractional"``) or an ``OddsFormat`` enum member.
- **market_names** (``List[str], optional``): Optional names for each market outcome; this parameter is included for API consistency but is not used in the conversion calculation.

Returns
-------

``List[float]``: A new list containing the odds converted to the **Decimal** format.

Usage Examples
==============

Converting American Odds to Decimal
------------------------------------

.. code-block:: python

   import penaltyblog as pb

   american_odds = [+170, +130, -110]

   decimal_odds = pb.betting.convert_odds(american_odds, "american")

   print(f"American: {american_odds}")
   print(f"Decimal:  {decimal_odds}")

.. code-block:: text

   American: [170, 130, -110]
   Decimal:  [2.7, 2.3, 1.9090909090909092]

Converting Fractional Odds to Decimal
--------------------------------------

.. code-block:: python

   import penaltyblog as pb

   fractional_odds = ['7/4', '13/10', '7/2']

   decimal_odds = pb.betting.convert_odds(fractional_odds, "fractional")

   print(f"Fractional: {fractional_odds}")
   print(f"Decimal:    {decimal_odds}")

.. code-block:: text

   Fractional: ['7/4', '13/10', '7/2']
   Decimal:    [2.75, 2.3, 4.5]
