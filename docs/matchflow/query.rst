================
The Query Method
================

The ``Flow`` class includes a powerful ``.query()`` method, enabling you to filter records using a concise, string-based expression. This improves the readability and flexibility of your data processing pipelines.

You can specify complex filtering conditions using standard Python comparison and logical operators, field access, and even built-in functions or string methods.

Use ``.query()`` when you want to:

- Prototype filters quickly.
- Write more readable and maintainable pipeline logic.
- Let users define filters without writing custom Python code.

How it Works
============

The ``.query()`` method parses a string expression using Python's Abstract Syntax Tree (AST) module. It then converts this AST into an efficient predicate function, which is applied to each record in the ``Flow`` stream. This approach ensures security (as it doesn't use ``eval()`` directly on arbitrary input) and allows for robust validation of the query syntax.

Basic Comparisons
=================

You can compare record fields to literal values using standard comparison operators:

- ``==`` (equals)
- ``!=`` (not equals)
- ``>`` (greater than)
- ``>=`` (greater than or equal to)
- ``<`` (less than)
- ``<=`` (less than or equal to)

.. code-block:: python

   # Example: Filter for matches with more than 3 goals
   flow.query("goals > 3")

   # Example: Filter for matches played by 'Man City'
   flow.query("home_team == 'Man City'")

Logical Operators
=================

Combine multiple conditions using ``and``, ``or``, and ``not``:

- ``and``
- ``or``
- ``not``

.. code-block:: python

   # Example: Home wins for 'Liverpool'
   flow.query("home_team == 'Liverpool' and home_goals > away_goals")

   # Example: Matches not involving 'Arsenal'
   flow.query("not (home_team == 'Arsenal' or away_team == 'Arsenal')")

Field Access
============

Access nested fields using dot notation:

- ``field.subfield``

.. code-block:: python

   # Example: Filter based on nested 'venue.city' field
   flow.query("venue.city == 'London'")

Membership Operators (``in``, ``not in``)
=========================================

Check if a field's value is present in a list or tuple:

- ``in``
- ``not in``

.. code-block:: python

   # Example: Filter for matches involving specific teams
   flow.query("home_team in ['Chelsea', 'Tottenham']")

   # Example: Filter for matches NOT involving a specific league
   flow.query("league not in ['Premier League', 'La Liga']")

.. warning::
   ``in`` and ``not in`` require the field to appear on the left-hand side of the expression. Reverse usage is not currently supported (e.g., "Man City" in home_team will raise an error).

Checking for NULLs
==================

Check for null/missing values:

- ``is None``
- ``is not None``

.. code-block:: python

   # Example: Find records where 'player.injury_status' is null
   flow.query("player.injury_status is None")

   # Example: Find records where 'player.injury_status' is not null
   flow.query("player.injury_status is not None")

String Methods
==============

Apply common string transformations for comparison. Note these are used *within* a comparison:

- ``len()``: Get the length of a string or list/tuple.
- ``.lower()``: Convert a string to lowercase.
- ``.upper()``: Convert a string to uppercase.

.. code-block:: python

   # Example: Find teams whose name is exactly 'manchester united' (case-insensitive)
   flow.query("home_team.lower() == 'manchester united'")

   # Example: Find teams with a short name
   flow.query("len(home_team) < 8")

Predicate-Style String Methods (Standalone)
===========================================

Directly check string properties using method calls:

- ``.contains(substring)``
- ``.startswith(prefix)``
- ``.endswith(suffix)``
- ``.regex(pattern, flags)`` or ``.match(pattern, flags)``

.. code-block:: python

   # Example: Find home teams containing 'united'
   flow.query("home_team.contains('united')")

   # Example: Find away teams starting with 'West'
   flow.query("away_team.startswith('West')")

   # Example: match player name using a regex
   flow.query("player.name.regex('^Mo')")

Referencing Local Python Variables (``@var``)
=============================================

To make your queries dynamic, you can inject external Python variables using the ``@`` symbol. This allows you to construct queries programmatically while maintaining readability. For example, ``@team_name`` will be replaced with the actual value of the variable ``team_name`` from your Python scope.

This is especially useful when working with date ranges, parameterized filters, or reusable queries.

.. code-block:: python

   import datetime

   min_goals = 2
   team_name = "Liverpool"
   start_date = datetime.date(2023, 1, 1)

   # Example: Using numeric and string variables
   flow.query("home_goals >= @min_goals and home_team == @team_name")

   # Example: Using a date object
   flow.query("match_date >= @start_date")

For regular expressions, you should pass flags such as ``re.IGNORECASE`` or ``re.MULTILINE`` by referencing them the same way:

.. code-block:: python

   import re

   pattern = r"liverpool"
   flags = re.IGNORECASE

   # Example: matching a string using a regular expression
   flow.query("home_team.regex(@pattern, @flags)")

Remember:

- Regex flags must be passed as values from the ``re`` module.
- The query parser substitutes ``@var`` with safe, scoped values - no arbitrary code execution occurs.

Filtering by Date and Time
==========================

You can filter results using ``datetime()`` and ``date()`` objects from Python's built-in ``datetime`` module.
These can be used directly in your query strings to create date or datetime values for comparison.

.. code-block:: python

   # Example: Matches after a specific date
   flow.query("match_date > date(2024, 6, 30)")
