========================================
Filtering Data with Predicates and Helpers
========================================

Flow includes a powerful set of **predicate helpers** that make it easier to filter nested, irregular JSON records without writing complex lambda functions every time.

These functions are designed to be:

- ✅ **Readable**: No more long inline lambdas
- ✅ **Safe**: Handle nested paths, nulls, and edge cases
- ✅ **Composable**: Chain filters with ``and_()``, ``or_()``, ``not_()``

✨ Why Use Predicates?
======================

Instead of writing:

.. code-block:: python

   flow.filter(lambda r: r.get("type") == "Shot" and r["xg"] > 0.1)

You can write:

.. code-block:: python

   from penaltyblog.matchflow import where_equals, where_gt, and_

   flow.filter(and_(
       where_equals("type", "Shot"),
       where_gt("xg", 0.1)
   ))

Cleaner, safer, and easier to maintain.

🔍 Core Predicate Helpers
=========================

``where_equals(field, value)``
------------------------------

Match records where a field equals a specific value.

.. code-block:: python

   where_equals("team.name", "Arsenal")

``where_not_equals(field, value)``
----------------------------------

Inverse of where_equals.

``where_in(field, values)``
---------------------------

Match records where a field value (or items in a list field) are in a list.

.. code-block:: python

   where_in("player.name", ["Messi", "Mbappe"])

If the field is a list, it checks if any item is in the list of values.

.. warning::
   Fails safely if the field is a dict or a list of dicts.

``where_not_in(field, values)``
-------------------------------

Inverse of where_in. Matches if none of the values appear.

``where_contains(field, substring)``
------------------------------------

Check if a substring appears in the string form of a field.

.. code-block:: python

   where_contains("player.name", "Haaland")

``where_exists(field)``
-----------------------

Check if the field exists and is not None.

.. code-block:: python

   where_exists("location")

``where_is_null(field)``
------------------------

Only matches records where the field is missing or explicitly None.

.. code-block:: python

   where_is_null("location")

Comparison Helpers
==================

+-------------------+----------------+
| Function          | Matches When   |
+===================+================+
| ``where_gt(f, x)``  | Field ``f > x``  |
+-------------------+----------------+
| ``where_gte(f, x)`` | Field ``f >= x`` |
+-------------------+----------------+
| ``where_lt(f, x)``  | Field ``f < x``  |
+-------------------+----------------+
| ``where_lte(f, x)`` | Field ``f <= x`` |
+-------------------+----------------+

🔗 Composing Predicates
=======================

Use logical combinators to build compound filters:

``and_(*predicates)``
---------------------

All must be true.

.. code-block:: python

   filter(and_(
       where_equals("type", "Shot"),
       where_gt("xg", 0.1)
   ))

``or_(*predicates)``
--------------------

Any can be true.

.. code-block:: python

   filter(or_(
       where_equals("type", "Shot"),
       where_equals("type", "Header")
   ))

``not_(predicate)``
-------------------

Negate any predicate.

.. code-block:: python

   filter(not_(where_equals("type", "Own Goal")))

🔧 Advanced Use: Nested + Typed Safety
======================================

All predicate helpers:

- Support dot notation for nested fields
- Handle missing fields safely (return False)
- Raise helpful errors for unsupported types (e.g. filtering a dict)

📦 How to Import
================

You can import individual helpers:

.. code-block:: python

   from penaltyblog.matchflow import where_equals, and_

Or import all in one go:

.. code-block:: python

   from penaltyblog.matchflow import predicates

   flow.filter(predicates.and_(
       predicates.where_equals("type", "Shot"),
       predicates.where_gt("xg", 0.1)
   ))

✅ Summary
==========

Predicate helpers make Flow filters:

- Safer on real-world nested JSON
- More expressive than bare lambdas
- Easier to reuse and compose

They're especially useful in pipelines that must remain readable, modular, or user-defined.

If you work with deeply nested data, predicates are the clearest way to say what you want.
