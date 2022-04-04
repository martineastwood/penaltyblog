.. _api_docs:

API Documentation
=================

``Poisson Goal Model``
--------------------------

.. autoclass:: penaltyblog.poisson.PoissonGoalsModel
    :members:
    :private-members:
    :member-order: bysource

``Dixon and Coles Goal Model``
-------------------------------

.. autoclass:: penaltyblog.poisson.DixonColesGoalModel
    :members:
    :private-members:
    :member-order: bysource    


``Dixon and Coles Goal Model With Rue and Salvesen Adjustment``
----------------------------------------------------------------

.. autoclass:: penaltyblog.poisson.RueSalvesenGoalModel
    :members:
    :private-members:
    :member-order: bysource   


``Calculate Implied Betting Odds``
----------------------------------

.. automodule:: penaltyblog.implied
    :members:
    :special-members:
    :exclude-members: __weakref__
    :member-order: bysource

.. autofunction:: multiplicative 

|

.. autofunction:: additive

|

.. autofunction:: power

|

.. autofunction:: shin

|

.. autofunction:: differential_margin_weighting

|

.. autofunction:: odds_ratio    

``Get data from footballdata.co.uk``
---------------------------------------

.. automodule:: penaltyblog.footballdata
    :members:
    :special-members:
    :exclude-members: __weakref__
    :member-order: bysource

.. autofunction:: list_countries 

| 

.. autofunction:: fetch_data


``Get data from clubelo.com``
-------------------------------

.. automodule:: penaltyblog.clubelo
    :members:
    :special-members:
    :exclude-members: __weakref__
    :member-order: bysource

.. autofunction:: list_all_teams 

|

.. autofunction:: fetch_rankings_by_date

|

.. autofunction:: fetch_rankings_by_team


``Get data from fantasy.premierleague.com``
--------------------------------------------------------------

.. automodule:: penaltyblog.fpl
    :members:
    :special-members:
    :exclude-members: __weakref__
    :member-order: bysource

.. autofunction:: get_current_gameweek 

|

.. autofunction:: get_gameweek_info

|

.. autofunction:: get_player_id_mappings

|

.. autofunction:: get_player_data

|

.. autofunction:: get_player_history

|

.. autofunction:: get_rankings

|

.. autofunction:: get_entry_picks_by_gameweek

|

.. autofunction:: get_entry_transfers



``Calculate Massey Ratings``
------------------------------

.. automodule:: penaltyblog.massey
    :members:
    :special-members:
    :exclude-members: __weakref__
    :member-order: bysource

.. autofunction:: ratings 


``Calculate Colley Ratings``
------------------------------

.. automodule:: penaltyblog.colley
    :members:
    :special-members:
    :exclude-members: __weakref__
    :member-order: bysource

.. autofunction:: ratings 


``Useful Metrics``
-------------------

.. automodule:: penaltyblog.metrics
    :members:
    :special-members:
    :exclude-members: __weakref__
    :member-order: bysource

.. autofunction:: rps 


