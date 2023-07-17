# import pandas as pd

# import penaltyblog as pb


# def test_current_gameweek():
#     week = pb.fpl.get_current_gameweek()
#     assert type(week) == int


# def test_gameweek_info():
#     info = pb.fpl.get_gameweek_info()
#     assert type(info) == pd.DataFrame


# def test_player_mappings_info():
#     mappings = pb.fpl.get_player_id_mappings()
#     assert type(mappings) == pd.DataFrame


# def test_get_player_data():
#     data = pb.fpl.get_player_data()
#     assert type(data) == pd.DataFrame


# def test_get_player_history():
#     data = pb.fpl.get_player_history(123)
#     assert type(data) == pd.DataFrame


# def test_get_rankings():
#     data = pb.fpl.get_rankings(1)
#     assert type(data) == pd.DataFrame


# def test_get_entry_picks_by_gameweek():
#     data = pb.fpl.get_entry_picks_by_gameweek(100, 1)
#     assert type(data) == pd.DataFrame


# def test_optimise_team():
#     res, data = pb.fpl.optimise_team(budget=100)
#     assert type(data) == pd.DataFrame
#     assert type(res) == dict


# def test_optimise_team():
#     res, data = pb.fpl.optimise_team(formation="1-3-5-2", budget=85)
#     assert type(data) == pd.DataFrame
#     assert type(res) == dict
