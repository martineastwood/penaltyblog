import requests
import pandas as pd
import pulp
from typing import Tuple


def get_current_gameweek() -> int:
    """
    Gets the current active gameweek

    Returns
    -------
    Returns current gameweek as an integer

    Examples
    --------
    >>> import penaltyblog as pb
    >>> pb.fpl.get_current_gameweek()
    """
    # get the data
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame(data["events"])
    current_gameweek = df.query("finished == False")["id"].min()
    return current_gameweek


def get_gameweek_info() -> pd.DataFrame:
    """
    Fetches data on the weekly events, e.g. most captained player, highest scoring player etc

    Returns
    -------
    Returns dataframe of events

    Examples
    --------
    >>> import penaltyblog as pb
    >>> pb.fpl.get_gameweek_info()
    """
    # get the data
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame(data["events"])

    # set column types that are not auto-recognised
    df["deadline_time"] = pd.to_datetime(df["deadline_time"])

    return df


def get_player_id_mappings() -> pd.DataFrame:
    """
    Fetches data mapping player names to IDs

    Returns
    -------
    Returns dataframe of player data

    Examples
    --------
    >>> import penaltyblog as pb
    >>> pb.fpl.get_player_id_mappings()
    """
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame(data["elements"])
    return df[["first_name", "second_name", "web_name", "id"]]


def get_player_data() -> pd.DataFrame:
    """
    Fetches top level data on all players, e.g. total points, total minutes played etc

    Returns
    -------
    Returns dataframe of player data

    Examples
    --------
    >>> import penaltyblog as pb
    >>> pb.fpl.get_player_data()
    """
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame(data["elements"])

    # set column types that are not auto-recognised
    df["influence"] = df["influence"].astype(float)
    df["creativity"] = df["creativity"].astype(float)
    df["threat"] = df["threat"].astype(float)
    df["ict_index"] = df["ict_index"].astype(float)
    df["form"] = df["form"].astype(float)
    df["points_per_game"] = df["points_per_game"].astype(float)
    df["selected_by_percent"] = df["selected_by_percent"].astype(float)
    df["value_form"] = df["value_form"].astype(float)
    df["value_season"] = df["value_season"].astype(float)

    # rescale price
    df["now_cost"] = df["now_cost"].astype(float) / 10.0

    # add positions into dataframe
    ets = pd.DataFrame(data["element_types"])
    ets = ets.rename(columns={"id": "pos_id"})

    df = (
        df.merge(
            ets[["pos_id", "singular_name", "singular_name_short"]],
            left_on="element_type",
            right_on="pos_id",
        )
        .drop("pos_id", axis=1)
        .rename(
            columns={
                "singular_name": "position",
                "singular_name_short": "position_short",
            }
        )
    )

    # add teams to dataframe
    teams = pd.DataFrame(data["teams"])
    teams = teams.rename(
        columns={"name": "team_name", "short_name": "team_name_short", "id": "team_id"}
    )
    df = df.merge(
        teams[["team_id", "team_name", "team_name_short"]],
        left_on="team",
        right_on="team_id",
    )
    df = df.drop("team", axis=1)

    return df


def get_player_history(player_id) -> pd.DataFrame:
    """
    Fetches player's history for current season

    Parameters
    ----------
    player_id : int
        The player's FPL Id, this can be determined from the `get_player_id_mappings` function

    Returns
    -------
    Returns dataframe of player data

    Examples
    --------
    >>> import penaltyblog as pb
    >>> pb.fpl.get_player_history(277)
    """
    url = "https://fantasy.premierleague.com/api/element-summary/{id}/".format(
        id=str(player_id)
    )
    r = requests.get(url)
    data = r.json()

    df = pd.DataFrame(data["history"])
    df["influence"] = df["influence"].astype(float)
    df["creativity"] = df["creativity"].astype(float)
    df["threat"] = df["threat"].astype(float)
    df["ict_index"] = df["ict_index"].astype(float)

    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"])

    df["value"] = df["value"] / 10.0

    return df


def get_rankings(page=1) -> pd.DataFrame:
    """
    Fetches a given page of fpl rankings. Each page contains 50 results, so the top fifty teams are on page 1, teams ranked 51-100 are on page 2 etc.

    Parameters
    ----------
    page : int
        The page number to get (each page contains upto 50 entries)

    Returns
    -------
    Returns dataframe of results

    Examples
    --------
    >>> import penaltyblog as pb
    >>> pb.fpl.get_rankings(page=1)
    """
    # get the data
    url = "https://fantasy.premierleague.com/api/leagues-classic/314/standings/?page_new_entries=1&page_standings={page}&phase=1".format(
        page=page
    )
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame(data["standings"]["results"])
    return df


def get_entry_picks_by_gameweek(entry_id, gameweek=1) -> dict:
    """
    Fetches the details for an entry's team on a given week

    Parameters
    ----------
    entry_id : int
        The entry's team ID, this can be found via the `get_rankings`  function or by looking at the URL for the entry on tHe FPL website
    gameweek : int
        The gameweek of interest

    Returns
    -------
    Returns dict of results

    Examples
    --------
    >>> import penaltyblog as pb
    >>> pb.fpl.get_entry_picks_by_gameweek(page=1)
    """
    # get the data
    url = "https://fantasy.premierleague.com/api/entry/{entry_id}/event/{gameweek}/picks/".format(
        entry_id=str(entry_id), gameweek=str(gameweek)
    )
    r = requests.get(url)
    data = r.json()

    # parse the data
    tmp = dict()
    tmp["team_id"] = entry_id
    tmp["active_chip"] = data["active_chip"]
    tmp["event"] = data["entry_history"]["event"]
    tmp["points"] = data["entry_history"]["points"]
    tmp["total_points"] = data["entry_history"]["total_points"]
    tmp["rank"] = data["entry_history"]["rank"]
    tmp["rank_sort"] = data["entry_history"]["rank_sort"]
    tmp["overall_rank"] = data["entry_history"]["overall_rank"]
    tmp["value"] = data["entry_history"]["value"] / 10
    tmp["bank"] = data["entry_history"]["bank"] / 10
    tmp["event_transfers"] = data["entry_history"]["event_transfers"]
    tmp["event_transfers"] = data["entry_history"]["event_transfers"]
    tmp["event_transfers_cost"] = data["entry_history"]["event_transfers_cost"] / 10
    tmp["points_on_bench"] = data["entry_history"]["points_on_bench"]

    # add in autosubs
    tmp["auto_sub_1"] = None
    tmp["auto_sub_2"] = None
    tmp["auto_sub_3"] = None
    tmp["auto_sub_4"] = None
    for i, x in enumerate(data["automatic_subs"]):
        tmp["auto_sub_{i}".format(i=i)] = x

    # add in player picks
    for i, x in enumerate(data["picks"]):
        tmp["player_pick_{i}".format(i=i)] = x["element"]

    # add in captain
    for i, x in enumerate(data["picks"]):
        if x["is_captain"] is True:
            tmp["captain_id"] = x["element"]

    # add in vice captain
    for i, x in enumerate(data["picks"]):
        if x["is_vice_captain"] is True:
            tmp["is_vice_captain"] = x["element"]

    return tmp


def get_entry_transfers(entry_id) -> pd.DataFrame:
    """
    Gets the transfer history for a given entry for the current season

    Parameters
    ----------
    entry_id : int
        The entry's team ID, this can be found via the `get_rankings`  function or by looking at the URL for the entry on tHe FPL website

    Returns
    -------
    Returns dataframe of results

    Examples
    --------
    >>> import penaltyblog as pb
    >>> pb.fpl.get_entry_transfers(page=1)
    """
    url = "https://fantasy.premierleague.com/api/entry/{entry_id}/transfers/".format(
        entry_id=str(entry_id)
    )
    r = requests.get(url)
    df = pd.DataFrame(r.json())
    df["element_in_cost"] = df["element_in_cost"] / 10.0
    df["element_out_cost"] = df["element_out_cost"] / 10.0
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(
        columns={
            "element_in": "player_id_in",
            "element_out": "player_id_out",
            "element_in_cost": "player_in_cost",
            "element_out_cost": "player_out_cost",
        }
    )
    return df


def optimise_team(formation="2-5-5-3", budget=100) -> Tuple[dict, pd.DataFrame]:
    """
    Gets the optimal team by maximising the total points based on formatation and budget

    Returns
    -------
    Returns tuple containgina

    Examples
    --------
    >>> import penaltyblog as pb
    >>> pb.fpl.get_current_gameweek()
    """
    # check formation is valid
    formation_len = len(formation.split("-"))
    if formation_len != 4:
        raise ValueError(
            "Error: formation is invalid, must contain values for gk, def, mid and fwd. For, example `2-5-5-3`, `1-4-3-3` etc"
        )

    # set up constraints
    constraints = dict()
    constraints["max_budget"] = budget
    constraints["position_gkp"] = int(formation.split("-")[0])
    constraints["position_def"] = int(formation.split("-")[1])
    constraints["position_mid"] = int(formation.split("-")[2])
    constraints["position_fwd"] = int(formation.split("-")[3])
    constraints["max_team"] = 3

    # get the player data
    def _one_hot_encode_positions(df, col):
        dummies = pd.get_dummies(df[col])
        df = pd.concat([df, dummies], axis=1)
        return df

    players = (
        get_player_data()
        .assign(full_name=lambda x: x["first_name"] + " " + x["second_name"])
        .loc[
            :,
            [
                "id",
                "full_name",
                "team_name_short",
                "position_short",
                "now_cost",
                "total_points",
            ],
        ]
        .rename(
            columns={
                "team_name_short": "team",
                "position_short": "position",
                "now_cost": "price",
            }
        )
        .pipe(_one_hot_encode_positions, "position")
        .pipe(_one_hot_encode_positions, "team")
    )

    # Initialise the PuLP problem to optimise
    problem = pulp.LpProblem("FPL", pulp.LpMaximize)

    # create dictionary from player names to use as PuLP variables
    x = pulp.LpVariable.dict(
        "player", players["full_name"].tolist(), 0, 1, cat=pulp.LpInteger
    )

    # create the objective function
    total_points = {x: y for x, y in zip(players["full_name"], players["total_points"])}
    problem += sum([total_points[i] * x[i] for i in total_points])

    # create constraint on budget
    player_cost = {x: y for x, y in zip(players["full_name"], players["price"])}
    problem += (
        sum([player_cost[i] * x[i] for i in player_cost]) <= constraints["max_budget"]
    )

    # apply constraints on positions
    is_gkp = {x: y for x, y in zip(players["full_name"], players["GKP"])}
    problem += sum([is_gkp[i] * x[i] for i in is_gkp]) == constraints["position_gkp"]

    is_def = {x: y for x, y in zip(players["full_name"], players["DEF"])}
    problem += sum([is_def[i] * x[i] for i in is_def]) == constraints["position_def"]

    is_mid = {x: y for x, y in zip(players["full_name"], players["MID"])}
    problem += sum([is_mid[i] * x[i] for i in is_mid]) == constraints["position_mid"]

    is_fwd = {x: y for x, y in zip(players["full_name"], players["FWD"])}
    problem += sum([is_fwd[i] * x[i] for i in is_fwd]) == constraints["position_fwd"]

    # add constraints on team
    for team in players["team"].unique():
        is_team = {x: y for x, y in zip(players["full_name"], players[team])}
        problem += sum([is_team[i] * x[i] for i in is_team]) <= constraints["max_team"]

    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    # get the selected players
    selected = [i for i in players["full_name"] if pulp.value(x[i]) == 1]
    selected_players = players[players["full_name"].isin(selected)]

    # map position so dataframe sorts nicely
    pos_map = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    selected_players = (
        selected_players.assign(pos_sort=lambda x: x["position"].map(pos_map))
        .sort_values(["pos_sort", "total_points"], ascending=[True, False])
        .reset_index(drop=True)
        .loc[:, ["id", "full_name", "team", "position", "price", "total_points"]]
    )

    output = dict()
    output["status"] = pulp.LpStatus[problem.status]
    output["total_points"] = pulp.value(problem.objective)
    output["total_price"] = round(selected_players["price"].sum(), 2)

    return output, selected_players
