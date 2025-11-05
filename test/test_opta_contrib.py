import pytest

from penaltyblog.matchflow.contrib.opta import Opta

# Get a default instance to check DEFAULT_CREDS
opta_instance = Opta()


def test_opta_events_plan():
    """Tests that the 'events' method builds the correct plan."""
    flow = Opta().events(
        fixture_uuid="fx123",
        contestant_uuid="ctst1",
        use_opta_names=True,
        person_uuid="pl123",
    )

    expected_args = {
        "fixture_uuid": "fx123",
        "ctst": "ctst1",
        "prsn": "pl123",
        "type": None,
        "_lcl": "en-op",
        "creds": opta_instance.DEFAULT_CREDS,
        "proxies": None,
    }

    assert flow.plan[0]["op"] == "from_opta"
    assert flow.plan[0]["source"] == "match_events"
    assert flow.plan[0]["args"] == expected_args


def test_opta_tournament_schedule_plan():
    """Tests that the 'tournament_schedule' method builds the correct plan."""
    flow = Opta().tournament_schedule(
        tournament_calendar_uuid="tmcl1",
        coverage_level=1,
        use_opta_names=True,
    )

    expected_args = {
        "tournament_calendar_uuid": "tmcl1",
        "cvlv": "1",
        "_lcl": "en-op",
        "creds": opta_instance.DEFAULT_CREDS,
        "proxies": None,
    }

    assert flow.plan[0]["op"] == "from_opta"
    assert flow.plan[0]["source"] == "tournament_schedule"
    assert flow.plan[0]["args"] == expected_args

    flow = Opta().tournament_schedule(
        tournament_calendar_uuid="tmcl1",
        coverage_level=[1, 2, 3],
        use_opta_names=False,
    )

    expected_args = {
        "tournament_calendar_uuid": "tmcl1",
        "cvlv": "1,2,3",
        "_lcl": None,
        "creds": opta_instance.DEFAULT_CREDS,
        "proxies": None,
    }

    assert flow.plan[0]["op"] == "from_opta"
    assert flow.plan[0]["source"] == "tournament_schedule"
    assert flow.plan[0]["args"] == expected_args


def test_opta_match_plan():
    """Tests that the 'match' method builds the correct plan."""
    flow = Opta().match(
        fixture_uuid="fx123",
        live=True,
        lineups=True,
        use_opta_names=True,
    )

    expected_args = {
        "fixture_uuid": "fx123",
        "live": "yes",
        "lineups": "yes",
        "_lcl": "en-op",
        "creds": opta_instance.DEFAULT_CREDS,
        "proxies": None,
    }

    assert flow.plan[0]["op"] == "from_opta"
    assert flow.plan[0]["source"] == "match_basic"
    assert flow.plan[0]["args"] == expected_args

    flow = Opta().match(fixture_uuid="fx123")
    expected_args = {
        "fixture_uuid": "fx123",
        "live": "no",
        "lineups": "no",
        "_lcl": None,
        "creds": opta_instance.DEFAULT_CREDS,
        "proxies": None,
    }
    assert flow.plan[0]["args"] == expected_args


def test_opta_matches_plan_params():
    """Tests that the 'matches' method handles complex parameters correctly."""
    flow = Opta().matches(
        tournament_calendar_uuid="tmcl1",
        contestant_uuid="ctst1",
        opponent_uuid="ctst2",
        contestant_position="home",
    )

    expected_args = {
        "fx": None,
        "tmcl": "tmcl1",
        "comp": None,
        "ctst": "ctst1",
        "ctst2": "ctst2",
        "ctstpos": "home",
        "mt_mDt": None,
        "_dlt": None,
        "live": "no",
        "lineups": "no",
        "_lcl": None,
        "creds": opta_instance.DEFAULT_CREDS,
        "proxies": None,
    }

    assert flow.plan[0]["source"] == "matches_basic"
    assert flow.plan[0]["args"] == expected_args


def test_opta_match_stats_plan():
    """Tests that the 'match_stats' method builds the correct plan."""
    flow = Opta().match_stats(
        fixture_uuids=["fx123", "fx456"],
        include_players=True,
        use_opta_names=True,
    )

    expected_args = {
        "fixture_uuids": ["fx123", "fx456"],
        "people": "yes",
        "_lcl": "en-op",
        "creds": opta_instance.DEFAULT_CREDS,
        "proxies": None,
    }

    assert flow.plan[0]["op"] == "from_opta"
    assert flow.plan[0]["source"] == "match_stats_basic"
    assert flow.plan[0]["args"] == expected_args

    flow = Opta().match_stats(fixture_uuids="fx123", include_players=False)
    expected_args = {
        "fixture_uuids": "fx123",
        "people": "no",
        "_lcl": None,
        "creds": opta_instance.DEFAULT_CREDS,
        "proxies": None,
    }
    assert flow.plan[0]["args"] == expected_args


def test_opta_player_season_stats_plan():
    """Tests that 'player_season_stats' builds the correct plan."""
    flow = Opta().player_season_stats(
        tournament_calendar_uuid="tmcl1", contestant_uuid="ctst1", detailed=False
    )

    expected_args = {
        "tmcl": "tmcl1",
        "ctst": "ctst1",
        "detailed": "no",
        "creds": opta_instance.DEFAULT_CREDS,
        "proxies": None,
    }

    assert flow.plan[0]["source"] == "player_season_stats"
    assert flow.plan[0]["args"] == expected_args


def test_opta_proxies_passed_to_plan():
    """Tests that the 'proxies' param is passed through to the plan args."""
    my_proxies = {"https": "http://my.proxy.com"}
    flow = Opta().teams(contestant_uuid="ctst1", proxies=my_proxies)

    assert flow.plan[0]["args"]["proxies"] == my_proxies


def test_opta_teams_plan():
    """Tests that the 'teams' method builds the correct plan."""
    flow = Opta().teams(
        tournament_calendar_uuid="tmcl1",
        contestant_uuid="ctst1",
        country_uuid="ctry1",
        stage_uuid="stg1",
        series_uuid="srs1",
    )

    expected_args = {
        "tmcl": "tmcl1",
        "ctst": "ctst1",
        "ctry": "ctry1",
        "stg": "stg1",
        "srs": "srs1",
        "creds": opta_instance.DEFAULT_CREDS,
        "proxies": None,
    }

    assert flow.plan[0]["op"] == "from_opta"
    assert flow.plan[0]["source"] == "teams"
    assert flow.plan[0]["args"] == expected_args

    with pytest.raises(ValueError):
        Opta().teams()


def test_opta_squads_plan():
    """Tests that the 'squads' method builds the correct plan."""
    flow = Opta().squads(
        tournament_calendar_uuid="tmcl1",
        contestant_uuid="ctst1",
        use_opta_names=True,
    )

    expected_args = {
        "tmcl": "tmcl1",
        "ctst": "ctst1",
        "_lcl": "en-op",
        "creds": opta_instance.DEFAULT_CREDS,
        "proxies": None,
    }

    assert flow.plan[0]["op"] == "from_opta"
    assert flow.plan[0]["source"] == "squads"
    assert flow.plan[0]["args"] == expected_args

    flow = Opta().squads(tournament_calendar_uuid="tmcl1")
    expected_args = {
        "tmcl": "tmcl1",
        "ctst": None,
        "_lcl": None,
        "creds": opta_instance.DEFAULT_CREDS,
        "proxies": None,
    }
    assert flow.plan[0]["args"] == expected_args

    with pytest.raises(ValueError):
        Opta().squads()


def test_opta_tournament_calendars_plan():
    """Tests that the 'tournament_calendars' method builds the correct plan."""
    flow = Opta().tournament_calendars(
        status="active",
        competition_uuid="comp1",
        contestant_uuid="ctst1",
        include_stages=True,
        include_coverage=True,
    )

    expected_args = {
        "status": "active",
        "comp": "comp1",
        "ctst": "ctst1",
        "stages": "yes",
        "coverage": "yes",
        "creds": opta_instance.DEFAULT_CREDS,
        "proxies": None,
    }

    assert flow.plan[0]["op"] == "from_opta"
    assert flow.plan[0]["source"] == "tournament_calendars"
    assert flow.plan[0]["args"] == expected_args


def test_opta_team_season_stats_plan():
    """Tests that 'team_season_stats' builds the correct plan."""
    flow = Opta().team_season_stats(
        tournament_calendar_uuid="tmcl1", contestant_uuid="ctst1", detailed=False
    )

    expected_args = {
        "tmcl": "tmcl1",
        "ctst": "ctst1",
        "detailed": "no",
        "creds": opta_instance.DEFAULT_CREDS,
        "proxies": None,
    }

    assert flow.plan[0]["op"] == "from_opta"
    assert flow.plan[0]["source"] == "team_season_stats"
    assert flow.plan[0]["args"] == expected_args

    flow = Opta().team_season_stats(
        tournament_calendar_uuid="tmcl1", contestant_uuid="ctst1", detailed=True
    )

    expected_args = {
        "tmcl": "tmcl1",
        "ctst": "ctst1",
        "detailed": "yes",
        "creds": opta_instance.DEFAULT_CREDS,
        "proxies": None,
    }

    assert flow.plan[0]["op"] == "from_opta"
    assert flow.plan[0]["source"] == "team_season_stats"
    assert flow.plan[0]["args"] == expected_args

    # Test optional contestant_uuid
    flow = Opta().team_season_stats(tournament_calendar_uuid="tmcl1", detailed=True)

    expected_args = {
        "tmcl": "tmcl1",
        "ctst": None,
        "detailed": "yes",
        "creds": opta_instance.DEFAULT_CREDS,
        "proxies": None,
    }

    assert flow.plan[0]["op"] == "from_opta"
    assert flow.plan[0]["source"] == "team_season_stats"
    assert flow.plan[0]["args"] == expected_args


def test_opta_matches_comprehensive_parameters():
    """Tests that 'matches' method handles all parameters correctly."""
    from datetime import datetime

    # Test with fixture_uuids list
    flow = Opta().matches(
        fixture_uuids=["fx123", "fx456"],
        competition_uuids=["comp1", "comp2"],
        date_from=datetime(2023, 1, 1),
        date_to=datetime(2023, 12, 31),
        delta_timestamp="2023-01-01T00:00:00Z",
        live=True,
        lineups=True,
        use_opta_names=True,
    )

    expected_args = {
        "fx": "fx123,fx456",
        "tmcl": None,
        "comp": "comp1,comp2",
        "ctst": None,
        "ctst2": None,
        "ctstpos": None,
        "mt_mDt": "[2023-01-01T00:00:00Z TO 2023-12-31T00:00:00Z]",
        "_dlt": "2023-01-01T00:00:00Z",
        "live": "yes",
        "lineups": "yes",
        "_lcl": "en-op",
        "creds": opta_instance.DEFAULT_CREDS,
        "proxies": None,
    }

    assert flow.plan[0]["source"] == "matches_basic"
    assert flow.plan[0]["args"] == expected_args

    # Test date range error
    with pytest.raises(
        ValueError, match="Both 'date_from' and 'date_to' must be provided"
    ):
        Opta().matches(date_from=datetime(2023, 1, 1))


def test_opta_events_event_types():
    """Tests that 'events' method handles event_types parameter correctly."""
    # Test with single event type
    flow = Opta().events(
        fixture_uuid="fx123",
        event_types=1,
    )

    expected_args = {
        "fixture_uuid": "fx123",
        "ctst": None,
        "prsn": None,
        "type": "1",
        "_lcl": None,
        "creds": opta_instance.DEFAULT_CREDS,
        "proxies": None,
    }

    assert flow.plan[0]["source"] == "match_events"
    assert flow.plan[0]["args"] == expected_args

    # Test with list of event types
    flow = Opta().events(
        fixture_uuid="fx123",
        event_types=[1, 2, 3],
    )

    expected_args = {
        "fixture_uuid": "fx123",
        "ctst": None,
        "prsn": None,
        "type": "1,2,3",
        "_lcl": None,
        "creds": opta_instance.DEFAULT_CREDS,
        "proxies": None,
    }

    assert flow.plan[0]["source"] == "match_events"
    assert flow.plan[0]["args"] == expected_args


def test_opta_custom_credentials_and_proxies():
    """Tests that custom credentials and proxies are passed through correctly."""
    custom_creds = {"auth_key": "custom_key", "rt_mode": "a"}
    custom_proxies = {"https": "http://my.proxy.com"}

    flow = Opta().events(
        fixture_uuid="fx123",
        creds=custom_creds,
        proxies=custom_proxies,
    )

    expected_args = {
        "fixture_uuid": "fx123",
        "ctst": None,
        "prsn": None,
        "type": None,
        "_lcl": None,
        "creds": custom_creds,
        "proxies": custom_proxies,
    }

    assert flow.plan[0]["source"] == "match_events"
    assert flow.plan[0]["args"] == expected_args


def test_opta_optimize_parameter():
    """Tests that the optimize parameter is passed through correctly."""
    flow = Opta().events(
        fixture_uuid="fx123",
        optimize=True,
    )

    assert flow.optimize is True

    flow = Opta().events(
        fixture_uuid="fx123",
        optimize=False,
    )

    assert flow.optimize is False


def test_format_opta_datetime():
    """Tests the _format_opta_datetime helper function."""
    from datetime import datetime

    from penaltyblog.matchflow.contrib.opta import _format_opta_datetime

    # Test with datetime object
    dt = datetime(2023, 1, 15, 14, 30, 0)
    result = _format_opta_datetime(dt)
    assert result == "2023-01-15T14:30:00Z"

    # Test with string (should return as-is)
    date_str = "2023-01-15T14:30:00Z"
    result = _format_opta_datetime(date_str)
    assert result == date_str


def test_opta_tournament_calendars_all_parameters():
    """Tests tournament_calendars with all parameter combinations."""
    # Test with all parameters
    flow = Opta().tournament_calendars(
        status="active",
        competition_uuid="comp1",
        contestant_uuid="ctst1",
        include_stages=True,
        include_coverage=True,
        creds={"auth_key": "test"},
        proxies={"https": "http://proxy.com"},
        optimize=True,
    )

    expected_args = {
        "status": "active",
        "comp": "comp1",
        "ctst": "ctst1",
        "stages": "yes",
        "coverage": "yes",
        "creds": {"auth_key": "test"},
        "proxies": {"https": "http://proxy.com"},
    }

    assert flow.plan[0]["source"] == "tournament_calendars"
    assert flow.plan[0]["args"] == expected_args
    assert flow.optimize is True

    # Test with default values
    flow = Opta().tournament_calendars()

    expected_args = {
        "status": "all",
        "comp": None,
        "ctst": None,
        "stages": None,
        "coverage": None,
        "creds": opta_instance.DEFAULT_CREDS,
        "proxies": None,
    }

    assert flow.plan[0]["args"] == expected_args
    assert flow.optimize is False


def test_opta_contestant_participation_plan():
    """Tests that the 'contestant_participation' method builds the correct plan."""
    # Test with single contestant UUID
    flow = Opta().contestant_participation(
        contestant_uuid="ctst1",
        active=True,
    )

    expected_args = {
        "contestant_uuid": "ctst1",
        "active": True,
        "creds": opta_instance.DEFAULT_CREDS,
        "proxies": None,
    }

    assert flow.plan[0]["op"] == "from_opta"
    assert flow.plan[0]["source"] == "contestant_participation"
    assert flow.plan[0]["args"] == expected_args

    # Test with list of contestant UUIDs
    flow = Opta().contestant_participation(
        contestant_uuid=["ctst1", "ctst2"],
        active=False,
    )

    expected_args = {
        "contestant_uuid": ["ctst1", "ctst2"],
        "active": False,
        "creds": opta_instance.DEFAULT_CREDS,
        "proxies": None,
    }

    assert flow.plan[0]["source"] == "contestant_participation"
    assert flow.plan[0]["args"] == expected_args

    # Test with custom credentials and proxies
    custom_creds = {"auth_key": "custom_key", "rt_mode": "a"}
    custom_proxies = {"https": "http://my.proxy.com"}

    flow = Opta().contestant_participation(
        contestant_uuid="ctst1",
        creds=custom_creds,
        proxies=custom_proxies,
        optimize=True,
    )

    expected_args = {
        "contestant_uuid": "ctst1",
        "active": False,
        "creds": custom_creds,
        "proxies": custom_proxies,
    }

    assert flow.plan[0]["source"] == "contestant_participation"
    assert flow.plan[0]["args"] == expected_args
    assert flow.optimize is True

    # Test error case - no contestant_uuid provided
    with pytest.raises(ValueError, match="'contestant_uuid' must be provided"):
        Opta().contestant_participation(contestant_uuid=None)
