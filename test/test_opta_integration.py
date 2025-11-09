# tests/test_opta_integration.py
from datetime import datetime

import pytest

from penaltyblog.matchflow.contrib.opta import opta

VALID_TMCL_UUID = "51r6ph2woavlbbpk8f29nynf8"
VALID_FIXTURE_UUID = "zhs8gg1hvcuqvhkk2itb54pg"
VALID_FIXTURE_UUID2 = "102dto55773ex3p58gp94ql90"
VALID_CONTESTANT_UUID = "c8h9bw1l82s06h77xxrelzhur"
VALID_PERSON_UUID = "5ilkkfbsss0bxd6ttdlqg0uz9"
VALID_VENUE_UUID = "bxpq91vq4x9r3q6eq3d0bwjuy"
VALID_AREA_UUID = "7yck0z0f9rlpeyatanjc1ylzp"


# This mark tells vcrpy to record/replay this test
@pytest.mark.vcr
def test_fetch_tournament_calendars_active():
    flow = opta.tournament_calendars(
        status="active",
        include_stages=True,
    )
    calendars = flow.collect()

    assert calendars is not None
    assert isinstance(calendars, list)
    assert len(calendars) > 0
    assert "competitionCode" in calendars[0]
    assert "competitionType" in calendars[0]


@pytest.mark.vcr
def test_fetch_tournament_calendars_all():
    flow = opta.tournament_calendars(status="all")
    calendars = flow.collect()

    assert calendars is not None
    assert isinstance(calendars, list)
    assert len(calendars) > 0
    assert "competitionCode" in calendars[0]
    assert "competitionType" in calendars[0]


@pytest.mark.vcr
def test_fetch_tournament_calendars_authorized():
    flow = opta.tournament_calendars(status="authorized")
    calendars = flow.collect()

    assert calendars is not None
    assert isinstance(calendars, list)
    assert len(calendars) == 1
    assert "competitionCode" in calendars[0]
    assert "competitionType" in calendars[0]


@pytest.mark.vcr
def test_venue_tmcl():
    flow = opta.venues(tournament_calendar_uuid=VALID_TMCL_UUID)
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) == 20
    assert "id" in data[0]
    assert "name" in data[0]


@pytest.mark.vcr
def test_venue_contestant():
    flow = opta.venues(contestant_uuid=VALID_CONTESTANT_UUID)
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) == 1
    assert "id" in data[0]
    assert "name" in data[0]


@pytest.mark.vcr
def test_venue_venue():
    flow = opta.venues(venue_uuid=VALID_VENUE_UUID, use_opta_names=True)
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) == 1
    assert "id" in data[0]
    assert "name" in data[0]


@pytest.mark.vcr
def test_areas_all():
    flow = opta.areas()
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert data[0]["name"] == "World"


@pytest.mark.vcr
def test_areas_area():
    flow = opta.areas(area_uuid=VALID_AREA_UUID)
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert data[0]["area"][0]["name"] == "Asia"


@pytest.mark.vcr
def test_tournament_schedule_tcml():
    flow = opta.tournament_schedule(tournament_calendar_uuid=VALID_TMCL_UUID)
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) == 380
    assert "id" in data[0]
    assert "homeContestantId" in data[0]


@pytest.mark.vcr
def test_matches_tmcl():
    flow = opta.matches(tournament_calendar_uuid=VALID_TMCL_UUID)
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) == 380
    assert "matchInfo" in data[0]
    assert "liveData" in data[0]


@pytest.mark.vcr
def test_matches_single_fixture():
    flow = opta.matches(fixture_uuids=VALID_FIXTURE_UUID)
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) == 1
    assert "matchInfo" in data[0]
    assert "liveData" in data[0]


@pytest.mark.vcr
def test_matches_multiple_fixtures():
    flow = opta.matches(fixture_uuids=[VALID_FIXTURE_UUID, VALID_FIXTURE_UUID2])
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) == 2
    assert "matchInfo" in data[0]
    assert "liveData" in data[0]


@pytest.mark.vcr
def test_matches_contestant():
    flow = opta.matches(
        tournament_calendar_uuid=VALID_TMCL_UUID, contestant_uuid=VALID_CONTESTANT_UUID
    )
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) == 38
    assert "matchInfo" in data[0]
    assert "liveData" in data[0]


@pytest.mark.vcr
def test_matches_contestant_home():
    flow = opta.matches(
        tournament_calendar_uuid=VALID_TMCL_UUID,
        contestant_uuid=VALID_CONTESTANT_UUID,
        contestant_position="home",
    )
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) == 19
    assert "matchInfo" in data[0]
    assert "liveData" in data[0]


@pytest.mark.vcr
def test_matches_contestant_date_str():
    flow = opta.matches(
        tournament_calendar_uuid=VALID_TMCL_UUID,
        contestant_uuid=VALID_CONTESTANT_UUID,
        date_from="2025-09-01",
        date_to="2025-10-01",
    )
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) == 3
    assert "matchInfo" in data[0]
    assert "liveData" in data[0]


@pytest.mark.vcr
def test_matches_lineups():
    flow = opta.matches(
        tournament_calendar_uuid=VALID_TMCL_UUID,
        contestant_uuid=VALID_CONTESTANT_UUID,
        lineups=True,
        live=True,
        date_from="2025-09-01",
        date_to="2025-10-01",
    )
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) == 3
    assert "lineUp" in data[0]["liveData"]


@pytest.mark.vcr
def test_matches_contestant_date_error():
    with pytest.raises(ValueError):
        flow = opta.matches(
            tournament_calendar_uuid=VALID_TMCL_UUID,
            contestant_uuid=VALID_CONTESTANT_UUID,
            date_from="2025-10-01",
            date_to="2025-09-01",
        )
        flow.collect()


@pytest.mark.vcr
def test_matches_contestant_date_dt():
    flow = opta.matches(
        tournament_calendar_uuid=VALID_TMCL_UUID,
        contestant_uuid=VALID_CONTESTANT_UUID,
        date_from=datetime(2025, 9, 1),
        date_to=datetime(2025, 10, 1),
    )
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) == 3
    assert "matchInfo" in data[0]
    assert "liveData" in data[0]


@pytest.mark.vcr
def test_parser_ma1_match_basic():
    """
    Tests: _handle_non_paginated_endpoint -> parse_match_basic
    """
    flow = opta.match(fixture_uuid=VALID_FIXTURE_UUID)
    data = flow.collect()
    assert data is not None
    assert len(data) == 1
    assert "matchInfo" in data[0]


@pytest.mark.vcr
def test_parser_ma1_match_lineups():
    """
    Tests: _handle_non_paginated_endpoint -> parse_match_basic
    """
    flow = opta.match(fixture_uuid=VALID_FIXTURE_UUID, lineups=True, live=True)
    data = flow.collect()
    assert data is not None
    assert len(data) == 1
    assert "matchInfo" in data[0]
    assert "lineUp" in data[0]["liveData"]


@pytest.mark.vcr
def test_parser_ma1_match_not_lineups():
    """
    Tests: _handle_non_paginated_endpoint -> parse_match_basic
    """
    flow = opta.match(fixture_uuid=VALID_FIXTURE_UUID, lineups=False, live=True)
    data = flow.collect()
    assert data is not None
    assert len(data) == 1
    assert "matchInfo" in data[0]
    assert "lineUp" not in data[0]["liveData"]


@pytest.mark.vcr
def test_parser_ma1_match_not_live():
    """
    Tests: _handle_non_paginated_endpoint -> parse_match_basic
    """
    flow = opta.match(fixture_uuid=VALID_FIXTURE_UUID, live=False)
    data = flow.collect()
    assert data is not None
    assert len(data) == 1
    assert "matchInfo" in data[0]
    assert "liveData" not in data[0]


@pytest.mark.vcr
def test_fetch_match_events():
    # Use a real match UUID you have access to this week
    MATCH_UUID = "zhs8gg1hvcuqvhkk2itb54pg"
    flow = opta.events(
        fixture_uuid=MATCH_UUID,
    )
    events = flow.collect()

    assert events is not None
    assert len(events) > 100
    assert "typeId" in events[0]
    assert "_match_info" in events[0]  # Test your parser


@pytest.mark.vcr
def test_parser_ma2_match_stats_players():
    """
    Tests: _handle_non_paginated_endpoint -> parse_match_stats_player
    Logic: Must test the new match_stats_player method.
    """
    flow = opta.match_stats_player(fixture_uuids=VALID_FIXTURE_UUID)
    data = flow.collect()
    assert data is not None
    assert len(data) > 2
    assert "playerId" in data[0] and "_match_uuid" in data[0]


@pytest.mark.vcr
def test_parser_ma2_match_stats_players_multiple():
    """
    Tests: _handle_non_paginated_endpoint -> parse_match_stats_player
    Logic: Must test the new match_stats_player method.
    """
    flow = opta.match_stats_player(
        fixture_uuids=[VALID_FIXTURE_UUID, VALID_FIXTURE_UUID2]
    )
    data = flow.collect()
    assert data is not None
    assert len(data) > 4
    assert "playerId" in data[0] and "_match_uuid" in data[0]


@pytest.mark.vcr
def test_parser_ma2_match_stats_teams():
    """
    Tests: _handle_non_paginated_endpoint -> parse_match_stats_team
    Logic: Must test the new match_stats_team method.
    """
    flow = opta.match_stats_team(fixture_uuids=VALID_FIXTURE_UUID)
    data = flow.collect()

    assert data is not None
    assert len(data) == 2  # Should be one record per team
    # Check for keys from extract_team_stats
    assert "contestantId" in data[0] and "_match_uuid" in data[0]


@pytest.mark.vcr
def test_parser_ma2_match_stats_teams_multiple():
    """
    Tests: _handle_non_paginated_endpoint -> parse_match_stats_team
    Logic: Must test the new match_stats_team method.
    """
    flow = opta.match_stats_team(
        fixture_uuids=[VALID_FIXTURE_UUID, VALID_FIXTURE_UUID2]
    )
    data = flow.collect()

    assert data is not None
    assert len(data) == 4  # Should be one record per team
    # Check for keys from extract_team_stats
    assert "contestantId" in data[0] and "_match_uuid" in data[0]


@pytest.mark.vcr
def test_parser_ma0_tournament_schedule():
    """
    Tests: _handle_non_paginated_endpoint -> parse_tournament_schedule
    """
    flow = opta.tournament_schedule(tournament_calendar_uuid=VALID_TMCL_UUID)
    data = flow.collect()
    assert data is not None
    assert len(data) > 0
    assert "_competition" in data[0]
    assert "_tournamentCalendar" in data[0]


@pytest.mark.vcr
def test_parser_ma3_match_events():
    """
    Tests: _handle_non_paginated_endpoint -> extract_match_events
    """
    flow = opta.events(fixture_uuid=VALID_FIXTURE_UUID)
    data = flow.collect()
    assert data is not None
    assert len(data) == 1733
    assert "typeId" in data[0] and "_match_info" in data[0]


@pytest.mark.vcr
def test_parser_ma3_match_events_contestant():
    """
    Tests: _handle_non_paginated_endpoint -> extract_match_events
    """
    flow = opta.events(
        fixture_uuid=VALID_FIXTURE_UUID, contestant_uuid=VALID_CONTESTANT_UUID
    )
    data = flow.collect()
    assert data is not None
    assert len(data) == 957
    assert "typeId" in data[0] and "_match_info" in data[0]


@pytest.mark.vcr
def test_parser_ma3_match_events_person():
    """
    Tests: _handle_non_paginated_endpoint -> extract_match_events
    """
    flow = opta.events(fixture_uuid=VALID_FIXTURE_UUID, person_uuid=VALID_PERSON_UUID)
    data = flow.collect()
    pytest.set_trace()
    assert data is not None
    assert len(data) == 957
    assert "typeId" in data[0] and "_match_info" in data[0]


@pytest.mark.vcr
def test_parser_tm4_player_season_stats():
    """
    Tests: _handle_non_paginated_endpoint -> extract_season_player_stats
    """
    flow = opta.player_season_stats(
        tournament_calendar_uuid=VALID_TMCL_UUID, contestant_uuid=VALID_CONTESTANT_UUID
    )
    data = flow.collect()
    assert data is not None
    assert len(data) > 0
    # Check for keys from extract_season_player_stats
    assert "id" in data[0]
    assert "_competition" in data[0]


@pytest.mark.vcr
def test_parser_tm4_team_season_stats():
    """
    Tests: _handle_non_paginated_endpoint -> extract_season_team_stats
    """
    flow = opta.team_season_stats(
        tournament_calendar_uuid=VALID_TMCL_UUID, contestant_uuid=VALID_CONTESTANT_UUID
    )
    data = flow.collect()

    assert data is not None
    assert len(data) == 1  # Should be one record for the team
    # Check for keys from extract_season_team_stats
    assert "id" in data[0]
    assert "name" in data[0]
    assert "_competition" in data[0]
    assert "_tournamentCalendar" in data[0]


@pytest.mark.vcr
def test_parser_tm16_contestant_participation():
    """
    Tests: _handle_non_paginated_endpoint -> extract_contestant_participation
    """
    flow = opta.contestant_participation(contestant_uuid=VALID_CONTESTANT_UUID)
    data = flow.collect()
    assert data is not None
    assert len(data) > 0
    # Check parser logic
    assert "id" in data[0] and "name" in data[0]


@pytest.mark.vcr
def test_dynamic_pagination_transfers_paginated():
    """
    Tests: is_paginated('transfers', ...) == True
    """
    flow = opta.transfers(contestant_uuid=VALID_CONTESTANT_UUID)
    data = flow.collect()
    assert data is not None
    assert "id" in data[0]


@pytest.mark.vcr
def test_dynamic_pagination_transfers_non_paginated():
    """
    Tests: is_paginated('transfers', ...) == False
    Logic: Uses 'person_uuid' to trigger non-paginated path
    """
    flow = opta.transfers(person_uuid=VALID_PERSON_UUID)
    data = flow.collect()

    assert data is not None
    assert "id" in data[0]
    assert "firstName" in data[0]
    assert "lastName" in data[0]


@pytest.mark.vcr
def test_params_tournament_calendars_status():
    """
    Tests: 'status' param in tournament_calendars
    """
    flow = opta.tournament_calendars(status="active", include_stages=True)
    data = flow.collect()
    assert data is not None
    assert len(data) > 0
    assert "competitionCode" in data[0]


@pytest.mark.vcr
def test_params_events_by_type():
    """
    Tests: 'event_types' param in events
    """
    # Event Type 1 = Pass, 16 = Goal
    flow = opta.events(fixture_uuid=VALID_FIXTURE_UUID, event_types=[1, 16])
    data = flow.collect()
    assert data is not None

    type_ids = {e["typeId"] for e in data}
    assert 1 in type_ids
    assert 16 in type_ids
    assert len(type_ids) == 2  # Should only have Passes and Goals


@pytest.mark.vcr
def test_error_handling_404_not_found():
    """
    Tests: OptaClient.make_request for 404
    """
    from penaltyblog.matchflow.steps.opta.exceptions import OptaRequestError

    flow = opta.match(fixture_uuid="this-is-not-a-real-uuid")

    with pytest.raises(OptaRequestError, match="404 Not Found"):
        flow.collect()


@pytest.mark.vcr
def test_error_handling_auth_failure(monkeypatch):
    """
    Tests: OptaClient.make_request for auth error
    """
    from penaltyblog.matchflow.steps.opta.exceptions import OptaRequestError

    # Temporarily patch the default creds to be invalid
    # This only affects this test
    invalid_creds = {
        "auth_key": "this-is-a-bad-key",
        "rt_mode": "b",
    }

    flow = opta.tournament_calendars(creds=invalid_creds)

    # The API might return 401, 403, or another HTTP error
    with pytest.raises(OptaRequestError):
        flow.collect()
