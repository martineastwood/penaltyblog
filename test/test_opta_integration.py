# tests/test_opta_integration.py
import pytest

from penaltyblog.matchflow.contrib.opta import opta

VALID_TMCL_UUID = "51r6ph2woavlbbpk8f29nynf8"
VALID_FIXTURE_UUID = "zhs8gg1hvcuqvhkk2itb54pg"
VALID_CONTESTANT_UUID = "c8h9bw1l82s06h77xxrelzhur"
VALID_PERSON_UUID = "5ilkkfbsss0bxd6ttdlqg0uz9"


# This mark tells vcrpy to record/replay this test
@pytest.mark.vcr
def test_fetch_tournament_calendars():
    flow = opta.tournament_calendars(
        status="active",
        include_stages=True,
    )
    calendars = flow.collect()

    # pytest.set_trace()

    assert calendars is not None
    assert isinstance(calendars, list)
    assert len(calendars) > 0
    assert "competitionCode" in calendars[0]
    assert "competitionType" in calendars[0]


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
def test_parser_ma0_tournament_schedule():
    """
    Tests: _handle_non_paginated_endpoint -> parse_tournament_schedule
    """
    flow = opta.tournament_schedule(tournament_calendar_uuid=VALID_TMCL_UUID)
    data = flow.collect()
    assert data is not None
    assert len(data) > 0
    # Check for your parsed keys
    assert "_competition" in data[0] and "_tournamentCalendar" in data[0]


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
def test_parser_ma2_match_stats_players():
    """
    Tests: _handle_non_paginated_endpoint -> parse_match_stats_basic (with players)
    Logic: Must test the 'include_players=True' path.
    """
    flow = opta.match_stats(fixture_uuids=VALID_FIXTURE_UUID, include_players=True)
    data = flow.collect()
    assert data is not None
    assert len(data) > 0
    # Check for keys from extract_player_stats
    assert "playerId" in data[0] and "_match_uuid" in data[0]


@pytest.mark.vcr
def test_parser_ma2_match_stats_teams():
    """
    Tests: _handle_non_paginated_endpoint -> parse_match_stats_basic (teams only)
    Logic: Must test the 'include_players=False' path.
    """
    flow = opta.match_stats(fixture_uuids=VALID_FIXTURE_UUID, include_players=False)
    data = flow.collect()
    assert data is not None
    assert len(data) == 2  # Should be one record per team
    # Check for keys from extract_team_stats
    assert "contestantId" in data[0] and "_match_uuid" in data[0]


@pytest.mark.vcr
def test_parser_ma3_match_events():
    """
    Tests: _handle_non_paginated_endpoint -> extract_match_events
    """
    flow = opta.events(fixture_uuid=VALID_FIXTURE_UUID)
    data = flow.collect()
    assert data is not None
    assert len(data) > 100
    # Check for your parsed keys
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
    assert "playerId" in data[0] and "_competition" in data[0]


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
    assert "contestantId" in data[0] and "_competition" in data[0]


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
def test_dynamic_pagination_injuries_paginated():
    """
    Tests: is_paginated('injuries_query', ...) == True
    """
    flow = opta.injuries(tournament_calendar_uuid=VALID_TMCL_UUID)
    data = flow.collect()
    assert data is not None
    assert "personId" in data[0]


@pytest.mark.vcr
def test_dynamic_pagination_injuries_non_paginated():
    """
    Tests: is_paginated('injuries_query', ...) == False
    Logic: Uses 'person_uuid' to trigger non-paginated path
    """
    flow = opta.injuries(person_uuid=VALID_PERSON_UUID)
    data = flow.collect()
    assert data is not None
    # This also tests the 'else: yield data' fallback in source_opta.py
    assert "personId" in data[0]


@pytest.mark.vcr
def test_dynamic_pagination_transfers_paginated():
    """
    Tests: is_paginated('transfers', ...) == True
    """
    flow = opta.transfers(contestant_uuid=VALID_CONTESTANT_UUID)
    data = flow.collect()
    assert data is not None
    assert "personId" in data[0]


@pytest.mark.vcr
def test_dynamic_pagination_transfers_non_paginated():
    """
    Tests: is_paginated('transfers', ...) == False
    Logic: Uses 'person_uuid' to trigger non-paginated path
    """
    flow = opta.transfers(person_uuid=VALID_PERSON_UUID)
    data = flow.collect()
    assert data is not None
    # This also tests the 'else: yield data' fallback in source_opta.py
    assert "personId" in data[0]


@pytest.mark.vcr
def test_params_tournament_calendars_status():
    """
    Tests: 'status' param in tournament_calendars
    """
    flow = opta.tournament_calendars(status="active", include_stages=True)
    data = flow.collect()
    assert data is not None
    assert len(data) > 0
    assert "stage" in data[0]  # Test 'include_stages'


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
def test_params_use_opta_names():
    """
    Tests: 'use_opta_names=True' param
    """
    # 'venues' is a good simple endpoint for this
    flow = opta.venues(venue_uuid="your_venue_uuid", use_opta_names=True)
    data = flow.collect()
    assert data is not None
    # No easy assert, but this captures a response with the _lcl=en-op param
    assert "name" in data[0]


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
