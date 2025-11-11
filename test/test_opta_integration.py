# tests/test_opta_integration.py
from datetime import datetime

import pytest

from penaltyblog.matchflow.contrib.opta import opta

VALID_TMCL_UUID = "51r6ph2woavlbbpk8f29nynf8"
VALID_COMPETITION_UUID = "2kwbbcootiqqgmrzs6o5inle5"
VALID_FIXTURE_UUID = "zhs8gg1hvcuqvhkk2itb54pg"
VALID_FIXTURE_UUID2 = "102dto55773ex3p58gp94ql90"
VALID_CONTESTANT_UUID = "c8h9bw1l82s06h77xxrelzhur"
VALID_CONTESTANT_UUID2 = "1pse9ta7a45pi2w2grjim70ge"
VALID_PERSON_UUID = "5ilkkfbsss0bxd6ttdlqg0uz9"
VALID_VENUE_UUID = "bxpq91vq4x9r3q6eq3d0bwjuy"
VALID_AREA_UUID = "7yck0z0f9rlpeyatanjc1ylzp"
VALID_COUNTRY_UUID = "1fk5l4hkqk12i7zske6mcqju6"
VALID_STAGE_UUID = "8qkvlx3f1s7z1h8r4y0g5p2s3"
VALID_SERIES_UUID = "9wmx8g4h2t3y1j7k6l5p4o1n"
VALID_REFEREE_UUID = "eh83vbp5w44denpmo9kaaj8ph"


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
    ids = [x["contestantId"] for x in data]
    assert all(i == VALID_CONTESTANT_UUID for i in ids)


@pytest.mark.vcr
def test_parser_ma3_match_events_person():
    """
    Tests: _handle_non_paginated_endpoint -> extract_match_events
    """
    flow = opta.events(fixture_uuid=VALID_FIXTURE_UUID, person_uuid=VALID_PERSON_UUID)
    data = flow.collect()
    assert data is not None
    assert len(data) == 70
    assert "typeId" in data[0] and "_match_info" in data[0]
    ids = [x["playerId"] for x in data]
    assert all(i == VALID_PERSON_UUID for i in ids)


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
    assert len(data) == 28
    assert "id" in data[0]
    assert "_competition" in data[0]
    ids = [x["_tournamentCalendar"]["id"] for x in data]
    assert all(i == VALID_TMCL_UUID for i in ids)


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
    assert len(data) == 1
    assert "id" in data[0]
    assert "name" in data[0]
    assert "_competition" in data[0]
    assert "_tournamentCalendar" in data[0]
    ids = [x["_tournamentCalendar"]["id"] for x in data]
    assert all(i == VALID_TMCL_UUID for i in ids)


@pytest.mark.vcr
def test_parser_tm16_contestant_participation():
    """
    Tests: _handle_non_paginated_endpoint -> extract_contestant_participation
    """
    flow = opta.contestant_participation(contestant_uuid=VALID_CONTESTANT_UUID)
    data = flow.collect()
    assert data is not None
    assert len(data) == 1
    assert "id" in data[0] and "name" in data[0]
    assert data[0]["id"] == VALID_CONTESTANT_UUID


@pytest.mark.vcr
def test_parser_tm16_contestant_participation_multiple():
    """
    Tests: _handle_non_paginated_endpoint -> extract_contestant_participation
    """
    flow = opta.contestant_participation(
        contestant_uuid=[VALID_CONTESTANT_UUID, VALID_CONTESTANT_UUID2]
    )
    data = flow.collect()
    assert data is not None
    assert len(data) == 2
    assert "id" in data[0] and "name" in data[0]
    ids = {x["id"] for x in data}
    assert ids == set([VALID_CONTESTANT_UUID, VALID_CONTESTANT_UUID2])


@pytest.mark.vcr
def test_dynamic_pagination_transfers_paginated():
    """
    Tests: is_paginated('transfers', ...) == True
    """
    flow = opta.transfers(contestant_uuid=VALID_CONTESTANT_UUID)
    data = flow.collect()
    assert len(data) > 0
    assert data is not None
    assert "id" in data[0]


@pytest.mark.vcr
def test_dynamic_pagination_transfers_paginated_tmcl():
    """
    Tests: is_paginated('transfers', ...) == True
    """
    flow = opta.transfers(tournament_calendar_uuid=VALID_TMCL_UUID)
    data = flow.collect()
    assert len(data) > 0
    assert data is not None
    assert "id" in data[0]


@pytest.mark.vcr
def test_dynamic_pagination_transfers_paginated_dates():
    """
    Tests: is_paginated('transfers', ...) == True with valid date parameters
    """
    flow = opta.transfers(
        competition_uuid=VALID_COMPETITION_UUID,
        start_date="2025-08-01",
        end_date="2025-09-01",
    )
    data = flow.collect()
    assert data is not None
    assert isinstance(data, list)
    assert len(data) > 0
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
    assert len(data) == 1
    assert data[0]["id"] == VALID_PERSON_UUID
    assert "firstName" in data[0]
    assert "lastName" in data[0]
    assert len(data[0]["membership"]) > 0


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

    invalid_creds = {
        "auth_key": "this-is-a-bad-key",
        "rt_mode": "b",
    }

    flow = opta.tournament_calendars(creds=invalid_creds)

    # The API might return 401, 403, or another HTTP error
    with pytest.raises(OptaRequestError):
        flow.collect()


def test_transfers_validation_invalid_date_combinations():
    """
    Tests: transfers() method validation for invalid parameter combinations
    """
    # Test: start_date without end_date should raise ValueError
    with pytest.raises(ValueError):
        opta.transfers(
            tournament_calendar_uuid=VALID_TMCL_UUID, start_date="2025-08-01"
        )

    # Test: end_date without start_date should raise ValueError
    with pytest.raises(ValueError):
        opta.transfers(tournament_calendar_uuid=VALID_TMCL_UUID, end_date="2025-09-01")

    # Test: date parameters with tournament_calendar_uuid should raise ValueError
    with pytest.raises(ValueError):
        opta.transfers(
            tournament_calendar_uuid=VALID_TMCL_UUID,
            start_date="2025-08-01",
            end_date="2025-09-01",
        )

    # Test: date parameters without competition_uuid should raise ValueError
    with pytest.raises(ValueError):
        opta.transfers(
            contestant_uuid=VALID_CONTESTANT_UUID,
            start_date="2025-08-01",
            end_date="2025-09-01",
        )

    # Test: date parameters with both tournament_calendar_uuid and competition_uuid should raise ValueError
    with pytest.raises(ValueError):
        opta.transfers(
            tournament_calendar_uuid=VALID_TMCL_UUID,
            competition_uuid="8qkvlx3f1s7z1h8r4y0g5p2s3",
            start_date="2025-08-01",
            end_date="2025-09-01",
        )


def test_transfers_validation_no_filters():
    """
    Tests: transfers() method validation when no filters are provided
    """
    # Test: no parameters should raise ValueError
    with pytest.raises(ValueError):
        opta.transfers()


def test_transfers_validation_valid_combinations():
    """
    Tests: transfers() method should accept valid parameter combinations
    """
    # These should not raise any exceptions
    try:
        # Valid: tournament_calendar_uuid alone
        opta.transfers(tournament_calendar_uuid=VALID_TMCL_UUID)

        # Valid: contestant_uuid alone
        opta.transfers(contestant_uuid=VALID_CONTESTANT_UUID)

        # Valid: person_uuid alone
        opta.transfers(person_uuid=VALID_PERSON_UUID)

        # Valid: competition_uuid alone
        opta.transfers(competition_uuid="8qkvlx3f1s7z1h8r4y0g5p2s3")

        # Valid: competition_uuid with date parameters
        opta.transfers(
            competition_uuid="8qkvlx3f1s7z1h8r4y0g5p2s3",
            start_date="2025-08-01",
            end_date="2025-09-01",
        )

    except ValueError:
        pytest.fail("Valid parameter combinations should not raise ValueError")


# ============================================================================
# NEW TESTS FOR MISSING COVERAGE
# ============================================================================

# --- Player Career Tests (Feed PE2) ---


@pytest.mark.vcr
def test_player_career_person():
    """
    Tests: player_career() with person_uuid (non-paginated)
    """
    flow = opta.player_career(person_uuid=VALID_PERSON_UUID)
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) == 1
    assert "id" in data[0]


@pytest.mark.vcr
def test_player_career_contestant():
    """
    Tests: player_career() with contestant_uuid (paginated)
    """
    flow = opta.player_career(contestant_uuid=VALID_CONTESTANT_UUID)
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) > 1
    assert "id" in data[0]


@pytest.mark.vcr
def test_player_career_contestant_inactive():
    """
    Tests: player_career() with contestant_uuid and active=False
    """
    flow = opta.player_career(contestant_uuid=VALID_CONTESTANT_UUID, active=False)
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) > 1
    assert "id" in data[0]


@pytest.mark.vcr
def test_player_career_opta_names():
    """
    Tests: player_career() with use_opta_names=True
    """
    flow = opta.player_career(person_uuid=VALID_PERSON_UUID, use_opta_names=True)
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) == 1


def test_player_career_validation():
    """
    Tests: player_career() parameter validation
    """
    # Test: no parameters should raise ValueError
    with pytest.raises(
        ValueError, match="Either 'person_uuid' or 'contestant_uuid' must be provided"
    ):
        opta.player_career()

    # Test: both parameters should raise ValueError
    with pytest.raises(
        ValueError, match="Cannot provide both 'person_uuid' and 'contestant_uuid'"
    ):
        opta.player_career(
            person_uuid=VALID_PERSON_UUID, contestant_uuid=VALID_CONTESTANT_UUID
        )


# --- Teams Tests (Feed TM1) ---


@pytest.mark.vcr
def test_teams_tournament_calendar():
    """
    Tests: teams() with tournament_calendar_uuid
    """
    flow = opta.teams(tournament_calendar_uuid=VALID_TMCL_UUID)
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) > 0
    assert "id" in data[0]
    assert "name" in data[0]


@pytest.mark.vcr
def test_teams_contestant():
    """
    Tests: teams() with contestant_uuid
    """
    flow = opta.teams(contestant_uuid=VALID_CONTESTANT_UUID)
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) == 1
    assert "id" in data[0]
    assert "name" in data[0]
    assert data[0]["id"] == VALID_CONTESTANT_UUID


def test_teams_validation():
    """
    Tests: teams() parameter validation
    """
    # Test: neither required parameter should raise ValueError
    with pytest.raises(
        ValueError,
        match="Either 'tournament_calendar_uuid' or 'contestant_uuid' must be provided for the teams feed.",
    ):
        opta.teams()


# --- Squads Tests (Feed TM3) ---


@pytest.mark.vcr
def test_squads_tournament_calendar():
    """
    Tests: squads() with tournament_calendar_uuid
    """
    flow = opta.squads(tournament_calendar_uuid=VALID_TMCL_UUID)
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) == 20
    assert len(data[0]["person"]) > 0


@pytest.mark.vcr
def test_squads_contestant():
    """
    Tests: squads() with contestant_uuid
    """
    flow = opta.squads(contestant_uuid=VALID_CONTESTANT_UUID)
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["contestantId"] == VALID_CONTESTANT_UUID


@pytest.mark.vcr
def test_squads_opta_names():
    """
    Tests: squads() with use_opta_names=True
    """
    flow = opta.squads(contestant_uuid=VALID_CONTESTANT_UUID, use_opta_names=True)
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) == 1


def test_squads_validation():
    """
    Tests: squads() parameter validation
    """
    # Test: neither required parameter should raise ValueError
    with pytest.raises(
        ValueError,
        match="Either 'tournament_calendar_uuid' or 'contestant_uuid' must be provided",
    ):
        opta.squads()


# --- Referees Tests (Feed PE3) ---


@pytest.mark.vcr
def test_referees_person():
    """
    Tests: referees() with person_uuid
    """
    flow = opta.referees(person_uuid=VALID_REFEREE_UUID)
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["id"] == VALID_REFEREE_UUID


@pytest.mark.vcr
def test_referees_tournament_calendar():
    """
    Tests: referees() with tournament_calendar_uuid
    """
    flow = opta.referees(tournament_calendar_uuid=VALID_TMCL_UUID)
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) > 1
    data[0]["id"] == VALID_REFEREE_UUID


@pytest.mark.vcr
def test_referees_opta_names():
    """
    Tests: referees() with use_opta_names=True
    """
    flow = opta.referees(person_uuid=VALID_REFEREE_UUID, use_opta_names=True)
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["id"] == VALID_REFEREE_UUID


def test_referees_validation():
    """
    Tests: referees() parameter validation
    """
    # Test: no parameters should raise ValueError
    with pytest.raises(
        ValueError,
        match="One of 'person_uuid', 'tournament_calendar_uuid', or 'stage_uuid' must be provided",
    ):
        opta.referees()

    # Test: multiple parameters should raise ValueError
    with pytest.raises(
        ValueError,
        match="Only one of 'person_uuid', 'tournament_calendar_uuid', or 'stage_uuid' can be provided at a time",
    ):
        opta.referees(
            person_uuid=VALID_PERSON_UUID, tournament_calendar_uuid=VALID_TMCL_UUID
        )


# --- Injuries Tests (Feed PE7) ---


@pytest.mark.vcr
def test_injuries_person():
    """
    Tests: injuries() with person_uuid only (path-based)
    """
    flow = opta.injuries(person_uuid=VALID_PERSON_UUID)
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) > 0
    assert "id" in data[0]
    assert "name" in data[0]


@pytest.mark.vcr
def test_injuries_tournament_calendar():
    """
    Tests: injuries() with tournament_calendar_uuid only
    """
    flow = opta.injuries(tournament_calendar_uuid=VALID_TMCL_UUID)
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) > 0


@pytest.mark.vcr
def test_injuries_tournament_calendar_contestant():
    """
    Tests: injuries() with tournament_calendar_uuid and contestant_uuid
    """
    flow = opta.injuries(
        tournament_calendar_uuid=VALID_TMCL_UUID, contestant_uuid=VALID_CONTESTANT_UUID
    )
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)


@pytest.mark.vcr
def test_injuries_opta_names():
    """
    Tests: injuries() with use_opta_names=True
    """
    flow = opta.injuries(person_uuid=VALID_PERSON_UUID, use_opta_names=True)
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) > 0


def test_injuries_validation():
    """
    Tests: injuries() parameter validation
    """
    # Test: no parameters should raise ValueError
    with pytest.raises(
        ValueError,
        match="Either 'person_uuid' or 'tournament_calendar_uuid' must be provided",
    ):
        opta.injuries()

    # Test: contestant_uuid without tournament_calendar_uuid should raise ValueError
    with pytest.raises(
        ValueError,
        match="'contestant_uuid' can only be used in combination with 'tournament_calendar_uuid'",
    ):
        opta.injuries(contestant_uuid=VALID_CONTESTANT_UUID)

    # Test: contestant_uuid and person_uuid together should raise ValueError
    with pytest.raises(
        ValueError,
        match="Cannot use 'contestant_uuid' and 'person_uuid' in the same request",
    ):
        opta.injuries(
            person_uuid=VALID_PERSON_UUID, contestant_uuid=VALID_CONTESTANT_UUID
        )


# --- Enhanced Tournament Schedule Tests ---


@pytest.mark.vcr
def test_tournament_schedule_coverage_level_single():
    """
    Tests: tournament_schedule() with single coverage_level
    """
    flow = opta.tournament_schedule(
        tournament_calendar_uuid=VALID_TMCL_UUID, coverage_level=1
    )
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) > 0
    assert "id" in data[0]


@pytest.mark.vcr
def test_tournament_schedule_coverage_level_list():
    """
    Tests: tournament_schedule() with list of coverage_levels
    """
    flow = opta.tournament_schedule(
        tournament_calendar_uuid=VALID_TMCL_UUID, coverage_level=[1, 2, 3]
    )
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) > 0


@pytest.mark.vcr
def test_tournament_schedule_opta_names():
    """
    Tests: tournament_schedule() with use_opta_names=True
    """
    flow = opta.tournament_schedule(
        tournament_calendar_uuid=VALID_TMCL_UUID, use_opta_names=True
    )
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) > 0


# --- Enhanced Matches Tests ---


@pytest.mark.vcr
def test_matches_competition_uuids():
    """
    Tests: matches() with competition_uuids parameter
    """
    flow = opta.matches(competition_uuids=[VALID_COMPETITION_UUID])
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) > 0
    assert "matchInfo" in data[0]


@pytest.mark.vcr
def test_matches_opponent_uuid():
    """
    Tests: matches() with opponent_uuid parameter
    """
    flow = opta.matches(
        tournament_calendar_uuid=VALID_TMCL_UUID,
        contestant_uuid=VALID_CONTESTANT_UUID,
        opponent_uuid=VALID_CONTESTANT_UUID2,
    )
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert "matchInfo" in data[0]


@pytest.mark.vcr
def test_matches_contestant_position_away():
    """
    Tests: matches() with contestant_position="away"
    """
    flow = opta.matches(
        tournament_calendar_uuid=VALID_TMCL_UUID,
        contestant_uuid=VALID_CONTESTANT_UUID,
        contestant_position="away",
    )
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) == 19  # Should be away games
    assert "matchInfo" in data[0]


@pytest.mark.vcr
def test_matches_delta_timestamp():
    """
    Tests: matches() with delta_timestamp parameter
    """
    flow = opta.matches(
        tournament_calendar_uuid=VALID_TMCL_UUID, delta_timestamp="2025-09-01T00:00:00Z"
    )
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert "matchInfo" in data[0]


# --- Enhanced Tournament Calendars Tests ---


@pytest.mark.vcr
def test_tournament_calendars_competition_uuid():
    """
    Tests: tournament_calendars() with competition_uuid parameter
    """
    flow = opta.tournament_calendars(competition_uuid=VALID_COMPETITION_UUID)
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) > 0
    assert "competitionCode" in data[0]


@pytest.mark.vcr
def test_tournament_calendars_contestant_uuid():
    """
    Tests: tournament_calendars() with contestant_uuid parameter
    """
    flow = opta.tournament_calendars(contestant_uuid=VALID_CONTESTANT_UUID)
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) > 0
    assert "competitionCode" in data[0]


@pytest.mark.vcr
def test_tournament_calendars_include_coverage():
    """
    Tests: tournament_calendars() with include_coverage=True
    """
    flow = opta.tournament_calendars(status="active", include_coverage=True)
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) > 0
    assert "competitionCode" in data[0]


# --- Enhanced Events Tests ---


@pytest.mark.vcr
def test_events_event_types_single():
    """
    Tests: events() with single event_type
    """
    flow = opta.events(fixture_uuid=VALID_FIXTURE_UUID, event_types=1)  # Pass events
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) > 0
    assert "typeId" in data[0]
    assert all(event["typeId"] == 1 for event in data)


@pytest.mark.vcr
def test_events_opta_names():
    """
    Tests: events() with use_opta_names=True
    """
    flow = opta.events(fixture_uuid=VALID_FIXTURE_UUID, use_opta_names=True)
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) > 0
    assert "typeId" in data[0]


def test_events_validation():
    """
    Tests: events() parameter validation for invalid combinations
    """
    # Note: events() doesn't have explicit validation in the current implementation
    # This test is a placeholder for future validation if added
    pass


# --- Enhanced Rankings Tests ---


@pytest.mark.vcr
def test_rankings_tmcl():
    """
    Tests: rankings() with tournament_calendar_uuid
    """
    flow = opta.rankings(tournament_calendar_uuid=VALID_TMCL_UUID)
    data = flow.collect()

    assert data is not None
    assert isinstance(data, list)
    assert len(data) > 0
    # Check for expected fields from the rankings parser
    assert "_record_type" in data[0]
    assert "_competition" in data[0]
    assert "_tournament_calendar" in data[0]

    # For match records, check that team information is extracted
    match_records = [r for r in data if r.get("_record_type") == "match"]
    if match_records:
        assert "_home_team_id" in match_records[0]
        assert "_away_team_id" in match_records[0]
