import pytest

from penaltyblog.matchflow.steps.opta.endpoints import OptaEndpointBuilder
from penaltyblog.matchflow.steps.opta.exceptions import OptaConfigurationError


class TestOptaEndpointBuilder:
    """Test cases for OptaEndpointBuilder class."""

    def test_init(self):
        """Test OptaEndpointBuilder initialization."""
        builder = OptaEndpointBuilder(
            base_url="http://api.test.com", asset_type="soccerdata", auth_key="test_key"
        )

        assert builder.base_url == "http://api.test.com/soccerdata"
        assert builder.auth_key == "test_key"

    def test_build_request_details_unknown_source(self):
        """Test build_request_details with unknown source."""
        builder = OptaEndpointBuilder(
            base_url="http://api.test.com", asset_type="soccerdata", auth_key="test_key"
        )

        with pytest.raises(OptaConfigurationError, match="Unknown Opta source type"):
            builder.build_request_details("unknown_source", {})

    def test_build_request_details_tournament_calendars_active(self):
        """Test tournament_calendars with active status."""
        builder = OptaEndpointBuilder(
            base_url="http://api.test.com", asset_type="soccerdata", auth_key="test_key"
        )

        url, params = builder.build_request_details(
            "tournament_calendars", {"status": "active"}
        )

        assert (
            url == "http://api.test.com/soccerdata/tournamentcalendar/test_key/active"
        )
        assert params["_fmt"] == "json"

    def test_build_request_details_tournament_calendars_all(self):
        """Test tournament_calendars with all status."""
        builder = OptaEndpointBuilder(
            base_url="http://api.test.com", asset_type="soccerdata", auth_key="test_key"
        )

        url, params = builder.build_request_details(
            "tournament_calendars", {"status": "all"}
        )

        assert url == "http://api.test.com/soccerdata/tournamentcalendar/test_key"
        assert params["_fmt"] == "json"

    def test_build_request_details_tournament_schedule(self):
        """Test tournament_schedule endpoint."""
        builder = OptaEndpointBuilder(
            base_url="http://api.test.com", asset_type="soccerdata", auth_key="test_key"
        )

        url, params = builder.build_request_details(
            "tournament_schedule", {"tournament_calendar_uuid": "tmcl123"}
        )

        assert (
            url == "http://api.test.com/soccerdata/tournamentschedule/test_key/tmcl123"
        )
        assert params["_fmt"] == "json"

    def test_build_request_details_tournament_schedule_missing_uuid(self):
        """Test tournament_schedule with missing tournament_calendar_uuid."""
        builder = OptaEndpointBuilder(
            base_url="http://api.test.com", asset_type="soccerdata", auth_key="test_key"
        )

        with pytest.raises(
            OptaConfigurationError,
            match="tournament_schedule source requires 'tournament_calendar_uuid'",
        ):
            builder.build_request_details("tournament_schedule", {})

    def test_build_request_details_match_basic(self):
        """Test match_basic endpoint."""
        builder = OptaEndpointBuilder(
            base_url="http://api.test.com", asset_type="soccerdata", auth_key="test_key"
        )

        url, params = builder.build_request_details(
            "match_basic", {"fixture_uuid": "fx123"}
        )

        assert url == "http://api.test.com/soccerdata/match/test_key/fx123"
        assert params["_fmt"] == "json"

    def test_build_request_details_match_basic_missing_uuid(self):
        """Test match_basic with missing fixture_uuid."""
        builder = OptaEndpointBuilder(
            base_url="http://api.test.com", asset_type="soccerdata", auth_key="test_key"
        )

        with pytest.raises(
            OptaConfigurationError,
            match="match_basic source requires 'fixture_uuid'",
        ):
            builder.build_request_details("match_basic", {})

    def test_build_request_details_match_events(self):
        """Test match_events endpoint."""
        builder = OptaEndpointBuilder(
            base_url="http://api.test.com", asset_type="soccerdata", auth_key="test_key"
        )

        url, params = builder.build_request_details(
            "match_events", {"fixture_uuid": "fx123"}
        )

        assert url == "http://api.test.com/soccerdata/matchevent/test_key/fx123"
        assert params["_fmt"] == "json"

    def test_build_request_details_match_events_missing_uuid(self):
        """Test match_events with missing fixture_uuid."""
        builder = OptaEndpointBuilder(
            base_url="http://api.test.com", asset_type="soccerdata", auth_key="test_key"
        )

        with pytest.raises(
            OptaConfigurationError,
            match="match_events source requires 'fixture_uuid'",
        ):
            builder.build_request_details("match_events", {})

    def test_build_request_details_match_stats_basic_single(self):
        """Test match_stats_basic with single fixture_uuid."""
        builder = OptaEndpointBuilder(
            base_url="http://api.test.com", asset_type="soccerdata", auth_key="test_key"
        )

        url, params = builder.build_request_details(
            "match_stats_basic", {"fixture_uuids": "fx123"}
        )

        assert url == "http://api.test.com/soccerdata/matchstats/test_key"
        assert params["_fmt"] == "json"

    def test_build_request_details_match_stats_basic_multiple(self):
        """Test match_stats_basic with multiple fixture_uuids."""
        builder = OptaEndpointBuilder(
            base_url="http://api.test.com", asset_type="soccerdata", auth_key="test_key"
        )

        url, params = builder.build_request_details(
            "match_stats_basic", {"fixture_uuids": ["fx123", "fx456"]}
        )

        assert url == "http://api.test.com/soccerdata/matchstats/test_key"
        assert params["_fmt"] == "json"
        assert params["fx"] == "fx123,fx456"

    def test_build_request_details_matches_basic(self):
        """Test matches_basic endpoint."""
        builder = OptaEndpointBuilder(
            base_url="http://api.test.com", asset_type="soccerdata", auth_key="test_key"
        )

        url, params = builder.build_request_details(
            "matches_basic",
            {
                "tournament_calendar_uuid": "tmcl123",
                "competition_uuids": ["comp1", "comp2"],
                "live": True,
                "lineups": False,
                "use_opta_names": True,
            },
        )

        assert url == "http://api.test.com/soccerdata/match/test_key"
        assert params["_fmt"] == "json"
        assert params["tmcl"] == "tmcl123"
        assert params["comp"] == "comp1,comp2"
        assert params["live"] == "yes"
        assert params["lineups"] == "no"
        assert params["_lcl"] is True

    def test_build_request_details_teams(self):
        """Test teams endpoint."""
        builder = OptaEndpointBuilder(
            base_url="http://api.test.com", asset_type="soccerdata", auth_key="test_key"
        )

        url, params = builder.build_request_details(
            "teams",
            {
                "tournament_calendar_uuid": "tmcl123",
                "contestant_uuid": "ctst123",
                "country_uuid": "ctry123",
            },
        )

        assert url == "http://api.test.com/soccerdata/team/test_key"
        assert params["_fmt"] == "json"
        assert params["tmcl"] == "tmcl123"
        assert params["ctst"] == "ctst123"
        assert params["ctry"] == "ctry123"

    def test_build_request_details_squads(self):
        """Test squads endpoint."""
        builder = OptaEndpointBuilder(
            base_url="http://api.test.com", asset_type="soccerdata", auth_key="test_key"
        )

        url, params = builder.build_request_details(
            "squads",
            {
                "tournament_calendar_uuid": "tmcl123",
                "contestant_uuid": "ctst123",
                "use_opta_names": True,
            },
        )

        assert url == "http://api.test.com/soccerdata/squads/test_key"
        assert params["_fmt"] == "json"
        assert params["tmcl"] == "tmcl123"
        assert params["ctst"] == "ctst123"
        assert params["_lcl"] is True

    def test_build_request_details_player_season_stats(self):
        """Test player_season_stats endpoint."""
        builder = OptaEndpointBuilder(
            base_url="http://api.test.com", asset_type="soccerdata", auth_key="test_key"
        )

        url, params = builder.build_request_details(
            "player_season_stats",
            {
                "tournament_calendar_uuid": "tmcl123",
                "contestant_uuid": "ctst123",
                "detailed": True,
            },
        )

        assert url == "http://api.test.com/soccerdata/seasonstats/test_key"
        assert params["_fmt"] == "json"
        assert params["tmcl"] == "tmcl123"
        assert params["ctst"] == "ctst123"
        assert params["detailed"] == "yes"

    def test_build_request_details_team_season_stats(self):
        """Test team_season_stats endpoint."""
        builder = OptaEndpointBuilder(
            base_url="http://api.test.com", asset_type="soccerdata", auth_key="test_key"
        )

        url, params = builder.build_request_details(
            "team_season_stats",
            {
                "tournament_calendar_uuid": "tmcl123",
                "contestant_uuid": "ctst123",
                "detailed": False,
            },
        )

        assert url == "http://api.test.com/soccerdata/seasonstats/test_key"
        assert params["_fmt"] == "json"
        assert params["tmcl"] == "tmcl123"
        assert params["ctst"] == "ctst123"
        assert params["detailed"] == "no"

    def test_build_parameters_boolean_conversion(self):
        """Test boolean parameter conversion to yes/no."""
        builder = OptaEndpointBuilder(
            base_url="http://api.test.com", asset_type="soccerdata", auth_key="test_key"
        )

        # Test various boolean parameters
        test_cases = [
            ("live", True, "yes"),
            ("live", False, "no"),
            ("lineups", True, "yes"),
            ("lineups", False, "no"),
            ("detailed", True, "yes"),
            ("detailed", False, "no"),
        ]

        for param_name, bool_value, expected_str in test_cases:
            url, params = builder.build_request_details(
                (
                    "matches_basic"
                    if param_name in ["live", "lineups"]
                    else "player_season_stats"
                ),
                {param_name: bool_value},
            )
            assert params[param_name] == expected_str

    def test_build_parameters_list_conversion(self):
        """Test list parameter conversion to comma-separated strings."""
        builder = OptaEndpointBuilder(
            base_url="http://api.test.com", asset_type="soccerdata", auth_key="test_key"
        )

        url, params = builder.build_request_details(
            "matches_basic",
            {
                "competition_uuids": ["comp1", "comp2", "comp3"],
                "event_types": [1, 2, 3],
            },
        )

        assert params["comp"] == "comp1,comp2,comp3"
        # event_types is not in matches_basic mapping, so it should be ignored

    def test_build_parameters_event_types_list(self):
        """Test event_types list conversion."""
        builder = OptaEndpointBuilder(
            base_url="http://api.test.com", asset_type="soccerdata", auth_key="test_key"
        )

        url, params = builder.build_request_details(
            "match_events", {"fixture_uuid": "fx123", "event_types": [1, 2, 3]}
        )

        assert params["type"] == "1,2,3"

    def test_build_parameters_none_values_filtered(self):
        """Test that None values are filtered out of parameters."""
        builder = OptaEndpointBuilder(
            base_url="http://api.test.com", asset_type="soccerdata", auth_key="test_key"
        )

        url, params = builder.build_request_details(
            "matches_basic",
            {
                "tournament_calendar_uuid": "tmcl123",
                "contestant_uuid": None,  # Should be filtered out
                "live": None,  # Should be filtered out
                "use_opta_names": None,  # Should be filtered out
            },
        )

        assert "tmcl" in params
        assert "ctst" not in params
        assert "live" not in params
        assert "_lcl" not in params

    def test_build_request_details_tournament_calendars_extra_params(self):
        """Test tournament_calendars with extra parameters."""
        builder = OptaEndpointBuilder(
            base_url="http://api.test.com", asset_type="soccerdata", auth_key="test_key"
        )

        url, params = builder.build_request_details(
            "tournament_calendars",
            {
                "status": "active",
                "comp": "comp123",
                "ctst": "ctst123",
                "stages": "yes",
                "coverage": "yes",
            },
        )

        assert params["comp"] == "comp123"
        assert params["ctst"] == "ctst123"
        assert params["stages"] == "yes"
        assert params["coverage"] == "yes"

    def test_extract_path_params_tournament_schedule(self):
        """Test _extract_path_params for tournament_schedule."""
        builder = OptaEndpointBuilder(
            base_url="http://api.test.com", asset_type="soccerdata", auth_key="test_key"
        )

        args = {"tournament_calendar_uuid": "tmcl123"}
        path_params = builder._extract_path_params("tournament_schedule", args)

        assert path_params == {"tournament_calendar_uuid": "tmcl123"}

    def test_extract_path_params_match_basic(self):
        """Test _extract_path_params for match_basic."""
        builder = OptaEndpointBuilder(
            base_url="http://api.test.com", asset_type="soccerdata", auth_key="test_key"
        )

        args = {"fixture_uuid": "fx123"}
        path_params = builder._extract_path_params("match_basic", args)

        assert path_params == {"fixture_uuid": "fx123"}

    def test_extract_path_params_match_events(self):
        """Test _extract_path_params for match_events."""
        builder = OptaEndpointBuilder(
            base_url="http://api.test.com", asset_type="soccerdata", auth_key="test_key"
        )

        args = {"fixture_uuid": "fx123"}
        path_params = builder._extract_path_params("match_events", args)

        assert path_params == {"fixture_uuid": "fx123"}

    def test_extract_path_params_match_stats_basic_single(self):
        """Test _extract_path_params for match_stats_basic with single UUID."""
        builder = OptaEndpointBuilder(
            base_url="http://api.test.com", asset_type="soccerdata", auth_key="test_key"
        )

        args = {"fixture_uuids": "fx123"}
        path_params = builder._extract_path_params("match_stats_basic", args)

        assert path_params == {"fixture_uuids": "fx123"}

    def test_extract_path_params_match_stats_basic_multiple(self):
        """Test _extract_path_params for match_stats_basic with multiple UUIDs."""
        builder = OptaEndpointBuilder(
            base_url="http://api.test.com", asset_type="soccerdata", auth_key="test_key"
        )

        args = {"fixture_uuids": ["fx123", "fx456"]}
        path_params = builder._extract_path_params("match_stats_basic", args)

        assert path_params == {}  # Multiple UUIDs go in params, not path

    def test_build_parameters_default_format(self):
        """Test that _fmt=json is always included."""
        builder = OptaEndpointBuilder(
            base_url="http://api.test.com", asset_type="soccerdata", auth_key="test_key"
        )

        url, params = builder.build_request_details("teams", {})

        assert params["_fmt"] == "json"

    def test_build_request_details_empty_args(self):
        """Test build_request_details with minimal args."""
        builder = OptaEndpointBuilder(
            base_url="http://api.test.com", asset_type="soccerdata", auth_key="test_key"
        )

        # Test with sources that don't require path parameters
        url, params = builder.build_request_details("teams", {})

        assert url == "http://api.test.com/soccerdata/team/test_key"
        assert params["_fmt"] == "json"
