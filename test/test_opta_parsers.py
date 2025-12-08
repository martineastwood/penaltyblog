import pytest

from penaltyblog.matchflow.steps.opta.parsers import (
    extract_match_events,
    extract_player_stats,
    extract_season_player_stats,
    extract_season_team_stats,
    extract_team_stats,
    flatten_stats,
    parse_match_basic,
    parse_match_stats_basic,
    parse_tournament_schedule,
)


class TestFlattenStats:
    """Test cases for flatten_stats function."""

    def test_flatten_stats_basic(self):
        """Test basic stats flattening."""
        stats_list = [
            {"type": "goals", "value": "5"},
            {"type": "assists", "value": "3"},
            {"type": "minutes", "value": "90"},
        ]

        result = flatten_stats(stats_list)

        expected = {
            "goals": "5",
            "assists": "3",
            "minutes": "90",
        }
        assert result == expected

    def test_flatten_stats_with_name_key(self):
        """Test stats flattening with 'name' key."""
        stats_list = [
            {"name": "goals", "value": "5"},
            {"name": "assists", "value": "3"},
        ]

        result = flatten_stats(stats_list, key_name="name")

        expected = {
            "goals": "5",
            "assists": "3",
        }
        assert result == expected

    def test_flatten_stats_with_float_conversion(self):
        """Test stats flattening with float conversion."""
        stats_list = [
            {"type": "rating", "value": "7.5"},
            {"type": "accuracy", "value": "85.2"},
        ]

        result = flatten_stats(stats_list)

        expected = {
            "rating": 7.5,
            "accuracy": 85.2,
        }
        assert result == expected

    def test_flatten_stats_with_invalid_float(self):
        """Test stats flattening with invalid float values."""
        stats_list = [
            {"type": "goals", "value": "5"},
            {"type": "name", "value": "Player Name"},
            {"type": "invalid", "value": "not_a_number"},
        ]

        result = flatten_stats(stats_list)

        expected = {
            "goals": "5",
            "name": "Player Name",
            "invalid": "not_a_number",
        }
        assert result == expected

    def test_flatten_stats_empty_list(self):
        """Test stats flattening with empty list."""
        result = flatten_stats([])
        assert result == {}

    def test_flatten_stats_missing_keys(self):
        """Test stats flattening with missing keys."""
        stats_list = [
            {"type": "goals", "value": "5"},
            {"value": "3"},  # Missing type
            {"type": "assists"},  # Missing value
        ]

        result = flatten_stats(stats_list)

        expected = {
            "goals": "5",
            "assists": None,
        }
        assert result == expected

    def test_flatten_stats_none_values(self):
        """Test stats flattening with None values."""
        stats_list = [
            {"type": "goals", "value": None},
            {"type": "assists", "value": "3"},
        ]

        result = flatten_stats(stats_list)

        expected = {
            "goals": None,
            "assists": "3",
        }
        assert result == expected


class TestExtractPlayerStats:
    """Test cases for extract_player_stats function."""

    def test_extract_player_stats_basic(self):
        """Test basic player stats extraction."""
        match = {
            "matchInfo": {"id": "match123"},
            "liveData": {
                "lineUp": [
                    {
                        "contestantId": "team1",
                        "player": [
                            {
                                "playerId": "player1",
                                "playerName": "Player 1",
                                "stat": [
                                    {"type": "goals", "value": "2"},
                                    {"type": "assists", "value": "1"},
                                ],
                            }
                        ],
                    }
                ]
            },
        }

        results = list(extract_player_stats(match))

        assert len(results) == 1
        player = results[0]
        assert player["playerId"] == "player1"
        assert player["playerName"] == "Player 1"
        assert player["goals"] == "2"
        assert player["assists"] == "1"
        assert player["_match_uuid"] == "match123"
        assert player["_contestant_id"] == "team1"
        assert "_match_info" in player
        assert "stat" not in player

    def test_extract_player_stats_multiple_teams(self):
        """Test player stats extraction with multiple teams."""
        match = {
            "matchInfo": {"id": "match123"},
            "liveData": {
                "lineUp": [
                    {
                        "contestantId": "team1",
                        "player": [
                            {
                                "playerId": "player1",
                                "stat": [{"type": "goals", "value": "1"}],
                            }
                        ],
                    },
                    {
                        "contestantId": "team2",
                        "player": [
                            {
                                "playerId": "player2",
                                "stat": [{"type": "goals", "value": "2"}],
                            }
                        ],
                    },
                ]
            },
        }

        results = list(extract_player_stats(match))

        assert len(results) == 2
        assert results[0]["playerId"] == "player1"
        assert results[0]["_contestant_id"] == "team1"
        assert results[1]["playerId"] == "player2"
        assert results[1]["_contestant_id"] == "team2"

    def test_extract_player_stats_no_stats(self):
        """Test player stats extraction with no stats."""
        match = {
            "matchInfo": {"id": "match123"},
            "liveData": {
                "lineUp": [
                    {
                        "contestantId": "team1",
                        "player": [{"playerId": "player1"}],  # No stats
                    }
                ]
            },
        }

        results = list(extract_player_stats(match))

        assert len(results) == 0

    def test_extract_player_stats_empty_lineup(self):
        """Test player stats extraction with empty lineup."""
        match = {"matchInfo": {"id": "match123"}, "liveData": {"lineUp": []}}

        results = list(extract_player_stats(match))

        assert len(results) == 0

    def test_extract_player_stats_missing_live_data(self):
        """Test player stats extraction with missing liveData."""
        match = {"matchInfo": {"id": "match123"}}

        results = list(extract_player_stats(match))

        assert len(results) == 0


class TestExtractTeamStats:
    """Test cases for extract_team_stats function."""

    def test_extract_team_stats_basic(self):
        """Test basic team stats extraction."""
        match = {
            "matchInfo": {"id": "match123"},
            "liveData": {
                "lineUp": [
                    {
                        "contestantId": "team1",
                        "stat": [
                            {"type": "goals", "value": "2"},
                            {"type": "possession", "value": "65"},
                        ],
                    }
                ]
            },
        }

        results = list(extract_team_stats(match))

        assert len(results) == 1
        team = results[0]
        assert team["contestantId"] == "team1"
        assert team["goals"] == "2"
        assert team["possession"] == "65"
        assert team["_match_uuid"] == "match123"
        assert "_match_info" in team
        assert "stat" not in team

    def test_extract_team_stats_multiple_teams(self):
        """Test team stats extraction with multiple teams."""
        match = {
            "matchInfo": {"id": "match123"},
            "liveData": {
                "lineUp": [
                    {
                        "contestantId": "team1",
                        "stat": [{"type": "goals", "value": "2"}],
                    },
                    {
                        "contestantId": "team2",
                        "stat": [{"type": "goals", "value": "1"}],
                    },
                ]
            },
        }

        results = list(extract_team_stats(match))

        assert len(results) == 2
        assert results[0]["contestantId"] == "team1"
        assert results[0]["goals"] == "2"
        assert results[1]["contestantId"] == "team2"
        assert results[1]["goals"] == "1"

    def test_extract_team_stats_no_stats(self):
        """Test team stats extraction with no stats."""
        match = {"matchInfo": {"id": "match123"}, "liveData": {"teamStats": []}}

        results = list(extract_team_stats(match))

        assert len(results) == 0


class TestExtractMatchEvents:
    """Test cases for extract_match_events function."""

    def test_extract_match_events_basic(self):
        """Test basic match events extraction."""
        data = {
            "matchInfo": {"id": "match123"},
            "liveData": {
                "matchDetails": {"score": "2-1"},
                "event": [
                    {"id": "event1", "type": "goal", "minute": "45"},
                    {"id": "event2", "type": "yellow_card", "minute": "60"},
                ],
            },
        }

        results = list(extract_match_events(data))

        assert len(results) == 2
        assert results[0]["id"] == "event1"
        assert results[0]["type"] == "goal"
        assert results[0]["_match_info"]["id"] == "match123"
        assert results[0]["_match_details"]["score"] == "2-1"

    def test_extract_match_events_nested_structure(self):
        """Test match events extraction with nested structure."""
        data = {
            "matchInfo": {"id": "match123"},
            "liveData": {
                "matchDetails": {"score": "1-0"},
                "events": {"event": [{"id": "event1", "type": "goal", "minute": "30"}]},
            },
        }

        results = list(extract_match_events(data))

        assert len(results) == 1
        assert results[0]["id"] == "event1"
        assert results[0]["type"] == "goal"

    def test_extract_match_events_no_events(self):
        """Test match events extraction with no events."""
        data = {"matchInfo": {"id": "match123"}, "liveData": {"matchDetails": {}}}

        results = list(extract_match_events(data))

        assert len(results) == 0

    def test_extract_match_events_missing_live_data(self):
        """Test match events extraction with missing liveData."""
        data = {"matchInfo": {"id": "match123"}}

        results = list(extract_match_events(data))

        assert len(results) == 0


class TestExtractSeasonPlayerStats:
    """Test cases for extract_season_player_stats function."""

    def test_extract_season_player_stats_basic(self):
        """Test basic season player stats extraction."""
        data = {
            "competition": {"id": "comp1", "name": "Premier League"},
            "tournamentCalendar": {"id": "tmcl1", "name": "2023/24"},
            "player": [
                {
                    "playerId": "player1",
                    "playerName": "Player 1",
                    "stat": [
                        {"name": "goals", "value": "10"},
                        {"name": "assists", "value": "5"},
                    ],
                }
            ],
        }

        results = list(extract_season_player_stats(data))

        assert len(results) == 1
        player = results[0]
        assert player["playerId"] == "player1"
        assert player["playerName"] == "Player 1"
        assert player["goals"] == "10"
        assert player["assists"] == "5"
        assert player["_competition"]["id"] == "comp1"
        assert player["_tournamentCalendar"]["id"] == "tmcl1"
        assert "stat" not in player

    def test_extract_season_player_stats_multiple_players(self):
        """Test season player stats extraction with multiple players."""
        data = {
            "competition": {"id": "comp1"},
            "tournamentCalendar": {"id": "tmcl1"},
            "player": [
                {"playerId": "player1", "stat": [{"name": "goals", "value": "10"}]},
                {"playerId": "player2", "stat": [{"name": "goals", "value": "5"}]},
            ],
        }

        results = list(extract_season_player_stats(data))

        assert len(results) == 2
        assert results[0]["playerId"] == "player1"
        assert results[0]["goals"] == "10"
        assert results[1]["playerId"] == "player2"
        assert results[1]["goals"] == "5"

    def test_extract_season_player_stats_no_player_data(self):
        """Test season player stats extraction with no player data."""
        data = {
            "competition": {"id": "comp1"},
            "tournamentCalendar": {"id": "tmcl1"},
            "player": None,
        }

        results = list(extract_season_player_stats(data))

        assert len(results) == 0


class TestExtractSeasonTeamStats:
    """Test cases for extract_season_team_stats function."""

    def test_extract_season_team_stats_basic(self):
        """Test basic season team stats extraction."""
        data = {
            "competition": {"id": "comp1", "name": "Premier League"},
            "tournamentCalendar": {"id": "tmcl1", "name": "2023/24"},
            "contestant": {
                "contestantId": "team1",
                "contestantName": "Team 1",
                "stat": [
                    {"name": "goals", "value": "50"},
                    {"name": "points", "value": "75"},
                ],
            },
        }

        results = list(extract_season_team_stats(data))

        assert len(results) == 1
        team = results[0]
        assert team["contestantId"] == "team1"
        assert team["contestantName"] == "Team 1"
        assert team["goals"] == "50"
        assert team["points"] == "75"
        assert team["_competition"]["id"] == "comp1"
        assert team["_tournamentCalendar"]["id"] == "tmcl1"
        assert "stat" not in team
        assert "player" not in team

    def test_extract_season_team_stats_with_nested_players(self):
        """Test season team stats extraction with nested player data."""
        data = {
            "competition": {"id": "comp1"},
            "tournamentCalendar": {"id": "tmcl1"},
            "contestant": {
                "contestantId": "team1",
                "stat": [{"name": "goals", "value": "50"}],
                "player": [{"playerId": "player1"}],  # Should be removed
            },
        }

        results = list(extract_season_team_stats(data))

        assert len(results) == 1
        assert "player" not in results[0]

    def test_extract_season_team_stats_no_team_data(self):
        """Test season team stats extraction with no team data."""
        data = {
            "competition": {"id": "comp1"},
            "tournamentCalendar": {"id": "tmcl1"},
            "contestant": None,
        }

        results = list(extract_season_team_stats(data))

        assert len(results) == 0


class TestParseTournamentSchedule:
    """Test cases for parse_tournament_schedule function."""

    def test_parse_tournament_schedule_basic(self):
        """Test basic tournament schedule parsing."""
        data = {
            "competition": {"id": "comp1", "name": "Premier League"},
            "tournamentCalendar": {"id": "tmcl1", "name": "2023/24"},
            "matchDate": [
                {
                    "date": "2023-08-01",
                    "match": [
                        {"id": "match1", "homeTeam": "Team A"},
                        {"id": "match2", "awayTeam": "Team B"},
                    ],
                },
                {
                    "date": "2023-08-02",
                    "match": [{"id": "match3", "homeTeam": "Team C"}],
                },
            ],
        }

        results = list(parse_tournament_schedule(data))

        assert len(results) == 3
        assert results[0]["id"] == "match1"
        assert results[0]["_matchDate"] == "2023-08-01"
        assert results[0]["_competition"]["id"] == "comp1"
        assert results[1]["id"] == "match2"
        assert results[1]["_matchDate"] == "2023-08-01"
        assert results[2]["id"] == "match3"
        assert results[2]["_matchDate"] == "2023-08-02"

    def test_parse_tournament_schedule_empty_dates(self):
        """Test tournament schedule parsing with empty match dates."""
        data = {
            "competition": {"id": "comp1"},
            "tournamentCalendar": {"id": "tmcl1"},
            "matchDate": [],
        }

        results = list(parse_tournament_schedule(data))

        assert len(results) == 0

    def test_parse_tournament_schedule_no_matches_in_date(self):
        """Test tournament schedule parsing with no matches in date."""
        data = {
            "competition": {"id": "comp1"},
            "tournamentCalendar": {"id": "tmcl1"},
            "matchDate": [{"date": "2023-08-01", "match": []}],
        }

        results = list(parse_tournament_schedule(data))

        assert len(results) == 0


class TestParseMatchBasic:
    """Test cases for parse_match_basic function."""

    def test_parse_match_basic(self):
        """Test basic match parsing."""
        data = {"id": "match1", "homeTeam": "Team A", "awayTeam": "Team B"}

        results = list(parse_match_basic(data))

        assert len(results) == 1
        assert results[0]["id"] == "match1"
        assert results[0]["homeTeam"] == "Team A"
        assert results[0]["awayTeam"] == "Team B"


class TestParseMatchStatsBasic:
    """Test cases for parse_match_stats_basic function."""

    def test_parse_match_stats_basic_single_match(self):
        """Test match stats parsing with single match."""
        data = {
            "matchInfo": {"id": "match1"},
            "liveData": {
                "lineUp": [
                    {
                        "contestantId": "team1",
                        "player": [
                            {
                                "playerId": "player1",
                                "stat": [{"type": "goals", "value": "1"}],
                            }
                        ],
                    }
                ]
            },
        }

        results = list(parse_match_stats_basic(data, include_players=True))

        assert len(results) == 1
        assert results[0]["playerId"] == "player1"
        assert results[0]["goals"] == "1"

    def test_parse_match_stats_basic_single_match_no_players(self):
        """Test match stats parsing with single match, no players."""
        data = {
            "matchInfo": {"id": "match1"},
            "liveData": {
                "lineUp": [
                    {"contestantId": "team1", "stat": [{"type": "goals", "value": "2"}]}
                ]
            },
        }

        results = list(parse_match_stats_basic(data, include_players=False))

        assert len(results) == 1
        assert results[0]["contestantId"] == "team1"
        assert results[0]["goals"] == "2"

    def test_parse_match_stats_basic_multiple_matches(self):
        """Test match stats parsing with multiple matches."""
        data = {
            "matchStats": [
                {
                    "matchInfo": {"id": "match1"},
                    "liveData": {
                        "lineUp": [
                            {
                                "contestantId": "team1",
                                "player": [
                                    {
                                        "playerId": "player1",
                                        "stat": [{"type": "goals", "value": "1"}],
                                    }
                                ],
                            }
                        ]
                    },
                },
                {
                    "matchInfo": {"id": "match2"},
                    "liveData": {
                        "lineUp": [
                            {
                                "contestantId": "team2",
                                "player": [
                                    {
                                        "playerId": "player2",
                                        "stat": [{"type": "goals", "value": "2"}],
                                    }
                                ],
                            }
                        ]
                    },
                },
            ]
        }

        results = list(parse_match_stats_basic(data, include_players=True))

        assert len(results) == 2
        assert results[0]["playerId"] == "player1"
        assert results[0]["goals"] == "1"
        assert results[1]["playerId"] == "player2"
        assert results[1]["goals"] == "2"

    def test_parse_match_stats_basic_empty_data(self):
        """Test match stats parsing with empty data."""
        data = {}

        results = list(parse_match_stats_basic(data, include_players=True))

        assert len(results) == 0
