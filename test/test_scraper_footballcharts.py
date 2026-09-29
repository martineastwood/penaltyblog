from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

import penaltyblog as pb

RESULTS = {
    "league": "premier",
    "season": "2025-2026",
    "matches": [
        {
            "id": 123,
            "date": "2025-08-01",
            "time": "19:00:00",
            "homeTeam": "Man Utd",
            "awayTeam": "Wolves",
            "score": "2:1",
            "ht_result": "1:0",
            "first_goal_time": 10,
        },
        {
            "id": 124,
            "date": "2025-08-02",
            "time": "15:00:00",
            "homeTeam": "Arsenal",
            "awayTeam": "Fulham",
            "score": None,
            "ht_result": None,
            "first_goal_time": None,
        },
    ],
}


def _response(payload=None, status_code=200):
    response = Mock()
    response.status_code = status_code
    response.json.return_value = payload
    return response


def test_footballcharts_get_fixtures_normalizes_results():
    response = _response(RESULTS)
    mappings = {
        "Manchester United": ["Man Utd"],
        "Wolverhampton Wanderers": ["Wolves"],
    }

    with patch(
        "penaltyblog.scrapers.footballcharts.requests.get",
        return_value=response,
    ) as mock_get:
        df = pb.scrapers.FootballCharts(
            "ENG Premier League",
            "2025-2026",
            team_mappings=mappings,
        ).get_fixtures()

    mock_get.assert_called_once_with(
        "https://footballcharts-backend.onrender.com/api/v1/leagues/premier/results/",
        params={"season": "2025-2026"},
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/102.0.0.0 Safari/537.36"
            )
        },
        timeout=30,
    )

    assert len(df) == 1
    assert df.index.name == "id"
    assert "1754006400---manchester_united---wolverhampton_wanderers" in df.index
    row = df.iloc[0]
    assert row["team_home"] == "Manchester United"
    assert row["team_away"] == "Wolverhampton Wanderers"
    assert row["goals_home"] == row["fthg"] == 2
    assert row["goals_away"] == row["ftag"] == 1
    assert row["hthg"] == 1
    assert row["htag"] == 0
    assert row["first_goal_minute"] == 10
    assert row["footballcharts_id"] == 123
    assert row["datetime"] == pd.Timestamp("2025-08-01 19:00:00")
    assert df.attrs["attribution"] == "Data by football-charts.com"


def test_footballcharts_from_league_supports_fc_native_codes():
    response = _response(
        {
            "matches": [
                {
                    "id": 456,
                    "date": "2025-08-01",
                    "time": "19:00:00",
                    "homeTeam": "RW Essen",
                    "awayTeam": "Munich 1860",
                    "score": "1:1",
                    "ht_result": "1:0",
                    "first_goal_time": 6,
                }
            ]
        }
    )

    with patch(
        "penaltyblog.scrapers.footballcharts.requests.get",
        return_value=response,
    ) as mock_get:
        scraper = pb.scrapers.FootballCharts.from_league("germany3", "2025-2026")
        df = scraper.get_fixtures()

    assert scraper.competition == "germany3"
    assert scraper.league == "germany3"
    assert df.iloc[0]["competition"] == "germany3"
    assert mock_get.call_args.kwargs["params"] == {"season": "2025-2026"}
    assert "leagues/germany3/results/" in mock_get.call_args.args[0]


def test_footballcharts_list_leagues_and_api_key_precedence(monkeypatch):
    response = _response(
        {
            "leagues": [
                {
                    "league": "germany3",
                    "name": "3. Liga",
                    "country": "Germany",
                    "seasons": ["2025-2026"],
                    "url": "https://example.test/germany3",
                }
            ]
        }
    )
    monkeypatch.setenv("FC_API_KEY", "environment-key")

    with patch(
        "penaltyblog.scrapers.footballcharts.requests.get",
        return_value=response,
    ) as mock_get:
        df = pb.scrapers.FootballCharts.list_leagues(api_key="explicit-key")

    assert list(df.columns) == ["league", "name", "country", "seasons"]
    assert df.iloc[0].to_dict() == {
        "league": "germany3",
        "name": "3. Liga",
        "country": "Germany",
        "seasons": ["2025-2026"],
    }
    assert mock_get.call_args.kwargs["headers"]["Authorization"] == (
        "Bearer explicit-key"
    )


def test_footballcharts_empty_results_have_schema_and_attribution():
    response = _response({"matches": []})

    with patch(
        "penaltyblog.scrapers.footballcharts.requests.get",
        return_value=response,
    ):
        df = pb.scrapers.FootballCharts(
            "ENG Premier League", "2025-2026"
        ).get_fixtures()

    assert df.empty
    assert df.index.name == "id"
    assert "footballcharts_id" in df.columns
    assert df.attrs["attribution"] == "Data by football-charts.com"


def test_footballcharts_403_is_a_permission_error():
    response = _response(status_code=403)

    with patch(
        "penaltyblog.scrapers.footballcharts.requests.get",
        return_value=response,
    ):
        with pytest.raises(PermissionError, match="paid API key"):
            pb.scrapers.FootballCharts("ENG Premier League", "2020-2021").get_fixtures()


def test_footballcharts_other_http_errors_are_preserved():
    response = _response(status_code=429)
    response.raise_for_status.side_effect = requests.HTTPError("rate limited")

    with patch(
        "penaltyblog.scrapers.footballcharts.requests.get",
        return_value=response,
    ):
        with pytest.raises(requests.HTTPError, match="rate limited"):
            pb.scrapers.FootballCharts("ENG Premier League", "2025-2026").get_fixtures()


def test_footballcharts_invalid_fc_native_code():
    with pytest.raises(ValueError, match="non-empty"):
        pb.scrapers.FootballCharts.from_league("", "2025-2026")
