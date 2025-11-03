import json
import os
from unittest.mock import MagicMock, patch

import pytest

from penaltyblog.matchflow.steps.source_opta import (
    _build_opta_request_details,
    from_opta,
)


# --- Mock Data Loader ---
# Create a 'tests/helpers.py' or just put this at the top of the test file
# This assumes your tests run from the root of the 'penaltyblog' repo
# and your data is in 'tests/data/opta/'
def load_sample_json(filename):
    """Loads a sample JSON file from the tests/data/opta directory."""
    path = os.path.join(os.path.dirname(__file__), "data", "opta", filename)
    if not os.path.exists(path):
        pytest.skip(f"Missing sample data file: {filename}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --- Test 1: Request Builder ---
# These tests run offline and check your URL/param logic


def test_build_request_events():
    """Tests _build_opta_request_details for MA3 Events."""
    step = {
        "source": "match_events",
        "base_url": "http://api.test.com",
        "asset_type": "soccerdata",
        "args": {
            "fixture_uuid": "fx1",
            "creds": {"auth_key": "my_key", "rt_mode": "b"},
        },
    }
    url, params, headers = _build_opta_request_details(step)
    assert url == "http://api.test.com/soccerdata/matchevent/my_key/fx1"
    assert params == {"_rt": "b", "_fmt": "json"}


def test_build_request_matches_filters():
    """Tests _build_opta_request_details for MA1 with multiple filters."""
    step = {
        "source": "matches_basic",
        "base_url": "http://api.test.com",
        "asset_type": "soccerdata",
        "args": {
            "tmcl": "tmcl1",
            "ctst": "ctst1",
            "ctst2": "ctst2",
            "creds": {"auth_key": "my_key", "rt_mode": "b"},
        },
    }
    url, params, headers = _build_opta_request_details(step)
    assert url == "http://api.test.com/soccerdata/match/my_key"
    assert params["tmcl"] == "tmcl1"
    assert params["ctst"] == "ctst1"
    assert params["ctst2"] == "ctst2"


def test_build_request_calendars_active():
    """Tests _build_opta_request_details for OT2 special URL."""
    step = {
        "source": "tournament_calendars",
        "base_url": "http://api.test.com",
        "asset_type": "soccerdata",
        "args": {
            "status": "active_authorized",
            "creds": {"auth_key": "my_key", "rt_mode": "b"},
        },
    }
    url, params, headers = _build_opta_request_details(step)
    assert (
        url
        == "http://api.test.com/soccerdata/tournamentcalendar/my_key/active/authorized"
    )


# --- Test 2: The Executor (from_opta) ---
# These tests use mocks and your saved data


@patch("penaltyblog.matchflow.steps.source_opta.requests.Session")
def test_from_opta_non_paginated(mock_Session):
    """Tests a single non-paginated feed (e.g., MA3 Events)."""
    # 1. Load your saved sample
    # SAMPLE_MA3 = load_sample_json("ma3_events_uuidDEF_sample.json")
    # For this example, I'll use the sample data you provided:
    SAMPLE_MA3 = {
        "matchInfo": {"id": "71pif9hi2vwzp6q0xzilyxst0"},
        "liveData": {
            "events": {
                "event": [
                    {"id": 2328542063, "typeId": 34},
                    {"id": 2328543167, "typeId": 34},
                ]
            }
        },
    }

    # 2. Configure the mock session and response
    mock_session_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = SAMPLE_MA3
    mock_session_instance.get.return_value = mock_response
    mock_Session.return_value.__enter__.return_value = mock_session_instance

    # 3. Define the step to run
    step = {
        "source": "match_events",
        "base_url": "http://api.test.com",
        "asset_type": "soccerdata",
        "args": {
            "fixture_uuid": "71pif9hi2vwzp6q0xzilyxst0",
            "creds": {"auth_key": "k", "rt_mode": "b"},
        },
    }

    # 4. Run and collect results
    results = list(from_opta(step))

    # 5. Assert
    mock_session_instance.get.assert_called_once()
    assert len(results) == 2  # Check that it flattened the event list
    assert results[0]["id"] == 2328542063
    assert results[1]["typeId"] == 34
    assert "_match_info" in results[0]  # Check that it was enriched


@patch("penaltyblog.matchflow.steps.source_opta.requests.Session")
def test_from_opta_paginated(mock_Session):
    """Tests a paginated feed (e.g., OT2) and flattens the items."""
    # 1. Load samples
    # SAMPLE_OT2_P1 = load_sample_json("ot2_page1.json")
    # SAMPLE_OT2_P2 = load_sample_json("ot2_page_last.json")
    # Using dummy data for this example:
    SAMPLE_OT2_P1 = {"competition": [{"id": "c1"}, {"id": "c2"}]}  # Page 1 (full)
    SAMPLE_OT2_P2 = {"competition": [{"id": "c3"}]}  # Page 2 (last page)

    # 2. Configure mock session
    mock_session_instance = MagicMock()
    mock_resp_p1 = MagicMock(status_code=200)
    mock_resp_p1.json.return_value = SAMPLE_OT2_P1

    mock_resp_p2 = MagicMock(status_code=200)
    mock_resp_p2.json.return_value = SAMPLE_OT2_P2

    mock_resp_404 = MagicMock(status_code=404)  # To stop pagination

    mock_session_instance.get.side_effect = [mock_resp_p1, mock_resp_p2, mock_resp_404]
    mock_Session.return_value.__enter__.return_value = mock_session_instance

    # 3. Define step
    step = {
        "source": "tournament_calendars",
        "base_url": "http://api.test.com",
        "asset_type": "soccerdata",
        "args": {"status": "active", "creds": {"auth_key": "k", "rt_mode": "b"}},
    }

    # 4. Run and collect
    results = list(from_opta(step))

    # 5. Assert
    assert mock_session_instance.get.call_count == 3  # p1, p2, p3(404)
    assert len(results) == 3  # 2 from p1 + 1 from p2
    assert results[0]["id"] == "c1"
    assert results[2]["id"] == "c3"


@patch("penaltyblog.matchflow.steps.source_opta.requests.Session")
def test_from_opta_proxies(mock_Session):
    """Tests that the proxies dict is correctly passed to the session."""
    mock_session_instance = MagicMock()
    mock_session_instance.get.return_value.status_code = 404  # Just stop immediately
    mock_Session.return_value.__enter__.return_value = mock_session_instance

    my_proxies = {"https": "socks5h://localhost:9090"}
    step = {
        "source": "teams",  # A paginated feed
        "base_url": "http://api.test.com",
        "asset_type": "soccerdata",
        "args": {
            "tmcl": "tmcl1",
            "proxies": my_proxies,
            "creds": {"auth_key": "k", "rt_mode": "b"},
        },
    }

    list(from_opta(step))  # Run the generator

    # Assert that the session's proxies attribute was set
    assert mock_session_instance.proxies == my_proxies


@patch("penaltyblog.matchflow.steps.source_opta.requests.Session")
def test_from_opta_ma2_parsing_logic(mock_Session):
    """Tests that MA2 (match_stats_basic) calls the correct helper."""
    # This sample is the single match root object from your example
    SAMPLE_MA2 = {"matchInfo": {"id": "m1"}, "liveData": {"lineUp": [...]}}

    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = SAMPLE_MA2
    mock_session_instance = MagicMock()
    mock_session_instance.get.return_value = mock_response
    mock_Session.return_value.__enter__.return_value = mock_session_instance

    # We also mock the helpers themselves to check they are called
    with (
        patch(
            "penaltyblog.matchflow.steps.source_opta._extract_player_stats"
        ) as mock_player_helper,
        patch(
            "penaltyblog.matchflow.steps.source_opta._extract_team_stats"
        ) as mock_team_helper,
    ):

        mock_player_helper.return_value = iter([{"player": 1}])
        mock_team_helper.return_value = iter([{"team": 1}])

        # Case 1: people=yes (default)
        step_players = {
            "source": "match_stats_basic",
            "base_url": "http://api.test.com",
            "asset_type": "soccerdata",
            "args": {
                "fixture_uuids": "fx1",
                "people": "yes",
                "creds": {"auth_key": "k", "rt_mode": "b"},
            },
        }
        results_players = list(from_opta(step_players))

        mock_player_helper.assert_called_once_with(SAMPLE_MA2, include_xg=False)
        mock_team_helper.assert_not_called()
        assert results_players == [{"player": 1}]

        # Reset mocks for next case
        mock_player_helper.reset_mock()
        mock_team_helper.reset_mock()
