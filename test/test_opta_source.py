import json
import os
from unittest.mock import MagicMock, patch

import pytest

from penaltyblog.matchflow.steps.opta.endpoints import OptaEndpointBuilder
from penaltyblog.matchflow.steps.source_opta import from_opta


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
    """Tests OptaEndpointBuilder for MA3 Events."""
    step = {
        "source": "match_events",
        "base_url": "http://api.test.com",
        "asset_type": "soccerdata",
        "args": {
            "fixture_uuid": "fx1",
            "creds": {"auth_key": "my_key", "rt_mode": "b"},
        },
    }
    builder = OptaEndpointBuilder(
        base_url=step["base_url"],
        asset_type=step["asset_type"],
        auth_key=step["args"]["creds"]["auth_key"],
    )
    url, params = builder.build_request_details(step["source"], step["args"])
    assert url == "http://api.test.com/soccerdata/matchevent/my_key/fx1"
    assert params["_fmt"] == "json"


def test_build_request_matches_filters():
    """Tests OptaEndpointBuilder for MA1 with multiple filters."""
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
    builder = OptaEndpointBuilder(
        base_url=step["base_url"],
        asset_type=step["asset_type"],
        auth_key=step["args"]["creds"]["auth_key"],
    )
    url, params = builder.build_request_details(step["source"], step["args"])
    assert url == "http://api.test.com/soccerdata/match/my_key"
    assert params["tmcl"] == "tmcl1"
    assert params["ctst"] == "ctst1"
    assert params["ctst2"] == "ctst2"


def test_build_request_calendars_active():
    """Tests OptaEndpointBuilder for OT2 special URL."""
    step = {
        "source": "tournament_calendars",
        "base_url": "http://api.test.com",
        "asset_type": "soccerdata",
        "args": {
            "status": "active_authorized",
            "creds": {"auth_key": "my_key", "rt_mode": "b"},
        },
    }
    builder = OptaEndpointBuilder(
        base_url=step["base_url"],
        asset_type=step["asset_type"],
        auth_key=step["args"]["creds"]["auth_key"],
    )
    url, params = builder.build_request_details(step["source"], step["args"])
    assert (
        url
        == "http://api.test.com/soccerdata/tournamentcalendar/my_key/active/authorized"
    )


# --- Test 2: The Executor (from_opta) ---
# These tests use mocks and your saved data


@patch("penaltyblog.matchflow.steps.source_opta.OptaClient")
def test_from_opta_non_paginated(mock_OptaClient):
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

    # 2. Configure the mock client
    mock_client_instance = MagicMock()
    mock_client_instance.make_request.return_value = SAMPLE_MA3
    mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
    mock_client_instance.__exit__ = MagicMock(return_value=None)
    mock_OptaClient.return_value = mock_client_instance

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
    mock_client_instance.make_request.assert_called_once()
    assert len(results) == 2  # Check that it flattened the event list
    assert results[0]["id"] == 2328542063
    assert results[1]["typeId"] == 34
    assert "_match_info" in results[0]  # Check that it was enriched


@patch("penaltyblog.matchflow.steps.source_opta.OptaPaginator")
def test_from_opta_paginated(mock_OptaPaginator):
    """Tests a paginated feed (e.g., OT2) and flattens the items."""
    # 1. Load samples
    # SAMPLE_OT2_P1 = load_sample_json("ot2_page1.json")
    # SAMPLE_OT2_P2 = load_sample_json("ot2_page_last.json")
    # Using dummy data for this example:
    SAMPLE_OT2_P1 = {"competition": [{"id": "c1"}, {"id": "c2"}]}  # Page 1 (full)
    SAMPLE_OT2_P2 = {"competition": [{"id": "c3"}]}  # Page 2 (last page)

    # 2. Configure mock paginator
    mock_paginator_instance = MagicMock()
    mock_paginator_instance.fetch_paginated_data.return_value = [
        {"id": "c1"},
        {"id": "c2"},
        {"id": "c3"},
    ]
    mock_OptaPaginator.return_value = mock_paginator_instance

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
    mock_paginator_instance.fetch_paginated_data.assert_called_once()
    assert len(results) == 3  # 2 from p1 + 1 from p2
    assert results[0]["id"] == "c1"
    assert results[2]["id"] == "c3"


@patch("penaltyblog.matchflow.steps.source_opta.OptaClient")
def test_from_opta_proxies(mock_OptaClient):
    """Tests that the proxies dict is correctly passed to the session."""
    mock_client_instance = MagicMock()
    mock_client_instance.fetch_paginated.return_value = []  # Return empty list
    mock_OptaClient.return_value = mock_client_instance

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

    # Assert that the client was initialized with proxies
    mock_OptaClient.assert_called_once()
    # Check that proxies were passed in the args
    call_args = mock_OptaClient.call_args
    assert call_args[1]["proxies"] == my_proxies


@patch("penaltyblog.matchflow.steps.source_opta.OptaClient")
def test_from_opta_ma2_parsing_logic(mock_OptaClient):
    """Tests that MA2 (match_stats_basic) calls correct helper."""
    # This sample is single match root object from your example
    SAMPLE_MA2 = {
        "matchInfo": {"id": "m1"},
        "liveData": {
            "lineUp": [
                {
                    "contestantId": "team1",
                    "player": [
                        {"id": "p1", "stat": [{"type": "goals", "value": "1"}]},
                        {"id": "p2", "stat": [{"type": "assists", "value": "2"}]},
                    ],
                }
            ]
        },
    }

    mock_client_instance = MagicMock()
    mock_client_instance.make_request.return_value = SAMPLE_MA2
    mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
    mock_client_instance.__exit__ = MagicMock(return_value=None)
    mock_OptaClient.return_value = mock_client_instance

    # We also mock the helpers themselves to check they are called
    with (
        patch(
            "penaltyblog.matchflow.steps.opta.parsers.extract_player_stats"
        ) as mock_player_helper,
        patch(
            "penaltyblog.matchflow.steps.opta.parsers.extract_team_stats"
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

        mock_player_helper.assert_called_once_with(SAMPLE_MA2)
        mock_team_helper.assert_not_called()
        assert results_players == [{"player": 1}]

        # Reset mocks for next case
        mock_player_helper.reset_mock()
        mock_team_helper.reset_mock()
