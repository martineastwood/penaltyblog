from unittest.mock import MagicMock, Mock, patch

import pytest

from penaltyblog.matchflow.steps.opta.exceptions import OptaParsingError
from penaltyblog.matchflow.steps.opta.paginator import OptaPaginator


class TestOptaPaginator:
    """Test cases for OptaPaginator class."""

    def test_init(self):
        """Test OptaPaginator initialization."""
        mock_client = Mock()
        paginator = OptaPaginator(mock_client)

        assert paginator.client is mock_client

    def test_is_paginated_true(self):
        """Test is_paginated returns True for paginated sources."""
        paginated_sources = [
            "tournament_calendars",
            "matches_basic",
            "teams",
            "squads",
        ]

        for source in paginated_sources:
            assert OptaPaginator.is_paginated(source) is True

    def test_is_paginated_false(self):
        """Test is_paginated returns False for non-paginated sources."""
        non_paginated_sources = [
            "tournament_schedule",
            "match_basic",
            "match_stats_basic",
            "match_events",
            "player_season_stats",
            "team_season_stats",
            "xg_shots",
            "xg_player_summary",
            "xg_team_summary",
        ]

        for source in non_paginated_sources:
            assert OptaPaginator.is_paginated(source) is False

    @patch("penaltyblog.matchflow.steps.opta.paginator.DEFAULT_PAGE_SIZE", 2)
    @patch("penaltyblog.matchflow.steps.opta.paginator.DEFAULT_PAGE_NUM", 1)
    def test_fetch_paginated_data_success(self):
        """Test successful paginated data fetching."""
        mock_client = Mock()
        paginator = OptaPaginator(mock_client)

        # Mock responses for 3 pages
        page1_response = {"match": [{"id": 1}, {"id": 2}]}
        page2_response = {"match": [{"id": 3}, {"id": 4}]}
        page3_response = {"match": [{"id": 5}]}  # Less than page size, last page

        mock_client.make_request.side_effect = [
            page1_response,
            page2_response,
            page3_response,
        ]

        base_url = "http://test.com"
        base_params = {"param1": "value1"}
        headers = {"Header1": "Value1"}

        results = list(
            paginator.fetch_paginated_data(
                "matches_basic", base_url, base_params, headers
            )
        )

        # Should get all 5 records
        assert len(results) == 5
        assert results[0]["id"] == 1
        assert results[4]["id"] == 5

        # Should have made 3 requests
        assert mock_client.make_request.call_count == 3

        # Check pagination parameters
        calls = mock_client.make_request.call_args_list
        assert calls[0][0][1]["_pgNm"] == 1
        assert calls[0][0][1]["_pgSz"] == 2
        assert calls[1][0][1]["_pgNm"] == 2
        assert calls[1][0][1]["_pgSz"] == 2
        assert calls[2][0][1]["_pgNm"] == 3
        assert calls[2][0][1]["_pgSz"] == 2

    @patch("penaltyblog.matchflow.steps.opta.paginator.DEFAULT_PAGE_SIZE", 2)
    def test_fetch_paginated_data_empty_response(self):
        """Test paginated data fetching with empty response."""
        mock_client = Mock()
        paginator = OptaPaginator(mock_client)

        # Mock empty response
        mock_client.make_request.return_value = {"match": []}

        base_url = "http://test.com"
        base_params = {"param1": "value1"}
        headers = {"Header1": "Value1"}

        results = list(
            paginator.fetch_paginated_data(
                "matches_basic", base_url, base_params, headers
            )
        )

        assert len(results) == 0
        assert mock_client.make_request.call_count == 1

    @patch("penaltyblog.matchflow.steps.opta.paginator.DEFAULT_PAGE_SIZE", 2)
    def test_fetch_paginated_data_no_records_key(self):
        """Test paginated data fetching when no records key found."""
        mock_client = Mock()
        paginator = OptaPaginator(mock_client)

        # Mock response with no records
        mock_client.make_request.return_value = {"other_data": "value"}

        base_url = "http://test.com"
        base_params = {"param1": "value1"}
        headers = {"Header1": "Value1"}

        results = list(
            paginator.fetch_paginated_data(
                "matches_basic", base_url, base_params, headers
            )
        )

        assert len(results) == 0
        assert mock_client.make_request.call_count == 1

    @patch("penaltyblog.matchflow.steps.opta.paginator.DEFAULT_PAGE_SIZE", 2)
    def test_fetch_paginated_data_request_exception(self):
        """Test paginated data fetching with request exception."""
        mock_client = Mock()
        paginator = OptaPaginator(mock_client)

        # Mock request exception
        mock_client.make_request.side_effect = Exception("Network error")

        base_url = "http://test.com"
        base_params = {"param1": "value1"}
        headers = {"Header1": "Value1"}

        with patch("builtins.print") as mock_print:
            results = list(
                paginator.fetch_paginated_data(
                    "matches_basic", base_url, base_params, headers
                )
            )

        assert len(results) == 0
        assert mock_client.make_request.call_count == 1
        mock_print.assert_called_once()
        assert "Warning" in mock_print.call_args[0][0]

    def test_extract_records_from_page_simple_key(self):
        """Test _extract_records_from_page with simple key."""
        mock_client = Mock()
        paginator = OptaPaginator(mock_client)

        data = {"match": [{"id": 1}, {"id": 2}]}
        records = paginator._extract_records_from_page("matches_basic", data)

        assert len(records) == 2
        assert records[0]["id"] == 1
        assert records[1]["id"] == 2

    def test_extract_records_from_page_nested_key(self):
        """Test _extract_records_from_page with nested key."""
        mock_client = Mock()
        paginator = OptaPaginator(mock_client)

        data = {"matches": {"match": [{"id": 1}, {"id": 2}]}}
        records = paginator._extract_records_from_page("matches_basic", data)

        assert len(records) == 2
        assert records[0]["id"] == 1
        assert records[1]["id"] == 2

    def test_extract_records_from_page_multiple_keys_first_match(self):
        """Test _extract_records_from_page with multiple keys, first matches."""
        mock_client = Mock()
        paginator = OptaPaginator(mock_client)

        data = {"match": [{"id": 1}], "other": [{"id": 2}]}
        records = paginator._extract_records_from_page("matches_basic", data)

        assert len(records) == 1
        assert records[0]["id"] == 1

    def test_extract_records_from_page_multiple_keys_second_match(self):
        """Test _extract_records_from_page with multiple keys, second matches."""
        mock_client = Mock()
        paginator = OptaPaginator(mock_client)

        data = {"other": "not a list", "match": [{"id": 1}, {"id": 2}]}
        records = paginator._extract_records_from_page("matches_basic", data)

        assert len(records) == 2
        assert records[0]["id"] == 1
        assert records[1]["id"] == 2

    def test_extract_records_from_page_no_records(self):
        """Test _extract_records_from_page with no records found."""
        mock_client = Mock()
        paginator = OptaPaginator(mock_client)

        data = {"other_data": "value"}
        records = paginator._extract_records_from_page("matches_basic", data)

        assert len(records) == 0

    def test_extract_records_from_page_non_list_value(self):
        """Test _extract_records_from_page when key exists but value is not a list."""
        mock_client = Mock()
        paginator = OptaPaginator(mock_client)

        data = {"match": "not a list"}
        records = paginator._extract_records_from_page("matches_basic", data)

        assert len(records) == 0

    def test_extract_records_from_page_nested_path_missing_intermediate(self):
        """Test _extract_records_from_page with missing intermediate key in nested path."""
        mock_client = Mock()
        paginator = OptaPaginator(mock_client)

        data = {"matches": {}}  # Missing "match" key
        records = paginator._extract_records_from_page("matches_basic", data)

        assert len(records) == 0

    def test_extract_records_from_page_nested_path_wrong_type(self):
        """Test _extract_records_from_page with wrong type in nested path."""
        mock_client = Mock()
        paginator = OptaPaginator(mock_client)

        data = {"matches": {"match": "not a list"}}
        records = paginator._extract_records_from_page("matches_basic", data)

        assert len(records) == 0

    def test_extract_records_from_page_unknown_source(self):
        """Test _extract_records_from_page with unknown source."""
        mock_client = Mock()
        paginator = OptaPaginator(mock_client)

        data = {"some_key": [{"id": 1}]}
        records = paginator._extract_records_from_page("unknown_source", data)

        assert len(records) == 0

    def test_extract_records_from_page_exception_handling(self):
        """Test _extract_records_from_page exception handling."""
        mock_client = Mock()
        paginator = OptaPaginator(mock_client)

        # Test the exception handling by calling the method directly
        # and verifying it raises OptaParsingError when an exception occurs
        with patch(
            "penaltyblog.matchflow.steps.opta.paginator.PAGINATION_RESPONSE_KEYS",
            [{"invalid": "key"}],  # This will cause a TypeError when iterated
        ):
            with pytest.raises(
                OptaParsingError, match="Failed to parse paginated response"
            ):
                paginator._extract_records_from_page(
                    "matches_basic", {"match": [{"id": 1}]}
                )

    def test_extract_records_from_page_tournament_calendars(self):
        """Test _extract_records_from_page for tournament_calendars source."""
        mock_client = Mock()
        paginator = OptaPaginator(mock_client)

        data = {"competition": [{"id": 1}, {"id": 2}]}
        records = paginator._extract_records_from_page("tournament_calendars", data)

        assert len(records) == 2
        assert records[0]["id"] == 1
        assert records[1]["id"] == 2

    def test_extract_records_from_page_teams(self):
        """Test _extract_records_from_page for teams source."""
        mock_client = Mock()
        paginator = OptaPaginator(mock_client)

        # Test with contestants.contestant structure
        data = {"contestants": {"contestant": [{"id": 1}, {"id": 2}]}}
        records = paginator._extract_records_from_page("teams", data)

        assert len(records) == 2
        assert records[0]["id"] == 1
        assert records[1]["id"] == 2

    def test_extract_records_from_page_teams_fallback(self):
        """Test _extract_records_from_page for teams source with fallback."""
        mock_client = Mock()
        paginator = OptaPaginator(mock_client)

        # Test with contestant structure (fallback)
        data = {"contestant": [{"id": 1}, {"id": 2}]}
        records = paginator._extract_records_from_page("teams", data)

        assert len(records) == 2
        assert records[0]["id"] == 1
        assert records[1]["id"] == 2

    def test_extract_records_from_page_squads(self):
        """Test _extract_records_from_page for squads source."""
        mock_client = Mock()
        paginator = OptaPaginator(mock_client)

        # Test with teamSquads.squad structure
        data = {"teamSquads": {"squad": [{"id": 1}, {"id": 2}]}}
        records = paginator._extract_records_from_page("squads", data)

        assert len(records) == 2
        assert records[0]["id"] == 1
        assert records[1]["id"] == 2

    def test_extract_records_from_page_squads_fallback(self):
        """Test _extract_records_from_page for squads source with fallback."""
        mock_client = Mock()
        paginator = OptaPaginator(mock_client)

        # Test with squad structure (fallback)
        data = {"squad": [{"id": 1}, {"id": 2}]}
        records = paginator._extract_records_from_page("squads", data)

        assert len(records) == 2
        assert records[0]["id"] == 1
        assert records[1]["id"] == 2

    @patch("penaltyblog.matchflow.steps.opta.paginator.DEFAULT_PAGE_SIZE", 3)
    @patch("penaltyblog.matchflow.steps.opta.paginator.DEFAULT_PAGE_NUM", 1)
    def test_fetch_paginated_data_single_page(self):
        """Test fetching a single page of data."""
        mock_client = Mock()
        mock_client.make_request.return_value = {"match": [{"id": 1}, {"id": 2}]}

        paginator = OptaPaginator(mock_client)
        results = list(
            paginator.fetch_paginated_data(
                "matches_basic", "http://test.com", {"_fmt": "json"}, {}
            )
        )

        assert len(results) == 2
        assert results == [{"id": 1}, {"id": 2}]
        mock_client.make_request.assert_called_once_with(
            "http://test.com", {"_fmt": "json", "_pgNm": 1, "_pgSz": 3}, {}
        )

        assert len(results) == 2
        assert mock_client.make_request.call_count == 1

        # Check pagination parameters
        call_args = mock_client.make_request.call_args
        assert call_args[0][1]["_pgNm"] == 1
        assert call_args[0][1]["_pgSz"] == 3

    @patch("penaltyblog.matchflow.steps.opta.paginator.DEFAULT_PAGE_SIZE", 2)
    def test_fetch_paginated_data_exact_page_size(self):
        """Test paginated data fetching with exact page size match."""
        mock_client = Mock()
        paginator = OptaPaginator(mock_client)

        # Mock response with exactly page size records (should still try next page)
        page1_response = {"match": [{"id": 1}, {"id": 2}]}
        page2_response = {"match": []}  # Empty next page

        mock_client.make_request.side_effect = [page1_response, page2_response]

        base_url = "http://test.com"
        base_params = {"param1": "value1"}
        headers = {"Header1": "Value1"}

        results = list(
            paginator.fetch_paginated_data(
                "matches_basic", base_url, base_params, headers
            )
        )

        assert len(results) == 2
        assert mock_client.make_request.call_count == 2

    def test_fetch_paginated_data_parameters_preserved(self):
        """Test that base parameters are preserved in pagination requests."""
        mock_client = Mock()
        paginator = OptaPaginator(mock_client)

        mock_client.make_request.return_value = {"match": []}

        base_url = "http://test.com"
        base_params = {"param1": "value1", "param2": "value2"}
        headers = {"Header1": "Value1"}

        list(
            paginator.fetch_paginated_data(
                "matches_basic", base_url, base_params, headers
            )
        )

        call_args = mock_client.make_request.call_args
        params = call_args[0][1]

        # Original parameters should be preserved
        assert params["param1"] == "value1"
        assert params["param2"] == "value2"

        # Pagination parameters should be added
        assert "_pgNm" in params
        assert "_pgSz" in params
