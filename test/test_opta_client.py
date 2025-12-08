from unittest.mock import MagicMock, Mock, patch

import pytest
import requests

from penaltyblog.matchflow.steps.opta.client import OptaClient
from penaltyblog.matchflow.steps.opta.exceptions import (
    OptaAPIError,
    OptaAuthenticationError,
    OptaRequestError,
)


class TestOptaClient:
    """Test cases for OptaClient class."""

    def test_init_without_proxies(self):
        """Test OptaClient initialization without proxies."""
        client = OptaClient()
        assert client.proxies is None
        assert client._session is None

    def test_init_with_proxies(self):
        """Test OptaClient initialization with proxies."""
        proxies = {"https": "http://proxy.com"}
        client = OptaClient(proxies=proxies)
        assert client.proxies == proxies
        assert client._session is None

    def test_session_property_creates_session(self):
        """Test that session property creates a new session."""
        client = OptaClient()
        session = client.session
        assert isinstance(session, requests.Session)
        assert client._session is session

    def test_session_property_returns_existing_session(self):
        """Test that session property returns existing session."""
        client = OptaClient()
        session1 = client.session
        session2 = client.session
        assert session1 is session2

    def test_session_property_with_proxies(self):
        """Test that session property configures proxies."""
        proxies = {"https": "http://proxy.com"}
        client = OptaClient(proxies=proxies)
        session = client.session
        assert session.proxies == proxies

    def test_validate_credentials_valid(self):
        """Test credential validation with valid credentials."""
        client = OptaClient()
        valid_creds = {"auth_key": "test_key", "rt_mode": "test_mode"}

        # Should not raise exception
        client.validate_credentials(valid_creds)

    def test_validate_credentials_missing_auth_key(self):
        """Test credential validation with missing auth_key."""
        client = OptaClient()
        invalid_creds = {"rt_mode": "test_mode"}

        with pytest.raises(OptaAuthenticationError, match="Invalid Opta credentials"):
            client.validate_credentials(invalid_creds)

    def test_validate_credentials_missing_rt_mode(self):
        """Test credential validation with missing rt_mode."""
        client = OptaClient()
        invalid_creds = {"auth_key": "test_key"}

        with pytest.raises(OptaAuthenticationError, match="Invalid Opta credentials"):
            client.validate_credentials(invalid_creds)

    def test_validate_credentials_empty_values(self):
        """Test credential validation with empty values."""
        client = OptaClient()
        invalid_creds = {"auth_key": "", "rt_mode": ""}

        with pytest.raises(OptaAuthenticationError, match="Invalid Opta credentials"):
            client.validate_credentials(invalid_creds)

    def test_validate_credentials_none_values(self):
        """Test credential validation with None values."""
        client = OptaClient()
        invalid_creds = {"auth_key": None, "rt_mode": None}

        with pytest.raises(OptaAuthenticationError, match="Invalid Opta credentials"):
            client.validate_credentials(invalid_creds)

    @patch("requests.Session.get")
    def test_make_request_success(self, mock_get):
        """Test successful API request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test"}
        mock_get.return_value = mock_response

        client = OptaClient()
        result = client.make_request("http://test.com")

        assert result == {"data": "test"}
        mock_get.assert_called_once_with("http://test.com", params=None, headers=None)

    @patch("requests.Session.get")
    def test_make_request_with_params_and_headers(self, mock_get):
        """Test API request with parameters and headers."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test"}
        mock_get.return_value = mock_response

        client = OptaClient()
        params = {"param1": "value1"}
        headers = {"Header1": "Value1"}
        result = client.make_request("http://test.com", params=params, headers=headers)

        assert result == {"data": "test"}
        mock_get.assert_called_once_with(
            "http://test.com", params=params, headers=headers
        )

    @patch("requests.Session.get")
    def test_make_request_404_error(self, mock_get):
        """Test 404 error handling."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.url = "http://test.com/notfound"
        mock_get.return_value = mock_response

        client = OptaClient()

        with pytest.raises(OptaRequestError, match="404 Not Found"):
            client.make_request("http://test.com")

    @patch("requests.Session.get")
    def test_make_request_http_error(self, mock_get):
        """Test HTTP error handling."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Error"
        )
        mock_get.return_value = mock_response

        client = OptaClient()

        with pytest.raises(OptaRequestError, match="HTTP request failed"):
            client.make_request("http://test.com")

    @patch("requests.Session.get")
    def test_make_request_connection_error(self, mock_get):
        """Test connection error handling."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")

        client = OptaClient()

        with pytest.raises(OptaRequestError, match="request failed"):
            client.make_request("http://test.com")

    @patch("requests.Session.get")
    def test_make_request_timeout_error(self, mock_get):
        """Test timeout error handling."""
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")

        client = OptaClient()

        with pytest.raises(OptaRequestError, match="request failed"):
            client.make_request("http://test.com")

    @patch("requests.Session.get")
    def test_make_request_json_decode_error(self, mock_get):
        """Test JSON decode error handling."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = requests.exceptions.JSONDecodeError(
            "Invalid JSON", "", 0
        )
        mock_response.text = "Invalid JSON response"
        mock_get.return_value = mock_response

        client = OptaClient()

        with pytest.raises(OptaRequestError, match="non-JSON response"):
            client.make_request("http://test.com")

    @patch("requests.Session.get")
    def test_make_request_api_error(self, mock_get):
        """Test API error response handling."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "errorCode": "API_ERROR",
            "message": "Test error",
        }
        mock_get.return_value = mock_response

        client = OptaClient()

        with pytest.raises(OptaAPIError, match="API_ERROR"):
            client.make_request("http://test.com")

    def test_close_session_exists(self):
        """Test closing an existing session."""
        client = OptaClient()
        session = client.session  # Create session

        with patch.object(session, "close") as mock_close:
            client.close()
            mock_close.assert_called_once()
            assert client._session is None

    def test_close_no_session(self):
        """Test closing when no session exists."""
        client = OptaClient()

        # Should not raise exception
        client.close()
        assert client._session is None

    def test_context_manager_success(self):
        """Test context manager functionality."""
        client = OptaClient()

        with patch.object(client, "close") as mock_close:
            with client as c:
                assert c is client
            mock_close.assert_called_once()

    def test_context_manager_with_exception(self):
        """Test context manager with exception."""
        client = OptaClient()

        with patch.object(client, "close") as mock_close:
            try:
                with client as c:
                    raise ValueError("Test exception")
            except ValueError:
                pass
            mock_close.assert_called_once()

    def test_session_reuse_after_close(self):
        """Test that new session is created after close."""
        client = OptaClient()

        session1 = client.session
        client.close()
        session2 = client.session

        assert session1 is not session2
        assert isinstance(session2, requests.Session)
