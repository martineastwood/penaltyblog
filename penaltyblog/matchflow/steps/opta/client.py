"""
HTTP client for Opta API requests.
"""

from typing import Any, Dict, Optional, Tuple

import requests

from .exceptions import OptaAPIError, OptaAuthenticationError, OptaRequestError


class OptaClient:
    """
    HTTP client for making requests to the Opta API.

    Handles authentication, session management, and error handling.
    """

    def __init__(self, proxies: Optional[Dict[str, str]] = None):
        """
        Initialize the Opta client.

        Args:
            proxies: Optional proxy configuration for requests
        """
        self.proxies = proxies
        self._session = None

    @property
    def session(self) -> requests.Session:
        """Get or create a requests session."""
        if self._session is None:
            self._session = requests.Session()
            if self.proxies:
                self._session.proxies = self.proxies
        return self._session

    def validate_credentials(self, creds: Dict[str, Any]) -> None:
        """
        Validate that required credentials are present.

        Args:
            creds: Credentials dictionary

        Raises:
            OptaAuthenticationError: If required credentials are missing
        """
        if not creds.get("auth_key") or not creds.get("rt_mode"):
            raise OptaAuthenticationError(
                "Invalid Opta credentials. Provide 'auth_key' and 'rt_mode' "
                "in DEFAULT_CREDS or via the 'creds' parameter."
            )

    def make_request(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Make a GET request to the Opta API.

        Args:
            url: The URL to request
            params: Query parameters
            headers: Request headers

        Returns:
            Parsed JSON response

        Raises:
            OptaRequestError: If the request fails
            OptaAPIError: If the API returns an error
        """
        try:
            response = self.session.get(url, params=params, headers=headers)

            if response.status_code == 404:
                raise OptaRequestError(
                    f"Opta API request returned 404 Not Found for URL: {response.url}"
                )

            response.raise_for_status()

            try:
                data = response.json()
            except requests.exceptions.JSONDecodeError as e:
                raise OptaRequestError(
                    f"Opta API returned non-JSON response: {response.text}"
                ) from e

            if "errorCode" in data:
                raise OptaAPIError(f"Opta API Error: {data['errorCode']}")

            return data

        except requests.exceptions.HTTPError as e:
            raise OptaRequestError(f"Opta API HTTP request failed: {e}") from e
        except requests.exceptions.RequestException as e:
            raise OptaRequestError(f"Opta API request failed: {e}") from e

    def close(self) -> None:
        """Close the session."""
        if self._session:
            self._session.close()
            self._session = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
