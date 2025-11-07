"""
Pagination handling for Opta API requests.
"""

from typing import Any, Dict, Iterator, List

from .client import OptaClient
from .config import (
    DEFAULT_PAGE_NUM,
    DEFAULT_PAGE_SIZE,
    NON_PAGINATED_SOURCES,
    PAGINATION_RESPONSE_KEYS,
)
from .exceptions import OptaParsingError, OptaRequestError


class OptaPaginator:
    """
    Handles pagination for Opta API endpoints.
    """

    def __init__(self, client: OptaClient):
        """
        Initialize paginator.

        Args:
            client: OptaClient instance for making requests
        """
        self.client = client

    def fetch_paginated_data(
        self,
        source: str,
        base_url: str,
        base_params: Dict[str, Any],
        headers: Dict[str, str],
    ) -> Iterator[Dict[str, Any]]:
        """
        Fetch data from a paginated endpoint.

        Args:
            source: The endpoint source name
            base_url: Base URL for requests
            base_params: Base parameters for requests
            headers: Request headers

        Yields:
            Individual records from paginated responses
        """
        page_num = DEFAULT_PAGE_NUM
        page_size = DEFAULT_PAGE_SIZE

        while True:
            params = base_params.copy()
            params["_pgNm"] = page_num
            params["_pgSz"] = page_size

            try:
                data = self.client.make_request(base_url, params, headers)
            except OptaRequestError as e:
                raise e
            except Exception as e:
                print(
                    f"Warning: Could not fetch page {page_num} for {source}. Stopping. Error: {e}"
                )
                break

            records_list = self._extract_records_from_page(source, data)

            if not records_list or not isinstance(records_list, list):
                break

            yield from records_list

            if len(records_list) < page_size:
                break

            page_num += 1

    def _extract_records_from_page(
        self, source: str, data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract records list from a paginated response.

        Args:
            source: The endpoint source name
            data: Response data from API

        Returns:
            List of records, or empty list if no records found

        Raises:
            OptaParsingError: If response structure is unexpected
        """
        if source == "referees":
            records = []
            if "stage" in data and isinstance(data["stage"], list):
                for stage in data["stage"]:
                    if "referee" in stage and isinstance(stage["referee"], list):
                        records.extend(stage["referee"])
            return records

        try:
            response_keys = PAGINATION_RESPONSE_KEYS.get(source, [])

            for key_path in response_keys:
                if "." in key_path:
                    # Handle nested keys like "matches.match"
                    keys = key_path.split(".")
                    current = data
                    for key in keys:
                        if isinstance(current, dict) and key in current:
                            current = current[key]
                        else:
                            current = None
                            break

                    if current and isinstance(current, list):
                        return current
                else:
                    # Handle simple keys
                    records = data.get(key_path)
                    if records and isinstance(records, list):
                        return records

            # If no records found, return empty list
            return []

        except Exception as e:
            raise OptaParsingError(
                f"Failed to parse paginated response for {source}: {e}"
            ) from e

    @staticmethod
    def is_paginated(source: str, args: Dict[str, Any]) -> bool:
        """
        Check if a source is paginated.

        Args:
            source: The endpoint source name

        Returns:
            True if source is paginated, False otherwise
        """
        # Handle dynamic pagination for injuries_query
        if source == "injuries_query":
            # It's non-paginated if 'prsn' (person_uuid) is provided
            if args.get("person_uuid") is not None:
                return False
            # Otherwise, it's paginated (e.g., when filtering by tmcl)
            return True

        # Handle dynamic pagination for transfers
        elif source == "transfers":
            # It's non-paginated if 'prsn' (person_uuid) is provided
            if args.get("person_uuid") is not None:
                return False
            # Otherwise, it's paginated
            return True

        return source not in NON_PAGINATED_SOURCES
