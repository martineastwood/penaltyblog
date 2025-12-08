"""
Custom exceptions for Opta API integration.
"""


class OptaAPIError(Exception):
    """Base exception for Opta API errors."""

    pass


class OptaAuthenticationError(OptaAPIError):
    """Raised when authentication fails (missing/invalid credentials)."""

    pass


class OptaRequestError(OptaAPIError):
    """Raised when HTTP request fails (network issues, 4xx/5xx errors)."""

    pass


class OptaParsingError(OptaAPIError):
    """Raised when response parsing fails."""

    pass


class OptaConfigurationError(OptaAPIError):
    """Raised when configuration is invalid."""

    pass
