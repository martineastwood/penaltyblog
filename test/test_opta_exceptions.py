import pytest

from penaltyblog.matchflow.steps.opta.exceptions import (
    OptaAPIError,
    OptaAuthenticationError,
    OptaConfigurationError,
    OptaParsingError,
    OptaRequestError,
)


class TestOptaAPIError:
    """Test cases for OptaAPIError base exception."""

    def test_inheritance(self):
        """Test that OptaAPIError inherits from Exception."""
        assert issubclass(OptaAPIError, Exception)

    def test_instantiation_without_message(self):
        """Test OptaAPIError instantiation without message."""
        error = OptaAPIError()
        assert str(error) == ""

    def test_instantiation_with_message(self):
        """Test OptaAPIError instantiation with message."""
        message = "Test API error"
        error = OptaAPIError(message)
        assert str(error) == message

    def test_instantiation_with_args(self):
        """Test OptaAPIError instantiation with multiple args."""
        error = OptaAPIError("Error 1", "Error 2")
        assert str(error) == "('Error 1', 'Error 2')"

    def test_exception_chaining(self):
        """Test exception chaining with OptaAPIError."""
        original_error = ValueError("Original error")
        api_error = OptaAPIError("API error")
        api_error.__cause__ = original_error

        assert api_error.__cause__ is original_error
        assert str(api_error) == "API error"


class TestOptaAuthenticationError:
    """Test cases for OptaAuthenticationError."""

    def test_inheritance(self):
        """Test that OptaAuthenticationError inherits from OptaAPIError."""
        assert issubclass(OptaAuthenticationError, OptaAPIError)
        assert issubclass(OptaAuthenticationError, Exception)

    def test_instantiation_without_message(self):
        """Test OptaAuthenticationError instantiation without message."""
        error = OptaAuthenticationError()
        assert str(error) == ""

    def test_instantiation_with_message(self):
        """Test OptaAuthenticationError instantiation with message."""
        message = "Authentication failed"
        error = OptaAuthenticationError(message)
        assert str(error) == message

    def test_exception_chaining(self):
        """Test exception chaining with OptaAuthenticationError."""
        original_error = ValueError("Invalid credentials")
        auth_error = OptaAuthenticationError("Auth failed")
        auth_error.__cause__ = original_error

        assert auth_error.__cause__ is original_error
        assert str(auth_error) == "Auth failed"

    def test_catch_as_base_exception(self):
        """Test that OptaAuthenticationError can be caught as OptaAPIError."""
        with pytest.raises(OptaAPIError):
            raise OptaAuthenticationError("Test")


class TestOptaRequestError:
    """Test cases for OptaRequestError."""

    def test_inheritance(self):
        """Test that OptaRequestError inherits from OptaAPIError."""
        assert issubclass(OptaRequestError, OptaAPIError)
        assert issubclass(OptaRequestError, Exception)

    def test_instantiation_without_message(self):
        """Test OptaRequestError instantiation without message."""
        error = OptaRequestError()
        assert str(error) == ""

    def test_instantiation_with_message(self):
        """Test OptaRequestError instantiation with message."""
        message = "Request failed"
        error = OptaRequestError(message)
        assert str(error) == message

    def test_exception_chaining(self):
        """Test exception chaining with OptaRequestError."""
        original_error = ConnectionError("Network error")
        request_error = OptaRequestError("Request failed")
        request_error.__cause__ = original_error

        assert request_error.__cause__ is original_error
        assert str(request_error) == "Request failed"

    def test_catch_as_base_exception(self):
        """Test that OptaRequestError can be caught as OptaAPIError."""
        with pytest.raises(OptaAPIError):
            raise OptaRequestError("Test")


class TestOptaParsingError:
    """Test cases for OptaParsingError."""

    def test_inheritance(self):
        """Test that OptaParsingError inherits from OptaAPIError."""
        assert issubclass(OptaParsingError, OptaAPIError)
        assert issubclass(OptaParsingError, Exception)

    def test_instantiation_without_message(self):
        """Test OptaParsingError instantiation without message."""
        error = OptaParsingError()
        assert str(error) == ""

    def test_instantiation_with_message(self):
        """Test OptaParsingError instantiation with message."""
        message = "Parsing failed"
        error = OptaParsingError(message)
        assert str(error) == message

    def test_exception_chaining(self):
        """Test exception chaining with OptaParsingError."""
        original_error = ValueError("Invalid JSON")
        parsing_error = OptaParsingError("Parse failed")
        parsing_error.__cause__ = original_error

        assert parsing_error.__cause__ is original_error
        assert str(parsing_error) == "Parse failed"

    def test_catch_as_base_exception(self):
        """Test that OptaParsingError can be caught as OptaAPIError."""
        with pytest.raises(OptaAPIError):
            raise OptaParsingError("Test")


class TestOptaConfigurationError:
    """Test cases for OptaConfigurationError."""

    def test_inheritance(self):
        """Test that OptaConfigurationError inherits from OptaAPIError."""
        assert issubclass(OptaConfigurationError, OptaAPIError)
        assert issubclass(OptaConfigurationError, Exception)

    def test_instantiation_without_message(self):
        """Test OptaConfigurationError instantiation without message."""
        error = OptaConfigurationError()
        assert str(error) == ""

    def test_instantiation_with_message(self):
        """Test OptaConfigurationError instantiation with message."""
        message = "Configuration error"
        error = OptaConfigurationError(message)
        assert str(error) == message

    def test_exception_chaining(self):
        """Test exception chaining with OptaConfigurationError."""
        original_error = KeyError("Missing config")
        config_error = OptaConfigurationError("Config error")
        config_error.__cause__ = original_error

        assert config_error.__cause__ is original_error
        assert str(config_error) == "Config error"

    def test_catch_as_base_exception(self):
        """Test that OptaConfigurationError can be caught as OptaAPIError."""
        with pytest.raises(OptaAPIError):
            raise OptaConfigurationError("Test")


class TestExceptionHierarchy:
    """Test cases for exception hierarchy and relationships."""

    def test_all_exceptions_inherit_from_base(self):
        """Test that all custom exceptions inherit from OptaAPIError."""
        exceptions = [
            OptaAuthenticationError,
            OptaRequestError,
            OptaParsingError,
            OptaConfigurationError,
        ]

        for exc_class in exceptions:
            assert issubclass(exc_class, OptaAPIError)

    def test_exception_type_checking(self):
        """Test exception type checking with isinstance."""
        base_error = OptaAPIError("Base")
        auth_error = OptaAuthenticationError("Auth")
        request_error = OptaRequestError("Request")
        parsing_error = OptaParsingError("Parsing")
        config_error = OptaConfigurationError("Config")

        # All should be instances of OptaAPIError
        assert isinstance(base_error, OptaAPIError)
        assert isinstance(auth_error, OptaAPIError)
        assert isinstance(request_error, OptaAPIError)
        assert isinstance(parsing_error, OptaAPIError)
        assert isinstance(config_error, OptaAPIError)

        # Each should be instance of its specific type
        assert isinstance(auth_error, OptaAuthenticationError)
        assert isinstance(request_error, OptaRequestError)
        assert isinstance(parsing_error, OptaParsingError)
        assert isinstance(config_error, OptaConfigurationError)

        # Should not be instances of other specific types
        assert not isinstance(auth_error, OptaRequestError)
        assert not isinstance(request_error, OptaParsingError)
        assert not isinstance(parsing_error, OptaConfigurationError)
        assert not isinstance(config_error, OptaAuthenticationError)

    def test_exception_catching_specificity(self):
        """Test that exceptions can be caught with appropriate specificity."""

        # Test catching specific exception
        try:
            raise OptaAuthenticationError("Auth failed")
        except OptaAuthenticationError:
            caught = True
        except OptaAPIError:
            caught = False
        assert caught is True

        # Test catching base exception when specific is not handled
        try:
            raise OptaAuthenticationError("Auth failed")
        except OptaRequestError:
            caught = False
        except OptaAPIError:
            caught = True
        assert caught is True

    def test_exception_message_formatting(self):
        """Test exception message formatting with different input types."""
        test_cases = [
            ("Simple string", "Simple string"),
            ("", ""),
            (123, "123"),
            (["error", "list"], "['error', 'list']"),
            ({"error": "dict"}, "{'error': 'dict'}"),
            (None, "None"),
        ]

        for input_val, expected_str in test_cases:
            error = OptaAPIError(input_val)
            assert str(error) == expected_str

    def test_exception_repr(self):
        """Test exception __repr__ method."""
        error = OptaAPIError("Test message")
        repr_str = repr(error)
        assert "OptaAPIError" in repr_str
        assert "Test message" in repr_str

    def test_multiple_exception_inheritance(self):
        """Test that exceptions can inherit from multiple levels."""
        # This tests the inheritance chain works correctly
        assert issubclass(OptaAuthenticationError, Exception)
        assert issubclass(OptaAuthenticationError, OptaAPIError)
        assert issubclass(OptaAuthenticationError, object)

    def test_exception_attributes(self):
        """Test that exceptions have expected attributes."""
        error = OptaAPIError("Test", "arg2")

        # Should have args attribute
        assert hasattr(error, "args")
        assert error.args == ("Test", "arg2")

        # Should have __cause__ attribute (even if None)
        assert hasattr(error, "__cause__")

        # Should have __context__ attribute (even if None)
        assert hasattr(error, "__context__")

    def test_exception_with_none_cause(self):
        """Test exception with explicit None cause."""
        error = OptaAPIError("Test")
        error.__cause__ = None
        assert error.__cause__ is None
        assert str(error) == "Test"
