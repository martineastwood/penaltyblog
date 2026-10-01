import faulthandler
import multiprocessing
import sys

import pytest

import penaltyblog as pb

# Enable faulthandler to help debug deadlocks and hangs on Windows/macOS
faulthandler.enable()


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": [
            "authorization",
            "auth_key",
            "_rt",
        ],
        "cassette_library_dir": "test/fixtures/vcr_cassettes",
        "ignore_localhost": True,
        "decode_compressed_response": False,
    }


@pytest.fixture(scope="session")
def _football_data_fixtures():
    """Download the sample fixtures once per test session.

    ``get_fixtures()`` performs a live HTTP fetch, so downloading it per test
    (the ``fixtures`` fixture is requested by over a hundred tests) dominated
    the runtime of the ``local`` test slice.
    """
    return pb.scrapers.FootballData("ENG Premier League", "2019-2020").get_fixtures()


@pytest.fixture()
def fixtures(_football_data_fixtures):
    # Hand each test its own copy so that in-place edits cannot leak between
    # tests now that the download itself is shared.
    return _football_data_fixtures.copy()


@pytest.fixture(autouse=True)
def cleanup_multiprocessing():
    """Automatically clean up any remaining child processes after each test.

    This is critical for Windows/macOS where spawn mode is used, to prevent
    test suite hangs from orphaned processes.
    """
    yield
    # Cleanup after test
    if sys.platform in ("win32", "darwin"):
        # On Windows/macOS, actively clean up any orphaned child processes
        # to prevent hangs in subsequent tests
        try:
            # Force cleanup of any child processes
            for child in multiprocessing.active_children():
                if child.is_alive():
                    child.terminate()
                    child.join(timeout=1.0)
                    if child.is_alive():
                        child.kill()
        except Exception:
            # Silently ignore cleanup errors to avoid masking actual test failures
            pass
