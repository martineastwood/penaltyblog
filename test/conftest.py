import pytest

import penaltyblog as pb


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


@pytest.fixture()
def fixtures():
    return pb.scrapers.FootballData("ENG Premier League", "2019-2020").get_fixtures()
