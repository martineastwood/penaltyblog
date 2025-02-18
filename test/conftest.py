import pytest

import penaltyblog as pb


@pytest.fixture()
def fixtures():
    return pb.scrapers.FootballData("ENG Premier League", "2019-2020").get_fixtures()
