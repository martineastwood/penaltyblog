from unittest.mock import patch

import pytest
from tls_requests import Request, Response
from tls_requests.exceptions import HTTPError

from penaltyblog.scrapers import FBRef
from penaltyblog.scrapers.base_scrapers import TLSRequestsScraper


def _response(status_code, url, body):
    response = Response(status_code, request=Request("GET", url), body=body)
    response.read()
    return response


@pytest.fixture
def tls_get():
    with patch("tls_requests.Client") as client:
        yield client.return_value.get


@pytest.mark.parametrize("status_code", [403, 429, 503])
@pytest.mark.parametrize("method", ["get_fixtures", "get_stats"])
def test_fbref_preserves_http_error(tls_get, status_code, method):
    def get_response(url):
        return _response(
            status_code,
            url,
            b"<html><h1>Service unavailable</h1></html>",
        )

    tls_get.side_effect = get_response
    scraper = FBRef("ENG Premier League", "2021-2022")

    with pytest.raises(HTTPError) as exc:
        getattr(scraper, method)()

    assert exc.value.response.status_code == status_code
    assert str(exc.value.response.url) == tls_get.call_args.args[0]


@pytest.mark.parametrize("status_code,body", [(200, b"<p>data</p>"), (204, b"")])
def test_tls_scraper_returns_successful_content(tls_get, status_code, body):
    url = "https://example.test/data"
    tls_get.return_value = _response(status_code, url, body)

    assert TLSRequestsScraper().get(url) == body.decode()
    tls_get.assert_called_once_with(url)


def test_fbref_rate_limits_retry_after_http_error(tls_get):
    url = "https://example.test/data"
    tls_get.side_effect = [
        _response(429, url, b"Too many requests"),
        _response(200, url, b"data"),
    ]
    scraper = FBRef("ENG Premier League", "2021-2022")

    with patch("penaltyblog.scrapers.fbref.time") as clock:
        clock.time.side_effect = [100, 101, 101, 104]
        with pytest.raises(HTTPError):
            scraper.get(url)
        assert scraper.get(url) == "data"

    clock.sleep.assert_called_once_with(3)
