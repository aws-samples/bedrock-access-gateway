"""Tests for the checks that keep caller-supplied image urls from reaching internal endpoints."""

import base64
import socket

import pytest
from fastapi import HTTPException

from api import image_url as image_url_module
from api.image_url import normalize_image_content_type, parse_image_url

PUBLIC_IP = "93.184.216.34"


class FakeResponse:
    """Minimal stand-in for requests.Response as the fetch path uses it."""

    def __init__(self, status_code=200, headers=None, body=b"", history_location=None):
        self.status_code = status_code
        self.headers = headers if headers is not None else {}
        self._body = body
        if history_location:
            self.headers["Location"] = history_location

    @property
    def is_redirect(self):
        return self.status_code in (301, 302, 303, 307, 308)

    @property
    def is_permanent_redirect(self):
        return self.status_code in (301, 308)

    def iter_content(self, chunk_size):
        for start in range(0, max(len(self._body), 1), chunk_size):
            yield self._body[start : start + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


@pytest.fixture
def resolve_to_public(monkeypatch):
    """Resolve every hostname to a public address so only the url logic is exercised."""

    def fake_getaddrinfo(host, port, **_kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (PUBLIC_IP, port))]

    monkeypatch.setattr(image_url_module.socket, "getaddrinfo", fake_getaddrinfo)


@pytest.fixture
def capture_request(monkeypatch):
    """Record the urls requests.get is called with and return canned responses."""
    calls = []
    responses = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return responses.pop(0) if responses else FakeResponse(body=b"image-bytes")

    monkeypatch.setattr(image_url_module.requests, "get", fake_get)
    return calls, responses


@pytest.fixture
def forbid_request(monkeypatch):
    """Fail the test if an outbound request is attempted."""

    def fail_get(*_args, **_kwargs):
        raise AssertionError("no outbound request expected")

    monkeypatch.setattr(image_url_module.requests, "get", fail_get)


@pytest.mark.parametrize(
    "url",
    [
        # ECS task credential endpoint and EC2 instance metadata service.
        "http://169.254.170.2/v2/credentials/abc",
        "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
        "http://[fd00:ec2::254]/latest/meta-data/",
        # Loopback.
        "http://127.0.0.1:8080/health",
        "http://127.1/health",
        "http://[::1]:8080/health",
        # Private and otherwise non-routable ranges.
        "http://10.0.0.5/",
        "http://172.16.3.4/",
        "http://192.168.1.1/",
        "http://100.64.0.1/",
        "http://[fd12:3456::1]/",
        "http://0.0.0.0/",
        # IPv4-mapped IPv6 form of the metadata endpoint.
        "http://[::ffff:169.254.169.254]/latest/meta-data/",
    ],
)
def test_rejects_non_routable_addresses(url):
    with pytest.raises(HTTPException) as exc_info:
        parse_image_url(url)
    assert exc_info.value.status_code == 400


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "file://localhost/etc/shadow",
        "gopher://127.0.0.1:6379/_INFO",
        "ftp://192.168.0.1/image.png",
        "dict://127.0.0.1:11211/stats",
        "http:/169.254.169.254/",
        "not-a-url",
        "",
    ],
)
def test_rejects_unsupported_schemes(url):
    with pytest.raises(HTTPException) as exc_info:
        parse_image_url(url)
    assert exc_info.value.status_code == 400


def test_blocked_url_is_rejected_before_any_request(forbid_request):
    """Validation must happen before the request, not after it has been sent."""
    with pytest.raises(HTTPException):
        parse_image_url("http://169.254.169.254/latest/meta-data/iam/security-credentials/")


def test_error_for_blocked_address_does_not_reveal_the_target():
    with pytest.raises(HTTPException) as exc_info:
        parse_image_url("http://169.254.170.2/v2/credentials/abc")
    assert "169.254" not in exc_info.value.detail


def test_unresolvable_host_is_rejected(monkeypatch):
    def fake_getaddrinfo(*_args, **_kwargs):
        raise socket.gaierror("no such host")

    monkeypatch.setattr(image_url_module.socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(HTTPException) as exc_info:
        parse_image_url("https://example.com/cat.png")
    assert exc_info.value.status_code == 400


def test_fetches_a_public_url(resolve_to_public, capture_request):
    calls, responses = capture_request
    responses.append(FakeResponse(headers={"Content-Type": "image/png"}, body=b"png-bytes"))

    data, content_type = parse_image_url("https://example.com/cat.png")

    assert (data, content_type) == (b"png-bytes", "image/png")
    assert calls[0][0] == "https://example.com/cat.png"
    # Redirects must be handled by the module so each hop is re-validated.
    assert calls[0][1]["allow_redirects"] is False


def test_redirect_to_a_public_url_is_followed(resolve_to_public, capture_request):
    calls, responses = capture_request
    responses.append(FakeResponse(status_code=302, history_location="https://cdn.example.com/cat.png"))
    responses.append(FakeResponse(headers={"Content-Type": "image/webp"}, body=b"webp-bytes"))

    data, content_type = parse_image_url("https://example.com/cat.png")

    assert (data, content_type) == (b"webp-bytes", "image/webp")
    assert [call[0] for call in calls] == [
        "https://example.com/cat.png",
        "https://cdn.example.com/cat.png",
    ]


def test_redirect_to_the_metadata_endpoint_is_blocked(monkeypatch, capture_request):
    """A public host must not be able to redirect the gateway to an internal address."""
    _calls, responses = capture_request
    responses.append(FakeResponse(status_code=302, history_location="http://169.254.169.254/latest/meta-data/"))

    def fake_getaddrinfo(host, port, **_kwargs):
        if host == "example.com":
            return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (PUBLIC_IP, port))]
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (host, port))]

    monkeypatch.setattr(image_url_module.socket, "getaddrinfo", fake_getaddrinfo)

    with pytest.raises(HTTPException) as exc_info:
        parse_image_url("https://example.com/cat.png")
    assert exc_info.value.status_code == 400


def test_redirect_loop_is_bounded(resolve_to_public, monkeypatch):
    attempts = []

    def fake_get(url, **_kwargs):
        attempts.append(url)
        return FakeResponse(status_code=302, history_location="https://example.com/next")

    monkeypatch.setattr(image_url_module.requests, "get", fake_get)

    with pytest.raises(HTTPException) as exc_info:
        parse_image_url("https://example.com/cat.png")
    assert exc_info.value.status_code == 400
    assert len(attempts) == image_url_module.MAX_REDIRECTS + 1


def test_host_allowlist_rejects_other_hosts(resolve_to_public, capture_request, monkeypatch):
    monkeypatch.setattr(image_url_module, "IMAGE_URL_ALLOWED_HOSTS", frozenset({"images.example.com"}))

    with pytest.raises(HTTPException) as exc_info:
        parse_image_url("https://evil.example.net/cat.png")
    assert exc_info.value.status_code == 400

    _data, content_type = parse_image_url("https://images.example.com/cat.png")
    assert content_type == "image/jpeg"


def test_fetching_can_be_disabled(monkeypatch):
    monkeypatch.setattr(image_url_module, "ENABLE_IMAGE_URL_FETCH", False)
    with pytest.raises(HTTPException) as exc_info:
        parse_image_url("https://example.com/cat.png")
    assert exc_info.value.status_code == 400


def test_oversized_body_is_rejected(resolve_to_public, capture_request, monkeypatch):
    monkeypatch.setattr(image_url_module, "IMAGE_URL_MAX_SIZE_MB", 1)
    _calls, responses = capture_request
    responses.append(FakeResponse(headers={"Content-Type": "image/png"}, body=b"x" * (2 * 1024 * 1024)))

    with pytest.raises(HTTPException) as exc_info:
        parse_image_url("https://example.com/big.png")
    assert exc_info.value.status_code == 413


def test_oversized_content_length_is_rejected_before_reading(resolve_to_public, capture_request, monkeypatch):
    monkeypatch.setattr(image_url_module, "IMAGE_URL_MAX_SIZE_MB", 1)
    _calls, responses = capture_request
    responses.append(
        FakeResponse(
            headers={"Content-Type": "image/png", "Content-Length": str(50 * 1024 * 1024)},
            body=b"",
        )
    )

    with pytest.raises(HTTPException) as exc_info:
        parse_image_url("https://example.com/big.png")
    assert exc_info.value.status_code == 413


def test_non_200_response_is_rejected(resolve_to_public, capture_request):
    _calls, responses = capture_request
    responses.append(FakeResponse(status_code=404))

    with pytest.raises(HTTPException) as exc_info:
        parse_image_url("https://example.com/missing.png")
    assert exc_info.value.status_code == 400


def test_connection_error_is_reported_as_a_client_error(resolve_to_public, monkeypatch):
    def fake_get(*_args, **_kwargs):
        raise image_url_module.requests.ConnectionError("refused")

    monkeypatch.setattr(image_url_module.requests, "get", fake_get)

    with pytest.raises(HTTPException) as exc_info:
        parse_image_url("https://example.com/cat.png")
    assert exc_info.value.status_code == 400


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("image/png", "image/png"),
        ("image/PNG", "image/png"),
        ("image/png; charset=binary", "image/png"),
        ("image/jpg", "image/jpeg"),
        ("image/gif", "image/gif"),
        ("image/webp", "image/webp"),
        # Anything Bedrock cannot take falls back to the default format.
        ("image/svg+xml", "image/jpeg"),
        ("text/html", "image/jpeg"),
        (None, "image/jpeg"),
        ("", "image/jpeg"),
    ],
)
def test_content_type_is_normalized_to_a_supported_format(raw, expected):
    assert normalize_image_content_type(raw) == expected


def test_missing_content_type_header_falls_back(resolve_to_public, capture_request):
    """A response with no Content-Type must not raise; the previous code crashed here."""
    _calls, responses = capture_request
    responses.append(FakeResponse(body=b"bytes"))

    data, content_type = parse_image_url("https://example.com/cat.png")
    assert (data, content_type) == (b"bytes", "image/jpeg")


def test_data_url_is_decoded_without_fetching(forbid_request):
    payload = base64.b64encode(b"png-bytes").decode()
    assert parse_image_url(f"data:image/png;base64,{payload}") == (b"png-bytes", "image/png")


def test_data_url_payload_may_be_wrapped(forbid_request):
    payload = base64.b64encode(b"png-bytes" * 20).decode()
    wrapped = "\n".join(payload[i : i + 16] for i in range(0, len(payload), 16))
    data, content_type = parse_image_url(f"data:image/png;base64,\n{wrapped}")
    assert (data, content_type) == (b"png-bytes" * 20, "image/png")


def test_invalid_data_url_payload_is_a_client_error(forbid_request):
    with pytest.raises(HTTPException) as exc_info:
        parse_image_url("data:image/png;base64,not!valid!base64")
    assert exc_info.value.status_code == 400
