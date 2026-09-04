"""Safe retrieval of caller-supplied image URLs.

The OpenAI chat completions schema lets an ``image_url`` content part carry
either an inline ``data:`` URI or a remote URL. Fetching a remote URL means the
gateway performs an outbound request on behalf of the caller, so an unvalidated
URL turns the gateway into a request proxy: a caller could point it at endpoints
that are only reachable from the task itself -- the ECS container credential
endpoint (``169.254.170.2``), the EC2 instance metadata service
(``169.254.169.254``), ``localhost``, or other VPC-internal services -- and the
fetched bytes would be sent to the model and echoed back in the completion.

To prevent that (server-side request forgery), a remote URL is only fetched when
all of the following hold:

* the scheme is ``http`` or ``https``;
* the host is in ``IMAGE_URL_ALLOWED_HOSTS`` when that allowlist is configured;
* every address the host resolves to is globally routable, which excludes the
  link-local range used by the metadata endpoints as well as private, loopback,
  shared-address-space, reserved, and multicast ranges;
* the response body stays within ``IMAGE_URL_MAX_SIZE_MB``.

Redirects are followed manually so that every hop is re-validated instead of
letting a public URL redirect to an internal one.

Note on residual risk: the checks above resolve the host and then let
``requests`` resolve it again when connecting, so a hostname backed by an
attacker-controlled DNS server with a very short TTL could in principle return a
permitted address to the check and an internal one to the connection (DNS
rebinding). Deployments that need to rule that out should set
``IMAGE_URL_ALLOWED_HOSTS``, restrict the task's egress with security groups, or
turn remote fetching off entirely with ``ENABLE_IMAGE_URL_FETCH=false``.
"""

import base64
import binascii
import ipaddress
import logging
import re
import socket
from urllib.parse import urljoin, urlparse

import requests
from fastapi import HTTPException

from api.setting import (
    ENABLE_IMAGE_URL_FETCH,
    IMAGE_URL_ALLOWED_HOSTS,
    IMAGE_URL_MAX_SIZE_MB,
)

logger = logging.getLogger(__name__)

ALLOWED_SCHEMES = frozenset({"http", "https"})
DEFAULT_PORTS = {"http": 80, "https": 443}

# Image formats accepted by the Bedrock Converse API.
# Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ImageBlock.html
SUPPORTED_IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})
DEFAULT_IMAGE_TYPE = "image/jpeg"
# Content types that name a supported format under a different label.
IMAGE_TYPE_ALIASES = {"image/jpg": "image/jpeg"}

CONNECT_TIMEOUT_SECONDS = 5
READ_TIMEOUT_SECONDS = 30
MAX_REDIRECTS = 3
CHUNK_SIZE = 64 * 1024

DATA_URL_PATTERN = r"^data:(image/[a-z]*);base64,\s*"

# Returned for every failure that depends on where the URL points, so that error
# messages cannot be used to probe which internal hosts exist.
_UNREACHABLE_DETAIL = "Unable to access the image url"


def normalize_image_content_type(content_type: str | None) -> str:
    """Map a content type onto an image format Bedrock accepts."""
    # Drop any parameters, e.g. "image/png; charset=binary".
    media_type = (content_type or "").split(";")[0].strip().lower()
    media_type = IMAGE_TYPE_ALIASES.get(media_type, media_type)
    if media_type in SUPPORTED_IMAGE_TYPES:
        return media_type
    return DEFAULT_IMAGE_TYPE


def _resolved_addresses(host: str, port: int) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Resolve a host to every address it maps to, IP literals included."""
    try:
        addr_infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError):
        logger.warning("Unable to resolve image url host")
        raise HTTPException(status_code=400, detail=_UNREACHABLE_DETAIL)

    addresses = []
    for addr_info in addr_infos:
        ip = ipaddress.ip_address(addr_info[4][0])
        # An IPv4-mapped IPv6 address such as ::ffff:169.254.169.254 reaches the
        # same host as the IPv4 address it wraps, so check the wrapped address.
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        addresses.append(ip)
    return addresses


def _validate_url(url: str) -> None:
    """Reject an image URL that the gateway must not fetch."""
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise HTTPException(
            status_code=400,
            detail="Image url must be an http(s) url or a base64 data url",
        )

    host = parsed.hostname
    if not host:
        raise HTTPException(status_code=400, detail="Image url is missing a host")

    if IMAGE_URL_ALLOWED_HOSTS and host.lower() not in IMAGE_URL_ALLOWED_HOSTS:
        raise HTTPException(
            status_code=400,
            detail=f"Image url host is not allowed: {host}",
        )

    try:
        port = parsed.port or DEFAULT_PORTS[parsed.scheme]
    except ValueError:
        raise HTTPException(status_code=400, detail="Image url has an invalid port")

    for ip in _resolved_addresses(host, port):
        if not ip.is_global:
            logger.warning("Blocked image url resolving to non-routable address %s", ip)
            raise HTTPException(status_code=400, detail=_UNREACHABLE_DETAIL)


def _read_capped(response: requests.Response, max_bytes: int) -> bytes:
    """Read a response body, refusing anything larger than ``max_bytes``."""
    declared_length = response.headers.get("Content-Length")
    if declared_length and declared_length.isdigit() and int(declared_length) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the maximum size of {IMAGE_URL_MAX_SIZE_MB} MB",
        )

    content = bytearray()
    for chunk in response.iter_content(CHUNK_SIZE):
        content += chunk
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Image exceeds the maximum size of {IMAGE_URL_MAX_SIZE_MB} MB",
            )
    return bytes(content)


def parse_image_url(image_url: str) -> tuple[bytes, str]:
    """Resolve an image url to a tuple of (image data, content type).

    Handles both an inline base64 ``data:`` url and, subject to the checks in
    this module, a remote http(s) url.
    """
    match = re.search(DATA_URL_PATTERN, image_url)
    if match:
        # Clients sometimes wrap the payload across lines, so drop whitespace
        # before decoding rather than letting it fail validation.
        payload = re.sub(r"\s+", "", re.sub(DATA_URL_PATTERN, "", image_url))
        try:
            image_data = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(status_code=400, detail="Image data url is not valid base64")
        return image_data, normalize_image_content_type(match.group(1))

    return _fetch_image(image_url)


def _fetch_image(image_url: str) -> tuple[bytes, str]:
    """Fetch a remote image url, returning a tuple of (image data, content type)."""
    if not ENABLE_IMAGE_URL_FETCH:
        raise HTTPException(
            status_code=400,
            detail="Fetching images by url is disabled, please pass a base64 data url instead",
        )

    max_bytes = IMAGE_URL_MAX_SIZE_MB * 1024 * 1024
    url = image_url
    for _ in range(MAX_REDIRECTS + 1):
        _validate_url(url)
        try:
            response = requests.get(
                url,
                timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
                # Followed manually below so that every hop is re-validated.
                allow_redirects=False,
                stream=True,
            )
        except requests.RequestException:
            logger.warning("Image url request failed", exc_info=True)
            raise HTTPException(status_code=400, detail=_UNREACHABLE_DETAIL)

        with response:
            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("Location")
                if not location:
                    raise HTTPException(status_code=400, detail=_UNREACHABLE_DETAIL)
                url = urljoin(url, location)
                continue

            if response.status_code != 200:
                raise HTTPException(status_code=400, detail=_UNREACHABLE_DETAIL)

            content_type = normalize_image_content_type(response.headers.get("Content-Type"))
            return _read_capped(response, max_bytes), content_type

    raise HTTPException(status_code=400, detail="Image url has too many redirects")
