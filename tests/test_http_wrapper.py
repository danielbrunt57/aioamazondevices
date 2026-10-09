# Copyright 2024 Simone Chemelli and contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the HTTP wrapper."""

import asyncio
from http import HTTPMethod
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from aioamazondevices import http_wrapper
from aioamazondevices.api import AmazonEchoApi
from aioamazondevices.const.http import CSRF_COOKIE
from aioamazondevices.implementation import request_metrics as module

from .const import TEST_CSRF


@pytest.mark.anyio
async def test_csrf_cookie_sent_on_next_request(api: AmazonEchoApi) -> None:
    """A CSRF cookie set by a response is sent as a header on later requests."""
    received_csrf: list[str | None] = []

    async def handler(request: web.Request) -> web.Response:
        received_csrf.append(request.headers.get(CSRF_COOKIE))
        response = web.Response(text="<html></html>", content_type="text/html")
        response.set_cookie(CSRF_COOKIE, TEST_CSRF)
        return response

    app = web.Application()
    app.router.add_get("/", handler)
    async with TestServer(app) as server:
        url = server.make_url("/")
        await api._http_wrapper.session_request(HTTPMethod.GET, url)
        await api._http_wrapper.session_request(HTTPMethod.GET, url)

    assert received_csrf == [None, TEST_CSRF]


@pytest.mark.anyio
async def test_request_metrics_count_retries_and_exclude_query(
    api: AmazonEchoApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 429 and its successful retry count as separate attempts."""
    calls = 0

    async def handler(_request: web.Request) -> web.Response:
        nonlocal calls
        calls += 1
        return web.Response(status=429 if calls == 1 else 200, text="<html></html>")

    monkeypatch.setattr(http_wrapper.asyncio, "sleep", AsyncMock())
    app = web.Application()
    app.router.add_get("/history", handler)
    async with TestServer(app) as server:
        await api._http_wrapper.session_request(
            HTTPMethod.GET, server.make_url("/history?customerId=private")
        )
    snapshot = api._http_wrapper.request_metrics
    expected_attempts = 2
    assert snapshot["total_since_start"] == expected_attempts
    assert snapshot["minute"] == {
        "requests": expected_attempts,
        "failures": 1,
        "http_429": 1,
        "endpoints": {
            "GET /history": {
                "requests": expected_attempts,
                "failures": 1,
                "statuses": {"429": 1, "200": 1},
            }
        },
    }


def test_request_metrics_rolling_windows_and_summary(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Rolling windows expire independently; logs are limited to once a minute."""
    caplog.set_level("INFO", logger="aioamazondevices")
    now = 0.0
    monkeypatch.setattr(module, "monotonic", lambda: now)
    metrics = module.RequestMetrics()
    metrics.start("POST /rah")
    now = 61.0
    metrics.start("GET /csd")
    metrics.maybe_log()
    metrics.maybe_log()
    snapshot = metrics.snapshot()
    assert snapshot["minute"]["endpoints"] == {
        "GET /csd": {
            "requests": 1,
            "failures": 0,
            "statuses": {"no_response": 1},
        }
    }
    expected_hour = 2
    assert snapshot["hour"]["requests"] == expected_hour
    assert caplog.text.count("HTTP request metrics") == 1
    now = 3601.0
    assert metrics.snapshot()["hour"]["endpoints"] == {
        "GET /csd": {
            "requests": 1,
            "failures": 0,
            "statuses": {"no_response": 1},
        }
    }
    now = 86462.0
    assert metrics.snapshot()["day"]["requests"] == 0
    assert metrics.snapshot()["total_since_start"] == expected_hour


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        (
            "GET /devicesTypes/TYPE_A/deviceId/SERIAL_A/preferences",
            "GET /devicesTypes/{deviceType}/deviceId/{deviceSerialNumber}/preferences",
        ),
        (
            "POST /alexashoppinglists/api/v2/lists/private-list/items/fetch",
            "POST /alexashoppinglists/api/v2/lists/{listId}/items/fetch",
        ),
        (
            "POST /alexashoppinglists/api/v2/lists/fetch",
            "POST /alexashoppinglists/api/v2/lists/fetch",
        ),
        ("POST /auth/token", "POST /auth/token"),
    ],
)
def test_request_metrics_normalizes_endpoint(endpoint: str, expected: str) -> None:
    """Variable device/list identifiers are grouped; fixed endpoints stay intact."""
    metrics = module.RequestMetrics()
    metrics.start(endpoint)
    assert list(metrics.snapshot()["minute"]["endpoints"]) == [expected]


def test_request_metrics_attributes_failures_and_statuses() -> None:
    """Device requests aggregate HTTP failures and failures without a response."""
    metrics = module.RequestMetrics()
    first = metrics.start("GET /devicesTypes/TYPE_A/deviceId/SERIAL_A/preferences")
    first.status = 503
    first.failed = True
    second = metrics.start("GET /devicesTypes/TYPE_B/deviceId/SERIAL_B/preferences")
    second.status = 200
    third = metrics.start("GET /devicesTypes/TYPE_C/deviceId/SERIAL_C/preferences")
    third.failed = True
    counts = metrics.snapshot()["minute"]["endpoints"]
    endpoint = (
        "GET /devicesTypes/{deviceType}/deviceId/{deviceSerialNumber}/preferences"
    )
    expected_requests = 3
    expected_failures = 2
    assert counts == {
        endpoint: {
            "requests": expected_requests,
            "failures": expected_failures,
            "statuses": {"503": 1, "200": 1, "no_response": 1},
        }
    }


@pytest.mark.anyio
@pytest.mark.parametrize("expires", [None, 900, 1030, "invalid"])
async def test_history_access_token_refresh_shared(
    api: AmazonEchoApi, monkeypatch: pytest.MonkeyPatch, expires: object
) -> None:
    """Concurrent history workers refresh an expired/unknown token only once."""
    wrapper = api._http_wrapper
    wrapper._session_state_data.login_stored_data = {
        "access_token": "old",
        "refresh_token": "refresh",
        "expires": expires,
    }
    monkeypatch.setattr(http_wrapper, "time", lambda: 1000)
    started = asyncio.Event()
    release = asyncio.Event()

    async def request(**_kwargs: object) -> tuple[dict, object]:
        started.set()
        await release.wait()
        return {}, SimpleNamespace(status=200)

    request_mock = AsyncMock(side_effect=request)
    monkeypatch.setattr(wrapper, "session_request", request_mock)
    monkeypatch.setattr(
        wrapper,
        "response_to_json",
        AsyncMock(return_value={"access_token": "new", "expires_in": "3600"}),
    )
    first = asyncio.create_task(wrapper.ensure_access_token())
    await started.wait()
    second = asyncio.create_task(wrapper.ensure_access_token())
    release.set()
    assert await asyncio.gather(first, second) == [True, True]
    request_mock.assert_awaited_once()
    expected_expiration = 4600
    assert (
        wrapper._session_state_data.login_stored_data["expires"] == expected_expiration
    )
    assert wrapper._session_state_data.login_stored_data["access_token"] == "new"  # noqa: S105


@pytest.mark.anyio
async def test_history_access_token_reused(
    api: AmazonEchoApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A token with sufficient lifetime causes no refresh request."""
    wrapper = api._http_wrapper
    wrapper._session_state_data.login_stored_data = {
        "access_token": "valid",
        "expires": 4600,
    }
    monkeypatch.setattr(http_wrapper, "time", lambda: 1000)
    refresh = AsyncMock()
    monkeypatch.setattr(wrapper, "refresh_data", refresh)
    assert await wrapper.ensure_access_token()
    refresh.assert_not_awaited()


@pytest.mark.anyio
async def test_history_access_token_refresh_failure_retries(
    api: AmazonEchoApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed refresh is not cached as a valid token."""
    wrapper = api._http_wrapper
    wrapper._session_state_data.login_stored_data = {
        "access_token": "old",
        "expires": 0,
    }
    refresh = AsyncMock(return_value=(False, {}))
    monkeypatch.setattr(wrapper, "refresh_data", refresh)
    assert not await wrapper.ensure_access_token()
    assert not await wrapper.ensure_access_token()
    expected_attempts = 2
    assert refresh.await_count == expected_attempts
