# Copyright 2024 Simone Chemelli and contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the HTTP wrapper."""

from http import HTTPMethod
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
        "endpoints": {"GET /history": expected_attempts},
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
    assert snapshot["minute"]["endpoints"] == {"GET /csd": 1}
    expected_hour = 2
    assert snapshot["hour"]["requests"] == expected_hour
    assert caplog.text.count("HTTP request metrics") == 1
    now = 3601.0
    assert metrics.snapshot()["hour"]["endpoints"] == {"GET /csd": 1}
    now = 86462.0
    assert metrics.snapshot()["day"]["requests"] == 0
    assert metrics.snapshot()["total_since_start"] == expected_hour
