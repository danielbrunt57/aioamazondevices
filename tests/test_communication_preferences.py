# Copyright 2024 Simone Chemelli and contributors
# SPDX-License-Identifier: Apache-2.0

"""Optional preferences reads must not delay every device on server errors."""

from collections.abc import Callable
from http import HTTPMethod, HTTPStatus
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from aioamazondevices import http_wrapper
from aioamazondevices.api import AmazonEchoApi
from aioamazondevices.exceptions import CannotAuthenticate, CannotRetrieveData
from aioamazondevices.implementation import communication
from aioamazondevices.structures import AmazonDevice


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status", "retry_server_errors", "attempts", "waits"),
    [
        (503, False, 1, 0),
        (500, False, 1, 0),
        (503, True, 3, 2),
        (500, True, 3, 2),
        (429, False, 3, 2),
        (401, False, 1, 0),
    ],
)
async def test_optional_server_retry_policy(  # noqa: PLR0913, PLR0917
    api: AmazonEchoApi,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    retry_server_errors: bool,
    attempts: int,
    waits: int,
) -> None:
    """Keep rate-limit backoff and authentication handling for optional reads."""
    calls = 0

    async def handler(_request: web.Request) -> web.Response:
        nonlocal calls
        calls += 1
        return web.Response(status=status)

    sleep = AsyncMock()
    monkeypatch.setattr(http_wrapper.asyncio, "sleep", sleep)
    app = web.Application()
    app.router.add_get("/", handler)
    async with TestServer(app) as server:
        with pytest.raises(
            CannotAuthenticate
            if status == HTTPStatus.UNAUTHORIZED
            else CannotRetrieveData
        ):
            await api._http_wrapper.session_request(
                HTTPMethod.GET,
                server.make_url("/"),
                retry_server_errors=retry_server_errors,
            )
    assert calls == attempts
    # aiohttp also sleeps for zero seconds while closing its test server.
    assert [call.args[0] for call in sleep.await_args_list if call.args[0]] == (
        [2, 5] if waits else []
    )
    assert api._http_wrapper.request_metrics["total_since_start"] == attempts


@pytest.mark.anyio
async def test_preferences_cache_continue_and_recover(
    api: AmazonEchoApi,
    make_device: Callable[..., AmazonDevice],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One failing device keeps its cache while other devices and later polls work."""
    failing = True
    calls: list[str] = []

    async def handler(request: web.Request) -> web.Response:
        serial = request.match_info["serial"]
        calls.append(serial)
        assert request.match_info["device_type"] == "A1B2C3"
        assert request.query.getall("devicePreferences") == [
            "communications",
            "dropin",
            "announcements",
        ]
        assert not await request.read()
        if serial == "first" and failing:
            return web.Response(status=503)
        return web.json_response(
            {
                "devicePermissionsPreferences": [
                    {
                        "devicePreference": "communications",
                        "state": "ON",
                        "allowed": True,
                    },
                ]
            }
        )

    sleep = AsyncMock()
    monkeypatch.setattr(http_wrapper.asyncio, "sleep", sleep)
    communications = api._communication_handler
    communications._communication_preferences["first"] = {"communications": "OFF"}
    app = web.Application()
    app.router.add_get(
        "/devicesTypes/{device_type}/deviceId/{serial}/preferences", handler
    )
    async with TestServer(app) as server:
        communications._communication_site = server.make_url("/")
        devices = [make_device("first"), make_device("second")]
        result = await communications.get_communication_preferences(devices)
        assert result == {
            "first": {"communications": "OFF"},
            "second": {"communications": "ON"},
        }
        assert calls == ["first", "second"]
        failing = False
        result = await communications.get_communication_preferences(devices)
        assert result["first"] == {"communications": "ON"}
    assert not [call for call in sleep.await_args_list if call.args[0]]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("invalid", None),
        ("-1", None),
        ("1.5", None),
        ("+10", None),
        ("１２", None),  # noqa: RUF001 -- delay-seconds must use ASCII digits
        ("9" * 400, None),
        ("0", 0.0),
        (" 600 ", 600.0),
        ("Wed, 21 Oct 2015 07:38:00 GMT", 600.0),
        ("Wed, 21 Oct 2015 07:20:00 GMT", 0.0),
        ("Wed, 21 Oct 2015 07:38:00", None),
    ],
)
def test_retry_after_formats(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: float | None
) -> None:
    """Accept both HTTP forms and reject malformed or unrepresentable delays."""
    monkeypatch.setattr(communication, "time", lambda: 1445412480.0)
    assert communication._retry_after_delay(value) == expected


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status", "header", "deferred"),
    [
        (503, "600", True),
        (503, "Wed, 21 Oct 2015 07:38:00 GMT", True),
        (503, "Wed, 21 Oct 2015 07:20:00 GMT", False),
        (503, "0", False),
        (503, "invalid", False),
        (503, None, False),
        (500, "600", False),
    ],
)
async def test_retry_after_defers_only_failing_device(  # noqa: PLR0913, PLR0917
    api: AmazonEchoApi,
    make_device: Callable[..., AmazonDevice],
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    header: str | None,
    deferred: bool,
) -> None:
    """Respect Retry-After across polls without sleeping or skipping other devices."""
    now = 100.0
    recovered = False
    calls: list[str] = []

    async def handler(request: web.Request) -> web.Response:
        serial = request.match_info["serial"]
        calls.append(serial)
        if serial == "first" and not recovered:
            return web.Response(
                status=status, headers={"Retry-After": header} if header else {}
            )
        return web.json_response(
            {
                "devicePermissionsPreferences": [
                    {
                        "devicePreference": "communications",
                        "state": "ON",
                        "allowed": True,
                    }
                ]
            }
        )

    sleep = AsyncMock()
    monkeypatch.setattr(http_wrapper.asyncio, "sleep", sleep)
    monkeypatch.setattr(communication, "monotonic", lambda: now)
    monkeypatch.setattr(communication, "time", lambda: 1445412480.0)
    communications = api._communication_handler
    communications._communication_preferences["first"] = {"communications": "OFF"}
    app = web.Application()
    app.router.add_get(
        "/devicesTypes/{device_type}/deviceId/{serial}/preferences", handler
    )
    async with TestServer(app) as server:
        communications._communication_site = server.make_url("/")
        devices = [make_device("first"), make_device("second")]
        await communications.get_communication_preferences(devices)
        assert calls == ["first", "second"]
        # A wall-clock adjustment after parsing cannot shorten the cooldown.
        monkeypatch.setattr(communication, "time", lambda: 2000000000.0)
        now = 699.0
        result = await communications.get_communication_preferences(devices)
        assert result["first"] == {"communications": "OFF"}
        assert result["second"] == {"communications": "ON"}
        assert calls == (
            ["first", "second", "second"]
            if deferred
            else ["first", "second", "first", "second"]
        )
        recovered = True
        now = 700.0
        result = await communications.get_communication_preferences(devices)
        assert result["first"] == {"communications": "ON"}
        assert "first" not in communications._preferences_retry_at
        assert calls[-2:] == ["first", "second"]
        assert api._http_wrapper.request_metrics["total_since_start"] == len(calls)
    assert not [call for call in sleep.await_args_list if call.args[0]]
