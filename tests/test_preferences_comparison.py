# Copyright 2024 Simone Chemelli and contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare combined and individual preferences reads for one experimental device."""

from collections.abc import Callable
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from aioamazondevices import http_wrapper
from aioamazondevices.api import AmazonEchoApi
from aioamazondevices.implementation import communication
from aioamazondevices.structures import AmazonDevice


@pytest.mark.anyio
@pytest.mark.parametrize("combined_success", [False, True])
async def test_comparison_partial_results_preserve_cache(
    api: AmazonEchoApi,
    make_device: Callable[..., AmazonDevice],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    combined_success: bool,
) -> None:
    """Singles merge partial results; other devices keep one combined read."""
    caplog.set_level("INFO", logger="aioamazondevices")
    target = communication._PREFERENCES_COMPARISON_SERIAL
    calls: list[tuple[str, tuple[str, ...]]] = []
    second_poll = False

    async def handler(request: web.Request) -> web.Response:
        serial = request.match_info["serial"]
        keys = tuple(request.query.getall("devicePreferences"))
        calls.append((serial, keys))
        if serial == target:
            if len(keys) > 1 and not combined_success:
                return web.Response(status=503)
            if keys == ("dropin",) and not second_poll:
                return web.Response(status=503)
            if keys == ("communications",) and second_poll:
                return web.Response(status=503)
        values = {"communications": "ON", "dropin": "Off", "announcements": "OFF"}
        return web.json_response(
            {
                "devicePermissionsPreferences": [
                    {
                        "devicePreference": key,
                        "state": values[key],
                        "allowed": not (second_poll and keys == ("announcements",)),
                    }
                    for key in keys
                ]
            }
        )

    sleep = AsyncMock()
    monkeypatch.setattr(http_wrapper.asyncio, "sleep", sleep)
    prefs = api._communication_handler
    prefs._communication_preferences[target] = {
        "communications": "OFF",
        "dropin": "Home",
        "announcements": "ON",
    }
    app = web.Application()
    app.router.add_get(
        "/devicesTypes/{device_type}/deviceId/{serial}/preferences", handler
    )
    async with TestServer(app) as server:
        prefs._communication_site = server.make_url("/")
        devices = [make_device(target), make_device("other")]
        result = await prefs.get_communication_preferences(devices)
        assert result[target] == {
            "communications": "ON",
            "dropin": "Off" if combined_success else "Home",
            "announcements": "OFF",
        }
        assert calls == [
            (target, ("communications", "dropin", "announcements")),
            (target, ("communications",)),
            (target, ("dropin",)),
            (target, ("announcements",)),
            ("other", ("communications", "dropin", "announcements")),
        ]
        second_poll = True
        result = await prefs.get_communication_preferences(devices)
        assert result[target] == {"communications": "ON", "dropin": "Off"}
        assert api._http_wrapper.request_metrics["total_since_start"] == len(calls)
    assert not [call for call in sleep.await_args_list if call.args[0]]
    for label in ("combined", "communications", "dropin", "announcements"):
        assert f"query={label} result=" in caplog.text


@pytest.mark.anyio
async def test_comparison_stops_for_retry_after(
    api: AmazonEchoApi,
    make_device: Callable[..., AmazonDevice],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server cooldown halts further variants and is respected across polls."""
    now = 100.0
    monkeypatch.setattr(communication, "monotonic", lambda: now)
    calls: list[str] = []

    async def handler(request: web.Request) -> web.Response:
        calls.append(request.match_info["serial"])
        return web.Response(status=503, headers={"Retry-After": "600"})

    app = web.Application()
    app.router.add_get(
        "/devicesTypes/{device_type}/deviceId/{serial}/preferences", handler
    )
    async with TestServer(app) as server:
        prefs = api._communication_handler
        prefs._communication_site = server.make_url("/")
        target = communication._PREFERENCES_COMPARISON_SERIAL
        devices = [make_device(target), make_device("other")]
        await prefs.get_communication_preferences(devices)
        assert calls == [target, "other"]
        now = 699.0
        await prefs.get_communication_preferences(devices)
        assert calls == [target, "other"]
