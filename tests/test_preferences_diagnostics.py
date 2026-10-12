# Copyright 2024 Simone Chemelli and contributors
# SPDX-License-Identifier: Apache-2.0

"""Preferences error diagnostics must be bounded and exclude sensitive values."""

from http import HTTPMethod
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from aioamazondevices.api import AmazonEchoApi
from aioamazondevices.exceptions import ServiceUnavailable


@pytest.mark.anyio
@pytest.mark.parametrize("large", [False, True])
async def test_preferences_error_sample_is_bounded_and_private(
    api: AmazonEchoApi, caplog: pytest.LogCaptureFixture, large: bool
) -> None:
    """Log fixed phrases and cookie names while preserving the original 503 error."""
    caplog.set_level("DEBUG", logger="aioamazondevices")
    secret = "PRIVATE_BODY_AND_COOKIE_VALUE"  # noqa: S105 -- synthetic privacy sentinel
    body = (
        "x" * 8192 + "cloudfront " + secret
        if large
        else "<html><title>503 Service Unavailable</title>cloudfront "
        + secret
        + "</html>"
    )

    async def handler(_request: web.Request) -> web.Response:
        return web.Response(
            status=503,
            text=body,
            content_type="text/html",
            headers={"Retry-After": "600"},
        )

    app = web.Application()
    app.router.add_get("/devicesTypes/type/deviceId/serial/preferences", handler)
    async with TestServer(app) as server:
        with pytest.raises(ServiceUnavailable) as exc:
            await api._http_wrapper.session_request(
                HTTPMethod.GET,
                server.make_url("/devicesTypes/type/deviceId/serial/preferences"),
                retry_server_errors=False,
                extended_headers={
                    "Cookie": f"session-token={secret}; at-main={secret}"
                },
            )
        assert exc.value.retry_after == "600"
    diagnostic = next(
        record.message
        for record in caplog.records
        if record.message.startswith("Preferences 503 diagnostics:")
    )
    assert secret not in diagnostic
    assert "session-token" in diagnostic
    assert "at-main" in diagnostic
    assert "read_failed=False" in diagnostic
    if large:
        assert "sample_bytes=8192" in diagnostic
        assert "sample_limit_reached=True" in diagnostic
        assert "cloudfront" not in diagnostic
    else:
        assert "service unavailable" in diagnostic
        assert "cloudfront" in diagnostic
        assert "sample_limit_reached=False" in diagnostic
    assert api._http_wrapper.request_metrics["total_since_start"] == 1


@pytest.mark.anyio
async def test_diagnostic_read_failure_keeps_error_handling(
    api: AmazonEchoApi, caplog: pytest.LogCaptureFixture
) -> None:
    """An error while sampling still releases the response and logs metadata."""
    caplog.set_level("DEBUG", logger="aioamazondevices")
    response = SimpleNamespace(
        content=SimpleNamespace(readexactly=AsyncMock(side_effect=TimeoutError)),
        release=Mock(),
        request_info=SimpleNamespace(headers={}),
        history=(),
    )
    await api._http_wrapper._log_preferences_503(response)
    response.release.assert_called_once()
    assert "read_failed=True" in caplog.text


@pytest.mark.anyio
@pytest.mark.parametrize("preferences", [True, False])
async def test_diagnostics_disabled_or_unrelated_endpoint(
    api: AmazonEchoApi,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    preferences: bool,
) -> None:
    """Only preferences failures at debug level are sampled."""
    caplog.set_level("INFO" if preferences else "DEBUG", logger="aioamazondevices")
    diagnostic = AsyncMock()
    monkeypatch.setattr(api._http_wrapper, "_log_preferences_503", diagnostic)
    app = web.Application()

    async def handler(_request: web.Request) -> web.Response:
        return web.Response(status=503)

    app.router.add_get("/{path:.*}", handler)
    async with TestServer(app) as server:
        with pytest.raises(ServiceUnavailable):
            await api._http_wrapper.session_request(
                HTTPMethod.GET,
                server.make_url(
                    "/devicesTypes/type/deviceId/serial/preferences"
                    if preferences
                    else "/other"
                ),
                retry_server_errors=False,
            )
    diagnostic.assert_not_awaited()
