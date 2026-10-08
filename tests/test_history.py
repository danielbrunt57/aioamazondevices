# Copyright 2024 Simone Chemelli and contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Alexa vocal history parsing."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from yarl import URL

from aioamazondevices.const.http import REFRESH_ACCESS_TOKEN, URI_HISTORY_DATA
from aioamazondevices.implementation import history as history_module
from aioamazondevices.implementation.history import AmazonHistoryHandler

from .const import TEST_SERIAL_1

PersonsInfo = dict[str, str] | list[dict[str, str]] | None


class _Absent:
    """Marker for a payload that carries no `personsInfo` key at all."""


ABSENT = _Absent()

TEST_PERSON = {
    "personId": "amzn1.actor.person.oid.PERSON_ID",
    "personFirstName": "Alice",
    "personType": "ADULT",
}


def _record(persons_info: PersonsInfo | _Absent) -> dict[str, Any]:
    """Build a minimal vocal history record, shaped like the Amazon payload."""
    record: dict[str, Any] = {
        "timestamp": 1757000000000,
        "utteranceType": "GENERAL",
        "intent": "PlayMusicIntent",
        "title": "play some music",
        "subTitle": "Echo Dot",
        "deviceInfo": {"deviceSerialNumber": TEST_SERIAL_1},
    }
    if not isinstance(persons_info, _Absent):
        record["personsInfo"] = persons_info
    return record


@pytest.fixture
def handler(monkeypatch: pytest.MonkeyPatch) -> AmazonHistoryHandler:
    """Return a history handler that skips the backend refresh wait."""
    monkeypatch.setattr(history_module, "BACKEND_REFRESH_WAIT_SECONDS", 0)
    return AmazonHistoryHandler(AsyncMock(), AsyncMock())


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("persons_info", "expected"),
    [
        pytest.param(
            TEST_PERSON,
            ("Alice", "ADULT"),
            id="recognised-speaker",
        ),
        pytest.param(
            [TEST_PERSON],
            ("Alice", "ADULT"),
            id="recognised-speaker-as-list",
        ),
        pytest.param(None, (None, None), id="voice-not-recognised"),
        pytest.param([], (None, None), id="empty-list"),
        pytest.param(ABSENT, (None, None), id="personsinfo-key-absent"),
    ],
)
async def test_vocal_history_exposes_speaker(
    handler: AmazonHistoryHandler,
    persons_info: PersonsInfo | _Absent,
    expected: tuple[str | None, str | None],
) -> None:
    """The recognised speaker is taken from personsInfo, absent when unknown."""
    handler._vocal_history_json = AsyncMock(  # type: ignore[method-assign]
        return_value={"alexaHistoryRecords": [_record(persons_info)]}
    )

    records = await handler.get_vocal_history()

    record = records[TEST_SERIAL_1]
    assert (record.person_first_name, record.person_type) == expected
    # personId is in the payload but account-scoped, so it is deliberately not exposed
    assert not hasattr(record, "person_id")


@pytest.mark.anyio
@pytest.mark.parametrize("filtered", [False, True])
async def test_rah_device_filter_query(
    handler: AmazonHistoryHandler, filtered: bool
) -> None:
    """Only device probes add both device parameters to the RAH POST."""
    handler._session_state_data = SimpleNamespace(
        retail_site_url=URL("https://www.amazon.ca"),
        login_stored_data={REFRESH_ACCESS_TOKEN: "test-token"},
    )
    handler._update_vocal_history_token = AsyncMock()
    handler._http_wrapper.refresh_data.return_value = (True, None)
    handler._http_wrapper.session_request.return_value = (None, "response")
    handler._http_wrapper.response_to_json.return_value = {"alexaHistoryRecords": []}
    kwargs = (
        {"device_serial_number": TEST_SERIAL_1, "device_type": "A1B2C3"}
        if filtered
        else {}
    )

    await handler.get_vocal_history(**kwargs)

    call = handler._http_wrapper.session_request.await_args.kwargs
    assert call["method"] == "POST"
    assert call["url"].path.endswith(URI_HISTORY_DATA)
    assert call["input_data"] == {"previousRequestToken": None}
    query = call["url"].query
    assert "startTime" in query
    assert "endTime" in query
    if filtered:
        assert query["deviceSerialNumber"] == TEST_SERIAL_1
        assert query["deviceType"] == "A1B2C3"
    else:
        assert "deviceSerialNumber" not in query
        assert "deviceType" not in query
