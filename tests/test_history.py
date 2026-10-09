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
    handler._http_wrapper.ensure_access_token.return_value = True
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


@pytest.mark.anyio
@pytest.mark.parametrize(
    "utterance_type", ["FALSE_WAKE_WORD_1P", "FALSE_WAKE_WORD_2P", "FALSE_WAKE_WORD"]
)
@pytest.mark.parametrize("with_older_record", [False, True])
async def test_false_wake_does_not_replace_qualifying_history(
    handler: AmazonHistoryHandler, utterance_type: str, with_older_record: bool
) -> None:
    """False wakes are excluded before choosing the latest record per device."""
    older = _record(None)
    false_wake = {
        **older,
        "timestamp": older["timestamp"] + 1,
        "utteranceType": utterance_type,
        "title": "",
        "subTitle": "",
    }
    payload = [false_wake, older] if with_older_record else [false_wake]
    handler._vocal_history_json = AsyncMock(  # type: ignore[method-assign]
        return_value={"alexaHistoryRecords": payload}
    )

    records = await handler.get_vocal_history()

    if with_older_record:
        assert records[TEST_SERIAL_1].timestamp == older["timestamp"]
        assert records[TEST_SERIAL_1].title == older["title"]
    else:
        assert records == {}


def _turn(  # noqa: PLR0913 - mirrors a conversation fragment
    purpose: str,
    text: str,
    timestamp: str,
    uri: str,
    *,
    related: str | None = None,
    variant: bool = False,
) -> dict[str, Any]:
    return {
        "utteranceId": uri,
        "createTime": timestamp,
        "fragment": {
            "fragmentURI": uri,
            "metadata": {
                "purpose": purpose,
                "relationships": [{"type": "RELATES_TO", "fragmentURI": related}]
                if related
                else [],
            },
            "content": None if variant else {"text": text},
            "variants": [{"content": {"text": text}}] if variant else [],
        },
    }


@pytest.mark.anyio
async def test_conversation_details_replace_labels(
    handler: AmazonHistoryHandler,
) -> None:
    """Keep the category but obtain the latest command/reply from CSD."""
    raw = {
        **_record(None),
        "recordType": "conversation",
        "title": "Date/Time Request",
        "subTitle": "What time is it?",
    }
    raw.pop("utteranceType")
    handler._vocal_history_json = AsyncMock(return_value={"alexaHistoryRecords": [raw]})
    detail = {
        "conversationTurns": [
            _turn("USER", "old question", "2026-10-07T08:30:00Z", "old"),
            _turn(
                "AGENT", "old reply", "2026-10-07T08:30:01Z", "old-agent", related="old"
            ),
            _turn("USER", "What time is it?", "2026-10-07T08:37:34.776Z", "new"),
            _turn(
                "AGENT",
                "It's 1:37 a.m.",
                "2026-10-07T08:37:35.334Z",
                "new-agent",
                related="new",
                variant=True,
            ),
        ]
    }
    handler._conversation_detail_json = AsyncMock(return_value=detail)
    record = (await handler.get_vocal_history())[TEST_SERIAL_1]
    assert record.activity_title == "Date/Time Request"
    assert record.voice_command == "What time is it?"
    assert record.voice_reply == "It's 1:37 a.m."
    expected_timestamp = 1791362254776
    assert record.timestamp == expected_timestamp


@pytest.mark.anyio
async def test_user_without_reply_and_empty_details(
    handler: AmazonHistoryHandler,
) -> None:
    """No reply is legitimate; empty CSD details must not become a baseline."""
    raw = {**_record(None), "recordType": "conversation", "title": "Request"}
    handler._vocal_history_json = AsyncMock(return_value={"alexaHistoryRecords": [raw]})
    handler._conversation_detail_json = AsyncMock(
        side_effect=[
            {"conversationTurns": []},
            {
                "conversationTurns": [
                    _turn("USER", "Good night.", "2026-10-07T14:36:07.295Z", "user")
                ]
            },
        ]
    )
    assert await handler.get_vocal_history() == {}
    record = (await handler.get_vocal_history())[TEST_SERIAL_1]
    assert record.voice_command == "Good night."
    assert record.voice_reply == ""


@pytest.mark.anyio
async def test_csd_query_uses_rah_identifiers(handler: AmazonHistoryHandler) -> None:
    """Use the RAH start time and account identifiers, keeping the GET method."""
    handler._session_state_data = SimpleNamespace(
        retail_site_url=URL("https://www.amazon.ca"),
        login_stored_data={REFRESH_ACCESS_TOKEN: "test-token"},
    )
    handler._http_wrapper.session_request.return_value = (
        None,
        SimpleNamespace(status=200),
    )
    handler._http_wrapper.response_to_json.return_value = {"conversationTurns": []}
    await handler._conversation_detail_json(
        {
            "conversationId": "conversation",
            "customerId": "customer",
            "startTime": 123,
            "timestamp": 456,
        }
    )
    call = handler._http_wrapper.session_request.await_args.kwargs
    assert call["method"] == "GET"
    assert call["url"].path == "/alexa-privacy/apd/csd/customer-conversation-detail"
    assert dict(call["url"].query) == {
        "conversationId": "conversation",
        "customerId": "customer",
        "timestamp": "123",
        "sort": "ASCENDING",
    }


@pytest.mark.anyio
async def test_utterance_uses_asr_and_tts_content(
    handler: AmazonHistoryHandler,
) -> None:
    """Routine reply-only content and actual ASR text retain their meanings."""
    raw = {
        **_record(None),
        "title": "category",
        "subTitle": "summary",
        "voiceHistoryRecordItems": [
            {"recordItemType": "ASR_REPLACEMENT_TEXT", "transcriptText": ""},
            {
                "recordItemType": "TTS_REPLACEMENT_TEXT",
                "transcriptText": "Home Assistant has started",
            },
        ],
    }
    handler._vocal_history_json = AsyncMock(return_value={"alexaHistoryRecords": [raw]})
    record = (await handler.get_vocal_history())[TEST_SERIAL_1]
    assert record.activity_title == "category"
    assert record.voice_command == ""
    assert record.voice_reply == "Home Assistant has started"


def test_latest_user_never_inherits_previous_reply() -> None:
    """A delayed AGENT fragment must remain linked to its original USER turn."""
    detail = {
        "conversationTurns": [
            _turn("USER", "old", "2026-10-07T08:30:00Z", "old"),
            _turn("USER", "new", "2026-10-07T08:31:00Z", "new"),
            _turn(
                "AGENT",
                "delayed old reply",
                "2026-10-07T08:32:00Z",
                "agent",
                related="old",
            ),
        ]
    }
    exchange = AmazonHistoryHandler._parse_conversation(detail)
    assert exchange is not None
    assert exchange[:2] == ("new", "")
