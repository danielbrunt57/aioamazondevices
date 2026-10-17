# Copyright 2024 Simone Chemelli and contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for AmazonDevice structure helpers."""

from collections.abc import Callable

import pytest

from aioamazondevices.structures import AmazonDevice, AmazonVocalRecord

from .const import TEST_SERIAL_1


@pytest.mark.parametrize(
    "expected",
    [
        pytest.param(True, id="voice-capable"),
        pytest.param(False, id="voice-incapable"),
    ],
)
def test_voice_control_supported(
    make_device: Callable[..., AmazonDevice],
    expected: bool,
) -> None:
    """The voice_control_supported field reflects what the device reports."""
    device = make_device(TEST_SERIAL_1, voice_control_supported=expected)

    assert device.voice_control_supported is expected


@pytest.mark.parametrize(
    ("history_type", "title", "activity_title"),
    [
        ("VOICE_HISTORY", "turn on the lights", "turn on the lights"),
        ("ROUTINES_3P", "", "Initiated by routine"),
        ("ROUTINES_3P", "Amazon title", "Amazon title"),
        ("ROUTINES_OR_TAP_TO_ALEXA", "", ""),
    ],
)
def test_vocal_record_event_attributes(
    history_type: str, title: str, activity_title: str
) -> None:
    """Custom event attributes preserve legacy parser text and routine labels."""
    record = AmazonVocalRecord(
        timestamp=123,
        history_type=history_type,
        intent="Unknown",
        title=title,
        sub_title="Okay",
    )
    assert record.activity_title == activity_title
    assert record.voice_command == title
    assert record.voice_reply == "Okay"
    assert record.title == title
    assert record.sub_title == "Okay"
    record.title = "updated command"
    record.sub_title = "updated reply"
    assert record.voice_command == "updated command"
    assert record.voice_reply == "updated reply"
