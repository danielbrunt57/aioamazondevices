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
    ("history_type", "title", "expected"),
    [
        ("ROUTINES_3P", "", "Activity initiated by routine"),
        ("ROUTINES_3P", "Amazon title", "Amazon title"),
        ("ROUTINES_OR_TAP_TO_ALEXA", "", ""),
        ("conversation", "Date/Time Request", "Date/Time Request"),
    ],
)
def test_activity_title_fallback(history_type: str, title: str, expected: str) -> None:
    """Only an empty routine title receives a descriptive fallback."""
    record = AmazonVocalRecord(
        timestamp=0,
        history_type=history_type,
        intent="Unknown",
        title=title,
        sub_title="Home Assistant has started",
        voice_reply="Home Assistant has started",
    )
    assert record.activity_title == expected
    assert record.title == title
    assert record.voice_command == ""
    assert record.voice_reply == "Home Assistant has started"
