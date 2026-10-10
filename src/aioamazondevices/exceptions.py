# Copyright 2024 Simone Chemelli and contributors
# SPDX-License-Identifier: Apache-2.0

"""Exceptions module for Amazon devices."""

from __future__ import annotations


class AmazonError(Exception):
    """Base class for aioamazondevices errors."""


class CannotConnect(AmazonError):
    """Exception raised when connection fails."""


class CannotAuthenticate(AmazonError):
    """Exception raised when authentication fails."""


class CannotRestartDevice(AmazonError):
    """Exception raised when device restart fails."""


class CannotRetrieveData(AmazonError):
    """Exception raised when data retrieval fails."""


class ServiceUnavailable(CannotRetrieveData):
    """A 503 response, including the server's optional retry guidance."""

    def __init__(self, message: str, retry_after: str | None = None) -> None:
        """Preserve Retry-After for callers that defer optional reads."""
        super().__init__(message)
        self.retry_after = retry_after


class NoOnlineDevicesError(AmazonError):
    """Exception raised when no online devices are found."""


class CannotRegisterDevice(AmazonError):
    """Exception raised when device registration fails."""


class WrongMethod(AmazonError):
    """Exception raised when the wrong login method is used."""


class UpdatedAVSSite(AmazonError):
    """Exception raised when the AVS site is updated."""


class AVSStreamEndedUnexpectedly(AmazonError):
    """Exception raised when the AVS stream ends unexpectedly."""
