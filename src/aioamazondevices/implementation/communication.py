# Copyright 2024 Simone Chemelli and contributors
# SPDX-License-Identifier: Apache-2.0

"""Communication module for Amazon devices."""

from email.utils import parsedate_to_datetime
from http import HTTPMethod
from math import isfinite
from time import monotonic, time

from yarl import URL

from aioamazondevices.const.devices import DEVICE_TYPE_AQM, SPEAKER_GROUP_FAMILY
from aioamazondevices.const.http import COMM_SITE, URI_COMM_PREFERENCES
from aioamazondevices.exceptions import CannotRetrieveData, ServiceUnavailable
from aioamazondevices.http_wrapper import AmazonHttpWrapper, AmazonSessionStateData
from aioamazondevices.structures import AmazonDevice, AmazonDropInStatus
from aioamazondevices.utils import _LOGGER


def _retry_after_delay(value: str | None) -> float | None:
    """Parse Retry-After as delay-seconds or an HTTP date."""
    if value is None:
        return None
    value = value.strip()
    try:
        if value.isascii() and value.isdecimal():
            delay = float(int(value))
        else:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                return None
            delay = max(0.0, date.timestamp() - time())
    except (ValueError, TypeError, OverflowError, OSError):
        return None
    return delay if isfinite(delay) else None


class AlexaCommunicationsHandler:
    """Class to handle Alexa communications."""

    def __init__(
        self,
        http_wrapper: AmazonHttpWrapper,
        session_state_data: AmazonSessionStateData,
    ) -> None:
        """Initialize AlexaCommunicationsHandler class."""
        self._session_state_data = session_state_data
        self._http_wrapper = http_wrapper
        self._communication_site = URL(COMM_SITE)
        self._communication_preferences: dict[str, dict[str, str]] = {}
        self._preferences_retry_at: dict[str, float] = {}

    async def _set_communications_state(
        self, preference: str, device: AmazonDevice, state: str
    ) -> None:
        payload = {"state": state}
        url = URL.joinpath(
            self._communication_site,
            URI_COMM_PREFERENCES.format(
                device_type=device.device_type,
                serial_number=device.serial_number,
            ),
            preference,
        )
        await self._http_wrapper.session_request(
            method=HTTPMethod.PATCH, url=url, input_data=payload, json_data=True
        )

    async def set_communication_status(self, device: AmazonDevice, state: bool) -> None:
        """Enable / disable communications for device."""
        await self._set_communications_state(
            "communications", device, "ON" if state else "OFF"
        )

    async def set_announcement_status(self, device: AmazonDevice, state: bool) -> None:
        """Enable / disable announcements for device."""
        await self._set_communications_state(
            "announcements", device, "ON" if state else "OFF"
        )

    async def set_dropin_status(
        self, device: AmazonDevice, state: AmazonDropInStatus
    ) -> None:
        """Set allowed dropin state for device."""
        await self._set_communications_state("dropin", device, state.value)

    async def get_communication_preferences(
        self, devices: list[AmazonDevice]
    ) -> dict[str, dict[str, str]]:
        """Get communication preferences for a device."""
        for device in devices:
            if (
                device.device_family == SPEAKER_GROUP_FAMILY
                or device.device_type == DEVICE_TYPE_AQM
            ):
                # avoid unnecessary call for devices that don't support communications
                continue

            if monotonic() < self._preferences_retry_at.get(device.serial_number, 0):
                continue
            self._preferences_retry_at.pop(device.serial_number, None)

            query_string = {
                "devicePreferences": [
                    "communications",
                    "dropin",
                    "announcements",
                ]
            }
            url = URL.joinpath(
                self._communication_site,
                URI_COMM_PREFERENCES.format(
                    device_type=device.device_type,
                    serial_number=device.serial_number,
                ),
            )
            url = url.with_query(query_string)
            try:
                _, resp = await self._http_wrapper.session_request(
                    # These optional reads are serialized across devices. Use
                    # cached state on server errors and try again next refresh,
                    # rather than adding seven seconds of waits per device.
                    method=HTTPMethod.GET,
                    url=url,
                    retry_server_errors=False,
                )
            except CannotRetrieveData as err:
                if isinstance(err, ServiceUnavailable):
                    delay = _retry_after_delay(err.retry_after)
                    if delay is not None:
                        self._preferences_retry_at[device.serial_number] = (
                            monotonic() + delay
                        )
                        _LOGGER.debug(
                            "Deferring communications preferences for device %s "
                            "for %s seconds per Retry-After",
                            device.account_name,
                            delay,
                        )
                _LOGGER.warning(
                    "Failed to refresh communications settings for device %s, used cached values.",  # noqa: E501
                    device.account_name,
                )
                continue
            resp_json = await self._http_wrapper.response_to_json(
                resp, "devicesTypes(preferences)"
            )

            device_communication_preferences: dict[str, str] = {}
            for device_permissions_pref in resp_json.get(
                "devicePermissionsPreferences", {}
            ):
                device_pref = device_permissions_pref["devicePreference"]
                pref_state = device_permissions_pref.get("state")
                pref_allowed = device_permissions_pref.get("allowed")

                if pref_allowed is True:
                    device_communication_preferences[device_pref] = pref_state

            self._communication_preferences[device.serial_number] = (
                device_communication_preferences
            )

        return self._communication_preferences
