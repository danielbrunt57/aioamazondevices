# Copyright 2024 Simone Chemelli and contributors
# SPDX-License-Identifier: Apache-2.0

"""Module to handle Alexa vocal history setting."""

import asyncio
from datetime import UTC, datetime, timedelta
from http import HTTPMethod, HTTPStatus
from typing import Any

from bs4 import Tag
from yarl import URL

from aioamazondevices.const.http import (
    CSRF_A2Z,
    REFRESH_ACCESS_TOKEN,
    URI_CONVERSATION_DETAIL,
    URI_HISTORY_DATA,
    URI_HISTORY_FRONTEND,
)
from aioamazondevices.exceptions import CannotRetrieveData
from aioamazondevices.http_wrapper import AmazonHttpWrapper, AmazonSessionStateData
from aioamazondevices.structures import AmazonVocalRecord
from aioamazondevices.utils import _LOGGER

BACKEND_REFRESH_WAIT_SECONDS = 2


class AmazonHistoryHandler:
    """Class to handle Alexa vocal history functionality."""

    def __init__(
        self,
        http_wrapper: AmazonHttpWrapper,
        session_state_data: AmazonSessionStateData,
    ) -> None:
        """Initialize AmazonHistoryHandler class."""
        self._session_state_data = session_state_data
        self._http_wrapper = http_wrapper
        self._csrf_a2z_token: str = ""
        # force initial refresh
        self._csrf_a2z_refresh_time = datetime.now(UTC) - timedelta(days=2)

    async def _vocal_history_json(
        self, *, device_serial_number: str | None = None, device_type: str | None = None
    ) -> dict[str, Any]:
        """Request vocal history data."""
        await self._update_vocal_history_token()

        refresh_successful = await self._http_wrapper.ensure_access_token()
        if not refresh_successful:
            _LOGGER.warning("Access token refresh failed")

        access_token = self._session_state_data.login_stored_data[REFRESH_ACCESS_TOKEN]

        start_time = (
            datetime.now(UTC).replace(hour=12, minute=0, second=0, microsecond=0)
            - timedelta(days=7)
        ).timestamp() * 1000
        end_time = datetime.now(UTC).timestamp() * 1000
        query_string: dict[str, int | str] = {
            "startTime": int(start_time),
            "endTime": int(end_time),
        }
        if device_serial_number is not None:
            if not device_type:
                raise ValueError("A device type is required for filtered history")
            query_string["deviceSerialNumber"] = device_serial_number
            query_string["deviceType"] = device_type
        url = URL.joinpath(self._session_state_data.retail_site_url, URI_HISTORY_DATA)
        url = url.with_query(query_string)
        _, raw_res = await self._http_wrapper.session_request(
            method=HTTPMethod.POST,
            url=url,
            input_data={"previousRequestToken": None},
            json_data=True,
            extended_headers={
                "Authorization": f"Bearer {access_token}",
                CSRF_A2Z: self._csrf_a2z_token,
            },
        )
        history = await self._http_wrapper.response_to_json(raw_res, "history")
        _LOGGER.debug("Vocal history data: %s", history)
        return history

    async def get_vocal_history(
        self, *, device_serial_number: str | None = None, device_type: str | None = None
    ) -> dict[str, AmazonVocalRecord]:
        """Get vocal history."""
        # Give backend the time to update
        await asyncio.sleep(BACKEND_REFRESH_WAIT_SECONDS)

        if device_serial_number is None:
            history_json = await self._vocal_history_json()
        else:
            history_json = await self._vocal_history_json(
                device_serial_number=device_serial_number, device_type=device_type
            )

        candidates: dict[str, dict[str, Any]] = {}
        records: dict[str, AmazonVocalRecord] = {}
        for record in history_json["alexaHistoryRecords"]:
            _LOGGER.debug("Processing vocal history record: %s", record)
            utterance_type = str(record.get("utteranceType") or "")
            device_info = record.get("deviceInfo")
            if (
                utterance_type
                in [
                    "ASR_TIMEOUT",
                    "DEVICE_ARBITRATION",
                    "NO_EXPRESSED_INTENT",
                    "WAKE_WORD_ONLY",
                ]
                or utterance_type.startswith("FALSE_WAKE_WORD")
                # InvokeRoutineIntent, AddToListIntent are not linked to a device
                or device_info is None
            ):
                continue

            if isinstance(device_info, list):
                device_info = device_info[0] if device_info else None
            if not isinstance(device_info, dict):
                continue
            serial = device_info["deviceSerialNumber"]
            if device_serial_number is not None and serial != device_serial_number:
                continue
            if (
                serial not in candidates
                or record["timestamp"] > candidates[serial]["timestamp"]
            ):
                candidates[serial] = record

        for serial, record in candidates.items():
            timestamp = record["timestamp"]
            utterance_type = str(record.get("utteranceType") or "")
            command = ""
            reply = ""
            if record.get("recordType") == "conversation":
                try:
                    detail = await self._conversation_detail_json(record)
                except CannotRetrieveData:
                    _LOGGER.exception(
                        "Conversation details unavailable for serial=%s", serial
                    )
                    continue
                exchange = self._parse_conversation(detail)
                if exchange is None:
                    # Do not advance the baseline while details are unavailable.
                    continue
                command, reply, turn_timestamp = exchange
                if turn_timestamp is not None:
                    timestamp = turn_timestamp
            else:
                command, reply = self._parse_utterance(record)
            person_info = record.get("personsInfo")
            if isinstance(person_info, list):
                person_info = person_info[0] if person_info else None
            if not isinstance(person_info, dict):
                person_info = {}
            new_record = AmazonVocalRecord(
                timestamp=timestamp,
                history_type=utterance_type or record.get("recordType") or "Unknown",
                intent=record.get("intent") or "Unknown",
                title=record["title"],
                sub_title=record["subTitle"],
                person_first_name=person_info.get("personFirstName"),
                person_type=person_info.get("personType"),
                voice_command=command,
                voice_reply=reply,
            )
            # Store only the latest record per serial number
            if serial not in records or timestamp > records[serial].timestamp:
                records[serial] = new_record

        return records

    @staticmethod
    def _parse_utterance(record: dict[str, Any]) -> tuple[str, str]:
        """Read spoken content without using conversation category labels."""
        items = record.get("voiceHistoryRecordItems")
        if not isinstance(items, list):
            return str(record.get("title") or ""), str(record.get("subTitle") or "")
        texts = {}
        for kind in ("ASR_REPLACEMENT_TEXT", "TTS_REPLACEMENT_TEXT"):
            texts[kind] = " ".join(
                str(item["transcriptText"])
                for item in items
                if isinstance(item, dict)
                and item.get("recordItemType") == kind
                and item.get("transcriptText")
            )
        return texts["ASR_REPLACEMENT_TEXT"], texts["TTS_REPLACEMENT_TEXT"]

    async def _conversation_detail_json(self, record: dict[str, Any]) -> dict[str, Any]:
        """Get conversation turns using the identifiers supplied by RAH."""
        if not record.get("conversationId") or not record.get("customerId"):
            raise CannotRetrieveData("Missing conversation identifiers")
        url = URL.joinpath(
            self._session_state_data.retail_site_url, URI_CONVERSATION_DETAIL
        ).with_query(
            {
                "conversationId": record["conversationId"],
                "timestamp": record.get("startTime") or record["timestamp"],
                "sort": "ASCENDING",
                "customerId": record["customerId"],
            }
        )
        access_token = self._session_state_data.login_stored_data[REFRESH_ACCESS_TOKEN]
        _, response = await self._http_wrapper.session_request(
            method=HTTPMethod.GET,
            url=url,
            extended_headers={
                "Authorization": f"Bearer {access_token}",
                CSRF_A2Z: self._csrf_a2z_token,
            },
        )
        if response.status != HTTPStatus.OK:
            raise CannotRetrieveData(f"Conversation detail returned {response.status}")
        return await self._http_wrapper.response_to_json(
            response, "conversation detail"
        )

    @staticmethod
    def _fragment_text(fragment: dict[str, Any]) -> str:
        """Read text from the primary content or its alternative variants."""
        content = fragment.get("content")
        if isinstance(content, dict) and isinstance(content.get("text"), str):
            return content["text"]
        for variant in fragment.get("variants") or []:
            if not isinstance(variant, dict):
                continue
            content = variant.get("content")
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                return content["text"]
        return ""

    @staticmethod
    def _turn_timestamp(turn: dict[str, Any]) -> int | None:
        """Convert the conversation turn timestamp to Amazon milliseconds."""
        fragment = turn.get("fragment") or {}
        value = turn.get("createTime") or fragment.get("timestamp")
        try:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                return None
            return int(parsed.timestamp() * 1000)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _parse_conversation(
        cls, detail: dict[str, Any]
    ) -> tuple[str, str, int | None] | None:
        """Select the latest USER turn and only its associated AGENT responses."""
        turns = detail.get("conversationTurns") or []
        users = [
            turn
            for turn in turns
            if isinstance(turn, dict)
            and isinstance(turn.get("fragment"), dict)
            and (turn["fragment"].get("metadata") or {}).get("purpose") == "USER"
            and cls._fragment_text(turn["fragment"]).strip()
        ]
        if not users:
            return None
        user = max(
            enumerate(users),
            key=lambda pair: (cls._turn_timestamp(pair[1]) or 0, pair[0]),
        )[1]
        fragment = user["fragment"]
        user_uri = fragment.get("fragmentURI")
        user_timestamp = cls._turn_timestamp(user)
        replies: list[str] = []
        for turn in turns:
            if not isinstance(turn, dict) or not isinstance(turn.get("fragment"), dict):
                continue
            agent = turn["fragment"]
            metadata = agent.get("metadata") or {}
            if metadata.get("purpose") != "AGENT":
                continue
            relationships = metadata.get("relationships") or []
            related = any(
                isinstance(rel, dict)
                and rel.get("type") == "RELATES_TO"
                and user_uri
                and rel.get("fragmentURI") == user_uri
                for rel in relationships
            )
            if not relationships:
                agent_timestamp = cls._turn_timestamp(turn)
                related = bool(
                    user.get("utteranceId")
                    and turn.get("utteranceId") == user["utteranceId"]
                    and user_timestamp is not None
                    and agent_timestamp is not None
                    and agent_timestamp >= user_timestamp
                )
            if related and (text := cls._fragment_text(agent)):
                replies.append(text)
        return cls._fragment_text(fragment), " ".join(replies), user_timestamp

    async def _update_vocal_history_token(self) -> None:
        """Find anti-csrftoken-a2z token."""
        csrf_token_age = datetime.now(UTC) - self._csrf_a2z_refresh_time
        if csrf_token_age < timedelta(hours=12):
            return

        bs_resp, _ = await self._http_wrapper.session_request(
            method=HTTPMethod.GET,
            url=URL.joinpath(
                self._session_state_data.retail_site_url, URI_HISTORY_FRONTEND
            ),
        )
        token_meta = bs_resp.find("meta", attrs={"name": "csrf-token"})
        if isinstance(token_meta, Tag):
            token = token_meta.get("content")
            if token:
                self._csrf_a2z_token = str(token)
                self._csrf_a2z_refresh_time = datetime.now(UTC)
                return
        raise CannotRetrieveData("Cannot find anti-csrftoken-a2z token")
