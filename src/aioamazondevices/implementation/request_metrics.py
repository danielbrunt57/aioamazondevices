# Copyright 2024 Simone Chemelli and contributors
# SPDX-License-Identifier: Apache-2.0

"""In-memory HTTP request metrics for live experiments."""

from collections import Counter, deque
from dataclasses import dataclass
from http import HTTPStatus
from time import monotonic
from typing import Any

from aioamazondevices.utils import _LOGGER

MINUTE_SECONDS = 60
HOUR_SECONDS = 3600
DAY_SECONDS = 86400


@dataclass
class RequestAttempt:
    """One wrapper HTTP attempt, including attempts still in flight."""

    started: float
    endpoint: str
    status: int | None = None
    failed: bool = False


class RequestMetrics:
    """Retain 24 hours of attempts, with lifetime totals since initialization."""

    def __init__(self) -> None:
        """Initialize counters and the summary clock."""
        self._attempts: deque[RequestAttempt] = deque()
        self._total = 0
        self._last_log = monotonic()

    def start(self, endpoint: str) -> RequestAttempt:
        """Count an attempt immediately before calling the HTTP client."""
        now = monotonic()
        self._prune(now)
        attempt = RequestAttempt(now, endpoint)
        self._attempts.append(attempt)
        self._total += 1
        return attempt

    def _prune(self, now: float) -> None:
        while self._attempts and self._attempts[0].started <= now - DAY_SECONDS:
            self._attempts.popleft()

    def snapshot(self) -> dict[str, Any]:
        """Return rolling counts, grouping attempts by their start time."""
        now = monotonic()
        self._prune(now)
        result: dict[str, Any] = {"total_since_start": self._total}
        for name, seconds in (
            ("minute", MINUTE_SECONDS),
            ("hour", HOUR_SECONDS),
            ("day", DAY_SECONDS),
        ):
            attempts = [a for a in self._attempts if a.started > now - seconds]
            result[name] = {
                "requests": len(attempts),
                "failures": sum(a.failed for a in attempts),
                "http_429": sum(
                    a.status == HTTPStatus.TOO_MANY_REQUESTS for a in attempts
                ),
                "endpoints": dict(Counter(a.endpoint for a in attempts)),
            }
        return result

    def maybe_log(self) -> None:
        """Log at most once a minute, when an HTTP attempt finishes."""
        now = monotonic()
        if now - self._last_log < MINUTE_SECONDS:
            return
        self._last_log = now
        _LOGGER.info("HTTP request metrics (rolling windows): %s", self.snapshot())
