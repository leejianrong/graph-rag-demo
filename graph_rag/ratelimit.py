"""Request rate limits (GRA-137): one budget per API key, and a throttle on bad keys.

Both are token buckets held in this process: a bucket holds up to ``per_minute`` tokens, refills
at ``per_minute / 60`` a second, and a request spends one. A burst of ``per_minute`` is allowed,
then requests are refused until tokens refill. With several service instances each keeps its own
buckets, so the effective limit is the sum; a shared store is not built (the key file has the same
single-instance limit).

The bad-key throttle is keyed by client address, because a bad key has no identity. It does not
stop key guessing (a key is 256 random bits, so there is nothing to guess); it stops a client from
using failed requests to flood the service, and answers it with 429 instead of 401. Behind a proxy
every client shares the proxy's address, so one noisy client throttles the failures of all of them;
good keys are checked first and are never affected.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable

__all__ = ["RateLimited", "RateLimiter"]

_MAX_TRACKED = 10_000  # keys held at once; idle ones are dropped past this


class RateLimited(Exception):
    """A request went over its budget. ``retry_after`` is whole seconds until one token is back."""

    def __init__(self, message: str, retry_after: int) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class RateLimiter:
    """A token bucket per key. ``per_minute`` of 0 or less turns the limiter off."""

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic) -> None:
        self._capacity = float(max(per_minute, 0))
        self._rate = self._capacity / 60.0
        self._clock = clock
        self._lock = threading.Lock()
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last refill time)

    @property
    def enabled(self) -> bool:
        return self._capacity > 0

    def _refilled(self, key: str, now: float) -> float:
        tokens, then = self._buckets.get(key, (self._capacity, now))
        return min(self._capacity, tokens + (now - then) * self._rate)

    def spend(self, key: str) -> int | None:
        """Take one token. ``None`` when it was granted, else the seconds to wait for one."""
        if not self.enabled:
            return None
        with self._lock:
            now = self._clock()
            tokens = self._refilled(key, now)
            if tokens >= 1:
                self._buckets[key] = (tokens - 1, now)
                self._forget_idle(now)
                return None
            self._buckets[key] = (tokens, now)
            return max(1, math.ceil((1 - tokens) / self._rate))

    def exhausted(self, key: str) -> int | None:
        """The seconds until ``key`` has a token again, or ``None`` if it has one now."""
        if not self.enabled:
            return None
        with self._lock:
            tokens = self._refilled(key, self._clock())
            return None if tokens >= 1 else max(1, math.ceil((1 - tokens) / self._rate))

    def _forget_idle(self, now: float) -> None:
        if len(self._buckets) <= _MAX_TRACKED:
            return
        full = [k for k in self._buckets if self._refilled(k, now) >= self._capacity]
        for key in full:
            del self._buckets[key]
