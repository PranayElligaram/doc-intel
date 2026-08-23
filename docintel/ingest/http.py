"""Polite HTTP for the ingest fetchers: one request at a time, bounded retries.

Split out of the fetcher because it is the one piece of it with nothing
source-specific about it. Any API we fetch a corpus from wants the same
treatment - a minimum gap between requests, backoff on 429 and 5xx, and a cap
on how long one bad response can park a run.

The minimum interval is 0.6s, which is well inside EDGAR's published ceiling of
10 requests per second. The limit is not the reason for the gap: a corpus fetch
is a background job with no deadline, and there is no benefit to being anywhere
near a rate limit that, if hit, gets your IP blocked.
"""

from collections.abc import Callable
from dataclasses import dataclass

import httpx

MIN_REQUEST_INTERVAL = 0.6
MAX_ATTEMPTS = 4
BASE_BACKOFF = 1.0
MAX_BACKOFF = 30.0


@dataclass
class Pacer:
    """Serialises requests: a minimum gap, then retries with bounded backoff.

    The clock, the sleep and the jitter are injected so the tests exercise the
    real retry path at zero wall-clock cost. A test that actually slept would
    either be slow or would quietly stop testing the backoff.
    """

    sleep: Callable[[float], None]
    monotonic: Callable[[], float]
    rand: Callable[[], float]
    min_interval: float = MIN_REQUEST_INTERVAL
    max_attempts: int = MAX_ATTEMPTS
    last_request_at: float | None = None

    def get(
        self,
        client: httpx.Client,
        url: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        last: httpx.Response | None = None
        for attempt in range(self.max_attempts):
            self._wait_turn()
            response = client.send(client.build_request("GET", url, headers=headers))
            self.last_request_at = self.monotonic()

            if response.status_code != 429 and response.status_code < 500:
                return response
            last = response
            if attempt == self.max_attempts - 1:
                break
            self.sleep(self._retry_delay(response, attempt))

        assert last is not None
        return last

    def _wait_turn(self) -> None:
        if self.last_request_at is None:
            return
        elapsed = self.monotonic() - self.last_request_at
        if elapsed < self.min_interval:
            self.sleep(self.min_interval - elapsed)

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        header = response.headers.get("retry-after")
        if header is not None:
            try:
                # Retry-After may also be an HTTP-date. We do not parse that
                # form: falling through to backoff is safe, whereas a
                # mis-parsed date could produce a delay of hours.
                return max(0.0, float(int(header)))
            except ValueError:
                pass
        # Exponential, jittered so that a batch of retries does not resynchronise
        # into a thundering herd, and capped so one bad response cannot park the
        # run for an unbounded time.
        backoff = min(BASE_BACKOFF * (2.0**attempt), MAX_BACKOFF)
        return backoff * (0.5 + self.rand())
