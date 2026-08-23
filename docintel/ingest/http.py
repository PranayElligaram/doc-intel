"""Polite HTTP for the ingest fetchers: one request at a time, bounded retries.

Split out of corpus.py because it is the one piece of that module with nothing
Companies House about it. Any API we fetch a corpus from wants the same
treatment - a minimum gap between requests, backoff on 429 and 5xx, and a cap
on how long one bad response can park a run.
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
        authed: bool = True,
    ) -> httpx.Response:
        last: httpx.Response | None = None
        for attempt in range(self.max_attempts):
            self._wait_turn()
            request = client.build_request("GET", url, headers=headers)
            # auth=None disables the client-level credential for this request.
            # See _download for why the second hop of a redirect must not carry
            # it.
            response = client.send(request) if authed else client.send(request, auth=None)
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
