"""Direct tests for the pacer: request spacing and bounded retries.

These assertions used to live in test_fetch_corpus.py, driven through
fetch_corpus with a Companies House handler. That made them integration tests
wearing a unit-test name: to prove the jitter formula you had to stand up a
filing history, a document metadata endpoint and a signed-URL redirect, none of
which the pacer knows about. Here the pacer is called directly and the
responses are queued by hand.

Time is still injected rather than real. The clock only advances when something
sleeps, so the delays are asserted on exactly and the suite runs in
milliseconds.
"""

import httpx
import pytest

from docintel.ingest.http import MAX_ATTEMPTS, Pacer

URL = "/anything"


class Clock:
    """A monotonic clock that only advances when something sleeps."""

    def __init__(self) -> None:
        self.t = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.t += seconds


class Responses:
    """Serves a queued list of responses, then 200s forever, logging requests."""

    def __init__(self, *queued: httpx.Response) -> None:
        self.queued = list(queued)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.queued:
            return self.queued.pop(0)
        return httpx.Response(200, json={"ok": True})


def make_pacer(clock: Clock) -> Pacer:
    # rand() is pinned so the jitter factor is 0.5 + 0.5 = 1.0 and the backoff
    # delays are the bare exponential.
    return Pacer(sleep=clock.sleep, monotonic=clock.monotonic, rand=lambda: 0.5)


def make_client(handler: Responses) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.example")


def test_requests_are_spaced_by_the_minimum_interval() -> None:
    clock = Clock()
    pacer = make_pacer(clock)
    handler = Responses()

    with make_client(handler) as client:
        for _ in range(3):
            pacer.get(client, URL)

    # Every request after the first waits, because the injected clock only
    # moves when we sleep, so no real time ever elapses between them.
    assert len(clock.slept) == len(handler.requests) - 1
    assert all(delay == pytest.approx(0.6) for delay in clock.slept)


def test_429_is_retried_and_retry_after_is_honoured() -> None:
    clock = Clock()
    pacer = make_pacer(clock)
    handler = Responses(httpx.Response(429, headers={"retry-after": "7"}))

    with make_client(handler) as client:
        response = pacer.get(client, URL)

    assert 7.0 in clock.slept
    # And the caller got the eventual success rather than the 429.
    assert response.status_code == 200


def test_429_without_retry_after_backs_off_with_jitter() -> None:
    clock = Clock()
    pacer = make_pacer(clock)
    handler = Responses(httpx.Response(429), httpx.Response(429))

    with make_client(handler) as client:
        pacer.get(client, URL)

    # rand() is pinned to 0.5, so the jitter factor is 0.5 + 0.5 = 1.0 and the
    # delays are the bare exponential: 1s then 2s.
    backoffs = [delay for delay in clock.slept if delay not in (0.6,)]
    assert backoffs[:2] == [1.0, 2.0]


def test_retries_are_bounded_and_the_last_response_is_returned() -> None:
    """A server that never recovers must not retry forever, and must not raise.

    Returning the final failure is what lets the caller record it as a rejection
    rather than lose the whole run to an exception.
    """
    clock = Clock()
    pacer = make_pacer(clock)
    handler = Responses(*[httpx.Response(429)] * 10)

    with make_client(handler) as client:
        response = pacer.get(client, URL)

    assert len(handler.requests) == MAX_ATTEMPTS
    assert response.status_code == 429


def test_an_http_date_retry_after_falls_back_to_backoff() -> None:
    """We do not parse the date form. A mis-parse could park the run for hours."""
    clock = Clock()
    pacer = make_pacer(clock)
    handler = Responses(
        httpx.Response(429, headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})
    )

    with make_client(handler) as client:
        pacer.get(client, URL)

    assert 1.0 in clock.slept
