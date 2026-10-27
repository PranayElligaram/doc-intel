"""The provider seam: the fake is deterministic, and every real failure is LLMError.

Two things are being protected here. First, that `FakeLLM` cannot quietly
become non-deterministic, because the whole offline suite rests on it. Second,
that no failure mode of the real client escapes as a bare exception: the
extraction layer turns `LLMError` into `Failed`, and anything that slips past
that net either crashes an eval run or gets mis-scored as an abstention.

No Ollama is installed or contacted. The transport is a mock, so the failure
paths are exercised exactly and the suite still runs in milliseconds.
"""

from collections.abc import Callable

import httpx
import pytest

from docintel.llm.base import LLMError, LLMProvider
from docintel.llm.fake import FakeLLM
from docintel.llm.ollama import OllamaProvider

PROMPT = "State the auditor named in this filing."

type Handler = Callable[[httpx.Request], httpx.Response]


def make_provider(handler: Handler, *, model: str = "test-model") -> OllamaProvider:
    return OllamaProvider(
        httpx.Client(transport=httpx.MockTransport(handler), base_url="http://localhost:11434"),
        model=model,
    )


def ok(payload: object) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return handler


def test_both_implementations_satisfy_the_protocol() -> None:
    """Structural conformance, asserted at runtime as well as by mypy.

    The annotation is what mypy checks. isinstance against a runtime_checkable
    Protocol only checks the members exist, which is weaker, but it catches a
    rename that a test written against the concrete class would sail past.
    """
    fake: LLMProvider = FakeLLM(default="x")
    real: LLMProvider = make_provider(ok({"response": "x"}))

    assert isinstance(fake, LLMProvider)
    assert isinstance(real, LLMProvider)


def test_fake_answers_from_a_cue_and_records_the_call() -> None:
    llm = FakeLLM({"auditor": "PricewaterhouseCoopers LLP"})

    assert llm.complete(PROMPT) == "PricewaterhouseCoopers LLP"
    assert llm.calls == [PROMPT]


def test_fake_is_deterministic_regardless_of_cue_insertion_order() -> None:
    """The same prompt and cues must answer the same way whatever the dict order."""
    forwards = FakeLLM({"auditor": "A", "employees": "B"})
    backwards = FakeLLM({"employees": "B", "auditor": "A"})

    assert forwards.complete(PROMPT) == backwards.complete(PROMPT) == "A"


def test_an_ambiguous_prompt_raises_rather_than_picking_one() -> None:
    """Two matching cues would otherwise resolve by ordering, which is a flaky fake."""
    llm = FakeLLM({"auditor": "A", "filing": "B"})

    with pytest.raises(KeyError, match="matches 2 cues"):
        llm.complete(PROMPT)


def test_an_unconfigured_prompt_raises_keyerror_not_llmerror() -> None:
    """A test that forgot to script a response is a bug, not a model failure."""
    llm = FakeLLM({"something else": "A"})

    with pytest.raises(KeyError):
        llm.complete(PROMPT)


def test_fake_falls_back_to_the_default() -> None:
    llm = FakeLLM({"nothing": "A"}, default="not stated")

    assert llm.complete(PROMPT) == "not stated"


def test_fake_can_simulate_a_model_failure() -> None:
    """This is how the Failed path is reached without a misbehaving real model."""
    llm = FakeLLM(fail_with="GPU fell over")

    with pytest.raises(LLMError, match="GPU fell over"):
        llm.complete(PROMPT)


def test_ollama_returns_the_completion() -> None:
    provider = make_provider(ok({"response": "PricewaterhouseCoopers LLP"}))

    assert provider.complete(PROMPT) == "PricewaterhouseCoopers LLP"
    assert provider.name == "test-model"


def test_ollama_sends_a_pinned_seed_and_non_streaming_request() -> None:
    """Both are load-bearing for reproducibility, so they are asserted, not assumed."""
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"response": "ok"})

    make_provider(handler).complete(PROMPT, temperature=0.0, max_tokens=64)

    body = httpx.Response(200, content=sent[0].content).json()
    assert body["stream"] is False
    assert body["options"]["seed"] == 0
    assert body["options"]["temperature"] == 0.0
    assert body["options"]["num_predict"] == 64


def test_a_transport_failure_becomes_llmerror() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(LLMError, match="ConnectError"):
        make_provider(handler).complete(PROMPT)


def test_a_non_200_becomes_llmerror() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="model not loaded")

    with pytest.raises(LLMError, match="HTTP 500"):
        make_provider(handler).complete(PROMPT)


def test_a_non_json_body_becomes_llmerror() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>proxy error</html>")

    with pytest.raises(LLMError, match="not JSON"):
        make_provider(handler).complete(PROMPT)


def test_a_reply_without_a_response_field_becomes_llmerror() -> None:
    with pytest.raises(LLMError, match="no 'response' string"):
        make_provider(ok({"error": "model not found"})).complete(PROMPT)


def test_an_empty_completion_becomes_llmerror_not_an_answer() -> None:
    """An empty string is one `if` away from being scored as a correct abstention."""
    with pytest.raises(LLMError, match="empty completion"):
        make_provider(ok({"response": "   \n  "})).complete(PROMPT)
