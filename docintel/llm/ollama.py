"""The real provider: a local model served by Ollama over loopback.

Ollama rather than llama-cpp-python or a Transformers pipeline because the
model then lives outside this process and outside this repo. The suite never
imports a multi-gigabyte dependency, the venv stays small enough to rebuild in
CI, and swapping `qwen2.5:7b-instruct` for something else is a string change
and a `ollama pull`, not a reinstall.

No API key exists anywhere in this path, by design. The only endpoint is
`http://localhost:11434`, and nothing in this project can be pointed at a paid
hosted model by editing a config file.

The client is injected rather than constructed here, for the same reason the
pacer takes its clock: it makes the failure paths testable with
`httpx.MockTransport` at zero wall-clock cost. Every one of the four ways this
can fail is exercised in the suite with no Ollama installed.

Generation is non-streaming. Streaming would buy a progress indicator for a
caller that has nothing to do with a partial answer, at the cost of parsing a
chunked protocol in the one place that must not misread a response.
"""

import json

import httpx

from docintel.llm.base import DEFAULT_MAX_TOKENS, DETERMINISTIC_TEMPERATURE, LLMError

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b-instruct"

# A 7B model on consumer hardware answers an extraction prompt in seconds, but
# a cold start has to load weights from disk first. Generous enough that a
# first call does not fail spuriously, bounded so a wedged server cannot park
# an eval run over 36 documents indefinitely.
DEFAULT_TIMEOUT = 120.0


class OllamaProvider:
    """Calls Ollama's /api/generate and returns the completion text."""

    def __init__(
        self,
        client: httpx.Client,
        *,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._client = client
        self._model = model
        self._timeout = timeout

    @property
    def name(self) -> str:
        return self._model

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = DETERMINISTIC_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                # Ollama's default seed is random. Pinning it means a rerun of
                # the eval against the same corpus reproduces the same numbers,
                # which is the difference between a measurement and an anecdote.
                "seed": 0,
            },
        }

        try:
            response = self._client.post("/api/generate", json=payload, timeout=self._timeout)
        except httpx.HTTPError as exc:
            # Covers connect errors, timeouts and read errors alike. The class
            # name is included because "connection refused" and "read timeout"
            # lead to completely different fixes on the operator's side.
            raise LLMError(f"{self._model}: request failed ({type(exc).__name__}: {exc})") from exc

        if response.status_code != httpx.codes.OK:
            raise LLMError(
                f"{self._model}: Ollama returned HTTP {response.status_code}: {response.text[:200]}"
            )

        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMError(f"{self._model}: response was not JSON: {response.text[:200]}") from exc

        if not isinstance(body, dict):
            raise LLMError(f"{self._model}: expected a JSON object, got {type(body).__name__}")

        completion = body.get("response")
        if not isinstance(completion, str):
            raise LLMError(f"{self._model}: reply has no 'response' string (keys: {sorted(body)})")

        # An empty or whitespace-only completion is a failure, not an answer.
        # Returning it would hand the extraction layer something it could only
        # mistake for an abstention.
        if not completion.strip():
            raise LLMError(f"{self._model}: model returned an empty completion")

        return completion


def connect(base_url: str = DEFAULT_BASE_URL, *, model: str = DEFAULT_MODEL) -> OllamaProvider:
    """Build a provider against a running Ollama, for scripts and the eval run.

    Tests do not use this: they construct `OllamaProvider` with a client backed
    by `httpx.MockTransport`, which is the whole reason the client is a
    parameter.
    """
    return OllamaProvider(httpx.Client(base_url=base_url), model=model)
