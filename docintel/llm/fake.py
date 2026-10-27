"""A provider that answers from a script, so the suite needs no model.

This is what keeps CI honest. Without it the choice is a test suite that
downloads a 7B model on every run, or a test suite that mocks out the very
layer it claims to be testing. `FakeLLM` is the real `LLMProvider` as far as
every caller is concerned; only the source of the text differs.

Determinism here is a stronger claim than "no randomness". The same prompt must
give the same answer on the same inputs regardless of dictionary ordering,
insertion order, or how many other cues happen to be configured. So a prompt
that matches two cues is an error rather than a coin flip resolved by ordering.
A flaky fake would be worse than no fake: it would produce an eval number that
moves between runs for reasons nobody can see.

A missing cue raises `KeyError`, not `LLMError`. They mean opposite things. One
is a test that forgot to configure a response, which should stop the suite and
name the prompt; the other is a model failure, which is a case the extraction
layer is supposed to handle gracefully and which tests deliberately provoke via
`fail_with`.
"""

from collections.abc import Mapping

from docintel.llm.base import DEFAULT_MAX_TOKENS, DETERMINISTIC_TEMPERATURE, LLMError


class FakeLLM:
    """Answers by matching cues against the prompt.

    A cue is matched when it appears anywhere in the prompt, which keeps tests
    readable: a caller asserts on the distinctive part of a generated prompt
    rather than reproducing the whole template and having every test break the
    next time the wording changes.

    `fail_with` makes every call raise `LLMError` with that message, which is
    how the `Failed` path gets exercised without waiting for a real model to
    misbehave.
    """

    def __init__(
        self,
        responses: Mapping[str, str] | None = None,
        *,
        default: str | None = None,
        fail_with: str | None = None,
        name: str = "fake",
    ) -> None:
        self._responses = dict(responses or {})
        self._default = default
        self._fail_with = fail_with
        self._name = name
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = DETERMINISTIC_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        self.calls.append(prompt)

        if self._fail_with is not None:
            raise LLMError(self._fail_with)

        # Every cue is checked, not just the first hit, so the result cannot
        # depend on the order the mapping was built in.
        matched = sorted(cue for cue in self._responses if cue in prompt)
        if len(matched) > 1:
            raise KeyError(
                f"prompt matches {len(matched)} cues ({', '.join(matched)}), so the "
                "answer would depend on ordering; make the cues unambiguous"
            )
        if matched:
            return self._responses[matched[0]]
        if self._default is not None:
            return self._default
        raise KeyError(
            f"FakeLLM has no response configured for this prompt and no default: {prompt[:200]!r}"
        )
