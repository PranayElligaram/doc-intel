"""The one seam between this project and a language model.

Everything that calls a model goes through `LLMProvider`. The point is not
abstraction for its own sake: it is that the test suite must run in CI with no
model, no GPU and no network, while the measured numbers must come from a real
model on a real machine. One protocol with two implementations is the smallest
thing that gives both.

It is a Protocol rather than a base class because a provider is defined by what
it does, not by what it inherits. Nothing here needs to import a framework or
subclass anything, and a test double does not have to be related to the real
client to be accepted in its place.

The interface is synchronous. A local 7B model answers one prompt at a time on
one GPU, so there is no concurrency to win, and async would put a colour on
every caller all the way up into the eval loop for no measured benefit. If a
future provider is remote and batched, that is when this changes, and the
change is contained here.

`name` is not decoration. Every number in the README has to say which model
produced it, and a run that cannot name its model cannot be reproduced or
defended. The eval record takes this string.
"""

from typing import Protocol, runtime_checkable

DEFAULT_MAX_TOKENS = 512

# Extraction wants the same prompt to give the same answer on every run.
# Anything above zero makes a measured F1 a sample from a distribution rather
# than a number, and re-running the eval would quietly move it.
DETERMINISTIC_TEMPERATURE = 0.0


class LLMError(RuntimeError):
    """The model did not produce a usable answer.

    Raised for every failure on the provider side: the host is down, the
    request timed out, the response was not valid JSON, the model returned an
    empty completion.

    This is the exception the extraction layer turns into `Failed`, and that is
    the whole reason it exists as one named type. `Failed` and `NotStated` are
    kept apart because correct abstention is a headline metric, so the path
    that produces `Failed` must be impossible to confuse with a model that
    answered "not stated". A bare `Exception` escaping a provider would either
    crash the run or, worse, get caught somewhere generic and scored as an
    abstention.
    """


@runtime_checkable
class LLMProvider(Protocol):
    """A thing that turns a prompt into text.

    Deliberately narrow. No chat roles, no tool calling, no streaming, no
    embeddings. The extraction prompt is a single instruction over a single
    retrieved span, and every capability added here is one more thing the fake
    has to imitate faithfully or the offline suite stops telling the truth.
    """

    @property
    def name(self) -> str:
        """Identifies the model well enough to reproduce a run, e.g. the tag."""
        ...

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = DETERMINISTIC_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """Return the completion, or raise `LLMError`.

        Never returns an empty string: an empty completion is a failure, not an
        answer, and a caller that had to check for it would be one `if` away
        from scoring it as an abstention.
        """
        ...
