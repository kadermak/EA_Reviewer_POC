"""Provider-neutral types for the model call surface.

Nothing in this module names a provider. Business logic imports from here and
from client.py, and from nowhere else in models/.

Note what is deliberately ABSENT: temperature, top_p, top_k. They are not merely
unused — exposing them would create a knob that is rejected outright by the
judgment model (Opus 4.8), and would encode one provider's sampling model into
the neutral surface. Reasoning depth is expressed as `effort`.

See docs/PHASE2_DESIGN.md §1.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum


class ModelRole(str, Enum):
    """Callers pass a ROLE, never a model id.

    Roles let the model behind a job change without a code change, and make the
    resolved model id an audited fact rather than a hardcoded one.
    """

    DEFAULT = "default"      # everyday agent work
    JUDGMENT = "judgment"    # conformance review, output review
    CLASSIFY = "classify"    # cheap classification (Phase 3)


class Reasoning(str, Enum):
    OFF = "off"
    ADAPTIVE = "adaptive"


class Effort(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class StopReason(str, Enum):
    """Normalised stop reasons. ALWAYS check this before reading content.

    On a refusal the content may be empty; code that reads the first block
    unconditionally breaks.
    """

    COMPLETE = "complete"
    TRUNCATED = "truncated"   # hit the output cap; structured output is unusable
    REFUSED = "refused"       # provider safety refusal
    TOOL_USE = "tool_use"     # not used in Phase 2 (the agent has no tools)
    #: A stop reason the adapter does not recognise — a provider added one we
    #: have never seen. Exists so the adapter can FAIL CLOSED. It previously
    #: defaulted unmapped values to COMPLETE, which meant a new truncation-like
    #: reason would be read as a finished generation and its partial output
    #: validated as a whole review. Never retried: the next call would return
    #: the same unknown reason and spend again for the same outcome.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ContentPart:
    """One piece of a user turn. Text only in Phase 2; images are a Phase 3+ seam."""

    text: str
    label: str | None = None  # e.g. "artifact under review", for prompt assembly


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model MAY call during a generation.

    The model decides IF and WHEN to call it — the caller only offers it. When
    the model emits a tool_use for this spec, the provider runs `handler` with
    the model-supplied input dict and feeds the returned string back as the tool
    result, then lets the model continue. The provider knows only this neutral
    shape (name / description / input_schema) and the callable; it never learns
    what the tool DOES, which keeps business logic out of the provider adapter.

    `handler` MUST be pure and side-effect-free. Its input is model-chosen text
    ultimately extracted from an UNTRUSTED artifact, so the handler must treat it
    strictly as data — a lookup key, never a command, a path, or a query. See the
    conformance agent's approved-technology lookup for the reference handler.
    """

    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict], str]


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def as_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
        }


@dataclass(frozen=True)
class ModelRequest:
    """A provider-neutral request.

    `user_content` is where UNTRUSTED artifact text goes. It never goes in
    `system`, which is a constant assembled by code — that placement is one of
    the four structural controls that make prompt injection harmless
    (design §3.4), and it is not negotiable per-call.
    """

    system: str
    user_content: list[ContentPart]
    purpose: str                       # audit label, e.g. "conformance.review"
    schema: dict | None = None         # JSON Schema; None = free text
    max_output_tokens: int = 16000
    reasoning: Reasoning = Reasoning.ADAPTIVE
    effort: Effort = Effort.HIGH
    #: Tools the model MAY call during this generation. Empty = no tools, the
    #: default, so existing callers and Phase 2's tool-free contract are unchanged.
    tools: tuple[ToolSpec, ...] = ()


#: `stop_reason` on a record for a call that never produced a response. Not a
#: member of StopReason, deliberately: StopReason describes how a generation
#: ENDED, and this describes a generation that never began. Code branching on a
#: StopReason must not be able to receive this by accident.
STOP_REASON_ERRORED = "errored"


@dataclass(frozen=True)
class ModelCallRecord:
    """What the caller persists to the audit log.

    models/ does NOT write this itself — it has no data-layer import and
    therefore no CallerScope and no org_id. The caller persists it inside its
    own scoped session. See design §1.6.
    """

    purpose: str
    role: str
    model_id: str
    stop_reason: str
    usage: dict
    prompt_sha256: str
    request_id: str | None = None
    #: The provider's own stop-reason string, unmapped. `stop_reason` above
    #: is OUR interpretation of it; this is what actually arrived.
    provider_stop_reason: str | None = None
    #: Set ONLY on a record for a call that raised. Its presence is what
    #: distinguishes an attempt from a completed call: `usage` is zeroed and
    #: there is no response to describe. A failed attempt is still an audited
    #: fact — the request left our infrastructure, and on some failures (a
    #: timeout mid-stream) it may even have been billed.
    error: str | None = None
    #: The model's extended-thinking / reasoning trace for THIS call, captured by
    #: the provider adapter. Deliberately ABSENT from as_dict(): it is carried in
    #: memory so the caller can persist it ONCE on the review row, and keeping it
    #: out of the audit-log serialisation avoids duplicating a large, unvalidated
    #: blob into every model.call entry. None for providers with no separate
    #: reasoning channel (e.g. the Ollama adapter).
    thinking: str | None = None

    def as_dict(self) -> dict:
        return {
            "purpose": self.purpose,
            "role": self.role,
            "model_id": self.model_id,
            "stop_reason": self.stop_reason,
            "usage": self.usage,
            "prompt_sha256": self.prompt_sha256,
            "request_id": self.request_id,
            "provider_stop_reason": self.provider_stop_reason,
            "error": self.error,
        }


@dataclass(frozen=True)
class ModelResponse:
    text: str | None
    structured: dict | None
    stop_reason: StopReason
    model_id: str
    usage: Usage
    call_record: ModelCallRecord
    request_id: str | None = None
    # Quarantined. Reading this outside the provider adapter defeats the whole
    # abstraction; a lint in the conformance tests forbids it.
    raw: object = field(default=None, repr=False, compare=False)


# --- neutral errors ----------------------------------------------------------
# Business logic catches these. It must never catch a provider's exception type.

class ModelError(Exception):
    """Base for every provider failure, after translation.

    Carries `call_record`: the audit record for the ATTEMPT, populated by
    client.call before the exception is re-raised. Everything in it is known
    before the request is sent (role, resolved model id, prompt fingerprint), so
    it does not depend on a response arriving — which is the point, because none
    did. A caller that handles a ModelError therefore always has something to
    audit, and cannot leave a spend or a data egress unrecorded because the call
    failed. See PHASE3_DESIGN.md §3c.
    """

    call_record: "ModelCallRecord | None" = None


class ModelRateLimited(ModelError):
    """429 / quota. The SDK has already retried with backoff before this raises."""


class ModelUnavailable(ModelError):
    """5xx / overloaded."""


class ModelBadRequest(ModelError):
    """4xx that will not succeed on retry — malformed request, bad schema."""


class ModelAuthError(ModelError):
    """Missing or rejected credentials."""


class ModelTransportError(ModelError):
    """Network failure before any response."""
