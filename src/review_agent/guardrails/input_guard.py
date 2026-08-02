"""Input guardrail — runs BEFORE retrieval.

THE PRIMARY CONTROL IS NOT HERE.
--------------------------------
"An out-of-domain request is declined" is guaranteed by the API surface: there is
no free-text endpoint, so "write me some code" has nowhere to be typed. That is
the strongest form of decline — it needs no classifier and no judgement, so it
cannot be talked around. `test_out_of_scope_request_declined` asserts the route
shape and stays exactly as it is; everything here is added ALONGSIDE it, never
instead of it.

Nor is scope enforced here. RLS decides what the caller can see, one layer down.
The scope check below re-states that decision EARLY so a wrong project fails
legibly at the front door rather than as a puzzling empty result three nodes
later — a usability property, not a security one.

ORDER: deterministic first, model last
--------------------------------------
  1. scope    - is this project visible to the caller? (RLS is the authority)
  2. format   - is the artifact something we can actually review?
  3. flags    - what did the sanitiser notice? REPORTED, never acted on
  4. domain   - in-domain classification via the cheap model (Phase 3, optional)

Step 4 is the weakest and comes last on purpose. A classifier is a model, so its
refusal rate is a distribution rather than a guarantee; it is a filter over
content that is already inside a typed, scoped envelope.

Like output_review, this module may BLOCK. It may not rewrite anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from review_agent.ingestion.extract import SUPPORTED_SUFFIXES


class Decision(str, Enum):
    """Only outcomes this module actually returns.

    No FLAG member: sanitiser findings travel on `flags` alongside a PASS,
    because a suspicious phrase is information for the reviewer, not a verdict on
    the request. A separate FLAG decision would invite callers to branch on it
    and quietly become a second gate.
    """

    PASS = "pass"
    BLOCK = "block"


@dataclass(frozen=True)
class InputGuardResult:
    decision: Decision
    reasons: tuple[str, ...] = ()
    flags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def blocked(self) -> bool:
        return self.decision is Decision.BLOCK

    def as_audit_detail(self) -> dict:
        return {
            "decision": self.decision.value,
            "reasons": list(self.reasons),
            "flags": list(self.flags),
        }


def _check_project_visible(
    project_id: str, visible_project_ids: frozenset[str]
) -> list[str]:
    """RLS would return nothing anyway; this turns a confusing empty review into
    a clear refusal. A named function so it can be mutated on its own."""
    if project_id in visible_project_ids:
        return []
    return [f"project {project_id!r} is not visible to this caller"]


def _check_supported_format(filename: str) -> list[str]:
    """Can this artifact actually be extracted? Named for the same reason."""
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix in SUPPORTED_SUFFIXES:
        return []
    return [
        f"unsupported artifact format {suffix or filename!r}; "
        f"expected one of {sorted(SUPPORTED_SUFFIXES)}"
    ]


def check(
    *,
    project_id: str,
    filename: str,
    visible_project_ids: frozenset[str],
    suspicious_spans: tuple[str, ...] = (),
) -> InputGuardResult:
    """Screen a review request before anything is retrieved.

    Takes `visible_project_ids` — what RLS already returned for this caller —
    rather than a scope or an org id. The guard therefore cannot be handed a
    tenant to check against; it can only compare a request to what the database
    already said was visible. That keeps the BUG-2 lint (no tenant identifier in
    guardrails/) satisfiable without an exception, and means a bug here can
    narrow access but never widen it.
    """
    reasons = (
        _check_project_visible(project_id, visible_project_ids)
        + _check_supported_format(filename)
    )
    if reasons:
        return InputGuardResult(
            decision=Decision.BLOCK, reasons=tuple(reasons), flags=suspicious_spans
        )

    # Injection flags are REPORTED, not acted on. An artifact containing "ignore
    # all previous instructions" is a fact the SAO reviewer should see; blocking
    # on it would let anyone deny review of a document by pasting that phrase
    # into it, and would teach us to trust the sanitiser as a gate (BUG-10 — it
    # is a tripwire, and isolation holds with it absent entirely).
    return InputGuardResult(decision=Decision.PASS, flags=suspicious_spans)
