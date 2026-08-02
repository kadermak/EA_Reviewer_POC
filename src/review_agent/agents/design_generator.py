"""Generative design agent (Task 1).

Takes a plain-text REQUIREMENTS prompt and drafts an architecture design as
Markdown, using the DEFAULT model. It designs from what the system must DO — not
from the EA rulebook.

The boundary is structural, not a promise in a prompt: this module never imports
the rulebook loader and never reads ``ea_standards.json``. A generator that could
see the rules it will be judged against would learn to satisfy the checker rather
than design well, and would collapse the generate/review separation the
autonomous loop exists to demonstrate. The ONLY channels into this agent are (a)
the operator's requirements and (b) the critic's fail-finding feedback — and (b)
carries only what a ``Finding`` carries (rule_id, severity, evidence, the
reviewer's reasoning), never rule text lifted from the catalogue.

Revision is driven by that feedback. The generator DECIDES what to change;
nothing here maps a finding to a fix. See scripts/generate_and_review.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from review_agent.models import client
from review_agent.models.types import (
    ContentPart,
    Effort,
    ModelCallRecord,
    ModelError,
    ModelRequest,
    ModelRole,
    Reasoning,
    StopReason,
)

PURPOSE = "design.generate"

# General architecture-doc guidance — the TOPICS a competent design covers, not
# the pass criteria of any rulebook. Naming "internet-facing?" or "data
# classification" is first-principles engineering; the specific standards (a WAF,
# TLS 1.2+, two availability zones, an approved secrets manager) are never stated
# here and reach the generator only as the critic's feedback.
DRAFT_SYSTEM = """\
You are a senior solution architect. You are handed a short set of REQUIREMENTS \
for a system and you write a concrete, realistic architecture DESIGN document in \
Markdown that a review board could assess.

Design to satisfy the requirements and sound engineering practice. You are given \
NO compliance checklist and NO standards catalogue — design well from first \
principles.

Structure the document with these Markdown sections, in order:
- `# <system name>`
- `## Overview` — what it does, how critical it is, and whether it handles \
personal data.
- `## Components` — each named component: what it is, whether it is \
internet-facing, how it authenticates, and where it is deployed.
- `## Data flows` — each significant flow and how data is protected while moving.
- `## Data` — data stores, their classification, and how data is protected at \
rest and where it is located.
- `## Operations` — logging and metrics, backup and recovery, and how secrets \
are handled.
- `## Notes` — anything else, including known gaps.

Be concrete and specific: name things, state classifications, and describe \
protections explicitly. Avoid "TBD" where a real decision is expected. Output \
ONLY the Markdown document — no preamble, no explanation of your choices.\
"""


@dataclass(frozen=True)
class GenerationResult:
    """A draft, or a reason none was produced. No partial output is returned."""

    markdown: str | None
    call_record: ModelCallRecord | None = None
    reject_reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.markdown is not None


def build_request(
    requirements: str,
    *,
    feedback: str | None = None,
    prior_design: str | None = None,
    effort: Effort = Effort.MEDIUM,
) -> ModelRequest:
    """Assemble the generation request.

    The requirements are the trusted spec and lead the user turn. On a revision,
    the prior draft and the critic's findings follow as a SECOND turn labelled as
    feedback — our own text, not instructions smuggled from elsewhere.
    """
    parts = [
        ContentPart(
            text="Requirements:\n\n" + requirements.strip(),
            label="requirements",
        )
    ]
    if prior_design and feedback:
        parts.append(
            ContentPart(
                text=(
                    "A conformance reviewer assessed your previous draft against "
                    "standards you cannot see and raised the findings below. "
                    "Produce a REVISED, COMPLETE design in the same Markdown "
                    "format that resolves them. You decide the fixes; keep what "
                    "was already sound and do not remove working parts of the "
                    "design.\n\n"
                    "--- your previous draft ---\n"
                    + prior_design.strip()
                    + "\n\n--- reviewer findings to address ---\n"
                    + feedback.strip()
                ),
                label="revision feedback",
            )
        )

    return ModelRequest(
        system=DRAFT_SYSTEM,
        user_content=parts,
        purpose=PURPOSE,
        schema=None,  # free-text Markdown, never structured output
        reasoning=Reasoning.ADAPTIVE,
        effort=effort,
    )


def generate_design(
    requirements: str,
    *,
    feedback: str | None = None,
    prior_design: str | None = None,
    effort: Effort = Effort.MEDIUM,
) -> GenerationResult:
    """Draft (or revise) an architecture design. Returns Markdown or a reason.

    A provider failure, a refusal, a truncation, or empty text all yield a
    result with ``ok == False`` and a ``reject_reason`` rather than an exception —
    the loop reports it and stops, the same posture the conformance agent takes.
    """
    request = build_request(
        requirements, feedback=feedback, prior_design=prior_design, effort=effort
    )
    try:
        response = client.call(ModelRole.DEFAULT, request)
    except ModelError as exc:
        return GenerationResult(
            None,
            exc.call_record,
            f"the generator's model call failed: {type(exc).__name__}",
        )

    if response.stop_reason is StopReason.REFUSED:
        return GenerationResult(
            None, response.call_record, "the model refused to generate a design"
        )
    if response.stop_reason is StopReason.TRUNCATED:
        return GenerationResult(
            None,
            response.call_record,
            "the draft was truncated at the output limit; no complete design",
        )
    if response.stop_reason is StopReason.UNKNOWN:
        return GenerationResult(
            None,
            response.call_record,
            "the provider returned a stop reason this adapter does not recognise",
        )

    text = (response.text or "").strip()
    if not text:
        return GenerationResult(
            None, response.call_record, "the model returned no text"
        )
    return GenerationResult(text, response.call_record)
