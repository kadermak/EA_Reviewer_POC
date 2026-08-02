"""Output review — runs AFTER the agent, BEFORE a human sees findings.

WHAT THIS IS NOT
----------------
It is NOT the isolation boundary. Isolation is enforced one layer down by RLS,
where a foreign row is never fetched — so there is nothing for the model to
paraphrase. The checks here are a TRIPWIRE ON THE DATA LAYER. If one ever fires
in production that is a P1 data-layer incident, not a successful defence
(PHASE1_DESIGN.md BUG-4).

It also does NOT depend on `distinct_markers`. Those exist only in
mock_organisations.json, as test fixtures. Real tenant data has no such field, so
a marker scan would catch a test payload and miss a paraphrase of another
tenant's actual architecture. Naming it as the control would claim a guarantee
that does not survive contact with real data.

WHAT IT ACTUALLY CHECKS, AND WHAT EACH CHECK CANNOT CATCH
---------------------------------------------------------
  1. evidence provenance     - every quote is verbatim from THIS run's artifact
                               cannot catch: a finding that quotes nothing
  2. retrieval containment   - every rule_id is in the loaded rulebook
                               cannot catch: content with no identifier in it
  3. reasoning id scan       - org-/proj-/uuid tokens absent from the source
                               cannot catch: PROSE. A paraphrase has no identifier.

A leak fabricated wholesale — no quote, no identifier, prose only — passes all
three and is caught only by the SAO reviewer. That gap is closed by RLS, not here.

THE ONE THING THIS MODULE MAY NEVER DO
--------------------------------------
Alter a verdict, rule_id or severity (CLAUDE.md standing rule, BUG-16). A leak is
answered by BLOCK, never by editing the finding: rewriting would mean a model
deciding which parts of a compromised output are safe to show, at the last point
anything could catch it. See _assert_verdicts_unchanged.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from enum import Enum

from review_agent.findings import Finding, normalise as _normalise
from review_agent.rules.loader import Rulebook

# Identifier shapes that must have come from the retrieved material.
_IDENTIFIER = re.compile(
    r"\b(?:org-[a-z0-9_-]+|proj-[a-z0-9_-]+"
    r"|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b",
    re.IGNORECASE,
)


class Decision(str, Enum):
    """Only outcomes this module actually returns.

    FLAG and REGENERATE were declared here and never returned. An unreachable
    enum member reads as implemented capability — and a declared REGENERATE is
    worse than merely unused, because it argues AGAINST BUG-16 to whoever builds
    the reviewer UI: it suggests the guardrail has a re-run path and therefore
    some licence over the findings. Regeneration is a graph-level concern
    (PHASE3_DESIGN §1.4) and belongs in the graph's vocabulary if it lands, not
    in this one.
    """

    PASS = "pass"
    REDACT = "redact"      # evidence/reasoning TEXT only
    BLOCK = "block"        # nothing reaches the reviewer


class GuardrailAlteredVerdict(RuntimeError):
    """A guardrail changed a verdict, rule_id or severity. Always fatal.

    Raised in the PRODUCTION path, not only under test, because the failure it
    guards is silent and reaches a human as authoritative.
    """


@dataclass(frozen=True)
class OutputReviewResult:
    decision: Decision
    findings: tuple[Finding, ...]
    reasons: tuple[str, ...] = ()
    redacted_rule_ids: tuple[str, ...] = ()

    @property
    def blocked(self) -> bool:
        return self.decision is Decision.BLOCK

    def as_audit_detail(self) -> dict:
        return {
            "decision": self.decision.value,
            "reasons": list(self.reasons),
            "redacted_rule_ids": list(self.redacted_rule_ids),
            "finding_count": len(self.findings),
        }


# --- the invariant -----------------------------------------------------------

def _signature(findings) -> list[tuple[str, str, str]]:
    """The part of a finding set a guardrail may never change.

    A SEQUENCE, not a set. A set of triples misses the most attractive alteration
    available to a guardrail — dropping a finding entirely — because every
    surviving triple is unchanged, and a set can also be defeated by substituting
    one finding for another that shares a triple. rule_id is a stable key here
    because the agent returns exactly one verdict per rule.
    """
    return [(f.rule_id, f.verdict, f.severity) for f in findings]


def _assert_verdicts_unchanged(before, after) -> None:
    if len(before) != len(after):
        raise GuardrailAlteredVerdict(
            f"a guardrail changed the number of findings: {len(before)} -> "
            f"{len(after)}. Dropping a finding is how an inconvenient one "
            "disappears; blocking the review is the permitted response."
        )
    if _signature(before) != _signature(after):
        raise GuardrailAlteredVerdict(
            "a guardrail altered a verdict, rule_id or severity. The permitted "
            "outcomes are block, or redact (text only)."
        )


# --- the checks --------------------------------------------------------------

def _check_evidence_provenance(findings, artifact_text: str) -> list[str]:
    haystack = _normalise(artifact_text)
    problems = []
    for finding in findings:
        if finding.evidence and _normalise(finding.evidence) not in haystack:
            problems.append(
                f"{finding.rule_id}: evidence is not present in the retrieved "
                f"artifact: {finding.evidence[:80]!r}"
            )
    return problems


def _check_retrieval_containment(findings, rulebook: Rulebook) -> list[str]:
    return [
        f"{f.rule_id}: rule is not in rulebook {rulebook.version}"
        for f in findings
        if f.rule_id not in rulebook.by_id
    ]


def _check_reasoning_identifiers(findings, artifact_text: str) -> list[str]:
    """Deliberately the WEAKEST check, and labelled as such.

    It catches foreign identifiers pasted into prose. It cannot catch prose: a
    paraphrase of another tenant's architecture contains no identifier at all and
    passes cleanly. Kept because it is free, not because it closes the gap.
    """
    known = {m.lower() for m in _IDENTIFIER.findall(artifact_text)}
    problems = []
    for finding in findings:
        for token in _IDENTIFIER.findall(finding.reasoning or ""):
            if token.lower() not in known:
                problems.append(
                    f"{finding.rule_id}: reasoning names {token!r}, which does "
                    "not appear in the retrieved artifact"
                )
    return problems


def _apply_redactions(findings, redactions: dict[str, dict]) -> tuple[Finding, ...]:
    """Rebuild findings with redacted TEXT only.

    The redactor never sees a Finding. It is handed, and returns, only
    {rule_id: {evidence, reasoning}} — a projection with no field corresponding
    to a verdict, so the alteration is not expressible in the type it
    manipulates. Note that dataclasses.replace would accept verdict= perfectly
    happily if someone wrote it here: `frozen` prevents in-place assignment, NOT
    replace. That is why _assert_verdicts_unchanged exists.
    """
    rebuilt = []
    for finding in findings:
        patch = redactions.get(finding.rule_id)
        if not patch:
            rebuilt.append(finding)
            continue
        rebuilt.append(
            dataclasses.replace(
                finding,
                evidence=patch.get("evidence", finding.evidence),
                reasoning=patch.get("reasoning", finding.reasoning),
            )
        )
    return tuple(rebuilt)


def deterministic_leak_checks(findings, artifact_text: str, rulebook) -> list[str]:
    """The three deterministic checks, as ONE function with TWO callers.

    Called by review_output (the guardrail) and by data.repository.insert_findings
    (the write path). The second caller is what makes the guardrail
    unskippable: output review lives inside the conformance node for good reasons
    (PHASE3_DESIGN §1.5 minimisation, and "block means nothing reaches the
    reviewer"), but that placement is invisible in the graph, so a later second
    path to findings — a re-review endpoint, a batch job, an API handler — would
    quietly bypass it. Enforcing at the write closes that structurally, the same
    way insert_artifact audits itself rather than trusting its callers.
    """
    return (
        _check_evidence_provenance(findings, artifact_text)
        + _check_retrieval_containment(findings, rulebook)
        + _check_reasoning_identifiers(findings, artifact_text)
    )


def review_output(
    findings,
    artifact_text: str,
    rulebook: Rulebook,
    *,
    redactor=None,
) -> OutputReviewResult:
    """Check drafted findings before a human sees them.

    `redactor(projection) -> {rule_id: {evidence?, reasoning?}}` is optional and
    receives only text. Any deterministic failure BLOCKS: the review is discarded
    whole, exactly as an invalid agent response is (PHASE2_DESIGN §4.5), because
    partial acceptance of a suspect output is a targeted-omission primitive.
    """
    findings = tuple(findings)
    before = list(findings)

    problems = deterministic_leak_checks(findings, artifact_text, rulebook)
    if problems:
        # Blocked, not repaired. The findings are returned unchanged so the
        # invariant below still holds and the audit records what was rejected.
        _assert_verdicts_unchanged(before, findings)
        return OutputReviewResult(
            decision=Decision.BLOCK, findings=findings, reasons=tuple(problems)
        )

    if redactor is not None:
        projection = {
            f.rule_id: {"evidence": f.evidence, "reasoning": f.reasoning}
            for f in findings
        }
        redactions = redactor(projection) or {}
        if redactions:
            redacted = _apply_redactions(findings, redactions)
            _assert_verdicts_unchanged(before, redacted)
            return OutputReviewResult(
                decision=Decision.REDACT,
                findings=redacted,
                reasons=("redacted on guardrail request",),
                redacted_rule_ids=tuple(sorted(redactions)),
            )

    _assert_verdicts_unchanged(before, findings)
    return OutputReviewResult(decision=Decision.PASS, findings=findings)
