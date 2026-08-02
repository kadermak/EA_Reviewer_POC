"""The finding contract, in a module that depends on nothing.

`Finding` and `normalise` live here rather than in `agents/` because the write
path needs them: `data.repository.insert_findings` runs the guardrail's
deterministic checks itself, so anything those checks import ends up in the
import graph of every database write.

They previously lived in `conformance_agent`, which imports `models.client` —
so `output_review` (and therefore `insert_findings`) transitively pulled the
provider SDK and the network client into the write path. That surfaces under
load, not in review: a slow or failing import in a code path that only ever
needed a substring comparison.

This module must keep importing nothing from `review_agent`. A lint asserts the
guardrails' transitive import closure excludes `review_agent.models`, and that
lint is only meaningful while this stays dependency-free.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

VERDICTS = ("pass", "fail", "unclear")
CONFIDENCE_LEVELS = ("high", "medium", "low")
MAX_REASONING_CHARS = 500

# Verdicts that ASSERT something, and therefore require an explicit human
# decision before a run can complete. `pass` is deliberately absent: see
# PHASE3_DESIGN §4.5. Requiring a click on every pass produces rubber-stamping —
# the reviewer accepts ten low-information items by reflex and brings that reflex
# to the ones that matter. An incorrect pass is caught by nobody whether or not
# they clicked, so mandatory acknowledgment buys nothing against the failure it
# appears to address, and costs attention where attention works.
VERDICTS_REQUIRING_DECISION = ("fail", "unclear")

# Verdicts that ASSERT A VIOLATION and therefore MUST quote the contradicting
# text. Only `fail`. `unclear` means the design is SILENT (a legitimate gap —
# there is nothing to quote), and a `pass` may cite supporting text but is not
# required to. A `fail` with empty evidence is an unsupported assertion, not a
# gap, so it is rejected in validate() the same way a fabricated quote is. Note
# this is DISTINCT from VERDICTS_REQUIRING_DECISION: unclear needs a human
# decision but not evidence.
VERDICTS_REQUIRING_EVIDENCE = ("fail",)
REVIEWER_ACTIONS = ("pending", "accepted", "overridden", "waived")

_WHITESPACE = re.compile(r"\s+")

# `[label](target)` -> `label`. The target is never quotable prose, and a model
# reading the rendered document never sees it.
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
# Inline emphasis and code markers. Removed WITHOUT introducing a space: they sit
# INSIDE a line, so `**Reporting job**` renders as one contiguous token and must
# normalise to one.
_MD_INLINE = re.compile(r"(\*\*|\*|__|_|`|~~)")
# Line-level markers: heading hashes, list bullets, ordered-list numerals,
# blockquote carets, and table pipes. Replaced by a SPACE, not deleted — see the
# note on adjacency below.
_MD_BLOCK = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)]|#{1,6}|>)[ \t]+", re.MULTILINE)
_MD_TABLE = re.compile(r"[|]")


def normalise(text: str) -> str:
    """Whitespace-, case- and MARKUP-insensitive form used for evidence provenance.

    The tolerances are deliberate: a quote differing only in case, spacing or
    Markdown syntax is still text FROM the artifact, so a match still proves
    origin. Foreign content fails however it is cased or formatted. See
    tests/test_output_review.py for the cases that deliberately pass, recorded so
    a later reader does not tighten this into something that rejects legitimate
    quotes.

    WHY MARKUP IS STRIPPED (Phase 4 §2b). The model quotes the document as
    RENDERED; this compared against raw Markdown. Every quote crossing an
    emphasis marker failed provenance, retried, and on org-a exhausted both
    attempts and was rejected whole — scoring 0/4 on defects it had actually
    found. The Phase 3 "retry rate" was this bug, misread as model quality.

    Stripping is applied SYMMETRICALLY to the quote and the artifact, so it
    cannot let foreign content in: it removes marker characters and never adds,
    reorders or deletes prose. Fabricated text fails however it is formatted.

    ADJACENCY IS PRESERVED ON PURPOSE — and it is the TABLE PIPE that needs it,
    not the list marker. Inline markers are deleted (they sit within a line, and
    `**A**B` renders as `AB`). Line-level markers become a space, but that is
    REDUNDANT: `- A\\n- B` keeps its newline, which whitespace-collapse turns
    into a space regardless. Verified by mutation — deleting instead of spacing
    changes nothing there.

    A table row is ONE line, so the pipe is the only separator it has:
    `|cell one|cell two|` deleted yields `cell onecell two`, containing the token
    `onecell`, which appears nowhere in the rendered document. That fabricates
    adjacency, so pipes MUST become a space. The list-marker space stays as
    cheap consistency, not as a control — recorded plainly so nobody later
    "simplifies" the pipe rule believing the two are equivalent.

    Consecutive list items DO become matchable as one span (`- A` `- B` matches
    "A B"). That is accepted: the text is contiguous in reading order and really
    is from this document, and this check exists to catch FABRICATION, not
    reformatting.
    """
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_BLOCK.sub(" ", text)
    text = _MD_TABLE.sub(" ", text)
    text = _MD_INLINE.sub("", text)
    return _WHITESPACE.sub(" ", text).strip().casefold()


@dataclass(frozen=True)
class Finding:
    """One judged rule. `severity` is joined from the rulebook, never modelled.

    Frozen prevents in-place assignment. It does NOT prevent
    `dataclasses.replace(f, verdict=...)` — verified by probe — which is why
    output_review carries a runtime invariant rather than relying on this.
    """

    rule_id: str
    verdict: str
    severity: str
    evidence: str
    confidence: str
    reasoning: str

    def as_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "verdict": self.verdict,
            "severity": self.severity,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }
