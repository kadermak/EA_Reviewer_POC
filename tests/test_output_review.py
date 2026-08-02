"""Task 11 — output_review.

Two properties dominate this file:

  * BUG-16: no path may alter a verdict, rule_id or severity. A leak is answered
    by BLOCK, never by rewriting the finding.
  * BUG-4: nothing here depends on distinct_markers, and none of it is the
    isolation boundary. The checks are a tripwire on the data layer.

The mutation tests exist because output_review is an isolation-adjacent control
under the standing rule in CLAUDE.md.
"""

import dataclasses
from pathlib import Path

import pytest

from review_agent.findings import Finding
from review_agent.guardrails import output_review as gr
from review_agent.guardrails.output_review import (
    Decision,
    GuardrailAlteredVerdict,
    review_output,
)
from review_agent.rules.loader import load_rulebook

SAMPLE = Path(__file__).resolve().parents[1] / "sample-data"
ARTIFACT = (SAMPLE / "artifact_org-a_proj-a1.md").read_text()


@pytest.fixture
def rulebook():
    return load_rulebook()


@pytest.fixture
def findings(rulebook):
    """A clean set: real quotes, one verdict per rule."""
    quote = "Deployed to a single availability zone"
    return tuple(
        Finding(
            rule_id=rule_id,
            verdict="unclear",
            severity=rulebook.severity_for(rule_id),
            evidence=quote if index == 0 else "",
            confidence="low",
            reasoning="not stated in the design",
        )
        for index, rule_id in enumerate(rulebook.ids)
    )


def signature(findings):
    return [(f.rule_id, f.verdict, f.severity) for f in findings]


# --- the happy path ----------------------------------------------------------

def test_clean_findings_pass_untouched(findings, rulebook):
    result = review_output(findings, ARTIFACT, rulebook)
    assert result.decision is Decision.PASS
    assert result.findings == findings


# --- BUG-16: no path alters a verdict ----------------------------------------

def test_no_outcome_alters_a_verdict(findings, rulebook):
    """Drive a fixed set through EVERY outcome; the signature must be identical."""
    before = signature(findings)

    outcomes = [
        review_output(findings, ARTIFACT, rulebook),                       # pass
        review_output(                                                     # redact
            findings, ARTIFACT, rulebook,
            redactor=lambda p: {"EA-SEC-01": {"evidence": "[redacted]"}},
        ),
    ]
    # block
    leaky = dataclasses.replace(findings[1], evidence="text from another tenant")
    outcomes.append(
        review_output((findings[0], leaky) + findings[2:], ARTIFACT, rulebook)
    )

    for result in outcomes:
        assert signature(result.findings) == before, result.decision
        assert len(result.findings) == len(findings)


def test_redaction_changes_only_text(findings, rulebook):
    result = review_output(
        findings, ARTIFACT, rulebook,
        redactor=lambda p: {"EA-SEC-01": {"evidence": "[redacted]",
                                          "reasoning": "[redacted]"}},
    )
    assert result.decision is Decision.REDACT
    assert result.redacted_rule_ids == ("EA-SEC-01",)
    redacted = {f.rule_id: f for f in result.findings}["EA-SEC-01"]
    assert redacted.evidence == "[redacted]"
    assert redacted.verdict == findings[0].verdict
    assert redacted.severity == findings[0].severity


def test_a_redactor_that_alters_a_verdict_raises(findings, rulebook):
    """The redactor's projection has no verdict field — so this is forced in."""
    def sneaky(projection):
        return {"EA-SEC-01": {"evidence": "ok"}}

    # The projection handed to the redactor must contain text and nothing else.
    captured = {}

    def capture(projection):
        captured.update(projection)
        return {}

    review_output(findings, ARTIFACT, rulebook, redactor=capture)
    assert set(next(iter(captured.values()))) == {"evidence", "reasoning"}, (
        "the redactor was handed a field it could use to change a verdict"
    )


def test_dropping_a_finding_is_caught(findings, rulebook, monkeypatch):
    """The invariant is a SEQUENCE with a length check, not a set of triples.

    Dropping a finding preserves every surviving triple, so a per-finding or
    set-based comparison passes — and dropping is the most attractive alteration
    a guardrail has, because it is how an inconvenient finding disappears.
    """
    monkeypatch.setattr(
        gr, "_apply_redactions", lambda f, r: tuple(f)[:-1]  # silently drop one
    )
    with pytest.raises(GuardrailAlteredVerdict, match="number of findings"):
        review_output(
            findings, ARTIFACT, rulebook,
            redactor=lambda p: {"EA-SEC-01": {"evidence": "x"}},
        )


def test_substituting_a_finding_is_caught(findings, rulebook, monkeypatch):
    """Same length, same triples as a SET — caught only because order is compared."""
    swapped = (findings[1], findings[0]) + findings[2:]
    monkeypatch.setattr(gr, "_apply_redactions", lambda f, r: swapped)
    with pytest.raises(GuardrailAlteredVerdict, match="altered a verdict"):
        review_output(
            findings, ARTIFACT, rulebook,
            redactor=lambda p: {"EA-SEC-01": {"evidence": "x"}},
        )


# --- BUG-4: what the detection actually is -----------------------------------

def test_nothing_depends_on_distinct_markers():
    """The control must not reference a field that exists only in test fixtures.

    AST-based, excluding docstrings — the module's own docstring EXPLAINS that it
    does not depend on distinct_markers, and a raw text search reads that
    explanation as a violation. Third time this repo has hit that false positive
    (the org_id lint and the GUC lint before it); a lint that fails on correct
    prose is a lint someone deletes.
    """
    from tests.test_isolation_redteam import names_in_code, string_constants_in_code

    path = Path(gr.__file__)
    used = names_in_code(path) | {
        token for literal in string_constants_in_code(path) for token in (literal,)
    }
    assert "distinct_markers" not in used
    haystack = " ".join(used)
    for marker in ("SANDPIPER", "BLUEJAY", "AURORA-PAY", "BOREALIS-LOG"):
        assert marker not in haystack


def test_evidence_not_in_the_artifact_blocks(findings, rulebook):
    leaky = dataclasses.replace(
        findings[1], evidence="The Borealis Fleet Tracker uses plaintext HTTP"
    )
    result = review_output((findings[0], leaky) + findings[2:], ARTIFACT, rulebook)
    assert result.decision is Decision.BLOCK
    assert any("not present in the retrieved artifact" in r for r in result.reasons)


def test_foreign_identifier_in_reasoning_blocks(findings, rulebook):
    leaky = dataclasses.replace(findings[1], reasoning="see proj-b1 for context")
    result = review_output((findings[0], leaky) + findings[2:], ARTIFACT, rulebook)
    assert result.decision is Decision.BLOCK
    assert any("proj-b1" in r for r in result.reasons)


def test_fabricated_prose_is_out_of_scope_here_and_closed_by_rls(findings, rulebook):
    """The documented boundary of this module, asserted so it stays visible.

    Prose with no quote and no identifier passes every check here. That gap is
    closed one layer down by RLS — the foreign row was never fetched, so there is
    nothing to paraphrase — and by the SAO reviewer. If this test ever starts
    failing, someone has added a check that claims to close it; verify that claim
    carefully before believing it.
    """
    fabricated = dataclasses.replace(
        findings[1],
        evidence="",
        reasoning="Another business unit runs an unauthenticated freight API.",
    )
    result = review_output((findings[0], fabricated) + findings[2:], ARTIFACT, rulebook)
    assert result.decision is Decision.PASS


# --- mutation tests (standing rule) ------------------------------------------

@pytest.mark.mutation
def test_disabling_evidence_provenance_lets_a_leak_through(
    findings, rulebook, monkeypatch
):
    leaky = dataclasses.replace(findings[1], evidence="text from another tenant")
    payload = (findings[0], leaky) + findings[2:]

    assert review_output(payload, ARTIFACT, rulebook).decision is Decision.BLOCK

    monkeypatch.setattr(gr, "_check_evidence_provenance", lambda f, a: [])
    result = review_output(payload, ARTIFACT, rulebook)
    assert result.decision is Decision.PASS, "the check was not load-bearing"
    assert any(f.evidence == "text from another tenant" for f in result.findings)


@pytest.mark.mutation
def test_disabling_retrieval_containment_lets_a_fake_rule_through(
    findings, rulebook, monkeypatch
):
    fake = dataclasses.replace(findings[1], rule_id="EA-FAKE-99")
    payload = (findings[0], fake) + findings[2:]

    assert review_output(payload, ARTIFACT, rulebook).decision is Decision.BLOCK

    monkeypatch.setattr(gr, "_check_retrieval_containment", lambda f, r: [])
    assert review_output(payload, ARTIFACT, rulebook).decision is Decision.PASS


@pytest.mark.mutation
def test_removing_the_invariant_lets_an_alteration_through(
    findings, rulebook, monkeypatch
):
    """Proves the ASSERTION, not the redactor, is what catches alteration.

    The redactor's projection makes the change inexpressible; this forces it
    anyway, then removes the assertion, and shows the altered verdict reaching
    the caller. That is why the invariant runs in the production path.
    """
    tampered = (dataclasses.replace(findings[0], verdict="pass"),) + findings[1:]
    monkeypatch.setattr(gr, "_apply_redactions", lambda f, r: tampered)

    with pytest.raises(GuardrailAlteredVerdict):
        review_output(findings, ARTIFACT, rulebook,
                      redactor=lambda p: {"EA-SEC-01": {"evidence": "x"}})

    monkeypatch.setattr(gr, "_assert_verdicts_unchanged", lambda b, a: None)
    result = review_output(findings, ARTIFACT, rulebook,
                           redactor=lambda p: {"EA-SEC-01": {"evidence": "x"}})
    assert result.findings[0].verdict == "pass", (
        "expected the altered verdict to reach the caller with the assertion gone"
    )
    assert findings[0].verdict == "unclear"  # the original is untouched


# --- item 3: what _normalise's casefolding actually permits ------------------

def test_normalise_tolerances_do_not_create_a_leak_path(findings, rulebook):
    """_normalise casefolds and collapses whitespace. Is that exploitable?

    CONCLUSION: no, and the reasoning is what matters. Provenance asks "did this
    text come from the artifact". Case and whitespace differences mean the text
    IS in the artifact, just typed differently — so a match still proves origin.
    A leak needs foreign CONTENT to appear; a re-cased quote of the caller's own
    document carries none.

    These pass deliberately. They are recorded so a later reader does not mistake
    them for holes and "fix" provenance into something stricter than it needs to
    be — a stricter match would reject legitimate quotes and train people to
    weaken the check.
    """
    for variant in (
        "DEPLOYED TO A SINGLE AVAILABILITY ZONE",       # case
        "Deployed  to a   single availability zone",     # internal whitespace
        "  Deployed to a single availability zone  ",    # padding
    ):
        payload = (dataclasses.replace(findings[0], evidence=variant),) + findings[1:]
        assert review_output(payload, ARTIFACT, rulebook).decision is Decision.PASS


def test_trivially_short_evidence_passes_and_that_is_acceptable(findings, rulebook):
    """A one-word quote satisfies provenance. Also not a leak path.

    "the" is a substring of almost any document, so it passes — but it carries no
    information, which is the property that matters. Provenance stops foreign
    CONTENT appearing; it was never a proof of relevance. Adding a minimum length
    would reject legitimate short evidence ("no WAF") in exchange for nothing.
    """
    payload = (dataclasses.replace(findings[0], evidence="the"),) + findings[1:]
    assert review_output(payload, ARTIFACT, rulebook).decision is Decision.PASS


def test_foreign_content_still_fails_however_it_is_cased(findings, rulebook):
    """The tolerances above must not extend to text that is genuinely absent."""
    for variant in (
        "the borealis fleet tracker",
        "THE BOREALIS FLEET TRACKER",
        "the   Borealis    Fleet   Tracker",
    ):
        payload = (dataclasses.replace(findings[0], evidence=variant),) + findings[1:]
        assert review_output(payload, ARTIFACT, rulebook).decision is Decision.BLOCK


# --- the write path must not import network code -----------------------------

def _first_party_imports(module: str) -> set[str]:
    import ast
    from pathlib import Path as _Path

    src = _Path(__file__).resolve().parents[1] / "src"
    path = src / (module.replace(".", "/") + ".py")
    if not path.is_file():
        return set()
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("review_agent"):
                found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(
                a.name for a in node.names if a.name.startswith("review_agent")
            )
    return found


def import_closure(module: str) -> set[str]:
    seen, stack = set(), [module]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(_first_party_imports(current))
    return seen


def test_guardrails_never_reach_the_model_layer():
    """TRANSITIVE, not direct — a direct-import check would pass vacuously.

    insert_findings runs the guardrail's checks itself, so everything the
    guardrails import lands in the import graph of every database write. When
    this lint was first proposed, output_review imported Finding from
    conformance_agent, which imports models.client — so the provider SDK and the
    network client were already in the write path, and a direct-import assertion
    would have reported success. The shared contract moved to review_agent.findings,
    which imports nothing.

    This is the kind of coupling that surfaces under load rather than in review:
    a slow or failing import in a path that only ever needed a substring compare.
    """
    for module in ("review_agent.guardrails.output_review",):
        closure = import_closure(module)
        offenders = {m for m in closure if m.startswith("review_agent.models")}
        assert offenders == set(), (
            f"{module} transitively imports {sorted(offenders)}; the write path "
            "must not depend on the model layer"
        )


def test_the_findings_contract_module_depends_on_nothing():
    """The lint above is only meaningful while this stays true."""
    assert _first_party_imports("review_agent.findings") == set()


# --- the Markdown widening, attacked (Phase 4 §2b) ---------------------------
#
# normalise() now strips Markdown so a quote of the RENDERED document matches the
# raw source. That is a deliberate WIDENING of a provenance check, so these tests
# attack it rather than confirm it.

from review_agent.findings import normalise


def test_markup_stripping_lets_a_rendered_quote_match():
    """The defect this widening exists to fix, at its narrowest."""
    artifact = "- **Reporting job** — nightly batch that reads directly from the DB."
    quoted_as_rendered = "Reporting job — nightly batch that reads directly from the DB."
    assert normalise(quoted_as_rendered) in normalise(artifact)


def test_every_markup_form_produces_the_same_match():
    """Fix the CLASS, not the instance.

    Bold was merely the form that happened to appear first. Italics, code spans,
    links, headers and list markers would each produce the identical failure the
    first time a planted defect sat behind one — and backticks are near-certain
    in an architecture document.
    """
    cases = [
        ("*single asterisk emphasis* here", "single asterisk emphasis here"),
        ("_underscore emphasis_ here", "underscore emphasis here"),
        ("`code span` here", "code span here"),
        ("~~struck~~ here", "struck here"),
        ("### Heading text", "Heading text"),
        ("> quoted line", "quoted line"),
        ("1. ordered item", "ordered item"),
        ("See [the standard](https://example.test/x) here", "See the standard here"),
        ("| cell one | cell two |", "cell one cell two"),
    ]
    for raw, rendered in cases:
        assert normalise(rendered) in normalise(raw), f"{raw!r} did not match"


def test_foreign_content_still_fails_however_it_is_formatted(org_markers):
    """THE PROPERTY THE WIDENING MUST NOT COST.

    Stripping removes marker characters; it never adds, reorders or deletes
    prose. So a quote naming another tenant's system fails no matter how it is
    dressed up — which is the whole point of doing it symmetrically.
    """
    artifact = "- **Checkout API** — public REST API behind the enterprise WAF."
    for marker in org_markers["org-b"]:
        for dressed in (marker, f"**{marker}**", f"`{marker}`", f"*{marker}*",
                        f"[{marker}](http://x.test)", f"### {marker}"):
            assert normalise(dressed) not in normalise(artifact), (
                f"foreign marker {dressed!r} became matchable"
            )


def test_inline_markers_do_not_fabricate_adjacency():
    """THE CONCATENATION PROBE.

    Inline markers are DELETED rather than spaced, because `**A**B` renders as
    `AB`. The risk is the mirror image: text separated only by markers becoming
    contiguous. `**A** **B**` keeps its space and must not collapse to `AB`.
    """
    assert normalise("**A** **B**") == "a b"
    assert "ab" not in normalise("**A** **B**")
    # Adjacent with no whitespace SHOULD join — that is how it renders.
    assert normalise("**A**B") == "ab"


def test_a_table_pipe_becomes_a_space_and_must_not_be_deleted():
    """WHERE the adjacency guard actually does its work.

    A first version of this test asserted list markers were the separator. A
    mutation disproved it: `- A\\n- B` keeps its NEWLINE, which whitespace
    collapse turns into a space whether or not the marker was replaced. The
    guard was redundant exactly where it was claimed to matter.

    A table row is ONE line, so the pipe is the only separator it has. Deleting
    it yields the token `onecell`, which appears nowhere in the rendered
    document — fabricated adjacency, the thing this must not do.
    """
    tight = "|cell one|cell two|"
    assert normalise(tight) == "cell one cell two"
    assert "onecell" not in normalise(tight), (
        "deleting the pipe joined two cells into a token that is nowhere in the "
        "rendered document"
    )
    assert normalise("| cell one | cell two |") == "cell one cell two"


def test_line_level_markers_are_separated_by_the_newline_not_the_substitution():
    """Recorded so the redundancy is not mistaken for a control.

    Asserted both ways round: bullets stay separate, and they stay separate for
    a reason that has nothing to do with what the marker is replaced by.
    """
    assert normalise("- Alpha\n- Beta") == "alpha beta"
    assert "alphabeta" not in normalise("- Alpha\n- Beta")
    assert normalise("# Title\nBody") == "title body"


def test_consecutive_bullets_match_as_one_span_deliberately():
    """A DECISION, recorded so it is not mistaken for an oversight.

    A quote spanning two consecutive list items now matches. The text is
    contiguous in reading order and genuinely from this document; this check
    exists to catch FABRICATION, not reformatting. Recorded because it was a
    real rejection before the change, so a later reader will meet it.
    """
    artifact = "- Client to Checkout API: TLS 1.3.\n- Checkout API to Payments: TLS 1.2."
    spanning = "Client to Checkout API: TLS 1.3. Checkout API to Payments: TLS 1.2."
    assert normalise(spanning) in normalise(artifact)
