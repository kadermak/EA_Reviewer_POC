"""Deterministic, zero-token test of the generate -> review -> revise LOOP.

This tests the WIRING in scripts/generate_and_review.py::run_loop — not the
generator or the critic, which are exercised elsewhere. Both model-calling
functions are stubbed at the module boundary, so nothing here touches a provider,
a credential, or the network:

  * design_generator.generate_design -> a scripted GenerationResult; records the
    feedback / prior_design it was handed on each call.
  * conformance_agent.review        -> a scripted ReviewResult with canned
    findings.

Two properties are pinned:
  1. A fail finding from iteration 1 reaches the generator as revision input in
     iteration 2 (and iteration 1's design is handed back as prior_design), and a
     clean review in iteration 2 exits the loop.
  2. A loop whose reviews never come back clean stops at the iteration limit
     rather than running forever.

The script is loaded BY PATH (it is not an importable package); loading it runs
its module-level import block but not main(), which is __name__-guarded.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

from review_agent.agents.design_generator import GenerationResult
from review_agent.agents.conformance_agent import ReviewResult
from review_agent.findings import Finding

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "generate_and_review.py"


@pytest.fixture
def gar():
    """The loaded script module (fresh per test)."""
    spec = importlib.util.spec_from_file_location("generate_and_review", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- stubs -------------------------------------------------------------------

class StubGenerator:
    """Stands in for design_generator.generate_design.

    Records the kwargs of every call (that is the whole point — the test asserts
    what the loop fed back), and returns the scripted design for that call.
    """

    def __init__(self, designs):
        self.designs = list(designs)
        self.calls = []

    def __call__(self, requirements, *, feedback=None, prior_design=None, effort=None):
        self.calls.append(
            {
                "requirements": requirements,
                "feedback": feedback,
                "prior_design": prior_design,
            }
        )
        index = min(len(self.calls) - 1, len(self.designs) - 1)
        return GenerationResult(markdown=self.designs[index])


class StubCritic:
    """Stands in for conformance_agent.review. Scripted results, clamped."""

    def __init__(self, results):
        self.results = list(results)
        self.designs_seen = []

    def __call__(self, design, rulebook, **kwargs):
        self.designs_seen.append(design)
        index = min(len(self.designs_seen) - 1, len(self.results) - 1)
        return self.results[index]


def _fail(rule_id="EA-INT-01"):
    return Finding(
        rule_id=rule_id,
        verdict="fail",
        severity="high",
        evidence="direct database connection",
        confidence="high",
        reasoning="uses direct database access instead of a published API",
    )


def _pass(rule_id="EA-SEC-01"):
    return Finding(
        rule_id=rule_id,
        verdict="pass",
        severity="critical",
        evidence="",
        confidence="high",
        reasoning="conforms",
    )


# --- tests -------------------------------------------------------------------

def test_iteration1_feedback_reaches_generator_and_clean_review_exits(gar, monkeypatch):
    """Feedback threads gen1 -> critic1 -> gen2, and a clean critic2 exits."""
    designs = ["# Draft one\n\nnaive design with direct database access",
               "# Draft two\n\nrevised design via a published API"]
    generator = StubGenerator(designs)
    critic = StubCritic(
        [
            # iteration 1: one fail (fed back), plus a pass
            ReviewResult(accepted=True, findings=(_fail("EA-INT-01"), _pass("EA-SEC-01"))),
            # iteration 2: all clean -> converge
            ReviewResult(accepted=True, findings=(_pass("EA-SEC-01"), _pass("EA-INT-01"))),
        ]
    )
    monkeypatch.setattr(gar.design_generator, "generate_design", generator)
    monkeypatch.setattr(gar.conformance_agent, "review", critic)

    rc = gar.run_loop("build a portal", max_iterations=3)

    assert rc == 0, "a clean iteration-2 review should converge and exit 0"
    # Stopped at iteration 2 — did NOT run a third round.
    assert len(generator.calls) == 2
    assert len(critic.designs_seen) == 2

    # Iteration 1 is a cold start: no feedback, no prior design.
    assert generator.calls[0]["feedback"] is None
    assert generator.calls[0]["prior_design"] is None

    # Iteration 2 received the critic's FAIL as revision input — rule id,
    # reasoning and quoted evidence all threaded through _format_feedback.
    feedback = generator.calls[1]["feedback"]
    assert feedback is not None
    assert "EA-INT-01" in feedback
    assert "direct database access instead of a published API" in feedback
    assert "direct database connection" in feedback
    # The passing finding must NOT be fed back (only fails drive revision).
    assert "EA-SEC-01" not in feedback

    # Iteration 1's design was handed back as prior_design...
    assert generator.calls[1]["prior_design"] == designs[0]
    # ...and the critic's second review saw iteration 2's (new) design.
    assert critic.designs_seen[1] == designs[1]


def test_loop_that_never_converges_stops_at_iteration_limit(gar, monkeypatch):
    """A review that never comes back clean stops at the limit, not forever."""
    generator = StubGenerator(["# design that never satisfies the critic"])
    critic = StubCritic([ReviewResult(accepted=True, findings=(_fail("EA-INT-01"),))])
    monkeypatch.setattr(gar.design_generator, "generate_design", generator)
    monkeypatch.setattr(gar.conformance_agent, "review", critic)

    rc = gar.run_loop("build a portal", max_iterations=3)

    assert rc == 1, "an unresolved loop should hit the limit and exit 1"
    # Exactly max_iterations rounds — the loop terminated rather than spinning.
    assert len(generator.calls) == 3
    assert len(critic.designs_seen) == 3
    # Every round after the first got the unresolved fail fed back for revision.
    assert generator.calls[1]["feedback"] is not None
    assert generator.calls[2]["feedback"] is not None
    assert "EA-INT-01" in generator.calls[2]["feedback"]


def test_generation_failure_exits_with_rc2(gar, monkeypatch):
    """If the generator cannot produce a draft, the loop stops before reviewing."""
    calls = []

    def failing_generator(requirements, *, feedback=None, prior_design=None, effort=None):
        calls.append((feedback, prior_design))
        return GenerationResult(
            markdown=None,
            reject_reason="the generator's model call failed: ModelBadRequest",
        )

    critic = StubCritic([ReviewResult(accepted=True, findings=(_pass(),))])
    monkeypatch.setattr(gar.design_generator, "generate_design", failing_generator)
    monkeypatch.setattr(gar.conformance_agent, "review", critic)

    rc = gar.run_loop("build a portal", max_iterations=3)

    assert rc == 2, "a generation failure should exit with rc 2"
    assert len(calls) == 1, "failed on the first draft — no further attempts"
    # The critic is never consulted when there is no design to review.
    assert critic.designs_seen == []


def test_review_not_accepted_exits_with_rc3(gar, monkeypatch):
    """The live credit-exhaustion path: generation succeeds, the critic's own
    model call fails, so review returns accepted=False and the loop stops."""
    generator = StubGenerator(["# a perfectly fine draft"])
    critic = StubCritic(
        [
            ReviewResult(
                accepted=False,
                reject_reason=(
                    "the model call failed and no review was produced: "
                    "ModelBadRequest"
                ),
            )
        ]
    )
    monkeypatch.setattr(gar.design_generator, "generate_design", generator)
    monkeypatch.setattr(gar.conformance_agent, "review", critic)

    rc = gar.run_loop("build a portal", max_iterations=3)

    assert rc == 3, "an unaccepted review should exit with rc 3"
    assert len(generator.calls) == 1, "one draft produced"
    # Reviewed once, then bailed — no feedback loop off an un-produced review.
    assert len(critic.designs_seen) == 1
