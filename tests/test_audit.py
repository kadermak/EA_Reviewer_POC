"""The append-only audit trail is a CLAUDE.md non-negotiable — so prove it runs.

This file exists because of a gap the provider lint exposed: audit/log.py was
syntactically invalid from Phase 1 until Phase 2 and nothing noticed, because
nothing imported it. 21 green tests coexisted with an audit module that could not
even be parsed. A trail that has never been exercised is not a trail.

Two things are asserted here:

  * COVERAGE — every operation in REQUIRED_AUDITED_OPERATIONS actually writes.
  * ATOMICITY — an operation that cannot write its entry does not happen. This
    is the property that stops the trail degrading silently: there is no state
    where the artifact landed but the record did not.
"""

import json
from pathlib import Path

import pytest
from sqlalchemy import text

from review_agent.agents import conformance_agent as agent
from review_agent.audit import actions
from review_agent.data.db import get_engine, get_owner_engine, scoped_session
from tests.conftest import make_run
from review_agent.data.repository import (
    insert_artifact,
    insert_findings,
    record_model_call,
    record_review_rejected,
)
from review_agent.data.scope import ScopeResolutionError, resolve_scope_for_subject
from review_agent.models import client
from review_agent.models.types import ModelCallRecord, ModelResponse, StopReason, Usage
from review_agent.rules.loader import load_rulebook

SAMPLE = Path(__file__).resolve().parents[1] / "sample-data"


def audit_actions(scope, action=None) -> list[dict]:
    """Read the audit trail back as the caller would see it (RLS applies)."""
    sql = "SELECT action, project_id, user_id, retrieved_ids, detail FROM audit_log"
    params = {}
    if action:
        sql += " WHERE action = :action"
        params["action"] = action
    with scoped_session(scope) as session:
        return [dict(r) for r in session.execute(text(sql), params).mappings()]


@pytest.fixture
def rulebook():
    return load_rulebook()


@pytest.fixture
def stub_model():
    class _Stub:
        name = "stub"

        def complete(self, model_id, request, prompt_sha256, role):
            payload = {
                "findings": [
                    {
                        "rule_id": rid,
                        "verdict": "unclear",
                        "evidence": "",
                        "confidence": "low",
                        "reasoning": "not stated",
                    }
                    for rid in load_rulebook().ids
                ]
            }
            return ModelResponse(
                text=json.dumps(payload),
                structured=payload,
                stop_reason=StopReason.COMPLETE,
                model_id=model_id,
                usage=Usage(input_tokens=10, output_tokens=20),
                call_record=ModelCallRecord(
                    purpose=request.purpose,
                    role=role,
                    model_id=model_id,
                    stop_reason="complete",
                    usage=Usage(input_tokens=10, output_tokens=20).as_dict(),
                    prompt_sha256=prompt_sha256,
                ),
                raw=object(),
            )

    client.set_provider(_Stub())
    yield
    client.set_provider(None)


# --- coverage: every required operation writes -------------------------------

def test_artifact_upload_is_audited(scope_a):
    """Ingestion. The entry is written by insert_artifact itself, not by a caller."""
    with scoped_session(scope_a) as session:
        artifact = insert_artifact(
            session, scope_a, project_id="proj-a1", filename="x.md", content="hello"
        )
        artifact_id = str(artifact.artifact_id)

    entries = audit_actions(scope_a, actions.ARTIFACT_UPLOAD)
    mine = [e for e in entries if artifact_id in (e["retrieved_ids"] or {}).get("artifacts", [])]
    assert len(mine) == 1
    assert mine[0]["project_id"] == "proj-a1"
    assert mine[0]["user_id"] == scope_a.user_id
    assert mine[0]["detail"]["filename"] == "x.md"


def test_scope_denial_is_audited(seeded_db):
    """A refused caller leaves a trace, recorded under the unscoped sentinel.

    This is the entry most likely to be missing, because the request failed and
    there is no tenant row afterwards to hint that anything happened.
    """
    with pytest.raises(ScopeResolutionError):
        resolve_scope_for_subject("intruder@nowhere")

    # Read as the sentinel org: these rows belong to no tenant.
    from review_agent.data.scope import CallerScope

    sentinel = CallerScope(user_id="audit-reader", org_id=actions.ORG_UNSCOPED)
    entries = audit_actions(sentinel, actions.SCOPE_DENIED)
    assert any(e["detail"]["subject"] == "intruder@nowhere" for e in entries)

    # And it is NOT visible to a tenant — a denial for an unknown subject is not
    # any org's business.
    from review_agent.data.scope import resolve_scope_for_subject as resolve

    org_a = resolve("user-a@org-a")
    assert audit_actions(org_a, actions.SCOPE_DENIED) == []


def test_model_call_is_audited(scope_a):
    """Which model ran, what it cost, and the hash that makes it reproducible."""
    record = ModelCallRecord(
        purpose="conformance.review",
        role="judgment",
        model_id="claude-opus-4-8",
        stop_reason="complete",
        usage=Usage(input_tokens=5, output_tokens=7).as_dict(),
        prompt_sha256="deadbeef",
    )
    record_model_call(scope_a, record, project_id="proj-a1")

    entries = audit_actions(scope_a, actions.MODEL_CALL)
    assert len(entries) == 1
    assert entries[0]["detail"]["model_id"] == "claude-opus-4-8"
    assert entries[0]["detail"]["prompt_sha256"] == "deadbeef"
    assert entries[0]["detail"]["usage"]["output_tokens"] == 7


def test_completed_review_is_audited(scope_a, rulebook, stub_model):
    """What was retrieved and what was decided, in one entry."""
    text_content = (SAMPLE / "artifact_org-a_proj-a1.md").read_text()
    result = agent.review(text_content, rulebook)
    assert result.accepted

    with scoped_session(scope_a) as session:
        artifact = insert_artifact(
            session, scope_a, project_id="proj-a1", filename="r.md",
            content=text_content,
        )
        run_id = make_run(session, scope_a, artifact)
        insert_findings(session, scope_a, artifact, result, run_id=run_id)
        artifact_id = str(artifact.artifact_id)  # capture before the session closes

    entries = audit_actions(scope_a, actions.REVIEW_COMPLETED)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["retrieved_ids"]["artifacts"] == [artifact_id]
    assert len(entry["retrieved_ids"]["findings"]) == len(rulebook.rules)
    assert entry["detail"]["rulebook_sha256"] == rulebook.sha256
    assert entry["detail"]["verdicts"]["unclear"] == len(rulebook.rules)
    assert entry["detail"]["model_calls"], "the model call must be recorded too"


def test_rejected_review_is_audited(scope_a, rulebook):
    """The case most worth auditing: a review that produced NOTHING.

    No findings row exists afterwards, so without this entry the attempt is
    indistinguishable from a review that was never run.
    """
    rejected = agent.ReviewResult(
        accepted=False,
        validation_errors=("EA-SEC-01: evidence does not appear in the artifact",),
        reject_reason="output failed validation after 2 attempts",
        rulebook_version=rulebook.version,
        rulebook_sha256=rulebook.sha256,
    )
    with scoped_session(scope_a) as session:
        artifact = insert_artifact(
            session, scope_a, project_id="proj-a1", filename="r.md", content="x"
        )
        artifact_id = artifact.artifact_id
    record_review_rejected(scope_a, artifact_id, "proj-a1", rejected)

    entries = audit_actions(scope_a, actions.REVIEW_REJECTED)
    assert len(entries) == 1
    assert "failed validation" in entries[0]["detail"]["reject_reason"]
    assert entries[0]["detail"]["validation_errors"]


def test_finding_decision_is_audited(scope_a, rulebook, stub_model):
    """The SAO's ruling on a finding. Written inside the decision's transaction.

    finding.decide records a STATE CHANGE (the reviewer_action write), so its
    entry shares that transaction — if the decision rolls back, so does its
    record. It was audited in code from Phase 3 but sat OUTSIDE
    REQUIRED_AUDITED_OPERATIONS until this test was added; the enumerated
    contract is only real if every member is exercised.
    """
    from review_agent.data.repository import apply_reviewer_decisions

    text_content = (SAMPLE / "artifact_org-a_proj-a1.md").read_text()
    result = agent.review(text_content, rulebook)
    assert result.accepted

    with scoped_session(scope_a) as session:
        artifact = insert_artifact(
            session, scope_a, project_id="proj-a1", filename="d.md",
            content=text_content,
        )
        run_id = make_run(session, scope_a, artifact)
        rows = insert_findings(session, scope_a, artifact, result, run_id=run_id)
        target = str(rows[0].finding_id)

    with scoped_session(scope_a) as session:
        apply_reviewer_decisions(
            session, scope_a, run_id, {target: "accepted"}
        )

    entries = audit_actions(scope_a, actions.FINDING_DECIDED)
    assert len(entries) == 1
    assert entries[0]["detail"]["decisions"] == {target: "accepted"}
    assert entries[0]["detail"]["run_id"] == run_id
    assert target in entries[0]["retrieved_ids"]["findings"]


def test_review_sent_is_audited(scope_a):
    """A submitter sending a draft to the SAO. A STATE CHANGE, audited in-txn.

    draft -> awaiting_review must leave a trace: it is the moment a submission
    becomes the architect's business. Recorded inside the status write's
    transaction, so it cannot land without the record.
    """
    from review_agent.data.models import ReviewRun
    from review_agent.data.repository import mark_run_sent

    with scoped_session(scope_a) as session:
        artifact = insert_artifact(
            session, scope_a, project_id="proj-a1", filename="s.md", content="x"
        )
        session.add(
            ReviewRun(
                run_id="draft-send-1", org_id=scope_a.org_id, project_id="proj-a1",
                artifact_id=artifact.artifact_id, status="draft",
            )
        )
        session.flush()

    with scoped_session(scope_a) as session:
        mark_run_sent(session, scope_a, "draft-send-1")

    entries = audit_actions(scope_a, actions.REVIEW_SENT)
    assert len(entries) == 1
    assert entries[0]["detail"]["run_id"] == "draft-send-1"
    assert entries[0]["detail"]["to"] == "awaiting_review"


def test_every_required_operation_has_a_test(seeded_db):
    """The enumerated contract, asserted as a set.

    Adding an action to REQUIRED_AUDITED_OPERATIONS without wiring and testing it
    fails here, so the table cannot drift ahead of the implementation.
    """
    tested = {
        actions.SCOPE_DENIED,
        actions.ARTIFACT_UPLOAD,
        actions.MODEL_CALL,
        actions.REVIEW_COMPLETED,
        actions.REVIEW_REJECTED,
        actions.FINDING_DECIDED,
        actions.REVIEW_SENT,
        actions.GUARDRAIL_OUTPUT,
        actions.GUARDRAIL_INPUT,
    }
    assert set(actions.REQUIRED_AUDITED_OPERATIONS) == tested


def test_records_of_discarded_work_survive_a_rollback(scope_a, rulebook):
    """review.rejected and model.call must OUTLIVE the transaction they died in.

    This is the stated exception to atomicity, and it is the half that matters
    under attack. A rejected review rolls its transaction back — the artifact
    goes away with it. If the rejection record went too, inducing rejections
    would give an attacker unlimited un-logged attempts, and the trail would show
    nothing at all.

    A rollback also cannot un-spend the tokens or un-send the content to the
    provider, so the model.call record describes something that really happened
    and must not be undone with the state change.
    """
    call = ModelCallRecord(
        purpose="conformance.review",
        role="judgment",
        model_id="claude-opus-4-8",
        stop_reason="complete",
        usage=Usage(input_tokens=99, output_tokens=1).as_dict(),
        prompt_sha256="rolledback",
    )
    rejected = agent.ReviewResult(
        accepted=False,
        validation_errors=("EA-SEC-01: fabricated evidence",),
        reject_reason="output failed validation after 2 attempts",
        call_records=(call,),
        rulebook_version=rulebook.version,
        rulebook_sha256=rulebook.sha256,
    )

    # A review whose surrounding transaction is thrown away entirely.
    with pytest.raises(RuntimeError):
        with scoped_session(scope_a) as session:
            artifact = insert_artifact(
                session, scope_a, project_id="proj-a1",
                filename="doomed-review.md", content="x",
            )
            artifact_id = artifact.artifact_id
            record_model_call(scope_a, call, project_id="proj-a1")
            record_review_rejected(scope_a, artifact_id, "proj-a1", rejected)
            raise RuntimeError("the surrounding work is discarded")

    # The artifact is gone — the state change rolled back, correctly.
    with scoped_session(scope_a) as session:
        assert session.execute(
            text("SELECT count(*) FROM artifacts WHERE filename = 'doomed-review.md'")
        ).scalar() == 0

    # The records of what happened are NOT gone.
    rejections = audit_actions(scope_a, actions.REVIEW_REJECTED)
    assert any("failed validation" in e["detail"]["reject_reason"] for e in rejections)

    calls = audit_actions(scope_a, actions.MODEL_CALL)
    assert any(e["detail"]["prompt_sha256"] == "rolledback" for e in calls)

    # They reference an artifact that no longer exists. That is by design:
    # audit_log carries no FK on its subjects, because the record outlives them.
    referenced = [
        e for e in rejections
        if str(artifact_id) in (e["retrieved_ids"] or {}).get("artifacts", [])
    ]
    assert referenced, "the rejection must still name what it was reviewing"


def test_guardrail_output_is_audited(scope_a, rulebook):
    """What output review decided, including a BLOCK that leaves no findings."""
    from review_agent.guardrails.output_review import Decision, OutputReviewResult
    from review_agent.data.repository import record_guardrail_output

    blocked = OutputReviewResult(
        decision=Decision.BLOCK,
        findings=(),
        reasons=("EA-SEC-01: evidence is not present in the retrieved artifact",),
    )
    with scoped_session(scope_a) as session:
        artifact = insert_artifact(
            session, scope_a, project_id="proj-a1", filename="g.md", content="x"
        )
        artifact_id = artifact.artifact_id
    record_guardrail_output(scope_a, "proj-a1", artifact_id, blocked)

    entries = audit_actions(scope_a, actions.GUARDRAIL_OUTPUT)
    assert len(entries) == 1
    assert entries[0]["detail"]["decision"] == "block"
    assert entries[0]["detail"]["reasons"]


def test_guardrail_input_is_audited(scope_a):
    """A refused request stops the run before anything exists — this is its only trace."""
    from review_agent.guardrails.input_guard import Decision, InputGuardResult
    from review_agent.data.repository import record_guardrail_input

    blocked = InputGuardResult(
        decision=Decision.BLOCK,
        reasons=("project 'proj-b1' is not visible to this caller",),
    )
    record_guardrail_input(scope_a, "proj-b1", blocked)

    entries = audit_actions(scope_a, actions.GUARDRAIL_INPUT)
    assert len(entries) == 1
    assert entries[0]["detail"]["decision"] == "block"
    assert entries[0]["project_id"] == "proj-b1"


# --- mutation: an audit trail that stops recording must not stay quiet --------

@pytest.mark.mutation
def test_audit_write_failure_aborts_the_operation(owner_engine, scope_a):
    """Break the audit write; assert the OPERATION fails and nothing persists.

    An audit trail that silently stops recording is indistinguishable from one
    that works — so the design does not rely on noticing. The audit INSERT shares
    a transaction with the operation it records, which means losing the ability
    to write the entry loses the ability to perform the operation.

    Asserts the real consequence, not a flag: with INSERT revoked, the artifact
    must be absent afterwards. If this ever passed while the artifact landed, the
    trail would have a hole exactly where the interesting events are.
    """
    before = len(audit_actions(scope_a, actions.ARTIFACT_UPLOAD))

    try:
        with owner_engine.begin() as conn:
            conn.execute(text('REVOKE INSERT ON audit_log FROM "review_app"'))

        with pytest.raises(Exception) as excinfo:
            with scoped_session(scope_a) as session:
                insert_artifact(
                    session, scope_a, project_id="proj-a1",
                    filename="unaudited.md", content="this must not survive",
                )
        assert "audit_log" in str(excinfo.value).lower()

        # The consequence: the artifact did NOT land.
        with scoped_session(scope_a) as session:
            orphans = session.execute(
                text("SELECT count(*) FROM artifacts WHERE filename = 'unaudited.md'")
            ).scalar()
        assert orphans == 0, "an artifact was stored with no audit entry"
    finally:
        with owner_engine.begin() as conn:
            conn.execute(text('GRANT INSERT ON audit_log TO "review_app"'))

    # Restored: the operation works again and the trail grew by exactly one.
    with scoped_session(scope_a) as session:
        insert_artifact(
            session, scope_a, project_id="proj-a1", filename="ok.md", content="fine"
        )
    assert len(audit_actions(scope_a, actions.ARTIFACT_UPLOAD)) == before + 1


@pytest.mark.mutation
def test_audit_coverage_assertions_are_not_vacuous(monkeypatch, scope_a):
    """Neuter the audit write; assert the coverage check notices.

    The coverage tests above would pass just as happily against a trail that
    records nothing, if the seeded data already contained matching rows. This
    proves they are reading what this operation wrote.
    """
    import review_agent.data.repository as repo

    def _silent(*args, **kwargs):
        return None  # the silent-degradation failure mode, simulated

    monkeypatch.setattr(repo, "record_audit", _silent)

    with scoped_session(scope_a) as session:
        artifact = insert_artifact(
            session, scope_a, project_id="proj-a1", filename="quiet.md", content="q"
        )
        artifact_id = str(artifact.artifact_id)

    entries = audit_actions(scope_a, actions.ARTIFACT_UPLOAD)
    assert not any(
        artifact_id in (e["retrieved_ids"] or {}).get("artifacts", []) for e in entries
    ), "the coverage assertion cannot distinguish a written entry from a missing one"
    # monkeypatch restores record_audit at teardown.


def test_audit_log_survives_a_rolled_back_operation(scope_a):
    """Atomicity cuts both ways: a failed operation leaves no audit entry either.

    An entry for an operation that did not happen is as bad as a missing entry
    for one that did — it would make the trail describe a system state that never
    existed.
    """
    before = len(audit_actions(scope_a, actions.ARTIFACT_UPLOAD))

    with pytest.raises(Exception):
        with scoped_session(scope_a) as session:
            insert_artifact(
                session, scope_a, project_id="proj-b1",  # not visible to org-a
                filename="doomed.md", content="x",
            )

    assert len(audit_actions(scope_a, actions.ARTIFACT_UPLOAD)) == before
