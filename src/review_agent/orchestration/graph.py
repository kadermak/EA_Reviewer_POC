"""LangGraph graph wiring the review flow.

Flow: scoped_retrieval -> conformance_agent -> human_review (INTERRUPT)
      -> finalize -> END
Guardrail nodes (input_guard, output_review) are inserted by tasks 10 and 11.

THREE THINGS THIS MODULE IS RESPONSIBLE FOR GETTING RIGHT
---------------------------------------------------------
1. **The checkpointer rides OUR scoped connection.** PostgresSaver is handed the
   psycopg connection that the data layer has already scoped. A
   pool with a `configure` hook was tried and REJECTED: psycopg_pool's configure
   is a connection-CREATION hook, not a per-checkout hook, so a checkout
   intending org-b silently ran with org-a's GUC still set. See
   PHASE3_DESIGN.md §1.3 / §9 Finding 2.

2. **No transaction is held across human review.** A run executes in bounded
   SEGMENTS: start -> interrupt commits and closes; resume -> end commits and
   closes. A days-long `idle in transaction` connection holds locks, blocks
   VACUUM and exhausts the pool — it fails slowly and invisibly, which is the
   worst way to fail. Asserted by tests/test_orchestration.py.

3. **The tenant scope is NEVER checkpointed** (BUG-19, removed structurally).
   It lives in a ContextVar set once per segment from a freshly resolved
   identity, so a resumed run cannot take its tenant from storage — there is
   nothing in state to take. Visibility of the run itself is decided by RLS, the
   same way artifact visibility is.

State minimisation (design §1.5): the checkpoint holds control flow and
identifiers only. Artifact content and finding bodies stay in the database, so a
purged checkpoint costs a run's control flow and not its findings.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from typing import Any, TypedDict

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from sqlalchemy import select, text

from dataclasses import replace

from review_agent.agents import conformance_agent
from review_agent.guardrails import input_guard, output_review
from review_agent.ingestion.sanitise import sanitise
from review_agent.audit import actions
from review_agent.data import repository
from review_agent.data.db import scoped_raw_connection, scoped_session
from review_agent.data.models import Artifact
from review_agent.data.scope import CallerScope, resolve_scope_for_subject
from review_agent.rules.loader import load_rulebook


class ScopeMismatch(RuntimeError):
    """A run is not visible to, or not scoped for, the identity operating on it.

    Always fatal. Never downgraded to a warning: a run whose tenant cannot be
    established must not execute.
    """


_ACTIVE_SCOPE: ContextVar[CallerScope | None] = ContextVar("active_scope", default=None)


def _current_scope() -> CallerScope:
    """The scope for the segment being executed.

    A ContextVar, NOT graph state. This is the structural removal of BUG-19: the
    caller's identity is never checkpointed, so a resumed run cannot take its
    tenant from storage — there is nothing there to take. The scope is set once
    per segment by _run_segment, from a freshly resolved identity.
    """
    scope = _ACTIVE_SCOPE.get()
    if scope is None:
        raise ScopeMismatch("graph node ran with no resolved scope")
    return scope


class ReviewState(TypedDict, total=False):
    """Graph state. Identifiers and control flow only.

    Deliberately absent: artifact text, finding bodies, prompts, model responses
    — and ANY tenant identifier. Nodes re-fetch through RLS, which makes each
    re-fetch a re-verification.
    """

    project_id: str
    artifact_id: str
    run_id: str
    # The status a successful review parks in AFTER findings are persisted, before
    # the human interrupt. Control flow, not tenant data. Default `awaiting_review`
    # (the run goes straight to the SAO queue). The submitter demo flow (FBR-4)
    # passes `draft`: findings are persisted and the run pauses, but it does NOT
    # enter the reviewer queue until the submitter explicitly sends it.
    hold_status: str
    finding_ids: list[str]
    verdict_counts: dict[str, int]
    accepted: bool
    reject_reason: str | None
    decisions: dict[str, str]   # finding_id -> reviewer_action


def _load_artifact(session, artifact_id: str) -> Artifact | None:
    return session.execute(
        select(Artifact).where(Artifact.artifact_id == uuid.UUID(artifact_id))
    ).scalar_one_or_none()


# --- nodes -------------------------------------------------------------------

def scoped_retrieval(state: ReviewState) -> dict:
    """Confirm the artifact is visible to this caller. RLS does the work.

    No org predicate is written here: an artifact belonging to another tenant is
    simply not in the result set, so the run fails with "not visible" rather than
    confirming its existence.
    """
    scope = _current_scope()
    with scoped_session(scope) as session:
        artifact = _load_artifact(session, state["artifact_id"])
        if artifact is None:
            raise ScopeMismatch(
                f"artifact {state['artifact_id']} is not visible in this scope"
            )
        # What RLS already returned — the guard compares against this rather than
        # being handed a tenant to check.
        visible = frozenset(
            session.execute(text("SELECT project_id FROM projects")).scalars()
        )
        project_id, filename, content = (
            artifact.project_id, artifact.filename, artifact.content
        )

    verdict = input_guard.check(
        project_id=project_id,
        filename=filename,
        visible_project_ids=visible,
        suspicious_spans=sanitise(content).suspicious_spans,
    )
    repository.record_guardrail_input(scope, project_id, verdict)
    if verdict.blocked:
        _set_run_status(scope, state["run_id"], "failed")
        raise ScopeMismatch(
            "input guard blocked the request: " + "; ".join(verdict.reasons)
        )
    return {"project_id": project_id}


def conformance(state: ReviewState) -> dict:
    """Run the review and persist findings as `pending` BEFORE the interrupt.

    Persisting first is what makes the database the system of record for the
    reviewer queue: a checkpoint lost or purged mid-review costs the run's
    control flow, not the findings.
    """
    scope = _current_scope()
    rulebook = load_rulebook()

    with scoped_session(scope) as session:
        artifact = _load_artifact(session, state["artifact_id"])
        content, project_id = artifact.content, artifact.project_id

    result = conformance_agent.review(content, rulebook)

    # Model calls are recorded independently: they already happened, and a
    # rollback cannot un-spend them.
    for record in result.call_records:
        repository.record_model_call(scope, record, project_id=project_id)

    # Output review runs HERE rather than as its own graph node, deliberately.
    # A separate node would need findings to cross graph state (violating the
    # §1.5 minimisation rule: no finding bodies in a checkpoint) or to be
    # persisted before review (violating "block means nothing reaches the
    # reviewer"). The node boundary matters less than either property.
    if result.accepted:
        guardrail = output_review.review_output(
            result.findings, content, rulebook
        )
        repository.record_guardrail_output(
            scope, project_id, state["artifact_id"], guardrail
        )
        if guardrail.blocked:
            blocked = replace(
                result,
                accepted=False,
                findings=(),
                reject_reason=(
                    "output review blocked the findings: "
                    + "; ".join(guardrail.reasons[:3])
                ),
            )
            repository.record_review_rejected(
                scope, state["artifact_id"], project_id, blocked
            )
            _set_run_status(scope, state["run_id"], "failed")
            return {"accepted": False, "reject_reason": blocked.reject_reason,
                    "finding_ids": []}
        result = replace(result, findings=guardrail.findings)

    if not result.accepted:
        repository.record_review_rejected(
            scope, state["artifact_id"], project_id, result
        )
        _set_run_status(scope, state["run_id"], "failed")
        return {"accepted": False, "reject_reason": result.reject_reason,
                "finding_ids": []}

    with scoped_session(scope) as session:
        artifact = _load_artifact(session, state["artifact_id"])
        rows = repository.insert_findings(
            session, scope, artifact, result, run_id=state["run_id"]
        )
        finding_ids = [str(r.finding_id) for r in rows]
        # Park in the caller-chosen hold status. Bound as a parameter and
        # constrained to the two legal holding states — never interpolated, and a
        # bad value fails loudly rather than writing an unknown status.
        hold_status = state.get("hold_status", "awaiting_review")
        if hold_status not in ("awaiting_review", "draft"):
            raise ScopeMismatch(f"illegal hold status {hold_status!r}")
        session.execute(
            text("UPDATE review_runs SET status=:s, updated_at=now() "
                 "WHERE run_id=:r"),
            {"s": hold_status, "r": state["run_id"]},
        )

    counts: dict[str, int] = {}
    for finding in result.findings:
        counts[finding.verdict] = counts.get(finding.verdict, 0) + 1

    return {"accepted": True, "finding_ids": finding_ids, "verdict_counts": counts}


def human_review(state: ReviewState) -> dict:
    """The mandatory human-in-the-loop pause.

    Everything above this point is advisory. Note what is NOT here: a decision.
    The interrupt hands the reviewer the finding ids and waits.
    """
    # BACKSTOP, not the control: _after_conformance routes a rejected review to
    # END, so this node should never see one. It stays because interrupting on a
    # rejected run would hand the SAO an empty queue entry to decide on. Tests
    # assert the ROUTING is what stops it — otherwise this guard would keep the
    # suite green while the edge was miswired.
    if not state.get("accepted"):
        return {"decisions": {}}
    decisions = interrupt(
        {
            "run_id": state["run_id"],
            "finding_ids": state.get("finding_ids", []),
            "verdict_counts": state.get("verdict_counts", {}),
        }
    )
    return {"decisions": decisions or {}}


def finalize(state: ReviewState) -> dict:
    """Apply the SAO's decisions and close the run."""
    scope = _current_scope()
    decisions = state.get("decisions") or {}

    if decisions:
        with scoped_session(scope) as session:
            for finding_id, action in decisions.items():
                # Scoped to THIS run, and to findings that are still current.
                # A bare finding_id would let a decision land on a superseded
                # row — history that a later review has already retired, and
                # which no reviewer is looking at.
                session.execute(
                    text(
                        "UPDATE findings SET reviewer_action=:a, reviewed_by=:u, "
                        "reviewed_at=now() WHERE finding_id=:f AND run_id=:run "
                        "AND superseded_by_run_id IS NULL"
                    ),
                    {"a": action, "u": scope.user_id, "f": uuid.UUID(finding_id),
                     "run": state["run_id"]},
                )
            repository.record_audit(
                session,
                scope,
                action=actions.FINDING_DECIDED,
                project_id=state.get("project_id"),
                retrieved_ids={"findings": list(decisions)},
                detail={"decisions": decisions, "run_id": state["run_id"]},
            )

    _set_run_status(scope, state["run_id"], "completed")
    return {}


def _set_run_status(scope: CallerScope, run_id: str, status: str) -> None:
    with scoped_session(scope) as session:
        session.execute(
            text("UPDATE review_runs SET status=:s, updated_at=now() WHERE run_id=:r"),
            {"s": status, "r": run_id},
        )


def _after_conformance(state: ReviewState) -> str:
    """A rejected review ENDS. It never reaches finalize.

    `conformance` has already set the run `failed` and recorded why. finalize
    sets `completed` unconditionally — so without this edge a rejected run flowed
    straight through human_review (which passes through when not accepted) into
    finalize, which OVERWROTE `failed` with `completed`. A run that produced no
    findings and was recorded as rejected would then read as a successful review
    of a clean artifact: the two states that must never be confusable.

    The status write is not made conditional instead, deliberately. finalize
    means "this review reached a reviewer and closed"; teaching it to recognise
    runs that did not is how it acquires a second meaning. Routing keeps each
    node's postcondition true whenever it runs.
    """
    return "human_review" if state.get("accepted") else END


def build_graph(checkpointer):
    builder = StateGraph(ReviewState)
    builder.add_node("scoped_retrieval", scoped_retrieval)
    builder.add_node("conformance", conformance)
    builder.add_node("human_review", human_review)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "scoped_retrieval")
    builder.add_edge("scoped_retrieval", "conformance")
    builder.add_conditional_edges(
        "conformance", _after_conformance, {"human_review": "human_review", END: END}
    )
    builder.add_edge("human_review", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer)


# --- running one bounded segment ---------------------------------------------

def _run_segment(scope: CallerScope, fn):
    """Execute fn(graph) in ONE scoped transaction, which then commits and closes.

    The commit is the point. A segment is start->interrupt or resume->end: both
    bounded by machine time, never by human time. After this returns there is no
    open transaction and no checked-out connection, so a reviewer can take days
    without a backend sitting `idle in transaction` holding locks and blocking
    VACUUM.
    """
    token = _ACTIVE_SCOPE.set(scope)
    try:
        with scoped_raw_connection(scope) as (session, raw_connection):
            # The checkpointer writes on this already-scoped connection.
            graph = build_graph(PostgresSaver(raw_connection))
            return fn(graph)
    finally:
        _ACTIVE_SCOPE.reset(token)


def start_review(
    scope: CallerScope, artifact_id: str, run_id: str | None = None,
    hold_status: str = "awaiting_review",
):
    """Begin a review. Returns (run_id, interrupt_payload_or_None).

    `hold_status` is where a successful review parks before the human interrupt:
    `awaiting_review` (default — straight to the SAO queue) or `draft` (the
    submitter demo flow: persisted but not yet sent). See FBR-4.
    """
    run_id = run_id or str(uuid.uuid4())

    with scoped_session(scope) as session:
        artifact = _load_artifact(session, artifact_id)
        if artifact is None:
            raise ScopeMismatch(f"artifact {artifact_id} is not visible in this scope")
        repository.create_review_run(session, scope, artifact, run_id)

    state: ReviewState = {
        "artifact_id": artifact_id, "run_id": run_id, "hold_status": hold_status,
    }
    config = {"configurable": {"thread_id": run_id}}
    result = _run_segment(scope, lambda g: g.invoke(state, config=config))
    return run_id, result.get("__interrupt__")


def resume_review(subject: str, run_id: str, decisions: dict[str, str]):
    """Resume after human review. Scope is RE-RESOLVED, never taken from state.

    The tenant is never taken from stored state. An attacker able to influence a
    checkpoint row could otherwise choose the tenant a resumed run operates in,
    bypassing resolve_scope() entirely (BUG-19).
    """
    scope = resolve_scope_for_subject(subject)   # the SOURCE of authority

    # Is this run visible to the resumed identity? RLS answers, exactly as it
    # does for artifacts: another tenant's run is not in the result set, so the
    # resume fails without confirming the run exists.
    with scoped_session(scope) as session:
        visible = session.execute(
            text("SELECT status FROM review_runs WHERE run_id=:r"), {"r": run_id}
        ).scalar()
    if visible is None:
        repository.record_audit_independently(
            scope, action="scope.mismatch",
            detail={"run_id": run_id, "reason": "run not visible in resolved scope"},
        )
        raise ScopeMismatch(f"run {run_id} is not visible in this scope")

    config = {"configurable": {"thread_id": run_id}}
    return _run_segment(
        scope, lambda g: g.invoke(Command(resume=decisions), config=config)
    )

