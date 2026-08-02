"""Canonical audit action names, and the operations required to write them.

A separate module with no imports so both the data layer and audit/log.py can
depend on it without a cycle.

CLAUDE.md makes a full append-only audit trail non-negotiable: who, org/project
scope, standards version, what was retrieved, what was decided — reproducible.
This table is the enumerated version of that sentence, and tests/test_audit.py
asserts every row of it actually writes.
"""

from __future__ import annotations

# --- action names ------------------------------------------------------------

SCOPE_DENIED = "scope.denied"        # a caller could not be resolved to an org
ARTIFACT_UPLOAD = "artifact.upload"  # an artifact entered the system
MODEL_CALL = "model.call"            # a model was invoked (which model, what usage)
# NB: this fires when findings are PERSISTED (inside insert_findings), which is
# BEFORE the human interrupt — not when the run reaches the terminal `completed`
# status. An auditor reading the trail cold will see `review.completed` while the
# run is still `awaiting_review`. The name refers to the REVIEW (the agent's work
# is done), not the RUN. The collision with review_runs.status='completed' is
# unfortunate but the action name is load-bearing across tests and stored rows.
REVIEW_COMPLETED = "review.completed"  # findings were accepted and persisted
REVIEW_REJECTED = "review.rejected"  # output failed validation; nothing persisted
FINDING_DECIDED = "finding.decide"   # Phase 3: the SAO ruled on a finding
REVIEW_SENT = "review.sent"          # submitter sent a DRAFT run to the SAO (FBR-4)
GUARDRAIL_INPUT = "guardrail.input"    # input guard screened a request
GUARDRAIL_OUTPUT = "guardrail.output"  # output review ran; what it decided

# Every operation that MUST leave a trace, with the phase that owns it.
# tests/test_audit.py iterates this; adding an action here without wiring it
# fails the coverage test.
REQUIRED_AUDITED_OPERATIONS: dict[str, str] = {
    SCOPE_DENIED: "scope resolution refused a caller",
    ARTIFACT_UPLOAD: "an artifact was ingested and stored",
    MODEL_CALL: "a model was called on a tenant's behalf",
    REVIEW_COMPLETED: "a review produced findings that were persisted",
    REVIEW_REJECTED: "a review was rejected and nothing was persisted",
    FINDING_DECIDED: "the SAO ruled on a finding",
    REVIEW_SENT: "a submitter sent a draft run to the SAO",
    GUARDRAIL_INPUT: "input guard screened a review request",
    GUARDRAIL_OUTPUT: "output review ran on drafted findings",
}

# The org recorded when there is no org — a denied scope resolution has no
# tenant by definition. audit_log deliberately has no FK on org_id (the record
# outlives its subject), so this sentinel needs no organisations row. It owns
# nothing, so RLS confines these rows to a view that holds no tenant data.
ORG_UNSCOPED = "__unscoped__"
