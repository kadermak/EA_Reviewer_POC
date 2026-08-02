"""Append-only audit trail.

Records every action: who, org/project scope, standards version, what was
retrieved, what was decided. Append-only and tamper-evident — no update/delete.
This is the compliance record (distinct from engineering observability like
Langfuse/LangSmith).

TWO PROPERTIES MAKE THIS REAL RATHER THAN INTENDED
--------------------------------------------------
1. **Append-only is enforced by the DATABASE**, not by this module's discipline:
   the app role holds INSERT and SELECT and no UPDATE/DELETE grant, plus a
   trigger that raises. See PHASE1_DESIGN.md §1.7.

2. **The audit write shares a transaction with the operation it records.** An
   operation that cannot write its entry does not happen — the whole transaction
   rolls back. This is what stops the trail from silently degrading: there is no
   state in which the artifact landed but the record did not. The one deliberate
   exception is scope denial (see data/scope.py::record_scope_denied), where the
   operation is a refusal that has already succeeded.

`retrieved_ids` is written from the rows actually returned, never from the
model's account of what it did: an audit trail must be evidence, not testimony
(PHASE1_DESIGN.md BUG-7).
"""

from review_agent.audit.actions import (
    ARTIFACT_UPLOAD,
    MODEL_CALL,
    ORG_UNSCOPED,
    REQUIRED_AUDITED_OPERATIONS,
    REVIEW_COMPLETED,
    REVIEW_REJECTED,
    SCOPE_DENIED,
)
from review_agent.data.repository import (
    record_audit,
    record_model_call,
    record_review_rejected,
)
from review_agent.data.scope import record_scope_denied

# There is deliberately no append(entry: dict) taking a free-form org_id — the
# scope comes from CallerScope, never from a caller-supplied dict.
__all__ = [
    "ARTIFACT_UPLOAD",
    "MODEL_CALL",
    "ORG_UNSCOPED",
    "REQUIRED_AUDITED_OPERATIONS",
    "REVIEW_COMPLETED",
    "REVIEW_REJECTED",
    "SCOPE_DENIED",
    "record_audit",
    "record_model_call",
    "record_review_rejected",
    "record_scope_denied",
]
