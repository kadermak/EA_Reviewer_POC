"""Database tables. Every tenant-scoped row carries org_id + project_id.

Tables:
  organisations   - the tenant boundary (org-a, org-b, ...)
  projects        - belong to exactly one organisation
  artifacts       - uploaded design submissions (tenant-scoped)
  findings        - review results (tenant-scoped, inherit artifact scope)
  audit_log       - append-only record of every action (see audit/log.py)
  users           - identity -> org mapping (NOT tenant-scoped; see below)

The standards rulebook is GLOBAL (not tenant-scoped) and is loaded from data
files, not stored per-tenant — keep it separate to avoid leakage bugs. That is
why EVERY table here except `users` is tenant-scoped: it turns the RLS drift
check into "this list must be empty" rather than a hand-maintained allow-list.

See docs/PHASE1_DESIGN.md §1.
"""

from __future__ import annotations

import uuid

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


# --- the invariant the drift check depends on -------------------------------
# Every table in the schema is tenant-scoped and must carry RLS, EXCEPT `users`,
# which is what establishes tenancy in the first place (an RLS policy on it would
# need the org variable that reading it produces). `users` is protected by
# privilege instead: the runtime app role has no grant on it at all.
TENANT_TABLES: tuple[str, ...] = (
    "organisations",
    "projects",
    "artifacts",
    "findings",
    "audit_log",
    "review_runs",
)
NON_TENANT_TABLES: frozenset[str] = frozenset({"users"})

# Created by LangGraph's PostgresSaver, not by this metadata. They are brought
# under RLS by data/checkpoint.py after setup(); the drift check covers them
# like anything else and gets no exemption. See PHASE3_DESIGN.md §1.3.
CHECKPOINT_STATE_TABLES: tuple[str, ...] = (
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
)
CHECKPOINT_BOOKKEEPING_TABLE = "checkpoint_migrations"


class Organisation(Base):
    """The tenant boundary. Self-scoping: a caller sees only their own org row.

    NOTE: `distinct_markers` from mock_organisations.json are deliberately NOT
    stored. They are test-fixture leak detectors, not application data — storing
    them would put every org's markers one query away from every code path.
    """

    __tablename__ = "organisations"

    org_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Project(Base):
    """Belongs to exactly one organisation.

    project_id is globally unique, not unique-per-org: a leaked or guessed
    project_id is then INERT — it matches no visible row rather than silently
    resolving to the caller's own first project.
    """

    __tablename__ = "projects"

    project_id: Mapped[str] = mapped_column(String, primary_key=True)
    org_id: Mapped[str] = mapped_column(
        String, ForeignKey("organisations.org_id"), nullable=False
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    criticality: Mapped[str] = mapped_column(String, nullable=False)
    handles_personal_data: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    created_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (Index("ix_projects_org", "org_id"),)


class Artifact(Base):
    """An uploaded design submission. `content` is UNTRUSTED and opaque here.

    org_id/project_id are stamped from CallerScope at upload. Nothing in the data
    layer ever parses `content` for identifiers — see design §3.4 / BUG-3.
    """

    __tablename__ = "artifacts"

    artifact_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    org_id: Mapped[str] = mapped_column(
        String, ForeignKey("organisations.org_id"), nullable=False
    )
    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.project_id"), nullable=False
    )
    filename: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    uploaded_by: Mapped[str] = mapped_column(String, nullable=False)
    uploaded_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_artifacts_org_project", "org_id", "project_id"),
        Index("ix_artifacts_org", "org_id"),
    )


class Finding(Base):
    """A review result. Advisory until a human rules on it.

    org_id is denormalised from artifacts ON PURPOSE (design §1.6): an RLS policy
    must decide a row's visibility from the row itself. A policy that joins to
    establish tenancy inherits the other table's bugs. A trigger keeps the two
    in step so the denormalisation cannot drift.
    """

    __tablename__ = "findings"

    finding_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    org_id: Mapped[str] = mapped_column(
        String, ForeignKey("organisations.org_id"), nullable=False
    )
    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.project_id"), nullable=False
    )
    artifact_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("artifacts.artifact_id"), nullable=False
    )
    # WHICH review produced this. Without it, re-reviewing an artifact appended a
    # second set of findings indistinguishable from the first — the same rule
    # appearing twice, at two severities if the rulebook had changed between
    # runs. A finding has to be attributable to the run that produced it before
    # anything can be said about that run.
    run_id: Mapped[str] = mapped_column(
        String, ForeignKey("review_runs.run_id"), nullable=False
    )
    # NULL means CURRENT. Set to the run that replaced this finding.
    #
    # Supersession is LOGICAL: the row stays. Deleting would break the
    # append-only posture and, worse, destroy the record of what a reviewer was
    # actually shown when they made their decision — the decision would survive
    # while its subject vanished, which is the one thing an audit trail exists to
    # prevent.
    #
    # A reviewer's decision does NOT carry forward to the replacement. See
    # repository.insert_findings.
    superseded_by_run_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("review_runs.run_id")
    )
    rule_id: Mapped[str] = mapped_column(String, nullable=False)
    rulebook_version: Mapped[str] = mapped_column(String, nullable=False)
    # Version alone is not enough: the SAO edits the rules file to change what is
    # checked, and nothing forces a version bump. The hash makes two reviews that
    # both claim "0.1-sample" against different rule text distinguishable.
    rulebook_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    verdict: Mapped[str] = mapped_column(String, nullable=False)
    severity: Mapped[str] = mapped_column(String, nullable=False)
    evidence: Mapped[str] = mapped_column(Text, nullable=False)
    # The model's per-rule RATIONALE. Advisory context, NOT a decision field and
    # NOT the evidence: unlike evidence it is not substring-verified against the
    # artifact, so it is shown to the reviewer clearly marked as unverified.
    # Persisted because it was already generated and previously discarded.
    # server_default keeps any insert path that omits it valid.
    reasoning: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    # Human-in-the-loop: the SAO decides on every finding.
    reviewer_action: Mapped[str] = mapped_column(
        String, nullable=False, server_default="pending"
    )
    reviewed_by: Mapped[str | None] = mapped_column(String)
    reviewed_at: Mapped[object | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_findings_org_project", "org_id", "project_id"),
        Index("ix_findings_artifact", "artifact_id"),
        Index("ix_findings_run", "run_id"),
        # The reviewer queue asks "current findings for this artifact" on every
        # page load; superseded rows accumulate and are never in that answer.
        Index("ix_findings_current", "artifact_id", "superseded_by_run_id"),
    )


class AuditLog(Base):
    """Append-only. Enforced by PRIVILEGE (no UPDATE/DELETE grant) + a trigger.

    No FK on org_id: the audit record must outlive its subject.
    `retrieved_ids` is written by the data layer from the rows actually returned —
    never from model narration (design BUG-7). It is what makes a review
    reproducible and what lets us prove after the fact that no foreign row was
    touched.
    """

    __tablename__ = "audit_log"

    audit_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    org_id: Mapped[str] = mapped_column(String, nullable=False)
    project_id: Mapped[str | None] = mapped_column(String)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    action: Mapped[str] = mapped_column(String, nullable=False)
    rulebook_version: Mapped[str | None] = mapped_column(String)
    retrieved_ids: Mapped[dict | None] = mapped_column(JSONB)
    detail: Mapped[dict | None] = mapped_column(JSONB)
    occurred_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (Index("ix_audit_org_time", "org_id", "occurred_at"),)


class ReviewRun(Base):
    """One orchestrated review. `run_id` is the LangGraph thread_id.

    Exists so the question "is this run still awaiting a human?" is answerable
    from the database rather than by inspecting opaque checkpoint state. The
    checkpoint purge depends on that answer: a sweep that deletes an in-flight
    review destroys work irrecoverably (PHASE3_DESIGN.md §1.5).
    """

    __tablename__ = "review_runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)  # graph thread_id
    org_id: Mapped[str] = mapped_column(
        String, ForeignKey("organisations.org_id"), nullable=False
    )
    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.project_id"), nullable=False
    )
    artifact_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("artifacts.artifact_id"), nullable=False
    )
    # running | draft | awaiting_review | completed | failed
    # `draft` is the submitter pre-send state (FBR-4): findings persisted, run
    # paused, NOT yet in the SAO queue. `review.sent` moves draft -> awaiting_review.
    status: Mapped[str] = mapped_column(String, nullable=False, server_default="running")
    created_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # The model's whole-review reasoning trace — ONE per review, not per finding
    # (the model reasons over all rules at once). Advisory and UNVALIDATED;
    # surfaced collapsed and clearly labelled. NULL for reviews with no trace
    # (a provider without a reasoning channel, or a rejected review).
    thinking_trace: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (Index("ix_review_runs_org_status", "org_id", "status"),)


class User(Base):
    """Identity -> org mapping. The ONE table not under tenant RLS (see module docstring).

    Protected by privilege: only `review_auth` may read it, and the runtime app
    role has no grant on it at all.
    """

    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String, primary_key=True)  # OIDC sub claim
    org_id: Mapped[str] = mapped_column(
        String, ForeignKey("organisations.org_id"), nullable=False
    )
    email: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)  # submitter | sao_reviewer
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    __table_args__ = (Index("ix_users_org", "org_id"),)
