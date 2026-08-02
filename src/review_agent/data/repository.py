"""Scoped reads and writes. The ONLY way the rest of the app touches tenant data.

Two rules hold everywhere in this module:

1. org_id and project_id on written rows come from CallerScope — never from a
   request body, never from parsed file content (design §3.4 / BUG-3, BUG-6).
2. Reads never add an org predicate by hand. RLS has already restricted the
   visible set; a client-supplied project_id is applied as an ordinary filter on
   top of it. Filtering for a row you cannot see is not an error, it is empty.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from review_agent.audit import actions
from review_agent.data.models import Artifact, AuditLog, Finding, Project, ReviewRun
from review_agent.data.scope import CallerScope


def sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(str(value))
        return True
    except (ValueError, AttributeError, TypeError):
        return False


@dataclass
class ReviewContext:
    """Everything a review is allowed to see. This is what the model gets fed.

    If a foreign row ever appears in here, isolation has already failed — the
    guardrails downstream are a tripwire, not the boundary (design BUG-4).
    """

    org_id: str
    project_id: str
    artifacts: list[Artifact] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    def retrieved_ids(self) -> dict[str, list[str]]:
        """Exactly which rows were retrieved — written to the audit log verbatim.

        Derived from the rows actually returned, never from model narration.
        """
        return {
            "artifacts": [str(a.artifact_id) for a in self.artifacts],
            "findings": [str(f.finding_id) for f in self.findings],
        }


def insert_artifact(
    session: Session,
    scope: CallerScope,
    project_id: str,
    filename: str,
    content: str,
) -> Artifact:
    """Store an uploaded artifact, stamped with the CALLER's scope.

    `content` is untrusted and is treated as an opaque blob here: nothing parses
    it for identifiers. The sample artifacts state their own org and project in
    their header text — that is display content, and auto-detecting it would be a
    privilege-escalation primitive.

    (Prompt-injection sanitisation is Phase 2 and is explicitly NOT part of the
    isolation guarantee: isolation holds even if sanitisation fails completely,
    because the file's bytes never reach a scoping decision.)
    """
    artifact = Artifact(
        org_id=scope.org_id,          # from the session identity, full stop
        project_id=project_id,        # client-supplied, but filtered by RLS + FK
        filename=filename,
        content=content,
        content_sha256=sha256(content),
        uploaded_by=scope.user_id,
    )
    session.add(artifact)
    session.flush()

    # The audit entry is written HERE, in the same transaction, not left to the
    # caller to remember. An operation that can happen without an audit entry is
    # an operation that will eventually happen without one — and because both
    # writes share a transaction, an audit failure rolls the artifact back too.
    record_audit(
        session,
        scope,
        action=actions.ARTIFACT_UPLOAD,
        project_id=project_id,
        retrieved_ids={"artifacts": [str(artifact.artifact_id)]},
        detail={"filename": filename, "content_sha256": artifact.content_sha256},
    )
    return artifact


def fetch_review_context(
    session: Session, scope: CallerScope, project_id: str
) -> ReviewContext:
    """Retrieve everything in scope for one project.

    Note the absence of `WHERE org_id = ...`: RLS supplies it. A caller passing
    another org's project_id gets an empty context — not because a check
    rejected it, but because those rows are not in the result set the filter was
    applied to. There is no branch to forget.
    """
    artifacts = list(
        session.execute(
            select(Artifact).where(Artifact.project_id == project_id)
        ).scalars()
    )
    findings = list(
        session.execute(
            select(Finding).where(Finding.project_id == project_id)
        ).scalars()
    )
    return ReviewContext(
        org_id=scope.org_id,
        project_id=project_id,
        artifacts=artifacts,
        findings=findings,
    )


def insert_findings(
    session: Session,
    scope: CallerScope,
    artifact: Artifact,
    review_result,
    run_id: str,
) -> list[Finding]:
    """Persist an ACCEPTED review's findings, stamped with the caller's scope.

    Lives here rather than in agents/ because this is the only layer permitted
    to know about tenancy. The agent's Finding objects carry no org_id and no
    project_id — those are stamped here from CallerScope, exactly like artifacts
    (design BUG-6: a model-authored security column is a model-authored
    security decision).

    Refuses a rejected review outright: there is no partial persistence path,
    because a caller that could persist "some of" a rejected review would
    reintroduce the targeted-omission primitive that whole-review rejection
    exists to remove.
    """
    if not review_result.accepted:
        raise ValueError(
            "refusing to persist a rejected review: "
            f"{review_result.reject_reason or 'validation failed'}"
        )

    # The write path enforces the guardrail's deterministic checks ITSELF, rather
    # than trusting that a caller ran output review first. Output review lives
    # inside the conformance node (PHASE3_DESIGN §1.5), which is correct but
    # invisible in the graph — so a second path to findings would silently skip
    # it. Checking here means there is no such path: every finding that reaches
    # the database has had its evidence verified against the artifact it is being
    # attached to. Same pattern as insert_artifact writing its own audit entry.
    from review_agent.guardrails.output_review import deterministic_leak_checks
    from review_agent.rules.loader import load_rulebook

    rulebook = load_rulebook()
    # The checks below use the rulebook loaded HERE; the findings were produced
    # against the one recorded on the review. While there is a single rulebook
    # file those are always the same object, which is exactly why this would go
    # unnoticed — data_risk_rules.json is already staged in sample-data/, and the
    # first day a second rulebook exists this becomes "validated against the
    # wrong rules" with nothing to show for it.
    if review_result.rulebook_sha256 and (
        review_result.rulebook_sha256 != rulebook.sha256
    ):
        raise ValueError(
            "refusing to persist findings produced against a different rulebook: "
            f"review used {review_result.rulebook_sha256[:12]}, "
            f"loader has {rulebook.sha256[:12]}"
        )

    problems = deterministic_leak_checks(
        review_result.findings, artifact.content, rulebook
    )
    if problems:
        raise ValueError(
            "refusing to persist findings that fail the leak checks: "
            + "; ".join(problems[:3])
        )

    # SUPERSEDE the previous review's findings for this artifact. Logical, not
    # physical: the rows stay and become read-only history.
    #
    # Done HERE, in the same transaction as the insert, rather than by the
    # caller — the same reasoning as audit writes living inside the operation.
    # A re-review that could land without retiring its predecessor would leave
    # two current sets, and the reviewer queue would show one rule twice with no
    # way to tell which verdict is live.
    #
    # DECISIONS DO NOT CARRY FORWARD. The new rows are `pending` even where the
    # old finding was identical and already accepted, and the old row keeps its
    # reviewer_action, reviewed_by and reviewed_at. Copying a decision onto a
    # finding a human never saw would be the system ruling on the reviewer's
    # behalf — the same line the guardrails must not cross when they touch a
    # verdict. "The evidence changed but the decision stood" is precisely the
    # state a human-in-the-loop design exists to make impossible.
    superseded = session.execute(
        text(
            "UPDATE findings SET superseded_by_run_id = :run "
            "WHERE artifact_id = :a AND superseded_by_run_id IS NULL "
            "RETURNING finding_id, reviewer_action"
        ),
        {"run": run_id, "a": artifact.artifact_id},
    ).all()

    rows: list[Finding] = []
    for finding in review_result.findings:
        row = Finding(
            org_id=scope.org_id,            # from the session identity, full stop
            project_id=artifact.project_id,
            artifact_id=artifact.artifact_id,
            run_id=run_id,
            rule_id=finding.rule_id,
            rulebook_version=review_result.rulebook_version,
            rulebook_sha256=review_result.rulebook_sha256,
            verdict=finding.verdict,
            severity=finding.severity,      # joined from the rulebook, not modelled
            evidence=finding.evidence,
            # The model's per-rule rationale — previously discarded here. Advisory
            # and unverified (evidence is the substring-checked field, not this).
            reasoning=finding.reasoning,
            # Advisory until a human rules on it. The SAO decides on every one.
            reviewer_action="pending",
        )
        session.add(row)
        rows.append(row)
    session.flush()

    # The whole-review reasoning trace goes ONCE on the review row, not on every
    # finding. It comes from the successful (last) model call; a review with no
    # trace (e.g. an Ollama run, which exposes no reasoning channel) leaves it
    # NULL. Same transaction as the findings it describes.
    thinking = (
        review_result.call_records[-1].thinking
        if review_result.call_records
        else None
    )
    if thinking:
        session.execute(
            text("UPDATE review_runs SET thinking_trace = :t WHERE run_id = :r"),
            {"t": thinking, "r": run_id},
        )

    record_audit(
        session,
        scope,
        action=actions.REVIEW_COMPLETED,
        project_id=artifact.project_id,
        rulebook_version=review_result.rulebook_version,
        # What the review was derived from and what it produced — the "what was
        # retrieved / what was decided" half of the CLAUDE.md requirement.
        retrieved_ids={
            "artifacts": [str(artifact.artifact_id)],
            "findings": [str(r.finding_id) for r in rows],
        },
        detail={
            "run_id": run_id,
            "rulebook_sha256": review_result.rulebook_sha256,
            "verdicts": _verdict_counts(review_result.findings),
            "model_calls": [r.as_dict() for r in review_result.call_records],
            # What this review RETIRED. The decided count is the number that
            # matters to the SAO: those are findings a human had already ruled
            # on, and whose replacements are back to `pending`.
            "superseded": {
                "count": len(superseded),
                "decided_count": sum(
                    1 for _, action in superseded if action != "pending"
                ),
                "finding_ids": [str(fid) for fid, _ in superseded],
            },
            # WHY the review retried, not merely that it did. Without this a
            # successful retry records the spend (two model calls, two prompt
            # hashes) and nothing about its cause — and a retry RATE whose causes
            # are invisible is a number nobody can act on (design §3e(c)).
            "validation_retries": [list(errors) for errors in review_result.corrections],
        },
    )
    return rows


def _verdict_counts(findings) -> dict[str, int]:
    """Verdict distribution, so "this review was 90% unclear" is visible later."""
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.verdict] = counts.get(finding.verdict, 0) + 1
    return counts


def create_review_run(
    session: Session,
    scope: CallerScope,
    artifact: Artifact,
    run_id: str,
):
    """Register an orchestrated run, stamped with the caller's scope.

    Lives here, not in orchestration/, for the same reason insert_artifact does:
    this is the only layer permitted to name a tenant. It also keeps the
    orchestration lint (no tenant identifier in agents/orchestration/guardrails)
    satisfiable without exceptions.
    """
    from review_agent.data.models import ReviewRun

    run = ReviewRun(
        run_id=run_id,
        org_id=scope.org_id,
        project_id=artifact.project_id,
        artifact_id=artifact.artifact_id,
        status="running",
    )
    session.add(run)
    session.flush()
    return run


# --- reviewer surface --------------------------------------------------------
# These exist so the API layer contains no SQL. Every one is scoped by RLS
# before any filter is applied.

def list_review_queue(session: Session, scope: CallerScope) -> list[dict]:
    """Runs awaiting this caller's review.

    No org predicate: RLS decides membership of the list. A run belonging to
    another tenant is not in the result set, so there is nothing to filter out.
    """
    rows = session.execute(
        select(ReviewRun).where(ReviewRun.status == "awaiting_review")
    ).scalars()
    return [
        {
            "run_id": run.run_id,
            "project_id": run.project_id,
            "artifact_id": str(run.artifact_id),
            "created_at": run.created_at.isoformat(),
        }
        for run in rows
    ]


def load_review(session: Session, scope: CallerScope, run_id: str) -> dict | None:
    """One review: the artifact, its document-level flags, and every finding.

    Passes are included IN FULL. They are shown and individually overridable;
    they simply do not block completion (PHASE3_DESIGN §4.5).

    Returns counts of findings NEEDING ATTENTION only — never a ratio, never a
    pass rate, never a denominator. A count of passes is a compliance claim
    wearing a number's clothing (§4.4).
    """
    run = session.execute(
        select(ReviewRun).where(ReviewRun.run_id == run_id)
    ).scalar_one_or_none()
    if run is None:
        return None

    artifact = session.execute(
        select(Artifact).where(Artifact.artifact_id == run.artifact_id)
    ).scalar_one()
    # Project NAME for the document header, from the stored project row (RLS
    # scopes it). Falls back to the id if the project is somehow gone.
    project = session.execute(
        select(Project).where(Project.project_id == run.project_id)
    ).scalar_one_or_none()
    project_name = project.name if project is not None else run.project_id
    # By RUN, not by artifact. A re-review supersedes rather than replaces, so
    # an artifact accumulates findings across runs — keyed by artifact this
    # returned every run's findings at once, showing the same rule repeatedly
    # (and at two severities if the rulebook moved between them).
    #
    # Keying by run also means an OLD run's page still shows what that reviewer
    # was actually looking at when they decided, which is the whole reason
    # supersession is logical rather than a delete.
    findings = list(
        session.execute(
            select(Finding).where(Finding.run_id == run.run_id)
        ).scalars()
    )

    from review_agent.findings import VERDICTS_REQUIRING_DECISION
    from review_agent.ingestion.sanitise import sanitise
    from review_agent.rules.loader import load_rulebook

    flags = sanitise(artifact.content).suspicious_spans

    # The rule STATEMENT, joined for display so a reviewer sees what each rule
    # says rather than needing all 14 ids by heart — and so a misattributed
    # finding fails a glance test (the statement sits directly above the quote).
    #
    # Shown ONLY when the finding's rulebook hash matches the loaded one. The
    # stored text is not the finding's own — only its hash is kept — so on a
    # mismatch the current wording could misrepresent what was actually judged.
    # Rendering the rule id without a statement is the honest failure: it is the
    # SAME refusal insert_findings makes on the write side (it rejects findings
    # whose rulebook_sha256 differs from the loader's), so the read path no
    # longer silently contradicts the version-pinning the hash exists to enforce.
    rulebook = load_rulebook()
    rules_by_id = rulebook.by_id

    def statement_for(f) -> str | None:
        if f.rulebook_sha256 != rulebook.sha256:
            return None  # different rulebook — do not show current wording
        rule = rules_by_id.get(f.rule_id)
        return rule.statement if rule is not None else None

    # WORK FIRST. Fails, then unclear, then passes; within fails, most severe
    # first. Rulebook order breaks ties. The neutral count says "N need
    # attention" — those N must be the top cards, not below a screen of passes.
    # Ordered HERE, so the reviewer and submitter views share it without a sort
    # control (the right default beats a dropdown). Severity ranks only fails:
    # on unclear it is *potential* exposure, and the request was fails-by-severity.
    verdict_rank = {"fail": 0, "unclear": 1, "pass": 2}
    severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    rule_order = {rid: i for i, rid in enumerate(rulebook.ids)}
    findings.sort(key=lambda f: (
        verdict_rank.get(f.verdict, 9),
        severity_rank.get(f.severity, 9) if f.verdict == "fail" else 0,
        rule_order.get(f.rule_id, len(rule_order)),
    ))

    return {
        "run_id": run.run_id,
        "status": run.status,
        # Whole-review reasoning trace (or None). UNVALIDATED model output — the
        # UI shows it collapsed, escaped, and labelled as not part of the
        # reviewed findings.
        "thinking_trace": run.thinking_trace,
        "project_id": run.project_id,
        "project_name": project_name,
        "artifact": {
            "artifact_id": str(artifact.artifact_id),
            "filename": artifact.filename,
            # The document header fields. All from the STORED artifact record,
            # never the request: `submitter` is `uploaded_by` — the identity
            # recorded at INGEST (insert_artifact stamps it from CallerScope) —
            # so it is who actually submitted, not who is viewing now.
            "submitter": artifact.uploaded_by,
            "submitted_at": artifact.uploaded_at.isoformat(),
            "content_sha256": artifact.content_sha256,
            "content": artifact.content,
            # Document-level, never per-finding: the sanitiser observed the
            # DOCUMENT, and attaching this to a finding would imply it judged
            # that finding. Phrased as an observation with a next step, because a
            # flag that appears often and means little gets ignored (§4.0).
            "flags": [
                {
                    "observation": (
                        "This document contains text resembling instructions to "
                        "the AI; findings may warrant closer scrutiny."
                    ),
                    "span": span,
                }
                for span in flags
            ],
        },
        "findings": [
            {
                "finding_id": str(f.finding_id),
                "rule_id": f.rule_id,
                "statement": statement_for(f),
                "verdict": f.verdict,
                "severity": f.severity,
                # Severity on an `unclear` describes POTENTIAL exposure: "we could
                # not tell whether a critical rule is met", not a confirmed
                # critical violation (PHASE2_DESIGN §4.2b).
                "severity_is_potential": f.verdict == "unclear",
                "evidence": f.evidence,
                # Per-rule rationale. Advisory/unverified — the UI shows it
                # collapsed on the card, below the validated fields.
                "reasoning": f.reasoning,
                "reviewer_action": f.reviewer_action,
                "decision_required": f.verdict in VERDICTS_REQUIRING_DECISION,
            }
            for f in findings
        ],
        "awaiting_decision": sum(
            1
            for f in findings
            if f.verdict in VERDICTS_REQUIRING_DECISION
            and f.reviewer_action == "pending"
        ),
    }


def apply_reviewer_decisions(
    session: Session, scope: CallerScope, run_id: str, decisions: dict[str, str]
) -> int:
    """Write reviewer_action, and nothing else.

    BUG-16 reaching the last layer: the verdict, rule_id and severity are what
    the SAO rules on, so no code between the agent and the human may edit them.
    This UPDATEs one named column — there is no path here that could touch
    another, which is the same technique as the guardrail's text-only projection.
    """
    from review_agent.findings import REVIEWER_ACTIONS

    invalid = {a for a in decisions.values() if a not in REVIEWER_ACTIONS or a == "pending"}
    if invalid:
        raise ValueError(f"invalid reviewer actions: {sorted(invalid)}")

    # A malformed finding_id is a CLIENT error, not a server one. Without this it
    # reached the UPDATE bind and psycopg raised on the bad UUID, surfacing as a
    # 500 — which reads as "the system broke" when someone fat-fingers a URL.
    # ValueError is caught by the /decide handler and returned as 400. Validated
    # here rather than in the API layer so every caller of this function gets it,
    # not only the one HTTP route.
    malformed = [fid for fid in decisions if not _is_uuid(fid)]
    if malformed:
        raise ValueError(f"malformed finding_id(s): {malformed}")

    run = session.execute(
        select(ReviewRun).where(ReviewRun.run_id == run_id)
    ).scalar_one_or_none()
    if run is None:
        raise ValueError(f"run {run_id} is not visible in this scope")

    updated = 0
    for finding_id, action in decisions.items():
        result = session.execute(
            text(
                "UPDATE findings SET reviewer_action=:a, reviewed_by=:u, "
                "reviewed_at=now() WHERE finding_id=:f AND run_id=:run "
                "AND superseded_by_run_id IS NULL"
            ),
            {"a": action, "u": scope.user_id, "f": finding_id, "run": run_id},
        )
        updated += result.rowcount

    record_audit(
        session,
        scope,
        action=actions.FINDING_DECIDED,
        project_id=run.project_id,
        retrieved_ids={"findings": list(decisions)},
        detail={"decisions": decisions, "run_id": run_id},
    )
    return updated


def findings_awaiting_decision(
    session: Session, scope: CallerScope, run_id: str
) -> list[str]:
    """Findings that ASSERT something and have no decision yet.

    Passes are excluded by design (§4.5): requiring a click on each of them
    produces rubber-stamping, and an incorrect pass is caught by nobody whether
    or not the click happened.
    """
    from review_agent.findings import VERDICTS_REQUIRING_DECISION

    run = session.execute(
        select(ReviewRun).where(ReviewRun.run_id == run_id)
    ).scalar_one_or_none()
    if run is None:
        raise ValueError(f"run {run_id} is not visible in this scope")

    rows = session.execute(
        select(Finding).where(
            Finding.run_id == run.run_id,
            Finding.reviewer_action == "pending",
            Finding.verdict.in_(VERDICTS_REQUIRING_DECISION),
        )
    ).scalars()
    return [str(f.finding_id) for f in rows]


def mark_run_completed(session: Session, scope: CallerScope, run_id: str) -> None:
    session.execute(
        text("UPDATE review_runs SET status='completed', updated_at=now() "
             "WHERE run_id=:r"),
        {"r": run_id},
    )


def list_projects(session: Session, scope: CallerScope) -> list[dict]:
    """Projects visible to this caller — the source for the submitter's picker.

    No org predicate: RLS decides membership, so the picker can only ever offer
    projects the caller may already see. A wrong-org submission is not policed;
    it is unrepresentable, because the id the submitter would need is not in the
    list (FBR-2). Returns id + name only.
    """
    rows = session.execute(select(Project).order_by(Project.project_id)).scalars()
    return [{"project_id": p.project_id, "name": p.name} for p in rows]


def mark_run_sent(session: Session, scope: CallerScope, run_id: str) -> None:
    """Move a DRAFT run into the SAO queue (draft -> awaiting_review). FBR-4.

    A STATE CHANGE, so its audit entry shares this transaction — if the status
    write rolls back, so does the record that it was sent. Refuses anything not
    currently `draft`: a run already sent, completed, failed, or still running
    cannot be (re-)sent, and an unknown run is not visible in this scope.
    """
    status = session.execute(
        select(ReviewRun.status).where(ReviewRun.run_id == run_id)
    ).scalar_one_or_none()
    if status is None:
        raise ValueError(f"run {run_id} is not visible in this scope")
    if status != "draft":
        raise ValueError(f"run {run_id} is {status!r}, not a draft awaiting send")

    updated = session.execute(
        text("UPDATE review_runs SET status='awaiting_review', updated_at=now() "
             "WHERE run_id=:r AND status='draft'"),
        {"r": run_id},
    ).rowcount
    if updated != 1:  # lost a race, or RLS hid the row from the write
        raise ValueError(f"run {run_id} could not be sent")

    run = session.execute(
        select(ReviewRun).where(ReviewRun.run_id == run_id)
    ).scalar_one()
    record_audit(
        session,
        scope,
        action=actions.REVIEW_SENT,
        project_id=run.project_id,
        detail={"run_id": run_id, "from": "draft", "to": "awaiting_review"},
    )


def record_audit_independently(scope: CallerScope, **kwargs) -> None:
    """Write an audit entry in its OWN transaction, on its own connection.

    THE ATOMICITY RULE HAS TWO HALVES, and this is the second one. The entry's
    transaction scope matches the REVERSIBILITY of what it records:

    * Records of STATE CHANGES (artifact.upload, review.completed) share the
      operation's transaction. If the state change rolls back the record must
      too — otherwise the trail describes a system state that never existed.

    * Records of THINGS THAT ALREADY HAPPENED IRREVERSIBLY (model.call,
      review.rejected) commit independently. A rollback cannot un-spend the
      tokens, un-send the content to the provider, or un-attempt the review.
      Rolling the record back would erase evidence of an event that really
      occurred.

    The second half matters most under attack: inducing a rejection is a
    plausible probe, and if the rejection record died with the rolled-back
    transaction an attacker would get unlimited un-logged attempts.

    These entries may reference an artifact that no longer exists after the
    rollback. That is fine and was designed for — audit_log deliberately carries
    no FK on its subjects, because the record outlives them.
    """
    from review_agent.data.db import scoped_session

    with scoped_session(scope) as session:
        record_audit(session, scope, **kwargs)


def record_review_rejected(
    scope: CallerScope,
    artifact_id,
    project_id: str,
    review_result,
) -> None:
    """Record a review that produced NOTHING. Commits independently.

    A rejected review is the case most worth auditing and the easiest to lose:
    no findings row exists afterwards, so without this entry the attempt leaves
    no trace and looks identical to a review that was never run.

    Takes ids rather than the ORM object, because the caller's transaction may
    be about to roll away the row this refers to.
    """
    record_audit_independently(
        scope,
        action=actions.REVIEW_REJECTED,
        project_id=project_id,
        rulebook_version=review_result.rulebook_version,
        retrieved_ids={"artifacts": [str(artifact_id)] if artifact_id else []},
        detail={
            "reject_reason": review_result.reject_reason,
            "validation_errors": list(review_result.validation_errors),
            "rulebook_sha256": review_result.rulebook_sha256,
            "model_calls": [r.as_dict() for r in review_result.call_records],
        },
    )


def record_guardrail_input(
    scope: CallerScope,
    project_id: str,
    result,
) -> None:
    """Record what the input guard decided. Commits INDEPENDENTLY.

    A BLOCK stops the run before anything is created, so like review.rejected
    this entry is the only trace the attempt leaves. Without it, refused requests
    would be an unlogged probe.
    """
    record_audit_independently(
        scope,
        action=actions.GUARDRAIL_INPUT,
        project_id=project_id,
        detail=result.as_audit_detail(),
    )


def record_guardrail_output(
    scope: CallerScope,
    project_id: str,
    artifact_id,
    result,
) -> None:
    """Record what output review decided. Commits INDEPENDENTLY.

    A BLOCK is a record of discarded work, like review.rejected: the review it
    stopped leaves no findings behind, so if this entry died with the rolled-back
    transaction the block would leave no trace at all — and inducing blocks would
    be an unlogged probe.
    """
    record_audit_independently(
        scope,
        action=actions.GUARDRAIL_OUTPUT,
        project_id=project_id,
        retrieved_ids={"artifacts": [str(artifact_id)]},
        detail=result.as_audit_detail(),
    )


def record_model_call(
    scope: CallerScope,
    call_record,
    project_id: str | None = None,
) -> None:
    """Record one model invocation: which model, what usage, which prompt.

    Commits independently: the call already happened. Tokens were spent and
    content left our infrastructure — a rollback cannot undo either, so the
    record must not be undone with it.

    `prompt_sha256` is what makes the call reproducible. Written by code from
    what was actually sent — never from the model's account of what it did
    (Phase 1 BUG-7: an audit trail must be evidence, not testimony).
    """
    record_audit_independently(
        scope,
        action=actions.MODEL_CALL,
        project_id=project_id,
        detail=call_record.as_dict(),
    )


def record_audit(
    session: Session,
    scope: CallerScope,
    action: str,
    project_id: str | None = None,
    rulebook_version: str | None = None,
    retrieved_ids: dict | None = None,
    detail: dict | None = None,
) -> AuditLog:
    """Append one immutable audit entry. Append-only is enforced by the database."""
    entry = AuditLog(
        org_id=scope.org_id,
        project_id=project_id or scope.project_id,
        user_id=scope.user_id,
        action=action,
        rulebook_version=rulebook_version,
        retrieved_ids=retrieved_ids,
        detail=detail,
    )
    session.add(entry)
    session.flush()
    return entry
