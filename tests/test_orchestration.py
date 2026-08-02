"""Task 12 — the graph, the checkpointer, and the three confirmations.

The confirmations, each asserted rather than documented:

  1. NO TRANSACTION IS HELD ACROSS HUMAN REVIEW. A days-long `idle in
     transaction` backend holds locks, blocks VACUUM and exhausts the pool. It
     fails slowly and invisibly, which is the worst way to fail — so it is
     checked against pg_stat_activity, not reasoned about.
  2. THE MIGRATION ORDERING CONSTRAINT IS STRUCTURAL. The app refuses to start
     if checkpoint tables exist without tenant isolation.
  3. THE PURGE REFUSES TO SWEEP LIVE RUNS, and audits the refusal.
"""

import json
import uuid
from pathlib import Path

import pytest
from sqlalchemy import text

from review_agent.data import checkpoint as ckpt
from review_agent.data.models import CHECKPOINT_BOOKKEEPING_TABLE  # noqa: F401
from review_agent.data import rls
from review_agent.data.db import (
    get_admin_engine,
    get_engine,
    get_owner_engine,
    scoped_raw_connection,
    scoped_session,
)
from review_agent.data import repository
from review_agent.data.models import Finding
from review_agent.data.repository import insert_artifact
from review_agent.data.scope import resolve_scope_for_subject
from review_agent.models import client
from review_agent.models.types import (
    STOP_REASON_ERRORED,
    ModelCallRecord,
    ModelResponse,
    ModelTransportError,
    StopReason,
    Usage,
)
from langgraph.graph import END

from review_agent.orchestration import graph as orch
from review_agent.rules.loader import load_rulebook
from tests.conftest import make_run

SAMPLE = Path(__file__).resolve().parents[1] / "sample-data"

pytestmark = pytest.mark.agent


@pytest.fixture(scope="session", autouse=True)
def checkpoint_schema(provisioned_db):
    """Create the saver's tables, then immediately bring them under isolation.

    Immediately is not stylistic: SET NOT NULL requires an empty table, so the
    migration has exactly one safe moment — after setup(), before any run.
    """
    # Provisioning now creates AND migrates these (data.provision.bootstrap), so
    # this fixture only asserts the deployment path actually did it — previously
    # the tables existed solely because this fixture made them, which is why a
    # fresh deployment had none.
    with get_engine().connect() as conn:
        assert ckpt.checkpoint_tables_unmigrated(conn) == [], (
            "provisioning did not leave the checkpoint tables migrated"
        )
    yield


@pytest.fixture
def stub_model():
    def _payload():
        return {
            "findings": [
                {"rule_id": rid, "verdict": "unclear", "evidence": "",
                 "confidence": "low", "reasoning": "not stated"}
                for rid in load_rulebook().ids
            ]
        }

    class _Stub:
        name = "stub"

        def complete(self, model_id, request, prompt_sha256, role):
            payload = _payload()
            return ModelResponse(
                text=json.dumps(payload), structured=payload,
                stop_reason=StopReason.COMPLETE, model_id=model_id,
                usage=Usage(input_tokens=1, output_tokens=1),
                call_record=ModelCallRecord(
                    purpose=request.purpose, role=role, model_id=model_id,
                    stop_reason="complete", usage=Usage().as_dict(),
                    prompt_sha256=prompt_sha256,
                ),
                raw=object(),
            )

    client.set_provider(_Stub())
    yield
    client.set_provider(None)


@pytest.fixture
def retrying_model():
    """First response omits a rule (fails validation); second is complete."""
    def _payload(drop_one: bool):
        ids = list(load_rulebook().ids)
        if drop_one:
            ids = ids[:-1]
        return {"findings": [
            {"rule_id": rid, "verdict": "unclear", "evidence": "",
             "confidence": "low", "reasoning": "not stated"} for rid in ids
        ]}

    class _Stub:
        name = "retrying-stub"
        calls = 0

        def complete(self, model_id, request, prompt_sha256, role):
            _Stub.calls += 1
            payload = _payload(drop_one=_Stub.calls == 1)
            return ModelResponse(
                text=json.dumps(payload), structured=payload,
                stop_reason=StopReason.COMPLETE, model_id=model_id,
                usage=Usage(input_tokens=1, output_tokens=1),
                call_record=ModelCallRecord(
                    purpose=request.purpose, role=role, model_id=model_id,
                    stop_reason="complete", usage=Usage().as_dict(),
                    prompt_sha256=prompt_sha256,
                ),
                raw=object(),
            )

    _Stub.calls = 0
    client.set_provider(_Stub())
    yield
    client.set_provider(None)


@pytest.fixture
def artifact_a(scope_a):
    with scoped_session(scope_a) as session:
        artifact = insert_artifact(
            session, scope_a, project_id="proj-a1", filename="run.md",
            content=(SAMPLE / "artifact_org-a_proj-a1.md").read_text(),
        )
        return str(artifact.artifact_id)


def idle_in_transaction_backends() -> list[tuple]:
    """Backends parked mid-transaction. Must be empty between segments."""
    with get_admin_engine().connect() as conn:
        return [
            tuple(r) for r in conn.execute(
                text(
                    "SELECT usename, state, query FROM pg_stat_activity "
                    "WHERE state LIKE 'idle in transaction%' AND usename='review_app'"
                )
            )
        ]


# --- confirmation 1: no transaction across human review ----------------------

def test_no_transaction_is_held_across_the_human_interrupt(
    scope_a, artifact_a, stub_model
):
    """The run pauses with NOTHING open. A reviewer may take days.

    Asserted against pg_stat_activity rather than reasoned about, because the
    failure mode is silent: an `idle in transaction` backend holds locks and
    blocks VACUUM for as long as the human takes, and nothing surfaces until the
    database is already in trouble.
    """
    assert idle_in_transaction_backends() == []

    run_id, interrupted = orch.start_review(scope_a, artifact_a)
    assert interrupted, "the graph must pause for the SAO"

    # THE ASSERTION: the segment committed and let go.
    assert idle_in_transaction_backends() == [], (
        "a transaction is still open while awaiting human review"
    )

    # And the work is durably visible from an independent connection, which is
    # what proves a COMMIT happened rather than the transaction merely closing.
    with get_admin_engine().connect() as conn:
        assert conn.execute(
            text("SELECT count(*) FROM checkpoints WHERE thread_id=:t"), {"t": run_id}
        ).scalar() > 0
        assert conn.execute(
            text("SELECT status FROM review_runs WHERE run_id=:t"), {"t": run_id}
        ).scalar() == "awaiting_review"


def test_findings_are_persisted_before_the_interrupt(scope_a, artifact_a, stub_model):
    """The database is the system of record for the reviewer queue.

    A checkpoint lost or purged mid-review then costs the run's control flow,
    not the findings.
    """
    orch.start_review(scope_a, artifact_a)
    with scoped_session(scope_a) as session:
        rows = session.execute(
            text("SELECT reviewer_action FROM findings")
        ).scalars().all()
    assert len(rows) == len(load_rulebook().rules)
    assert set(rows) == {"pending"}  # HITL on every one


def test_full_run_completes_through_resume(scope_a, artifact_a, stub_model):
    """Start -> interrupt -> resume -> finalize, in two bounded segments."""
    run_id, _ = orch.start_review(scope_a, artifact_a)

    with scoped_session(scope_a) as session:
        finding_ids = [
            str(r) for r in session.execute(
                text("SELECT finding_id FROM findings")
            ).scalars()
        ]
    decisions = {fid: "accepted" for fid in finding_ids}

    orch.resume_review("user-a@org-a", run_id, decisions)

    assert idle_in_transaction_backends() == []
    with scoped_session(scope_a) as session:
        assert session.execute(
            text("SELECT status FROM review_runs WHERE run_id=:r"), {"r": run_id}
        ).scalar() == "completed"
        assert set(
            session.execute(text("SELECT reviewer_action FROM findings")).scalars()
        ) == {"accepted"}


# --- BUG-19: the tenant is never taken from stored state ---------------------

def test_scope_is_never_checkpointed():
    """Structural removal of BUG-19: there is nothing in state to tamper with."""
    fields = set(orch.ReviewState.__annotations__)
    for banned in ("org_id", "user_id", "scope", "tenant"):
        assert banned not in fields, (
            f"{banned!r} in graph state would let a resumed run take its tenant "
            "from storage instead of from a resolved identity"
        )


def test_resume_by_another_tenant_is_refused(scope_a, artifact_a, stub_model):
    """RLS decides run visibility, exactly as it does for artifacts."""
    run_id, _ = orch.start_review(scope_a, artifact_a)

    with pytest.raises(orch.ScopeMismatch):
        orch.resume_review("user-b@org-b", run_id, {})

    # org-b cannot even see the checkpoint rows.
    scope_b = resolve_scope_for_subject("user-b@org-b")
    with scoped_session(scope_b) as session:
        assert session.execute(
            text("SELECT count(*) FROM checkpoints WHERE thread_id=:t"), {"t": run_id}
        ).scalar() == 0


def test_node_without_a_resolved_scope_refuses_to_run():
    """A node that somehow executes unscoped fails loudly, not quietly."""
    with pytest.raises(orch.ScopeMismatch):
        orch.scoped_retrieval({"artifact_id": str(uuid.uuid4())})


# --- checkpoint isolation: both halves ---------------------------------------

def test_checkpoints_are_stamped_and_invisible_cross_tenant(
    scope_a, artifact_a, stub_model
):
    """Stamping and visibility are DIFFERENT properties; assert both."""
    run_id, _ = orch.start_review(scope_a, artifact_a)

    with get_admin_engine().connect() as conn:
        stamped = conn.execute(
            text("SELECT DISTINCT org_id FROM checkpoints WHERE thread_id=:t"),
            {"t": run_id},
        ).scalars().all()
    assert stamped == ["org-a"]  # half one: the row carries its tenant

    scope_b = resolve_scope_for_subject("user-b@org-b")
    with scoped_session(scope_b) as session:
        assert session.execute(text("SELECT count(*) FROM checkpoints")).scalar() == 0
    with scoped_session(scope_a) as session:
        assert session.execute(text("SELECT count(*) FROM checkpoints")).scalar() > 0


def test_unscoped_checkpoint_write_raises_even_on_a_recycled_connection(scope_a):
    """An unscoped checkpoint write must fail LOUDLY. Both connection states.

    The subtle half: `current_setting('app.current_org')` raises only while the
    parameter has NEVER been set on that connection. After a scoped session has
    used it — the normal case for a pooled connection — the parameter stays
    defined and reverts to the EMPTY STRING, so the default silently stamps
    org_id = ''. That is not NULL, so NOT NULL does not catch it, and the row is
    invisible to every tenant while accumulating forever.

    The CHECK constraint is what makes the guarantee unconditional. This test
    exercises a connection that has already carried a scope, because that is the
    state the spike could not see.
    """
    insert = text(
        "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id,"
        " checkpoint, metadata) VALUES ('x','','1','{}'::jsonb,'{}'::jsonb)"
    )

    # A connection that has previously been scoped, then released.
    with scoped_session(scope_a) as session:
        session.execute(text("SELECT 1"))

    with get_engine().connect() as conn:
        guc = conn.execute(
            text("SELECT current_setting('app.current_org', true)")
        ).scalar()
        with pytest.raises(Exception) as excinfo:
            conn.execute(insert)

    # THREE independent controls can catch this, and which one fires depends on
    # the connection's history. All are loud; none is silent:
    #   * GUC never set   -> current_setting() raises (undefined parameter)
    #   * GUC set-and-reverted -> the nullif policy rejects the write
    #   * if the policy were weakened -> the CHECK constraint rejects it
    # The test accepts any of them and then asserts the property that actually
    # matters: nothing landed under the empty-string pseudo-tenant.
    message = str(excinfo.value).lower()
    assert any(
        marker in message
        for marker in ("app.current_org", "not_blank", "row-level security")
    ), f"unscoped write did not fail loudly (guc={guc!r}): {message[:200]}"

    # And nothing landed under the empty-string pseudo-tenant.
    with get_admin_engine().connect() as conn:
        assert conn.execute(
            text("SELECT count(*) FROM checkpoints WHERE org_id = ''")
        ).scalar() == 0


# --- confirmation 2: the ordering constraint is structural -------------------

@pytest.mark.mutation
def test_startup_refuses_when_checkpoints_are_unmigrated(owner_engine):
    """Drop the org_id column; assert the app refuses to start.

    This is the structural replacement for "run the migration at the right
    moment". The consequence asserted is not a flag but the boot gate itself:
    verify_isolation_or_raise() must refuse.
    """
    with get_engine().connect() as conn:
        assert ckpt.checkpoint_tables_unmigrated(conn) == []

    try:
        with owner_engine.begin() as conn:
            # CASCADE because the policies depend on the column — which means
            # this mutation removes BOTH halves of checkpoint isolation, the
            # stamping and the visibility control, exactly as an un-migrated
            # table would be.
            conn.execute(text("ALTER TABLE checkpoints DROP COLUMN org_id CASCADE"))

        with get_engine().connect() as conn:
            problems = ckpt.checkpoint_tables_unmigrated(conn)
            assert any("no org_id column" in p for p in problems), problems

        with owner_engine.begin() as conn:
            with pytest.raises(rls.IsolationVerificationError) as excinfo:
                rls.verify_isolation_or_raise(conn)
            assert "checkpoint" in str(excinfo.value)
    finally:
        with owner_engine.begin() as conn:
            ckpt.apply_checkpoint_isolation(conn)

    with get_engine().connect() as conn:
        assert ckpt.checkpoint_tables_unmigrated(conn) == []
        assert rls.tables_missing_rls(conn) == []


@pytest.mark.mutation
def test_drift_check_still_covers_checkpoints_with_no_allow_list(owner_engine):
    """The trap must keep working on library tables — no skip-list, ever."""
    try:
        with owner_engine.begin() as conn:
            conn.execute(text("ALTER TABLE checkpoints DISABLE ROW LEVEL SECURITY"))
        with get_engine().connect() as conn:
            assert "checkpoints" in rls.tables_missing_rls(conn)
    finally:
        with owner_engine.begin() as conn:
            conn.execute(text("ALTER TABLE checkpoints ENABLE ROW LEVEL SECURITY"))
            conn.execute(text("ALTER TABLE checkpoints FORCE ROW LEVEL SECURITY"))
    with get_engine().connect() as conn:
        assert rls.tables_missing_rls(conn) == []


# --- confirmation 3: the purge refuses to sweep live work --------------------

def test_purge_refuses_runs_awaiting_review(scope_a, artifact_a, stub_model):
    """A sweep that removes an in-flight review destroys work irrecoverably.

    The 30-day TTL exceeds an SAO SLA that does not exist yet, so age is NOT the
    only gate: status is. Until the SAO sets an SLA, a live run is never swept
    however old it is.
    """
    run_id, _ = orch.start_review(scope_a, artifact_a)

    result = ckpt.purge_checkpoints(scope_a, run_ids=[run_id])
    assert result == {"purged": [], "refused": [run_id]}

    # The checkpoint survived, so the run is still resumable.
    with scoped_session(scope_a) as session:
        assert session.execute(
            text("SELECT count(*) FROM checkpoints WHERE thread_id=:t"), {"t": run_id}
        ).scalar() > 0

    # And the refusal is auditable, not silent.
    with scoped_session(scope_a) as session:
        entry = session.execute(
            text("SELECT detail FROM audit_log WHERE action='checkpoint.purged'")
        ).scalar()
    assert entry["refused"] == [run_id]
    assert "no SAO SLA" in entry["refused_reason"]


def test_purge_removes_terminal_runs(scope_a, artifact_a, stub_model):
    """Completed runs are swept; the findings and audit trail are not."""
    run_id, _ = orch.start_review(scope_a, artifact_a)
    with scoped_session(scope_a) as session:
        finding_ids = [
            str(r) for r in session.execute(
                text("SELECT finding_id FROM findings")
            ).scalars()
        ]
    orch.resume_review("user-a@org-a", run_id, {f: "accepted" for f in finding_ids})

    result = ckpt.purge_checkpoints(scope_a, run_ids=[run_id])
    assert result["purged"] == [run_id]

    with scoped_session(scope_a) as session:
        assert session.execute(
            text("SELECT count(*) FROM checkpoints WHERE thread_id=:t"), {"t": run_id}
        ).scalar() == 0
        # What a purge costs is control flow, never the record or the findings.
        assert session.execute(text("SELECT count(*) FROM findings")).scalar() == len(
            load_rulebook().rules
        )
        assert session.execute(
            text("SELECT count(*) FROM audit_log WHERE action='review.completed'")
        ).scalar() == 1


# --- migration crash-safety --------------------------------------------------

@pytest.mark.mutation
def test_migration_failure_cannot_leave_rls_lowered(owner_engine):
    """The lower/restore window must be crash-safe, including on bookkeeping.

    apply_checkpoint_isolation DISABLEs RLS on each table before restoring it. If
    that window survived a failure, a crash mid-migration would leave graph state
    readable by every tenant — and it would look like a successful deploy.

    DDL is transactional in Postgres, so the window closes on rollback. This
    proves it rather than assuming it: a statement is made to fail after the
    DISABLE, and every checkpoint table must come back with RLS on.
    """
    before = {}
    with get_admin_engine().connect() as conn:
        for t in (*ckpt.CHECKPOINT_STATE_TABLES, "checkpoint_migrations"):
            before[t] = conn.execute(
                text("SELECT relrowsecurity, relforcerowsecurity FROM pg_class "
                     "WHERE relname=:t"),
                {"t": t},
            ).one()
    assert all(row.relrowsecurity for row in before.values())

    with pytest.raises(Exception):
        with owner_engine.begin() as conn:
            ckpt.apply_checkpoint_isolation(conn)          # lowers, then restores
            conn.execute(text("ALTER TABLE checkpoints DISABLE ROW LEVEL SECURITY"))
            conn.execute(text("SELECT 1/0"))               # crash mid-migration

    with get_admin_engine().connect() as conn:
        for t in (*ckpt.CHECKPOINT_STATE_TABLES, "checkpoint_migrations"):
            after = conn.execute(
                text("SELECT relrowsecurity, relforcerowsecurity FROM pg_class "
                     "WHERE relname=:t"),
                {"t": t},
            ).one()
            assert after == before[t], f"{t} lost RLS through a failed migration"


def test_migration_refuses_to_run_without_a_transaction():
    """Autocommit would make the lower/restore window survive a failure."""
    with get_owner_engine().connect().execution_options(
        isolation_level="AUTOCOMMIT"
    ) as conn:
        with pytest.raises(RuntimeError, match="inside a transaction"):
            ckpt.apply_checkpoint_isolation(conn)


# --- the scope holder is on its third implementation: prove it isolates ------

def _run_one(subject, artifact_id, out, key):
    try:
        scope = resolve_scope_for_subject(subject)
        run_id, _ = orch.start_review(scope, artifact_id)
        out[key] = ("ok", run_id)
    except Exception as exc:  # noqa: BLE001
        out[key] = ("error", f"{type(exc).__name__}: {exc}")


def test_concurrent_segments_for_different_orgs_do_not_bleed(seeded_db, stub_model):
    """Two orgs reviewing at once. The scope holder must not be shared.

    The scope has now had three implementations (graph state, then a comparison
    against the checkpoint, now a ContextVar). A ContextVar is per-context and
    threads start with a fresh one — but that is a property to verify, not to
    assume, because the failure mode is a silent cross-tenant stamp under load
    rather than an error.
    """
    import threading

    scope_a = resolve_scope_for_subject("user-a@org-a")
    scope_b = resolve_scope_for_subject("user-b@org-b")
    with scoped_session(scope_a) as session:
        art_a = str(insert_artifact(
            session, scope_a, project_id="proj-a1", filename="a.md", content="A design"
        ).artifact_id)
    with scoped_session(scope_b) as session:
        art_b = str(insert_artifact(
            session, scope_b, project_id="proj-b1", filename="b.md", content="B design"
        ).artifact_id)

    out: dict = {}
    threads = [
        threading.Thread(target=_run_one, args=("user-a@org-a", art_a, out, "a")),
        threading.Thread(target=_run_one, args=("user-b@org-b", art_b, out, "b")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert out["a"][0] == "ok", out["a"]
    assert out["b"][0] == "ok", out["b"]
    run_a, run_b = out["a"][1], out["b"][1]

    # Every row each run produced must carry its own tenant, and only its own.
    with get_admin_engine().connect() as conn:
        for run_id, expected in ((run_a, "org-a"), (run_b, "org-b")):
            stamped = conn.execute(
                text("SELECT DISTINCT org_id FROM checkpoints WHERE thread_id=:t"),
                {"t": run_id},
            ).scalars().all()
            assert stamped == [expected], f"{run_id} stamped {stamped}, want {expected}"
        assert conn.execute(
            text("SELECT count(*) FROM checkpoints WHERE org_id NOT IN "
                 "('org-a','org-b')")
        ).scalar() == 0

    # And each tenant sees only its own run.
    for scope, mine, theirs in ((scope_a, run_a, run_b), (scope_b, run_b, run_a)):
        with scoped_session(scope) as session:
            visible = set(
                session.execute(text("SELECT run_id FROM review_runs")).scalars()
            )
        assert mine in visible and theirs not in visible


@pytest.mark.mutation
def test_a_shared_scope_holder_would_bleed(seeded_db, stub_model, monkeypatch):
    """Replace the ContextVar with a process-wide holder; assert bleed appears.

    This is what makes the test above worth having: it proves the ContextVar is
    load-bearing rather than incidental. A plain module global passes every
    single-threaded test in this repo and fails only under concurrency — which is
    exactly the bug class that reaches production.
    """
    import threading

    class _SharedHolder:
        """A module-global scope holder — the naive implementation."""

        _value = None

        def get(self):
            return _SharedHolder._value

        def set(self, value):
            _SharedHolder._value = value
            return value

        def reset(self, _token):
            _SharedHolder._value = None

    monkeypatch.setattr(orch, "_ACTIVE_SCOPE", _SharedHolder())

    scope_a = resolve_scope_for_subject("user-a@org-a")
    scope_b = resolve_scope_for_subject("user-b@org-b")
    with scoped_session(scope_a) as session:
        art_a = str(insert_artifact(
            session, scope_a, project_id="proj-a1", filename="a2.md", content="A"
        ).artifact_id)

    observed: list[str] = []
    barrier = threading.Barrier(2, timeout=30)

    def _racer(scope, label):
        orch._ACTIVE_SCOPE.set(scope)
        try:
            barrier.wait()          # both threads have now set the holder
            observed.append(orch._current_scope().org_id)
        except Exception:  # noqa: BLE001
            observed.append("error")

    threads = [
        threading.Thread(target=_racer, args=(scope_a, "a")),
        threading.Thread(target=_racer, args=(scope_b, "b")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    # With a shared holder both threads read the SAME org — one of them is wrong.
    assert len(set(observed)) == 1, (
        f"expected a shared holder to collapse both scopes into one, got {observed}"
    )
    assert art_a  # the fixture work is real, not elided


@pytest.mark.mutation
def test_absent_checkpoint_tables_are_reported_not_silently_clean(owner_engine):
    """"Nothing to check" and "checked and clean" must not return the same result.

    Before provisioning created these tables, absence was legitimate and the
    check returned []. Now that provisioning creates them, absence means graph
    state has nowhere isolated to live — and the dangerous version is subtle: a
    later setup() by some other path would create them UNMIGRATED, and rows could
    land before anyone noticed.

    This is the same silent-no-op shape as apply_checkpoint_isolation() quietly
    doing nothing when the tables were missing.
    """
    with get_engine().connect() as conn:
        assert ckpt.checkpoint_tables_unmigrated(conn) == []  # present and clean

    try:
        with owner_engine.begin() as conn:
            # checkpoint_migrations MUST go too: setup() consults it for
            # idempotency, so leaving it behind makes the recreate a silent
            # no-op — which is how this test first failed, and is now a loud
            # error in create_checkpoint_tables().
            for table in (*ckpt.CHECKPOINT_STATE_TABLES,
                          ckpt.CHECKPOINT_BOOKKEEPING_TABLE):
                conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))

        with get_engine().connect() as conn:
            problems = ckpt.checkpoint_tables_unmigrated(conn)
        assert problems, "absent tables were reported as clean"
        assert "absent" in problems[0]

        # And the boot gate refuses, rather than starting with nowhere to
        # checkpoint.
        with owner_engine.begin() as conn:
            with pytest.raises(rls.IsolationVerificationError) as excinfo:
                rls.verify_isolation_or_raise(conn)
        assert "checkpoint" in str(excinfo.value)
    finally:
        ckpt.create_checkpoint_tables(owner_engine)
        with owner_engine.begin() as conn:
            rls.apply_all(conn)

    with get_engine().connect() as conn:
        assert ckpt.checkpoint_tables_unmigrated(conn) == []
        assert rls.tables_missing_rls(conn) == []


# --- §3c: a provider failure is a handled outcome, not an escaping exception --
#
# Found by the FIRST live API call, which returned a billing 400. Every CONTENT
# failure was already handled (refusal, truncation, invalid output, guardrail
# block); a TRANSPORT failure was not. The run stayed `running` forever — neither
# resumable nor sweepable, since the purge only touches terminal runs by design —
# and `record_model_call` ran only after a successful call, so nothing recorded
# that a call had even been attempted.

@pytest.fixture
def failing_model():
    """A provider that always raises, as the live one did on a billing error."""
    class _Failing:
        name = "failing-stub"
        calls = 0

        def complete(self, model_id, request, prompt_sha256, role):
            _Failing.calls += 1
            raise ModelTransportError("connection reset before any response")

    _Failing.calls = 0
    stub = _Failing()
    client.set_provider(stub)
    yield stub
    client.set_provider(None)


def _model_call_attempts(scope) -> list[dict]:
    """Audited model calls that FAILED — the record of a spend with no result."""
    with scoped_session(scope) as session:
        details = session.execute(
            text("SELECT detail FROM audit_log WHERE action='model.call'")
        ).scalars().all()
    return [d for d in details if d.get("error")]


def _run_status(scope, run_id: str) -> str | None:
    with scoped_session(scope) as session:
        return session.execute(
            text("SELECT status FROM review_runs WHERE run_id=:r"), {"r": run_id}
        ).scalar()


def test_a_transport_failure_leaves_a_terminal_run(scope_a, artifact_a, failing_model):
    """The run must not be left `running`. That state is a defect in itself.

    `running` is not merely untidy: the purge refuses to sweep non-terminal runs
    (deliberately — it must never destroy a live review), and there is no
    checkpoint to resume from because the failure happened inside the first
    node. So a stuck run is permanent, and accumulates one row per outage.
    """
    run_id, interrupted = orch.start_review(scope_a, artifact_a)

    assert not interrupted, "a failed review must not pause for a reviewer"
    assert _run_status(scope_a, run_id) == "failed"

    with scoped_session(scope_a) as session:
        assert session.execute(text("SELECT count(*) FROM findings")).scalar() == 0


def test_a_transport_failure_records_the_attempt(scope_a, artifact_a, failing_model):
    """The attempt is audited even though no response ever arrived.

    This is the same reasoning that makes `review.rejected` commit
    independently: the request left our infrastructure and may have been
    billed, and a rollback cannot undo either. Without it, inducing failures is
    an unlimited un-logged probe — and retries spend money the trail never shows.
    """
    orch.start_review(scope_a, artifact_a)

    attempts = _model_call_attempts(scope_a)
    assert attempts, "a failed model call left no trace in the audit log"

    # The error CLASS, not just that something went wrong: a billing 400 and a
    # network drop need different responses, and the trail is what tells them
    # apart after the fact.
    assert "ModelTransportError" in attempts[0]["error"]
    assert attempts[0]["stop_reason"] == STOP_REASON_ERRORED
    # Written by code from what was SENT, so it survives having no response.
    assert attempts[0]["prompt_sha256"]
    assert attempts[0]["model_id"]

    with scoped_session(scope_a) as session:
        rejected = session.execute(
            text("SELECT count(*) FROM audit_log WHERE action='review.rejected'")
        ).scalar()
    assert rejected == 1


def test_the_sdk_is_not_retried_over(scope_a, artifact_a, failing_model):
    """One attempt. The SDK has already retried by the time this raises.

    Retrying in the agent would multiply spend on a failure the SDK judged
    non-transient — and the review loop's `max_attempts` exists for INVALID
    OUTPUT, which is a different failure with a different remedy.
    """
    orch.start_review(scope_a, artifact_a)
    assert failing_model.calls == 1


def test_a_rejected_review_is_never_marked_completed(scope_a, artifact_a, failing_model):
    """Routing, not finalize's own judgement, is what stops it.

    Asserted at the EDGE because a rejected run previously flowed through
    human_review (which passes through when not accepted) into finalize, whose
    status write is unconditional — turning `failed` into `completed`. A review
    that produced nothing would then be indistinguishable from a clean pass.
    """
    assert orch._after_conformance({"accepted": False}) is END
    assert orch._after_conformance({}) is END
    assert orch._after_conformance({"accepted": True}) == "human_review"

    run_id, _ = orch.start_review(scope_a, artifact_a)
    assert _run_status(scope_a, run_id) != "completed"


@pytest.mark.mutation
def test_recording_only_successful_calls_makes_the_attempt_untraceable(
    scope_a, artifact_a, failing_model
):
    """MUTATION — restore the pre-fix behaviour and assert the CONSEQUENCE.

    The pre-fix code recorded a model call only after one succeeded. This puts
    that back, at the recording layer, and asserts what it actually costs: the
    audit log contains no record that the provider was ever contacted. Not a
    flag flipped — the trail is genuinely blind to the spend.

    Restored in try/finally, then re-asserted, so a control that has never been
    OBSERVED failing is not being taken on trust.
    """
    original = repository.record_model_call
    try:
        def only_on_success(scope, call_record, project_id=None):
            if call_record.error:      # the pre-fix behaviour, exactly
                return
            original(scope, call_record, project_id=project_id)

        repository.record_model_call = only_on_success
        run_id, _ = orch.start_review(scope_a, artifact_a)

        assert _model_call_attempts(scope_a) == [], (
            "the mutation did not take effect; the rest of this test proves nothing"
        )
        # THE CONSEQUENCE: the run is recorded as failed, but nothing says a
        # provider was contacted, which model, or against which prompt. An
        # operator reading the trail cannot tell an outage from a review that
        # was never attempted.
        with scoped_session(scope_a) as session:
            actions_logged = set(session.execute(
                text("SELECT action FROM audit_log")
            ).scalars())
        assert "model.call" not in actions_logged
        assert _run_status(scope_a, run_id) == "failed"
    finally:
        repository.record_model_call = original

    # The gate fires once the control is back.
    orch.start_review(scope_a, artifact_a)
    assert _model_call_attempts(scope_a), "restoring the control did not restore it"


# --- run_id + LOGICAL supersession -------------------------------------------
#
# `findings` was keyed to artifact_id alone, so re-reviewing an artifact
# APPENDED a second set indistinguishable from the first. run_id attributes a
# finding to the review that produced it; supersession is logical so the record
# of what a reviewer was shown survives.

def _findings(scope, run_id=None):
    sql = ("SELECT rule_id, run_id, superseded_by_run_id, reviewer_action "
           "FROM findings")
    params = {}
    if run_id:
        sql += " WHERE run_id = :r"
        params = {"r": run_id}
    with scoped_session(scope) as session:
        return session.execute(text(sql), params).all()


def test_a_re_review_supersedes_rather_than_appends(scope_a, artifact_a, stub_model):
    """Both runs' findings survive; only the newer set is current."""
    first, _ = orch.start_review(scope_a, artifact_a)
    second, _ = orch.start_review(scope_a, artifact_a)

    rules = len(load_rulebook().rules)
    assert len(_findings(scope_a)) == rules * 2, "history was destroyed"

    old = _findings(scope_a, first)
    new = _findings(scope_a, second)
    assert all(row.superseded_by_run_id == second for row in old), (
        "the first run's findings were not retired by the second"
    )
    assert all(row.superseded_by_run_id is None for row in new), (
        "the current run's findings are marked superseded"
    )


def test_a_reviewers_decision_does_not_carry_forward_to_a_re_review(
    scope_a, artifact_a, stub_model
):
    """THE POINT OF SUPERSESSION BEING LOGICAL.

    The old decision stays attached to the old finding — that is the record of
    what a human actually ruled on. The replacement is undecided, even where the
    rule and verdict are identical, because copying the decision across would be
    the SYSTEM ruling on the reviewer's behalf. "The evidence changed but the
    decision stood" is the state a human-in-the-loop design exists to prevent.
    """
    first, _ = orch.start_review(scope_a, artifact_a)
    decided = [
        str(r) for r in _run_ids_of_findings(scope_a, first)
    ]
    orch.resume_review("user-a@org-a", first, {fid: "accepted" for fid in decided})

    assert {row.reviewer_action for row in _findings(scope_a, first)} == {"accepted"}

    second, _ = orch.start_review(scope_a, artifact_a)

    # The old decisions are untouched history...
    assert {row.reviewer_action for row in _findings(scope_a, first)} == {"accepted"}
    # ...and the new findings are undecided.
    assert {row.reviewer_action for row in _findings(scope_a, second)} == {"pending"}


def _run_ids_of_findings(scope, run_id):
    with scoped_session(scope) as session:
        return session.execute(
            text("SELECT finding_id FROM findings WHERE run_id = :r"), {"r": run_id}
        ).scalars().all()


def test_a_decision_cannot_land_on_a_superseded_finding(
    scope_a, artifact_a, stub_model
):
    """Superseded findings are read-only history.

    Asserted through the real decision path, because the guard is a WHERE clause
    and a WHERE clause that silently matches nothing looks identical to one that
    works — so the test checks the row did not change, not that a call returned.
    """
    first, _ = orch.start_review(scope_a, artifact_a)
    stale_ids = [str(f) for f in _run_ids_of_findings(scope_a, first)]
    orch.start_review(scope_a, artifact_a)      # supersedes them

    with scoped_session(scope_a) as session:
        updated = repository.apply_reviewer_decisions(
            session, scope_a, first, {stale_ids[0]: "accepted"}
        )
    assert updated == 0, "a superseded finding accepted a decision"
    assert {row.reviewer_action for row in _findings(scope_a, first)} == {"pending"}


def test_the_review_page_shows_one_runs_findings_not_the_artifacts_history(
    scope_a, artifact_a, stub_model
):
    """load_review is keyed by run. Previously it returned every run at once —
    the same rule repeatedly, and at two severities if the rulebook had moved."""
    first, _ = orch.start_review(scope_a, artifact_a)
    second, _ = orch.start_review(scope_a, artifact_a)

    rules = len(load_rulebook().rules)
    with scoped_session(scope_a) as session:
        for run_id in (first, second):
            review = repository.load_review(session, scope_a, run_id)
            assert len(review["findings"]) == rules, (
                f"run {run_id} returned {len(review['findings'])} findings for "
                f"{rules} rules — the artifact's whole history leaked in"
            )


def test_a_finding_cannot_be_attributed_to_another_tenants_run(scope_a, scope_b):
    """run_id is an FK to a TENANT-SCOPED table, and an FK does not check tenancy.

    Without the trigger check a foreign key would happily accept org-b's run_id
    on an org-a finding. Asserted at the DATABASE, because that is where the
    control lives — application code could be bypassed by a second write path,
    which is exactly why the artifact_id equivalent is enforced there too.
    """
    from review_agent.data.models import ReviewRun
    from sqlalchemy.exc import DatabaseError

    with scoped_session(scope_b) as session:
        artifact_b = insert_artifact(
            session, scope_b, project_id="proj-b1", filename="b.md", content="x",
        )
        foreign_run = make_run(session, scope_b, artifact_b)

    # The consistency triggers are DEFERRABLE INITIALLY DEFERRED, so this fires
    # at COMMIT, not at flush — the whole block is what must fail.
    with pytest.raises(DatabaseError) as excinfo:
        with scoped_session(scope_a) as session:
            artifact_a_row = insert_artifact(
                session, scope_a, project_id="proj-a1", filename="a.md", content="y",
            )
            session.add(
                Finding(
                    org_id=scope_a.org_id, project_id="proj-a1",
                    artifact_id=artifact_a_row.artifact_id,
                    run_id=foreign_run,               # org-b's run
                    rule_id="EA-RES-01", rulebook_version="0.1-sample",
                    rulebook_sha256="x" * 64, verdict="fail", severity="high",
                    evidence="y", reviewer_action="pending",
                )
            )

    # WHERE it failed matters: RLS hides org-b's run from this scope, so the
    # trigger reports "not visible" rather than an org mismatch. Asserting the
    # message keeps a future change that widened visibility from passing here
    # on the mismatch branch while the primary control had already gone.
    assert "run" in str(excinfo.value) and "not visible" in str(excinfo.value)


def test_a_successful_retry_records_WHY_it_retried(scope_a, artifact_a, retrying_model):
    """§3e(c): the spend was audited, the cause was not.

    validation_errors is only written by review.rejected, so a review that
    failed validation once and then succeeded recorded two model calls, two
    prompt hashes, and nothing about what went wrong. Phase 4 measures the retry
    RATE; a rate whose causes are invisible cannot be acted on.
    """
    orch.start_review(scope_a, artifact_a)

    with scoped_session(scope_a) as session:
        detail = session.execute(
            text("SELECT detail FROM audit_log WHERE action='review.completed'")
        ).scalar()

    assert len(detail["model_calls"]) == 2, "the stub did not force a retry"
    retries = detail["validation_retries"]
    assert len(retries) == 1, "a retry happened but was not recorded"
    assert any("EA-" in e or "verdict" in e for e in retries[0]), (
        f"the retry was recorded without a usable cause: {retries[0]}"
    )


def test_a_first_attempt_success_records_no_retries(scope_a, artifact_a, stub_model):
    """The counterpart: an empty list, not a missing key.

    Absent and zero must not be the same thing, or Phase 4 cannot tell "no
    retries" from "this run predates the field".
    """
    orch.start_review(scope_a, artifact_a)
    with scoped_session(scope_a) as session:
        detail = session.execute(
            text("SELECT detail FROM audit_log WHERE action='review.completed'")
        ).scalar()
    assert detail["validation_retries"] == []


# --- FBR-4: the submitter DRAFT state (pre-send) -----------------------------

def test_a_draft_review_is_persisted_but_not_in_the_reviewer_queue(
    scope_a, artifact_a, stub_model
):
    """hold_status='draft' parks the run before the SAO queue.

    Findings are persisted and the run pauses at the interrupt, but a draft is
    the submitter's private working state — it must not appear to the architect
    until explicitly sent.
    """
    from review_agent.data.repository import list_review_queue

    run_id, _ = orch.start_review(scope_a, artifact_a, hold_status="draft")

    assert _run_status(scope_a, run_id) == "draft"
    with scoped_session(scope_a) as session:
        rows = session.execute(
            text("SELECT count(*) FROM findings WHERE run_id=:r"), {"r": run_id}
        ).scalar()
        queue = [r["run_id"] for r in list_review_queue(session, scope_a)]
    assert rows == len(load_rulebook().rules), "draft findings must be persisted"
    assert run_id not in queue, "a draft must not be in the reviewer queue"


def test_sending_a_draft_moves_it_into_the_queue(scope_a, artifact_a, stub_model):
    from review_agent.data.repository import list_review_queue, mark_run_sent

    run_id, _ = orch.start_review(scope_a, artifact_a, hold_status="draft")
    with scoped_session(scope_a) as session:
        mark_run_sent(session, scope_a, run_id)

    assert _run_status(scope_a, run_id) == "awaiting_review"
    with scoped_session(scope_a) as session:
        queue = [r["run_id"] for r in list_review_queue(session, scope_a)]
    assert run_id in queue


def test_send_refuses_anything_that_is_not_a_draft(scope_a, artifact_a, stub_model):
    """Only a draft can be sent. An awaiting_review/completed/failed run cannot."""
    from review_agent.data.repository import mark_run_sent

    # A normal run goes straight to awaiting_review, never draft.
    run_id, _ = orch.start_review(scope_a, artifact_a)
    with scoped_session(scope_a) as session:
        with pytest.raises(ValueError, match="not a draft"):
            mark_run_sent(session, scope_a, run_id)


def test_draft_send_then_the_architect_resumes_to_completed(
    scope_a, artifact_a, stub_model
):
    """The full submitter->architect handoff: draft -> send -> resume -> completed.

    The architect's pause and decision rules are unchanged: the run was paused at
    the human interrupt the whole time; sending only flipped its status, and the
    resume applies decisions and finalises exactly as for a non-draft run.
    """
    from review_agent.data.repository import mark_run_sent

    run_id, _ = orch.start_review(scope_a, artifact_a, hold_status="draft")
    with scoped_session(scope_a) as session:
        mark_run_sent(session, scope_a, run_id)
        finding_ids = [
            str(r) for r in session.execute(
                text("SELECT finding_id FROM findings WHERE run_id=:r"), {"r": run_id}
            ).scalars()
        ]

    orch.resume_review(
        "user-a@org-a", run_id, {fid: "accepted" for fid in finding_ids}
    )
    assert _run_status(scope_a, run_id) == "completed"


def test_an_illegal_hold_status_is_refused(scope_a, artifact_a, stub_model):
    """The status write is bound + constrained, not interpolated.

    A hold_status outside the two legal holding states must fail loudly rather
    than write an unknown status the queue and purge would then misread.
    """
    from review_agent.orchestration.graph import ScopeMismatch
    with pytest.raises(ScopeMismatch, match="illegal hold status"):
        orch.start_review(scope_a, artifact_a, hold_status="totally-bogus")
