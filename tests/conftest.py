"""Shared test fixtures built on the sample data.

The database fixtures deliberately FAIL rather than skip when no PostgreSQL is
reachable (design D1). Row-level security is a Postgres feature; a suite that
skips could report success on a machine with no database, which manufactures
false confidence — the exact failure mode the red-team suite exists to prevent.
"""

import json
import pathlib
from pathlib import Path

import pytest
from sqlalchemy import text

from review_agent.data import rls
from review_agent.data.db import (
    DEFAULT_ADMIN_URL,
    dispose_engines,
    get_engine,
    get_owner_engine,
)
from review_agent.data.provision import bootstrap
from review_agent.data.scope import CallerScope, resolve_scope_for_subject
from review_agent.data.seed import seed_sample_data, truncate_all

SAMPLE = Path(__file__).resolve().parents[1] / "sample-data"
SRC = Path(__file__).resolve().parents[1] / "src"


def pytest_configure(config):
    """Register markers and REFUSE to run in parallel.

    This suite is globally destructive: every test truncates and reseeds the
    shared database, and the mutation tests disable RLS, weaken policies and
    widen grants on shared schema. Under more than one xdist worker a test could
    observe a deliberately-mutated database and either flake or — far worse —
    report a FALSE GREEN, because "no rows leaked" is trivially true against a
    half-truncated table.

    FAIL-LOUD DETECTION is the chosen design, deliberately in preference to
    per-worker database isolation:

    * xdist_group markers are NOT a solution. Pinning the mutation tests to a
      single worker does nothing while every worker shares one DATABASE_URL —
      the other workers still query the mutated schema. Grouping would only
      advertise a supported parallel mode that does not exist.
    * Per-worker databases would make parallelism genuinely safe, and were
      rejected on cost/benefit: this suite runs in ~1.5s, so there is no problem
      to solve, and "parallelism is supported" removes this guard while adding a
      new silent failure mode (a worker that falls back to the shared database
      rejoins shared state with nothing left to catch it).

    If someone later has a real reason to parallelise, the correct move is to
    implement per-worker databases AND KEEP a guard that refuses to run when
    workers share one — never to relax this check.

    Whoever adds `-n auto` in six months must hit this wall, not a comment.
    """
    config.addinivalue_line(
        "markers",
        "mutation: deliberately breaks an isolation control; globally destructive. "
        "Documentation and a selector — NOT an isolation mechanism.",
    )
    config.addinivalue_line(
        "markers",
        "agent: exercises the conformance agent through a stubbed model provider. "
        "Never in the isolation gate, which must stay offline.",
    )

    workers = getattr(config.option, "numprocesses", None)
    parallel = workers in ("auto", "logical") or (
        isinstance(workers, int) and workers > 1
    )
    # `workerinput` exists only inside an xdist worker process — belt and braces
    # in case the controller-side check is ever bypassed.
    if parallel or hasattr(config, "workerinput"):
        raise pytest.UsageError(
            "The isolation red-team suite must run single-worker: it truncates and "
            "reseeds shared tables and mutates shared schema, so parallel workers "
            "can produce false GREENS, not just flakes. Re-run without -n "
            "(or with -n0). Do not weaken this check to enable parallelism — "
            "see the standing rule in CLAUDE.md."
        )


# --- sample-data fixtures (fixtures only; never loaded into the database) ----

@pytest.fixture
def organisations() -> dict:
    return json.loads((SAMPLE / "mock_organisations.json").read_text())


@pytest.fixture
def rulebook() -> list[dict]:
    return json.loads((SAMPLE / "ea_standards.json").read_text())["rules"]


@pytest.fixture
def org_markers(organisations) -> dict:
    """Map org_id -> list of distinct markers that must never cross tenants."""
    return {o["org_id"]: o["distinct_markers"] for o in organisations["organisations"]}


# --- database fixtures -------------------------------------------------------

@pytest.fixture(scope="session")
def provisioned_db():
    """Provision the isolation core once per session. Errors out if no DB."""
    try:
        bootstrap(reset=True)
    except Exception as exc:  # noqa: BLE001 — we want the raw cause surfaced
        raise pytest.UsageError(
            "The isolation red-team suite requires a real PostgreSQL 15+ and will "
            "not skip. Start one and set ADMIN_DATABASE_URL "
            f"(default: {DEFAULT_ADMIN_URL}). Underlying error: {exc}"
        ) from exc
    yield
    dispose_engines()


@pytest.fixture(scope="session", autouse=True)
def assert_suite_runs_unprivileged(provisioned_db):
    """Design §4.0 — the whole suite must run as the unprivileged app role.

    A red-team suite connected as a superuser proves nothing: every query
    succeeds and every row is visible. This is autouse and session-scoped, so if
    it fails NOTHING in the file reports a pass.
    """
    with get_engine().connect() as conn:
        problems = rls.check_connection_privileges(conn, expected_role=rls.ROLE_APP)
    assert problems == [], (
        "the red-team suite must run as the unprivileged app role; "
        f"refusing to report passes because: {problems}"
    )


def repair_isolation() -> None:
    """Restore the full isolation state. Idempotent; safe to call at any time.

    Deliberately delegates to rls.apply_all rather than listing the steps: this
    fixture has under-repaired TWICE by maintaining a parallel list (first the
    compliance grants, then the checkpoint grants). Provisioning calls the same
    function, so the two cannot drift.
    """
    with get_owner_engine().begin() as conn:
        rls.apply_all(conn)


@pytest.fixture(scope="session")
def source_manifest() -> dict[str, str]:
    """Contents of every source file, captured once before any test runs."""
    return {str(p): p.read_text() for p in SRC.rglob("*.py")}


@pytest.fixture(autouse=True)
def source_tree_unmodified(source_manifest):
    """No test may leave a source file modified.

    The database mutations are restored by the fixture below, but a test that
    edits a file under src/ is outside its reach — an interrupted run would leave
    the change in place, and every later result would be measuring different code
    with nothing to say so. That is the same silent-degradation shape the
    isolation repair fixture exists for, one layer out.

    Tests SHOULD mutate a copy in tmp_path rather than rely on this. It is a
    backstop, and it generalises if another source-mutating test ever appears.
    """
    yield

    changed = [
        path for path, original in source_manifest.items()
        if pathlib.Path(path).read_text() != original
    ]
    if changed:
        for path in changed:
            pathlib.Path(path).write_text(source_manifest[path])
        raise AssertionError(
            f"this test modified source files: {sorted(changed)}. They have been "
            "restored, but every later result in this run would have been "
            "measuring different code. Mutate a copy in tmp_path instead."
        )


@pytest.fixture(autouse=True)
def isolation_intact_after_every_test(provisioned_db):
    """Guarantee no test can leave the database mutated — even if it fails.

    try/finally inside each mutation test is the primary mechanism, but it is
    exactly the thing a future contributor forgets, and the consequence is
    severe and SILENT: if a test fails mid-mutation with RLS disabled, every
    subsequent test in the run is meaningless and most of them still report
    PASS, because an unscoped query against a table with no RLS happily returns
    rows that the assertions were never written to notice.

    So restoration does not depend on the test author remembering. This runs
    after every test, repairs any drift, and fails LOUDLY so the gap is fixed
    rather than absorbed.
    """
    yield

    with get_engine().connect() as conn:
        problems = {k: v for k, v in rls.verify_isolation(conn).items() if v}

    if problems:
        repair_isolation()
        raise AssertionError(
            f"this test left the isolation state mutated: {problems}. It has been "
            "repaired, but every result after it in this run would have been "
            "meaningless. Wrap the mutation in try/finally (see CLAUDE.md)."
        )


@pytest.fixture
def seeded_db(provisioned_db):
    """Fresh sample data for each test, so tests cannot leak state into each other."""
    truncate_all()
    seed_sample_data()
    yield


@pytest.fixture
def scope_a(seeded_db) -> CallerScope:
    """org-a submitter, resolved server-side exactly as a request would."""
    return resolve_scope_for_subject("user-a@org-a")


@pytest.fixture
def scope_b(seeded_db) -> CallerScope:
    """org-b submitter — every leak test runs in both directions."""
    return resolve_scope_for_subject("user-b@org-b")


@pytest.fixture
def owner_engine(provisioned_db):
    """Owner engine, for tests that deliberately break and restore the schema."""
    return get_owner_engine()


def make_run(session, scope, artifact, run_id: str | None = None) -> str:
    """Create a ReviewRun for `artifact` and return its id.

    Findings now carry an FK to review_runs, so any test that inserts findings
    needs a run to attribute them to. Shared here rather than duplicated per
    file — a run built slightly differently in each test is how "which run
    produced this" quietly stops meaning one thing.
    """
    import uuid as _uuid

    from review_agent.data.models import ReviewRun

    run_id = run_id or str(_uuid.uuid4())
    session.add(
        ReviewRun(
            run_id=run_id,
            org_id=scope.org_id,
            project_id=artifact.project_id,
            artifact_id=artifact.artifact_id,
            status="running",
        )
    )
    session.flush()
    return run_id
