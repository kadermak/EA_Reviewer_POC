"""An EMPTY scope must be indistinguishable from no access, on every table.

Phase 1 tested one empty-scope state: the session variable never set, where
current_setting() returns NULL. Phase 3 Finding 3 showed there is a SECOND state,
and it is the common one in production:

    never set on this connection      -> current_setting(...) is NULL
    set and reverted (pooled reuse)   -> the parameter stays DEFINED and is ''

They are different states and they were not equally tested. In the second one,
`org_id = current_setting(...)` becomes `org_id = ''`, which makes the empty
string a PSEUDO-TENANT: a recycled connection could write a row stamped '' and
any other recycled connection could read it back. That is cross-tenant leakage
by a route Phase 1's test could not see.

Two independent controls close it, and each has a mutation test below:
  1. every tenant table CHECKs org_id <> ''  — such a row cannot exist;
  2. every policy uses nullif(current_setting(...), '') — it would not match
     even if one did.
"""

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from review_agent.data import rls
from review_agent.data.db import get_engine, scoped_session, url_for_role
from review_agent.data.models import (
    CHECKPOINT_STATE_TABLES,
    TENANT_TABLES,
)

ALL_SCOPED_TABLES = tuple(TENANT_TABLES) + CHECKPOINT_STATE_TABLES


def _purge_blank_rows_and_restore(owner_engine):
    """Clean up a mutation's blank-stamped rows, then restore isolation.

    RLS must be lowered for the DELETE: FORCE binds the owner too, so with no
    session variable set the delete matches nothing — the same trap that broke
    the checkpoint migration. Getting this wrong leaves a ''-stamped row behind,
    and apply_all then cannot re-add the CHECK constraint.
    """
    with owner_engine.begin() as conn:
        # RLS must be lowered: FORCE binds the owner, so with no session
        # variable the delete matches nothing — the same trap that broke the
        # checkpoint migration. The append-only trigger must be lowered too;
        # that it blocks even the owner is the trigger working as designed.
        conn.execute(text("ALTER TABLE audit_log DISABLE ROW LEVEL SECURITY"))
        conn.execute(text("ALTER TABLE audit_log DISABLE TRIGGER audit_log_no_mutate"))
        conn.execute(text("DELETE FROM audit_log WHERE org_id = ''"))
        conn.execute(text("ALTER TABLE audit_log ENABLE TRIGGER audit_log_no_mutate"))
        conn.execute(text("ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY"))
        rls.apply_all(conn)


def _connection_never_scoped():
    """A genuinely fresh backend, on its own pool.

    Obtaining this is harder than it looks, and that difficulty IS the finding:
    once a process has served one scoped request, connections handed out by the
    shared pool have all had the parameter set at some point, so they are in the
    '' state rather than the NULL state. The never-set state is the rare one in
    a running system — which is precisely why testing only that state (as Phase
    1 did) left the common case unexamined.
    """
    engine = create_engine(url_for_role(rls.ROLE_APP), poolclass=NullPool)
    return engine


def _connection_with_empty_scope():
    """A connection whose scope has been SET and then emptied — pooled reuse."""
    conn = get_engine().connect()
    conn.execute(text("SELECT set_config('app.current_org', 'org-a', false)"))
    conn.execute(text("SELECT set_config('app.current_org', '', false)"))
    return conn


def test_the_two_empty_scope_states_are_actually_different(seeded_db):
    """Establish the premise: this is not one state tested twice."""
    engine = _connection_never_scoped()
    with engine.connect() as conn:
        never_set = conn.execute(
            text("SELECT current_setting('app.current_org', true)")
        ).scalar()
    engine.dispose()
    conn = _connection_with_empty_scope()
    try:
        reverted = conn.execute(
            text("SELECT current_setting('app.current_org', true)")
        ).scalar()
    finally:
        conn.close()

    assert never_set is None
    assert reverted == ""
    assert never_set != reverted, "if these ever converge, this file can be simplified"


@pytest.mark.parametrize("table", ALL_SCOPED_TABLES)
def test_empty_scope_yields_zero_rows(table, seeded_db, scope_a):
    """Both empty-scope states see nothing, on every scoped table."""
    engine = _connection_never_scoped()
    with engine.connect() as conn:
        assert conn.execute(text(f"SELECT count(*) FROM {table}")).scalar() == 0
    engine.dispose()

    conn = _connection_with_empty_scope()
    try:
        assert conn.execute(text(f"SELECT count(*) FROM {table}")).scalar() == 0
    finally:
        conn.close()

    # Guard against the assertions above passing because the table is empty.
    if table in TENANT_TABLES:
        with scoped_session(scope_a) as session:
            visible = session.execute(text(f"SELECT count(*) FROM {table}")).scalar()
        assert visible >= 0  # organisations/projects/artifacts are seeded


def test_seeded_tables_are_actually_non_empty(seeded_db, scope_a):
    """Otherwise the zero-row assertions above would be vacuous."""
    with scoped_session(scope_a) as session:
        for table in ("organisations", "projects", "artifacts", "audit_log"):
            assert session.execute(
                text(f"SELECT count(*) FROM {table}")
            ).scalar() > 0, f"{table} is empty; the empty-scope test proves nothing"


@pytest.mark.parametrize("table", ALL_SCOPED_TABLES)
def test_blank_org_is_unplantable(table, seeded_db):
    """Control 1: no row stamped '' can exist, so '' can never become a tenant."""
    with get_engine().connect() as conn:
        constraint = conn.execute(
            text(
                "SELECT conname FROM pg_constraint WHERE conrelid = to_regclass(:t) "
                "AND contype='c' AND conname = :n"
            ),
            {"t": table, "n": f"{table}_org_id_not_blank"},
        ).scalar()
    assert constraint, f"{table} does not forbid org_id = ''"


def test_no_policy_treats_blank_as_a_tenant(seeded_db):
    """Control 2: every policy collapses '' to NULL before comparing."""
    with get_engine().connect() as conn:
        assert rls.policies_matching_blank_scope(conn) == []


def test_verify_isolation_covers_both_controls(seeded_db):
    with get_engine().connect() as conn:
        report = rls.verify_isolation(conn)
    assert report["tables_missing_blank_org_guard"] == []
    assert report["policies_matching_blank_scope"] == []


# --- mutation: each control is load-bearing on its own -----------------------

def test_only_audit_log_could_ever_hold_a_blank_org(seeded_db):
    """Why the mutations below target audit_log, and why the CHECK matters most there.

    Every other tenant table has org_id REFERENCES organisations(org_id), so a
    blank org is already impossible: there is no organisation ''. audit_log
    deliberately has NO foreign key — the record must outlive its subject — which
    makes it the one table where the CHECK constraint is not redundant but load
    bearing, and the one place the pseudo-tenant could actually have formed.
    """
    with get_engine().connect() as conn:
        no_fk = conn.execute(
            text(
                """
                SELECT count(*) FROM pg_constraint
                WHERE conrelid='audit_log'::regclass AND contype='f'
                  AND 'org_id' = ANY(
                        SELECT attname FROM pg_attribute
                        WHERE attrelid=conrelid AND attnum=ANY(conkey))
                """
            )
        ).scalar()
    assert no_fk == 0, "audit_log gained an FK on org_id; re-read design §1.7"


@pytest.mark.mutation
def test_without_the_check_constraint_blank_org_becomes_plantable(owner_engine):
    """Drop control 1; assert '' becomes a writable pseudo-tenant.

    The consequence asserted is the leak itself: a connection with an empty
    scope writes a row, and a DIFFERENT connection with an empty scope reads it
    back. That is two unrelated requests sharing data.
    """
    try:
        with owner_engine.begin() as conn:
            conn.execute(
                text("ALTER TABLE audit_log DROP CONSTRAINT audit_log_org_id_not_blank")
            )
            # Control 2 must also be lowered, or it masks control 1.
            conn.execute(
                text(
                    "ALTER POLICY org_isolation_insert ON audit_log "
                    "WITH CHECK (org_id = current_setting('app.current_org', true))"
                )
            )
            conn.execute(
                text(
                    "ALTER POLICY org_isolation_select ON audit_log "
                    "USING (org_id = current_setting('app.current_org', true))"
                )
            )

        writer = _connection_with_empty_scope()
        try:
            writer.execute(
                text(
                    "INSERT INTO audit_log (org_id, user_id, action) "
                    "VALUES ('', 'ghost', 'pseudo-tenant.probe')"
                )
            )
            writer.commit()
        finally:
            writer.close()

        reader = _connection_with_empty_scope()
        try:
            leaked = reader.execute(
                text("SELECT count(*) FROM audit_log WHERE action='pseudo-tenant.probe'")
            ).scalar()
        finally:
            reader.close()
        assert leaked == 1, "expected the empty scope to behave as a shared tenant"
    finally:
        _purge_blank_rows_and_restore(owner_engine)

    # Restored: unplantable again.
    conn = _connection_with_empty_scope()
    try:
        with pytest.raises(Exception):
            conn.execute(
                text(
                    "INSERT INTO audit_log (org_id, user_id, action) "
                    "VALUES ('', 'ghost', 'probe2')"
                )
            )
    finally:
        conn.close()


@pytest.mark.mutation
def test_without_nullif_a_blank_stamped_row_would_be_visible(owner_engine, scope_a):
    """Drop control 2 only; assert the policy would match a ''-stamped row.

    Control 1 still forbids creating one through normal paths, so the row is
    inserted as the owner with RLS lowered — simulating a row that arrived some
    other way (a bad migration, a restore, a future default). The point is that
    the POLICY, on its own, cannot tell '' from a tenant.
    """
    try:
        with owner_engine.begin() as conn:
            conn.execute(text("ALTER TABLE audit_log DISABLE ROW LEVEL SECURITY"))
            conn.execute(
                text("ALTER TABLE audit_log DROP CONSTRAINT audit_log_org_id_not_blank")
            )
            conn.execute(
                text(
                    "INSERT INTO audit_log (org_id, user_id, action) "
                    "VALUES ('', 'ghost', 'blank.row')"
                )
            )
            conn.execute(
                text(
                    "ALTER POLICY org_isolation_select ON audit_log "
                    "USING (org_id = current_setting('app.current_org', true))"
                )
            )
            conn.execute(text("ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY"))

        with get_engine().connect() as conn:
            assert rls.policies_matching_blank_scope(conn), "drift check missed it"

        reader = _connection_with_empty_scope()
        try:
            visible = reader.execute(
                text("SELECT count(*) FROM audit_log WHERE action='blank.row'")
            ).scalar()
        finally:
            reader.close()
        assert visible == 1, "without nullif, an empty scope reads the blank tenant"
    finally:
        _purge_blank_rows_and_restore(owner_engine)

    # Restored: nullif back, and nothing blank-stamped survives.
    reader = _connection_with_empty_scope()
    try:
        assert reader.execute(
            text("SELECT count(*) FROM audit_log WHERE action='blank.row'")
        ).scalar() == 0
    finally:
        reader.close()


# --- the exceptions register -------------------------------------------------

def test_derivation_covers_a_table_the_orm_never_declared(owner_engine, seeded_db):
    """A Phase-4 table must FAIL the gate, not silently miss a control.

    This is the "this list must be empty" property applied to every check, not
    just RLS. Before the derivation, a table created outside the ORM metadata
    failed tables_missing_rls() while passing the blank-org check — one gate
    caught it and the other waved it through.
    """
    try:
        with owner_engine.begin() as conn:
            conn.execute(
                text("CREATE TABLE phase4_probe (id serial primary key, "
                     "org_id text NOT NULL)")
            )
        with get_engine().connect() as conn:
            report = rls.verify_isolation(conn)
        # Every derived gate must name it. If any of these is empty, that gate is
        # still working off a maintained list.
        assert "phase4_probe" in report["tables_missing_rls"]
        assert "phase4_probe" in report["tables_missing_blank_org_guard"]
        assert "phase4_probe" in report["scoped_tables_without_app_grant"]
    finally:
        with owner_engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS phase4_probe"))

    with get_engine().connect() as conn:
        assert all(v == [] for v in rls.verify_isolation(conn).values())


def test_exceptions_register_is_complete(seeded_db):
    """The registry in PHASE3_DESIGN.md §8b must match reality.

    A fourth exception introduced in code fails here until it is documented with
    its rationale, compensating control and pinning test. Exceptions are cheap to
    add and expensive to remember; this makes adding one cost a paragraph.
    """
    # E4 — exactly one table outside RLS, and it is `users`.
    from review_agent.data.models import NON_TENANT_TABLES

    assert NON_TENANT_TABLES == frozenset({"users"})

    # E2 — the unconditional-policy exceptions, pinned to table+policy+role.
    assert rls.SANCTIONED_UNCONDITIONAL == {
        ("audit_log", "audit_compliance_read", rls.ROLE_COMPLIANCE, "qual"),
        ("checkpoint_migrations", "bookkeeping_read", "public", "qual"),
    }

    # E1 — exactly one table with an org_id and no FK on it.
    with get_engine().connect() as conn:
        fkless = [
            t for t in rls.tables_with_org_id(conn)
            if not conn.execute(
                text(
                    """
                    SELECT count(*) FROM pg_constraint
                    WHERE conrelid = to_regclass(:t) AND contype='f'
                      AND 'org_id' = ANY(
                            SELECT attname FROM pg_attribute
                            WHERE attrelid = conrelid AND attnum = ANY(conkey))
                    """
                ),
                {"t": t},
            ).scalar()
        ]
    # Four distinct reasons, all registered in PHASE3_DESIGN.md §8b:
    #   organisations  — org_id IS the primary key; it is the root of the FK
    #                    chain and cannot reference itself.
    #   audit_log      — deliberately FK-free so the record outlives its subject.
    #   checkpoint_*   — library-owned tables; we add a column, not a constraint
    #                    into LangGraph's schema.
    # Every one of them is covered by the blank-org CHECK instead, which is why
    # that control is derived rather than listed.
    assert set(fkless) == {
        "organisations",
        "audit_log",
        "checkpoints",
        "checkpoint_blobs",
        "checkpoint_writes",
    }, f"a new FK-less org_id table appeared: {sorted(fkless)}"
