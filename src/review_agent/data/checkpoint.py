"""Bring LangGraph's checkpoint tables under the tenant-isolation core.

LangGraph's PostgresSaver creates its own tables holding graph state — which is
tenant data. They arrive with no org_id and no RLS, and the Phase 1 drift check
fails on them the moment `setup()` runs. That is the new-table trap working.

THE RESOLUTION IS RLS, NEVER AN EXEMPTION. Adding the table names to a skip-list
in tables_missing_rls() would invert the drift check into a curated allow-list
and leave tenant data un-policied. See PHASE1_DESIGN.md §2.6 and
PHASE3_DESIGN.md §1.3 / §9.

Both halves are needed and they do different jobs:
  * the org_id DEFAULT stamps each row with the writing tenant, and raises if
    there is no tenant — but says nothing about who can READ it;
  * the RLS policies restrict visibility — but say nothing about stamping.
The spike confirmed a correctly-stamped row is readable cross-tenant with RLS
off, so neither substitutes for the other.
"""

from __future__ import annotations

from sqlalchemy import text

from review_agent.data.models import (
    CHECKPOINT_BOOKKEEPING_TABLE,
    CHECKPOINT_STATE_TABLES,
)

def _match() -> str:
    """The canonical scope comparison, imported rather than restated.

    nullif collapses the two empty-scope states (never-set -> NULL,
    set-and-reverted -> '') into one that matches nothing. Imported from rls so
    a change there cannot leave these tables on an older expression.
    """
    from review_agent.data.rls import SCOPE_MATCH

    return SCOPE_MATCH


def checkpoint_tables_present(connection) -> list[str]:
    rows = connection.execute(
        text(
            """
            SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
            WHERE n.nspname='public' AND c.relkind='r' AND c.relname = ANY(:names)
            ORDER BY c.relname
            """
        ),
        {"names": list(CHECKPOINT_STATE_TABLES) + [CHECKPOINT_BOOKKEEPING_TABLE]},
    ).scalars().all()
    return list(rows)


def create_checkpoint_tables(owner_engine) -> None:
    """Create LangGraph's checkpoint tables. Part of PROVISIONING, not of tests.

    This was previously called only from a test fixture, which meant a fresh
    deployment had no checkpoint tables at all and the first review failed when
    the saver tried to write. That is a deployment bug rather than a missing
    command: everything looked provisioned, and the gap only appeared under the
    first real run.

    langgraph is imported lazily so the isolation core's module graph stays free
    of it — rls.py imports this module, and the red-team gate imports rls.
    """
    from langgraph.checkpoint.postgres import PostgresSaver

    with owner_engine.connect() as conn:
        raw = conn.connection.driver_connection
        # setup() manages its own transactions and expects autocommit.
        previous, raw.autocommit = raw.autocommit, True
        try:
            PostgresSaver(raw).setup()
        finally:
            raw.autocommit = previous

    # setup() is idempotent VIA `checkpoint_migrations`, which records how far it
    # has run. If the state tables are dropped but that bookkeeping table is
    # left behind, setup() reads "already migrated" and does NOTHING — silently.
    # Provisioning would then report success with nowhere to store graph state.
    # Verified rather than assumed, because the failure is quiet and the recovery
    # is non-obvious.
    with owner_engine.connect() as conn:
        created = set(checkpoint_tables_present(conn))
    missing = [t for t in CHECKPOINT_STATE_TABLES if t not in created]
    if missing:
        raise RuntimeError(
            f"PostgresSaver.setup() did not create {missing}. This happens when "
            f"{CHECKPOINT_BOOKKEEPING_TABLE} survives while the state tables are "
            "dropped: setup() consults it for idempotency and skips the work. "
            f"Drop {CHECKPOINT_BOOKKEEPING_TABLE} as well, then re-provision."
        )


def apply_checkpoint_isolation(connection) -> list[str]:
    """Add org_id + RLS to the checkpoint tables. Idempotent. Returns tables done.

    MUST RUN IMMEDIATELY AFTER PostgresSaver.setup(), BEFORE ANY RUN WRITES A
    CHECKPOINT. `SET NOT NULL` requires an empty table, and pre-existing
    checkpoints have no tenant — they must be purged before migrating, never
    backfilled with a guess.

    The three-step column form is not stylistic. The single-statement version,

        ADD COLUMN org_id text NOT NULL DEFAULT current_setting('app.current_org')

    fails with `unrecognized configuration parameter`, because the default is
    evaluated during the ALTER in a session where the GUC was never set, and a
    custom GUC read without missing_ok raises. Confirmed by the Phase 3 spike.
    """
    if not connection.in_transaction():
        # The migration lowers RLS on each table before restoring it. DDL is
        # transactional in Postgres, so a failure anywhere in between rolls the
        # DISABLE back with everything else — but ONLY if there is a transaction
        # to roll back. Running this on an autocommit connection would leave a
        # table with RLS lowered if a later statement failed, which is the exact
        # window this check exists to make impossible.
        raise RuntimeError(
            "apply_checkpoint_isolation must run inside a transaction: it lowers "
            "RLS mid-migration, and only a transaction makes that window "
            "crash-safe."
        )

    present = set(checkpoint_tables_present(connection))
    done: list[str] = []

    for table in CHECKPOINT_STATE_TABLES:
        if table not in present:
            continue
        # Lower RLS for the duration of the migration. FORCE binds the owner
        # too, so with the session variable unset every policy matches nothing —
        # and the NULL-row cleanup below would silently delete zero rows before
        # SET NOT NULL failed. Re-enabled a few lines down; the table is
        # unreachable by review_app throughout, which has no DDL rights.
        connection.execute(text(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY"))
        connection.execute(
            text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS org_id text")
        )
        # No missing_ok: an unscoped checkpoint write must RAISE, not write a
        # NULL-org row that belongs to nobody.
        connection.execute(
            text(
                f"ALTER TABLE {table} ALTER COLUMN org_id "
                f"SET DEFAULT current_setting('app.current_org')"
            )
        )
        # Rows that predate the column have no tenant, and a tenant cannot be
        # inferred for them — guessing would attribute one org's graph state to
        # another. They are also already dead: with RLS on, a NULL org_id matches
        # no session variable, so no tenant can ever read them. Delete rather
        # than backfill. In a correctly ordered deployment this removes nothing,
        # because the migration runs before the first run writes a checkpoint.
        connection.execute(text(f"DELETE FROM {table} WHERE org_id IS NULL OR org_id=''"))
        connection.execute(text(f"ALTER TABLE {table} ALTER COLUMN org_id SET NOT NULL"))

        # NOT NULL IS NOT ENOUGH, and the reason is subtle enough to be worth
        # stating. `current_setting('app.current_org')` raises only while the
        # parameter has NEVER been set on that connection. Once a scoped session
        # has run on it — which, on a pooled connection, is almost always — the
        # parameter stays DEFINED and reverts to the EMPTY STRING at commit. So
        # an unscoped write on a recycled connection does not raise: it stamps
        # org_id = '', which is not NULL, passes the NOT NULL check, and creates
        # a row invisible to every tenant that quietly accumulates forever.
        #
        # The spike missed this because it used fresh connections. This CHECK is
        # what makes the "unscoped write fails loudly" guarantee unconditional.
        constraint = f"{table}_org_id_not_blank"
        connection.execute(
            text(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {constraint}")
        )
        connection.execute(
            text(f"ALTER TABLE {table} ADD CONSTRAINT {constraint} "
                 f"CHECK (org_id <> '')")
        )
        connection.execute(text(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY"))
        connection.execute(text(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY"))

        for policy in ("org_isolation_select", "org_isolation_insert",
                       "org_isolation_update", "org_isolation_delete"):
            connection.execute(text(f"DROP POLICY IF EXISTS {policy} ON {table}"))
        connection.execute(
            text(f"CREATE POLICY org_isolation_select ON {table} "
                 f"FOR SELECT USING ({_match()})")
        )
        connection.execute(
            text(f"CREATE POLICY org_isolation_insert ON {table} "
                 f"FOR INSERT WITH CHECK ({_match()})")
        )
        connection.execute(
            text(f"CREATE POLICY org_isolation_update ON {table} "
                 f"FOR UPDATE USING ({_match()}) WITH CHECK ({_match()})")
        )
        # The purge deletes checkpoints for terminal runs; it does so as the
        # owner, but the policy keeps a scoped delete confined to its own org.
        connection.execute(
            text(f"CREATE POLICY org_isolation_delete ON {table} "
                 f"FOR DELETE USING ({_match()})")
        )
        connection.execute(
            text(f'GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO "review_app"')
        )
        done.append(table)

    if CHECKPOINT_BOOKKEEPING_TABLE in present:
        # Library bookkeeping: a migration version number, no tenant data. It
        # still gets RLS so the drift check needs no exemption, with a read-only
        # unconditional policy — the single sanctioned entry in
        # rls.SANCTIONED_UNCONDITIONAL for checkpoints.
        t = CHECKPOINT_BOOKKEEPING_TABLE
        connection.execute(text(f"ALTER TABLE {t} ENABLE ROW LEVEL SECURITY"))
        connection.execute(text(f"ALTER TABLE {t} FORCE ROW LEVEL SECURITY"))
        connection.execute(text(f"DROP POLICY IF EXISTS bookkeeping_read ON {t}"))
        connection.execute(
            text(f"CREATE POLICY bookkeeping_read ON {t} FOR SELECT USING (true)")
        )
        connection.execute(text(f'GRANT SELECT ON {t} TO "review_app"'))
        done.append(t)

    return done


# --- purge -------------------------------------------------------------------

# The TTL exceeds no SLA, because the SAO has not defined one — see
# PHASE3_DESIGN.md §8. Until it does, the purge REFUSES to touch any run still
# awaiting a human, whatever its age.
CHECKPOINT_TTL_DAYS = 30
PROTECTED_STATUSES = ("running", "awaiting_review")


def purge_checkpoints(scope, run_ids: list[str] | None = None) -> dict:
    """Delete checkpoints for TERMINAL runs. Refuse, and audit, for live ones.

    A sweep that removes an in-flight review destroys work irrecoverably: the
    reviewer's queue survives (findings are persisted before the interrupt), but
    the run can never be resumed, and nothing about the deletion is visible
    afterwards. So the age check is NOT the only gate — status is.

    Returns {"purged": [...], "refused": [...]}.
    """
    from review_agent.data.db import scoped_session
    from review_agent.data.repository import record_audit

    purged: list[str] = []
    refused: list[str] = []

    with scoped_session(scope) as session:
        params: dict = {"ttl": CHECKPOINT_TTL_DAYS}
        sql = (
            "SELECT run_id, status FROM review_runs "
            "WHERE updated_at < now() - make_interval(days => :ttl)"
        )
        if run_ids is not None:
            sql = "SELECT run_id, status FROM review_runs WHERE run_id = ANY(:ids)"
            params = {"ids": run_ids}

        for row in session.execute(text(sql), params).mappings():
            if row["status"] in PROTECTED_STATUSES:
                refused.append(row["run_id"])
                continue
            for table in CHECKPOINT_STATE_TABLES:
                session.execute(
                    text(f"DELETE FROM {table} WHERE thread_id = :t"),
                    {"t": row["run_id"]},
                )
            purged.append(row["run_id"])

        if purged or refused:
            record_audit(
                session,
                scope,
                action="checkpoint.purged",
                detail={
                    "purged": purged,
                    "refused": refused,
                    "refused_reason": (
                        "run still awaiting human review; no SAO SLA is defined, "
                        "so live runs are never swept"
                    ) if refused else None,
                    "ttl_days": CHECKPOINT_TTL_DAYS,
                },
            )

    return {"purged": purged, "refused": refused}


def checkpoint_tables_unmigrated(connection) -> list[str]:
    """Checkpoint tables that exist but are not under tenant isolation.

    THIS IS THE STRUCTURAL VERSION OF THE ORDERING CONSTRAINT. Relying on
    "run the migration at the right moment" is relying on someone remembering, on
    a path where forgetting means graph state accumulates with no tenant column
    and no policy — and once rows exist, SET NOT NULL can no longer be applied,
    so the mistake is not even cheaply reversible.

    Wired into verify_isolation(), so the app refuses to start rather than
    serving with unisolated checkpoints.
    """
    problems: list[str] = []
    present = set(checkpoint_tables_present(connection))

    # ABSENT is not the same as CLEAN, and returning [] for both is the silent
    # no-op shape this project keeps finding. Now that provisioning creates these
    # tables, their absence means graph state has nowhere isolated to live — the
    # first review would fail, or worse, a later `setup()` by some other path
    # would create them UNMIGRATED and rows would land before anyone noticed.
    missing = [t for t in CHECKPOINT_STATE_TABLES if t not in present]
    if missing:
        return [
            f"checkpoint tables are absent: {missing}. Provisioning creates them "
            "(data.checkpoint.create_checkpoint_tables); without them graph state "
            "has no isolated home."
        ]

    for table in CHECKPOINT_STATE_TABLES:
        column = connection.execute(
            text(
                """
                SELECT a.attnotnull,
                       pg_get_expr(d.adbin, d.adrelid) AS default_expr
                FROM   pg_attribute a
                LEFT JOIN pg_attrdef d ON d.adrelid=a.attrelid AND d.adnum=a.attnum
                WHERE  a.attrelid = to_regclass(:t) AND a.attname='org_id'
                  AND  a.attnum > 0 AND NOT a.attisdropped
                """
            ),
            {"t": table},
        ).first()

        if column is None:
            problems.append(f"{table} has no org_id column")
            continue
        if not column.attnotnull:
            problems.append(f"{table}.org_id is nullable")
        if not column.default_expr or "current_setting" not in column.default_expr:
            problems.append(f"{table}.org_id has no current_setting default")
        elif "true" in column.default_expr.replace(" ", ""):
            # missing_ok would turn an unscoped write into a silent NULL-org row.
            problems.append(f"{table}.org_id default uses missing_ok; it must raise")

    return sorted(problems)
