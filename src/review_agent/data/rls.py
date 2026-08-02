"""Row-level security — THE isolation guarantee.

Isolation is enforced HERE, at the database, not in application logic and never
by prompting the model. A session scoped to org-a must be physically unable to
read org-b rows, even if the application (or an attacker) asks for them.

This module owns: the roles, the RLS policies, the grants, the append-only and
consistency triggers, the one sanctioned cross-org compliance path, and the drift
checks that prove the mechanism is still switched on.

See docs/PHASE1_DESIGN.md §2 and §1.10.
"""

from __future__ import annotations

import hashlib

from sqlalchemy import text

from review_agent.data.checkpoint import checkpoint_tables_unmigrated
from review_agent.data.models import NON_TENANT_TABLES, TENANT_TABLES

# --- roles ------------------------------------------------------------------
ROLE_OWNER = "review_owner"          # owns tables, runs migrations
ROLE_APP = "review_app"              # every runtime query; owns nothing
ROLE_AUTH = "review_auth"            # reads `users`, nothing else
ROLE_COMPLIANCE = "review_compliance"  # offline cross-org audit METADATA only

RUNTIME_ROLES = (ROLE_OWNER, ROLE_APP, ROLE_AUTH, ROLE_COMPLIANCE)

# The session variable the policies read. This name must appear NOWHERE else in
# src/ except this module and db.py — asserted by a lint test.
ORG_GUC = "app.current_org"

# Columns the compliance role may read across orgs. Everything else on audit_log
# — notably `detail` and `retrieved_ids` — is unreachable, not merely unselected.
AUDIT_METADATA_COLUMNS = (
    "audit_id",
    "org_id",
    "project_id",
    "user_id",
    "action",
    "rulebook_version",
    "occurred_at",
)

# The ONE sanctioned unconditional expression in the whole schema (design §1.10):
# the compliance role's cross-org READ of audit metadata. Pinned to the exact
# table, policy, role AND expression column, so it cannot widen into a write.
SANCTIONED_UNCONDITIONAL = {
    ("audit_log", "audit_compliance_read", ROLE_COMPLIANCE, "qual"),
    # LangGraph bookkeeping: a schema version number, no tenant data. RLS is on
    # so the drift check needs no table exemption; the policy is unconditional
    # because there is nothing to scope. Pinned to this exact table/policy.
    ("checkpoint_migrations", "bookkeeping_read", "public", "qual"),
}

# Which expression a policy MUST define, per command. Note UPDATE and ALL are
# absent from the with_check side: Postgres reuses `qual` as the check when
# WITH CHECK is omitted, so a NULL there is safe. An explicitly weak one is not.
# The canonical scope comparison. Defined ONCE so policies, the checkpoint
# migration, and the tests that restore after a mutation cannot drift apart —
# they did, and stale copies of an older expression silently reintroduced the
# empty-scope pseudo-tenant.
SCOPE_MATCH = f"org_id = nullif(current_setting('{ORG_GUC}', true), '')"

_REQUIRES_QUAL = frozenset({"SELECT", "UPDATE", "DELETE", "ALL"})
_REQUIRES_WITH_CHECK = frozenset({"INSERT"})


class IsolationVerificationError(RuntimeError):
    """RLS is not in the state this design requires. Never continue past this.

    A running service with broken isolation is worse than an outage, so the
    correct response to this at boot is to refuse to start.
    """


# --- provisioning (runs as superuser) ---------------------------------------

def _quote_literal(value: str) -> str:
    """Quote a string literal for a utility statement.

    CREATE/ALTER ROLE are utility statements: Postgres does not accept bind
    parameters in them, so the password has to be composed into the SQL text.
    These values come from the deployment environment, not from any request, but
    the escaping and the character check below are non-negotiable anyway — this
    is the one place in the codebase that builds SQL by concatenation.
    """
    if "\x00" in value or "\\" in value:
        raise ValueError("role password may not contain backslashes or null bytes")
    return "'" + value.replace("'", "''") + "'"


def provision_roles(connection, passwords: dict[str, str]) -> None:
    """Create the four roles. Superuser-only; run once per database.

    None of them are superusers and none have BYPASSRLS — asserted later by
    roles_with_bypass(). `review_app` deliberately owns nothing: in Postgres the
    table owner is exempt from its own RLS by default, so an app connecting as
    the owner has no isolation at all.
    """
    existing = set(
        connection.execute(
            text("SELECT rolname FROM pg_roles WHERE rolname = ANY(:roles)"),
            {"roles": list(RUNTIME_ROLES)},
        ).scalars()
    )

    for role in RUNTIME_ROLES:
        # Role names are module constants, never user input.
        if role not in existing:
            connection.execute(text(f'CREATE ROLE "{role}" LOGIN'))
        # Idempotent, and re-asserted on every provision so a hand-edited role
        # cannot quietly acquire privileges.
        connection.execute(
            text(
                f'ALTER ROLE "{role}" NOSUPERUSER NOBYPASSRLS NOCREATEDB '
                f"NOCREATEROLE LOGIN PASSWORD {_quote_literal(passwords[role])}"
            )
        )

    connection.execute(text(f'GRANT CREATE, USAGE ON SCHEMA public TO "{ROLE_OWNER}"'))
    for role in (ROLE_APP, ROLE_AUTH, ROLE_COMPLIANCE):
        connection.execute(text(f'GRANT USAGE ON SCHEMA public TO "{role}"'))


# --- the policies ------------------------------------------------------------

def apply_rls_policies(connection) -> None:
    """Enable RLS + install isolation policies on all tenant tables.

    Idempotent. Run as the table owner, after the schema exists.
    """
    for table in TENANT_TABLES:
        connection.execute(text(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY"))
        # FORCE so the policies bind the table OWNER too — a migration script or
        # an ops console session cannot read across tenants either.
        connection.execute(text(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY"))

        for policy in ("org_isolation_select", "org_isolation_insert",
                       "org_isolation_update", "audit_compliance_read"):
            connection.execute(text(f"DROP POLICY IF EXISTS {policy} ON {table}"))

        # THERE ARE TWO EMPTY-SCOPE STATES, and they are not the same state.
        #   * GUC never set on this connection  -> current_setting(...) is NULL
        #   * GUC set and reverted (the normal case on a POOLED connection)
        #     -> the parameter stays DEFINED and reverts to the EMPTY STRING
        # Phase 1 only tested the first. nullif() collapses the second into the
        # first, so `org_id = NULL` is not true and both fail CLOSED. Without it
        # an empty scope matches any row stamped '' — a shared pseudo-tenant that
        # any recycled connection could read. The CHECK constraint below stops
        # such a row existing; this stops it matching even if one did.
        match = SCOPE_MATCH

        connection.execute(
            text(f"CREATE POLICY org_isolation_select ON {table} "
                 f"FOR SELECT USING ({match})")
        )
        # WITH CHECK on writes matters as much as USING on reads: without it,
        # org-a could INSERT a row stamped org-b — invisible to org-a, but it
        # would surface in org-b's view. That is a write-direction leak.
        connection.execute(
            text(f"CREATE POLICY org_isolation_insert ON {table} "
                 f"FOR INSERT WITH CHECK ({match})")
        )
        if table != "audit_log":  # append-only: no UPDATE path at all
            connection.execute(
                text(f"CREATE POLICY org_isolation_update ON {table} "
                     f"FOR UPDATE USING ({match}) WITH CHECK ({match})")
            )

    _apply_compliance_path(connection)


def _apply_blank_org_guard(connection, table: str) -> None:
    """Make the empty string unplantable as an org_id.

    Generalised from the checkpoint fix (PHASE3_DESIGN.md Finding 3). An empty
    scope is not merely "sees nothing": without this, a connection whose scope
    has been set and reverted can INSERT a row stamped '' and any other such
    connection can read it back. That is a shared pseudo-tenant, and it is
    cross-tenant leakage by a different route than the one Phase 1 tested.
    """
    constraint = f"{table}_org_id_not_blank"
    connection.execute(
        text(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {constraint}")
    )
    connection.execute(
        text(f"ALTER TABLE {table} ADD CONSTRAINT {constraint} CHECK (org_id <> '')")
    )


def _apply_compliance_path(connection) -> None:
    """The ONE deliberate cross-org read: audit metadata, for compliance (§1.10).

    Note the shape. The view alone cannot do this job: under FORCE ROW LEVEL
    SECURITY the view's owner is itself bound by the audit_log policies, so a
    `security_invoker = false` view would return zero rows. So the exception is
    expressed where it is visible and auditable — a role-targeted policy plus
    COLUMN-LEVEL grants — and the view is merely the ergonomic surface over it.
    """
    connection.execute(
        text(
            f"CREATE POLICY audit_compliance_read ON audit_log "
            f'FOR SELECT TO "{ROLE_COMPLIANCE}" USING (true)'
        )
    )
    cols = ", ".join(AUDIT_METADATA_COLUMNS)
    connection.execute(text("DROP VIEW IF EXISTS audit_log_metadata"))
    connection.execute(
        text(
            f"CREATE VIEW audit_log_metadata WITH (security_invoker = true) AS "
            f"SELECT {cols} FROM audit_log"
        )
    )
    # Column-level privilege is the real enforcement: `detail` and
    # `retrieved_ids` are not merely absent from the view, they are ungrantable
    # through it, and `SELECT * FROM audit_log` is denied outright.
    connection.execute(
        text(f'GRANT SELECT ({cols}) ON audit_log TO "{ROLE_COMPLIANCE}"')
    )
    connection.execute(
        text(f'GRANT SELECT ON audit_log_metadata TO "{ROLE_COMPLIANCE}"')
    )


def apply_grants(connection) -> None:
    """Least privilege for each role. Run as the table owner, after create_all."""
    connection.execute(text("REVOKE ALL ON ALL TABLES IN SCHEMA public FROM PUBLIC"))
    # ROLE_COMPLIANCE is revoked here and re-granted by _apply_compliance_path,
    # which runs after this. That ordering makes apply_grants + apply_rls_policies
    # a complete RESTORE of the privilege state, not just an additive patch — the
    # mutation tests and the repair fixture depend on that.
    for role in (ROLE_APP, ROLE_AUTH, ROLE_COMPLIANCE):
        connection.execute(
            text(f'REVOKE ALL ON ALL TABLES IN SCHEMA public FROM "{role}"')
        )

    connection.execute(
        text(f'GRANT SELECT, INSERT, UPDATE ON artifacts, findings, projects, '
             f'review_runs TO "{ROLE_APP}"')
    )
    connection.execute(text(f'GRANT SELECT ON organisations TO "{ROLE_APP}"'))
    # Append-only by privilege: no UPDATE, no DELETE. Ever.
    connection.execute(text(f'GRANT SELECT, INSERT ON audit_log TO "{ROLE_APP}"'))
    connection.execute(
        text(f'GRANT USAGE, SELECT ON SEQUENCE audit_log_audit_id_seq TO "{ROLE_APP}"')
    )
    # review_app gets NO grant of any kind on users. The auth path is applied by
    # apply_all AFTER this function, because this one revokes from review_auth.


AUTH_FUNCTION = "resolve_user_scope"

# The body is a constant so it can be HASHED. Privileges on this function were
# checked while its definition was not — making it the only control in the
# system invisible to the drift regime that guards everything else. Anyone with
# review_owner could have rewritten the body to return an arbitrary org while
# every privilege check stayed green.
AUTH_FUNCTION_BODY = """
                SELECT u.user_id, u.org_id
                FROM   public.users u
                WHERE  u.user_id = p_subject AND u.active
            """


def _apply_auth_path(connection) -> None:
    """The ONLY interface to `users`: one function, one subject, one row.

    `users` is the single table without RLS — it is what establishes tenancy, so
    a policy on it would need the org variable that reading it produces. The
    original compensating control was a column grant plus "one permitted query
    shape", and that second half was a CONVENTION held by the calling code, not a
    control. It did not cover cross-tenant READS: with the column grant,
    review_auth could

        SELECT org_id, count(*) FROM users GROUP BY org_id

    which enumerates every organisation that exists, every subject, and who
    belongs to where. The access matrix (§1.10) treats metadata as its own
    permission level — PMO admin gets cross-org audit metadata as a REGISTERED
    exception with four compensating controls — while this handed cross-org
    membership metadata to an unregistered role for free.

    So the query shape is now structural. review_auth holds no table privilege at
    all; it may execute one SECURITY DEFINER function that takes a subject and
    returns at most one row. You must already know a subject to learn its org,
    and you learn nothing about any other org.
    """
    connection.execute(text(f'REVOKE ALL ON users FROM "{ROLE_AUTH}"'))
    connection.execute(
        text(
            f"""
            CREATE OR REPLACE FUNCTION {AUTH_FUNCTION}(p_subject text)
            RETURNS TABLE (user_id text, org_id text)
            LANGUAGE sql
            STABLE
            SECURITY DEFINER
            -- Pinned search_path: a SECURITY DEFINER function without one can be
            -- redirected to an attacker-controlled `users` earlier in the path.
            SET search_path = pg_catalog, public
            AS $fn${AUTH_FUNCTION_BODY}$fn$
            """
        )
    )
    # EXECUTE defaults to PUBLIC on new functions — revoke before granting, or
    # every role gains a subject->org oracle.
    connection.execute(
        text(f"REVOKE ALL ON FUNCTION {AUTH_FUNCTION}(text) FROM PUBLIC")
    )
    connection.execute(
        text(f'GRANT EXECUTE ON FUNCTION {AUTH_FUNCTION}(text) TO "{ROLE_AUTH}"')
    )


def apply_triggers(connection) -> None:
    """Append-only enforcement + org-consistency for the denormalised org_id.

    The consistency checks are DEFERRABLE CONSTRAINT triggers on purpose: they
    fire at COMMIT, which leaves the RLS WITH CHECK policy as the first thing an
    attacker hits. RLS is the boundary; these triggers only stop honest drift.
    """
    connection.execute(
        text(
            """
            CREATE OR REPLACE FUNCTION raise_append_only() RETURNS trigger
            LANGUAGE plpgsql AS $$
            BEGIN
                RAISE EXCEPTION 'audit_log is append-only: % is not permitted', TG_OP;
            END $$;
            """
        )
    )
    connection.execute(text("DROP TRIGGER IF EXISTS audit_log_no_mutate ON audit_log"))
    connection.execute(
        text(
            "CREATE TRIGGER audit_log_no_mutate BEFORE UPDATE OR DELETE ON audit_log "
            "FOR EACH ROW EXECUTE FUNCTION raise_append_only()"
        )
    )
    connection.execute(
        text("DROP TRIGGER IF EXISTS audit_log_no_truncate ON audit_log")
    )
    connection.execute(
        text(
            "CREATE TRIGGER audit_log_no_truncate BEFORE TRUNCATE ON audit_log "
            "FOR EACH STATEMENT EXECUTE FUNCTION raise_append_only()"
        )
    )

    connection.execute(
        text(
            """
            CREATE OR REPLACE FUNCTION enforce_artifact_org() RETURNS trigger
            LANGUAGE plpgsql AS $$
            DECLARE parent_org text;
            BEGIN
                SELECT org_id INTO parent_org FROM projects
                 WHERE project_id = NEW.project_id;
                IF parent_org IS NULL THEN
                    RAISE EXCEPTION 'project % is not visible in this scope',
                                    NEW.project_id;
                END IF;
                IF parent_org <> NEW.org_id THEN
                    RAISE EXCEPTION 'artifact org_id % does not match project org %',
                                    NEW.org_id, parent_org;
                END IF;
                RETURN NULL;
            END $$;
            """
        )
    )
    connection.execute(
        text(
            """
            CREATE OR REPLACE FUNCTION enforce_finding_org() RETURNS trigger
            LANGUAGE plpgsql AS $$
            DECLARE parent_org text; run_org text; superseding_org text;
            BEGIN
                SELECT org_id INTO parent_org FROM artifacts
                 WHERE artifact_id = NEW.artifact_id;
                IF parent_org IS NULL THEN
                    RAISE EXCEPTION 'artifact % is not visible in this scope',
                                    NEW.artifact_id;
                END IF;
                IF parent_org <> NEW.org_id THEN
                    RAISE EXCEPTION 'finding org_id % does not match artifact org %',
                                    NEW.org_id, parent_org;
                END IF;
                -- run_id and superseded_by_run_id are FKs to a TENANT-SCOPED
                -- table, and a foreign key does not check tenancy: it would
                -- happily accept another org's run_id. Checked here for the same
                -- reason artifact_id is — a finding must not be attributable to,
                -- or retired by, a run belonging to someone else.
                SELECT org_id INTO run_org FROM review_runs
                 WHERE run_id = NEW.run_id;
                IF run_org IS NULL THEN
                    RAISE EXCEPTION 'run % is not visible in this scope', NEW.run_id;
                END IF;
                IF run_org <> NEW.org_id THEN
                    RAISE EXCEPTION 'finding org_id % does not match run org %',
                                    NEW.org_id, run_org;
                END IF;
                IF NEW.superseded_by_run_id IS NOT NULL THEN
                    SELECT org_id INTO superseding_org FROM review_runs
                     WHERE run_id = NEW.superseded_by_run_id;
                    IF superseding_org IS NULL THEN
                        RAISE EXCEPTION 'superseding run % is not visible in this scope',
                                        NEW.superseded_by_run_id;
                    END IF;
                    IF superseding_org <> NEW.org_id THEN
                        RAISE EXCEPTION
                            'finding org_id % does not match superseding run org %',
                            NEW.org_id, superseding_org;
                    END IF;
                END IF;
                RETURN NULL;
            END $$;
            """
        )
    )
    for table, fn in (("artifacts", "enforce_artifact_org"),
                      ("findings", "enforce_finding_org")):
        trig = f"{table}_org_consistency"
        connection.execute(text(f"DROP TRIGGER IF EXISTS {trig} ON {table}"))
        connection.execute(
            text(
                f"CREATE CONSTRAINT TRIGGER {trig} AFTER INSERT OR UPDATE ON {table} "
                f"DEFERRABLE INITIALLY DEFERRED "
                f"FOR EACH ROW EXECUTE FUNCTION {fn}()"
            )
        )


# --- THE derivation ----------------------------------------------------------
# Every enumeration of tenant tables comes from here. Maintained lists have now
# caused three bugs in this repo — the repair fixture under-repaired twice, and
# truncate_all silently skipped the checkpoint tables — because each parallel
# list drifted independently of the schema. A derived list cannot drift: a table
# added in a later phase is in it the moment it exists.

def scoped_tables(connection) -> list[str]:
    """Every base table in the schema that is NOT a documented exception.

    Deliberately "everything minus the exceptions" rather than "these tables":
    that is what makes an unregistered new table FAIL the gate rather than
    silently miss a control. Same property as tables_missing_rls().
    """
    rows = connection.execute(
        text(
            """
            SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
            WHERE n.nspname='public' AND c.relkind='r' ORDER BY c.relname
            """
        )
    ).scalars().all()
    return [t for t in rows if t not in NON_TENANT_TABLES]


def tables_with_org_id(connection) -> list[str]:
    """Every base table carrying an org_id column, derived from the catalogue.

    Includes `users` — it has an org_id, so the blank guard applies to it, and
    covering it removes an exception rather than adding one.
    """
    rows = connection.execute(
        text(
            """
            SELECT DISTINCT c.relname
            FROM   pg_class c
            JOIN   pg_namespace n ON n.oid = c.relnamespace
            JOIN   pg_attribute a ON a.attrelid = c.oid
            WHERE  n.nspname='public' AND c.relkind='r'
              AND  a.attname='org_id' AND a.attnum > 0 AND NOT a.attisdropped
            ORDER BY c.relname
            """
        )
    ).scalars().all()
    return list(rows)


# --- drift detection ---------------------------------------------------------
# A tenant table that loses RLS leaks everything and raises no error: queries just
# quietly start returning more rows. These checks turn that into a build failure.

def tables_missing_rls(connection) -> list[str]:
    """Tables that should have RLS enabled AND forced, but don't.

    Written as "every table except the known non-tenant ones", never as an
    allow-list of tables to check. A Phase 2 developer who adds a table without
    RLS breaks the build immediately; an allow-list would silently accept it.
    """
    rows = connection.execute(
        text(
            """
            SELECT c.relname
            FROM   pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE  n.nspname = 'public' AND c.relkind = 'r'
              AND  NOT (c.relrowsecurity AND c.relforcerowsecurity)
            ORDER BY c.relname
            """
        )
    ).scalars().all()
    return [t for t in rows if t not in NON_TENANT_TABLES]


def tenant_tables_without_policies(connection) -> list[str]:
    """Tenant tables carrying no policy at all (RLS on but nothing defined)."""
    have = connection.execute(
        text("SELECT DISTINCT tablename FROM pg_policies WHERE schemaname = 'public'")
    ).scalars().all()
    return sorted(set(TENANT_TABLES) - set(have))


def unconditional_policies(connection) -> list[tuple[str, str, str, str, str]]:
    """Policies that constrain nothing, minus the one sanctioned exception.

    `qual` (the read side) and `with_check` (the write side) are inspected
    INDEPENDENTLY, never collapsed together.

    An earlier version of this check used COALESCE(qual, with_check), which was
    wrong in a way that mattered: for UPDATE and ALL policies both columns can be
    populated, so a policy with a sound USING but a weakened `WITH CHECK (true)`
    read as healthy — COALESCE returned the good `qual` and the broken write side
    was never examined. That is precisely the write-direction leak the design is
    most worried about (org-a stamping a row into org-b's view), hidden by the
    check meant to catch it.

    Returns (table, policy, roles, column, reason) tuples; empty means healthy.
    """
    rows = connection.execute(
        text(
            """
            SELECT tablename, policyname, cmd, qual, with_check,
                   COALESCE(array_to_string(roles, ','), '') AS roles
            FROM   pg_policies WHERE schemaname = 'public'
            """
        )
    ).all()

    offenders: list[tuple[str, str, str, str, str]] = []
    for r in rows:
        sides = (
            ("qual", r.qual, r.cmd in _REQUIRES_QUAL),
            ("with_check", r.with_check, r.cmd in _REQUIRES_WITH_CHECK),
        )
        for column, expression, required in sides:
            entry = (r.tablename, r.policyname, r.roles, column)
            if entry in SANCTIONED_UNCONDITIONAL:
                continue
            if expression is None:
                if required:
                    offenders.append((*entry, f"missing on {r.cmd}"))
            elif expression.strip().lower() == "true":
                offenders.append((*entry, f"unconditional on {r.cmd}"))
    return offenders


def check_connection_privileges(
    connection, expected_role: str = ROLE_APP
) -> list[str]:
    """Problems with the privileges of THIS connection. Empty means safe.

    Shared by the red-team suite's precondition fixture and by the test that
    proves that fixture actually rejects a privileged connection — so the guard
    itself is tested rather than trusted.
    """
    problems: list[str] = []

    current_user = connection.execute(text("SELECT current_user")).scalar()
    if current_user != expected_role:
        problems.append(f"connected as {current_user!r}, expected {expected_role!r}")

    # current_user's real attributes, not the connection string: a SET ROLE, or a
    # URL that disagrees with the authenticated role, makes a username check
    # meaningless on its own.
    attrs = connection.execute(
        text("SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user")
    ).one()
    if attrs.rolsuper:
        problems.append(f"{current_user!r} is a superuser")
    if attrs.rolbypassrls:
        problems.append(f"{current_user!r} has BYPASSRLS")

    # A role can INHERIT the bypass from a role it is a member of while its own
    # attribute reads false.
    inherited = connection.execute(
        text(
            """
            SELECT count(*) FROM pg_roles g
            WHERE (g.rolsuper OR g.rolbypassrls)
              AND pg_has_role(current_user, g.oid, 'USAGE')
            """
        )
    ).scalar()
    if inherited:
        problems.append(f"{current_user!r} inherits superuser/BYPASSRLS from a granted role")

    if connection.execute(text("SHOW row_security")).scalar() != "on":
        problems.append("row_security is not on for this session")

    return problems


def roles_with_bypass(connection) -> list[str]:
    """Runtime roles that are superuser or can bypass RLS, directly or inherited.

    Checking a role's own attributes is not enough: a role can INHERIT BYPASSRLS
    from a role it is a member of while its own rolbypassrls is false.
    """
    rows = connection.execute(
        text(
            """
            SELECT r.rolname
            FROM   pg_roles r
            WHERE  r.rolname = ANY(:roles)
              AND  EXISTS (
                     SELECT 1 FROM pg_roles g
                     WHERE (g.rolsuper OR g.rolbypassrls)
                       AND pg_has_role(r.rolname, g.oid, 'USAGE')
                   )
            ORDER BY r.rolname
            """
        ),
        {"roles": list(RUNTIME_ROLES)},
    ).scalars().all()
    return list(rows)


def compliance_role_overreach(connection) -> list[str]:
    """Ways the ONE sanctioned cross-org exception (§1.10) has widened.

    The compliance path is the only thing in this system that reads across orgs,
    so it is the one privilege boundary that can be widened without tripping any
    of the RLS checks: a single `GRANT SELECT ON audit_log TO review_compliance`
    turns a seven-column metadata window into full cross-tenant read of every
    audit `detail` payload, and nothing above would notice.

    Permitted, exhaustively: SELECT on the `audit_log_metadata` view, and SELECT
    on exactly the AUDIT_METADATA_COLUMNS of `audit_log`. Anything else is drift.

    Deliberately uses has_table_privilege / has_column_privilege rather than the
    information_schema.role_*_grants views. Those views only expose grants where
    the CURRENT user is the grantor, the grantee, or a member of the grantee —
    so this check, run as `review_app`, saw an empty result set whether the
    compliance role was correctly scoped or granted the whole table. It reported
    "healthy" unconditionally. The pg_catalog functions are visible to any role
    and answer about the named role rather than the caller.
    """
    problems: list[str] = []
    writes = ("INSERT", "UPDATE", "DELETE", "TRUNCATE", "REFERENCES", "TRIGGER")

    def has_table(table: str, privilege: str) -> bool:
        return connection.execute(
            text("SELECT has_table_privilege(:role, :table, :privilege)"),
            {"role": ROLE_COMPLIANCE, "table": table, "privilege": privilege},
        ).scalar()

    def has_column(table: str, column: str, privilege: str) -> bool:
        return connection.execute(
            text("SELECT has_column_privilege(:role, :table, :column, :privilege)"),
            {
                "role": ROLE_COMPLIANCE,
                "table": table,
                "column": column,
                "privilege": privilege,
            },
        ).scalar()

    # 1. No access of any kind to tenant CONTENT.
    content_tables = tuple(t for t in TENANT_TABLES if t != "audit_log")
    for table in content_tables + tuple(sorted(NON_TENANT_TABLES)):
        for privilege in ("SELECT", *writes):
            if has_table(table, privilege):
                problems.append(f"{privilege} on {table}")

    # 2. No TABLE-level privilege on audit_log — column grants only.
    for privilege in ("SELECT", *writes):
        if has_table("audit_log", privilege):
            problems.append(f"table-level {privilege} on audit_log")

    # 3. Exactly the metadata columns are readable, and nothing is writable.
    columns = connection.execute(
        text(
            """
            SELECT attname FROM pg_attribute
            WHERE  attrelid = 'audit_log'::regclass AND attnum > 0
              AND  NOT attisdropped
            ORDER BY attnum
            """
        )
    ).scalars().all()
    for column in columns:
        readable = has_column("audit_log", column, "SELECT")
        if column in AUDIT_METADATA_COLUMNS and not readable:
            # The capability must EXIST, not just be constrained: a check that
            # only proved denial would pass with the path broken entirely, and
            # nobody would find out until an auditor asked.
            problems.append(f"compliance path broken: cannot read audit_log.{column}")
        elif column not in AUDIT_METADATA_COLUMNS and readable:
            problems.append(f"SELECT on audit_log.{column}, which is not metadata")
        for privilege in ("INSERT", "UPDATE"):
            if has_column("audit_log", column, privilege):
                problems.append(f"{privilege} on audit_log.{column} (must be read-only)")

    # 4. The view is readable and read-only.
    if not has_table("audit_log_metadata", "SELECT"):
        problems.append("compliance path broken: cannot read audit_log_metadata")
    for privilege in writes:
        if has_table("audit_log_metadata", privilege):
            problems.append(f"{privilege} on audit_log_metadata (must be read-only)")

    return sorted(problems)


def tables_missing_blank_org_guard(connection) -> list[str]:
    """Tables carrying an org_id where '' is still plantable.

    DERIVED over tables_with_org_id(), not checked against a maintained list.
    The list-based version passed a table created outside the ORM metadata while
    tables_missing_rls() correctly failed it — one gate caught the new table and
    the other waved it through, which is the worst possible split.
    """
    problems = []
    for table in tables_with_org_id(connection):
        found = connection.execute(
            text(
                """
                SELECT 1 FROM pg_constraint
                WHERE conrelid = to_regclass(:t) AND contype='c' AND conname = :n
                """
            ),
            {"t": table, "n": f"{table}_org_id_not_blank"},
        ).scalar()
        if not found:
            problems.append(table)
    return problems


def scoped_tables_without_app_grant(connection) -> list[str]:
    """Tables review_app cannot touch at all — almost always a forgotten grant.

    Added after `review_runs` shipped with policies and no grant: every isolation
    check was green while the application simply could not use the table.
    """
    problems = []
    for table in scoped_tables(connection):
        reachable = connection.execute(
            text(
                "SELECT has_table_privilege(:r, :t, 'SELECT') OR "
                "has_table_privilege(:r, :t, 'INSERT')"
            ),
            {"r": ROLE_APP, "t": table},
        ).scalar()
        if not reachable:
            problems.append(table)
    return problems


def policies_matching_blank_scope(connection) -> list[tuple[str, str]]:
    """Policies that would match a row stamped '' under an empty scope.

    A policy comparing org_id directly to current_setting() without nullif()
    treats the empty string as a tenant.
    """
    rows = connection.execute(
        text(
            """
            SELECT tablename, policyname, COALESCE(qual,'') || COALESCE(with_check,'')
                   AS expr
            FROM   pg_policies WHERE schemaname='public'
            """
        )
    ).all()
    # Postgres renders the stored expression as NULLIF(...) in upper case, so
    # this compares case-insensitively. A case-sensitive check here reported
    # every policy as broken while they were all correct — a drift check that
    # cries wolf is a drift check someone switches off.
    return sorted(
        (r.tablename, r.policyname)
        for r in rows
        if "current_setting" in r.expr.lower() and "nullif" not in r.expr.lower()
    )


def apply_all(connection) -> None:
    """Apply the COMPLETE isolation state. The single source of truth.

    Both provisioning and the test-suite repair path call this. They used to
    maintain parallel lists of what to restore, and that list under-repaired
    TWICE — first missing the compliance grants, then the checkpoint grants.
    apply_grants() begins by revoking everything from review_app, so any repair
    that stops short of the full set silently strips privileges rather than
    restoring them. Deriving both callers from one function removes the class of
    bug rather than the two instances of it.
    """
    from review_agent.data.checkpoint import apply_checkpoint_isolation

    apply_grants(connection)         # revokes, then re-grants...
    # MUST follow apply_grants, which revokes everything from review_auth. If
    # these two are ever reordered or this call dropped, auth_role_overreach()
    # fires at boot — the ordering is held by a derived gate, not by a comment.
    _apply_auth_path(connection)
    apply_rls_policies(connection)   # ...including the compliance path
    apply_triggers(connection)
    apply_checkpoint_isolation(connection)
    # Last, and over the DERIVED list, so it covers ORM tables, library tables,
    # and anything a later phase adds — without anyone maintaining a list.
    for table in tables_with_org_id(connection):
        _apply_blank_org_guard(connection, table)


def auth_path_unavailable(connection) -> list[str]:
    """Scope resolution CANNOT RUN. A broken deploy, not a breach.

    Split out because the response differs: re-run apply_all. Bundling it with
    tamper-evidence would mean a half-applied migration and a rewritten function
    body raised the same undifferentiated alarm, and someone would learn to
    treat both as "just re-provision".
    """
    problems: list[str] = []
    exists = connection.execute(
        text("SELECT count(*) FROM pg_proc WHERE proname = :n"), {"n": AUTH_FUNCTION}
    ).scalar()
    if not exists:
        return [f"{AUTH_FUNCTION} does not exist: the auth path was never applied"]

    if not connection.execute(
        text(f"SELECT has_function_privilege(:r, '{AUTH_FUNCTION}(text)', 'EXECUTE')"),
        {"r": ROLE_AUTH},
    ).scalar():
        problems.append(f"{ROLE_AUTH} cannot execute {AUTH_FUNCTION}")
    return sorted(problems)


def auth_function_integrity(connection) -> list[str]:
    """The definer function's DEFINITION has changed. Treat as a COMPROMISE.

    Distinct from privilege drift: every privilege can be correct while the body
    returns whatever org it likes. See the exceptions register (E5) for what this
    control does and does not prove — it is tamper-EVIDENCE, not authorisation.
    """
    definition = connection.execute(
        text(
            """
            SELECT prosrc, prosecdef, proconfig, pg_get_userbyid(proowner) AS owner
            FROM   pg_proc WHERE proname = :n
            """
        ),
        {"n": AUTH_FUNCTION},
    ).first()
    if definition is None:
        return []  # absence is auth_path_unavailable's incident, not this one

    problems: list[str] = []
    actual = hashlib.sha256(definition.prosrc.encode("utf-8")).hexdigest()
    expected = hashlib.sha256(AUTH_FUNCTION_BODY.encode("utf-8")).hexdigest()
    if actual != expected:
        problems.append(
            f"{AUTH_FUNCTION} body has been modified "
            f"(sha256 {actual[:12]}, expected {expected[:12]})"
        )
    if not definition.prosecdef:
        problems.append(f"{AUTH_FUNCTION} is no longer SECURITY DEFINER")
    if not any(c.startswith("search_path=") for c in (definition.proconfig or [])):
        problems.append(f"{AUTH_FUNCTION} has no pinned search_path")
    if definition.owner != ROLE_OWNER:
        problems.append(
            f"{AUTH_FUNCTION} is owned by {definition.owner}, not {ROLE_OWNER}"
        )
    return sorted(problems)


def auth_role_overreach(connection) -> list[str]:
    """A path to ENUMERATE tenants has reappeared. Privilege drift.

    Narrowed to privileges only. `users` carries org membership, so any direct
    read is a cross-tenant metadata disclosure — the census the function-only
    interface exists to prevent.
    """
    problems: list[str] = []

    for privilege in ("SELECT", "INSERT", "UPDATE", "DELETE"):
        if connection.execute(
            text("SELECT has_table_privilege(:r, 'users', :p)"),
            {"r": ROLE_AUTH, "p": privilege},
        ).scalar():
            problems.append(f"{ROLE_AUTH} holds table-level {privilege} on users")

    for column in ("user_id", "org_id", "role", "active", "email"):
        if connection.execute(
            text("SELECT has_column_privilege(:r, 'users', :c, 'SELECT')"),
            {"r": ROLE_AUTH, "c": column},
        ).scalar():
            problems.append(f"{ROLE_AUTH} can read users.{column} directly")

    if connection.execute(
        text("SELECT count(*) FROM pg_proc WHERE proname = :n"), {"n": AUTH_FUNCTION}
    ).scalar():
        for role in (ROLE_APP, ROLE_COMPLIANCE):
            if connection.execute(
                text(f"SELECT has_function_privilege(:r, '{AUTH_FUNCTION}(text)', "
                     f"'EXECUTE')"),
                {"r": role},
            ).scalar():
                problems.append(
                    f"{role} can execute {AUTH_FUNCTION}; it is a subject->org oracle"
                )
    return sorted(problems)


def verify_isolation(connection) -> dict[str, list]:
    """Run every drift check and return the findings (empty lists == healthy)."""
    return {
        "tables_missing_rls": tables_missing_rls(connection),
        "tenant_tables_without_policies": tenant_tables_without_policies(connection),
        "unconditional_policies": unconditional_policies(connection),
        "roles_with_bypass": roles_with_bypass(connection),
        "compliance_role_overreach": compliance_role_overreach(connection),
        # Structural, not documented: refuse to serve if graph state is stored
        # without tenant isolation (PHASE3_DESIGN.md §1.3).
        "checkpoint_tables_unmigrated": checkpoint_tables_unmigrated(connection),
        # Finding 3, generalised: an empty scope must be indistinguishable from
        # no access on EVERY tenant table, not just the checkpoint tables.
        "tables_missing_blank_org_guard": tables_missing_blank_org_guard(connection),
        "policies_matching_blank_scope": policies_matching_blank_scope(connection),
        "scoped_tables_without_app_grant": scoped_tables_without_app_grant(connection),
        # Three separate keys on purpose: a broken deploy, a rewritten function
        # body, and privilege drift are different incidents with different
        # responses, and one flat list makes them indistinguishable.
        "auth_path_unavailable": auth_path_unavailable(connection),
        "auth_function_integrity": auth_function_integrity(connection),
        "auth_role_overreach": auth_role_overreach(connection),
    }


def verify_isolation_or_raise(connection) -> None:
    """Boot-time assertion. Refuse to start if isolation is not intact."""
    problems = {k: v for k, v in verify_isolation(connection).items() if v}
    if problems:
        raise IsolationVerificationError(
            f"row-level security is not in the required state: {problems}"
        )
