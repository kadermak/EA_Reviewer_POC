"""Seed the two mock organisations from ./sample-data.

The seeder connects as `review_owner` and still sets an org scope per row —
because FORCE ROW LEVEL SECURITY binds the owner too. That is deliberate: if
seeding could bypass RLS, the tests would be running against a different security
model than production does.

Real data swaps in later with no code change (design goal); only the loader path
below knows these are samples.
"""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import text

from review_agent.data.db import get_owner_engine, scoped_session
from review_agent.data.models import NON_TENANT_TABLES, Organisation, Project, User
from review_agent.data.repository import insert_artifact
from review_agent.data.scope import CallerScope

SAMPLE_DATA = Path(__file__).resolve().parents[3] / "sample-data"

# One submitter + one SAO reviewer per org. Each user belongs to exactly ONE org
# for the POC, which keeps the leak tests crisp.
SEED_USERS = {
    "org-a": [("user-a@org-a", "submitter"), ("reviewer-a@org-a", "sao_reviewer")],
    "org-b": [("user-b@org-b", "submitter"), ("reviewer-b@org-b", "sao_reviewer")],
}

ARTIFACTS = {
    "org-a": ("proj-a1", "artifact_org-a_proj-a1.md"),
    "org-b": ("proj-b1", "artifact_org-b_proj-b1.md"),
}


def load_organisations() -> dict:
    return json.loads((SAMPLE_DATA / "mock_organisations.json").read_text())


def rulebook_version() -> str:
    meta = json.loads((SAMPLE_DATA / "ea_standards.json").read_text())["rulebook_meta"]
    return meta["version"]


def truncate_all() -> None:
    """Clear every table that holds data, DERIVED from the catalogue.

    Owner-only. This used to be a hardcoded list and silently skipped the
    checkpoint tables when they arrived, leaking graph state across tests — the
    third bug caused by a parallel table list. Deriving means a table added in a
    later phase is cleared the moment it exists.

    checkpoint_migrations is excluded: it is LangGraph's schema-version record,
    not data. Truncating it would make the saver believe it had never migrated.
    """
    from review_agent.data.rls import scoped_tables

    with get_owner_engine().begin() as conn:
        tables = [
            t for t in scoped_tables(conn)
            if t != "checkpoint_migrations"
        ] + sorted(NON_TENANT_TABLES)

        # The append-only trigger blocks TRUNCATE even for the owner — that it
        # does is the trigger working, so it is lowered explicitly here rather
        # than worked around.
        conn.execute(text("ALTER TABLE audit_log DISABLE TRIGGER audit_log_no_truncate"))
        conn.execute(
            text(f"TRUNCATE {', '.join(tables)} RESTART IDENTITY CASCADE")
        )
        conn.execute(text("ALTER TABLE audit_log ENABLE TRIGGER audit_log_no_truncate"))


def seed_sample_data() -> None:
    """Load both orgs, their projects, users, artifacts and a starter audit entry."""
    data = load_organisations()
    version = rulebook_version()
    owner = get_owner_engine()

    for org in data["organisations"]:
        org_id = org["org_id"]
        # A seeding identity, scoped like any other caller.
        scope = CallerScope(user_id="seed@system", org_id=org_id)

        with scoped_session(scope, engine=owner) as session:
            session.add(
                Organisation(
                    org_id=org_id,
                    name=org["name"],
                    description=org.get("description"),
                )
            )
            session.flush()

            for project in org["projects"]:
                session.add(
                    Project(
                        project_id=project["project_id"],
                        org_id=org_id,
                        name=project["name"],
                        criticality=project["criticality"],
                        handles_personal_data=project["handles_personal_data"],
                    )
                )
            session.flush()

            for user_id, role in SEED_USERS[org_id]:
                session.add(
                    User(
                        user_id=user_id,
                        org_id=org_id,
                        email=user_id,
                        role=role,
                        active=True,
                    )
                )
            session.flush()

            project_id, filename = ARTIFACTS[org_id]
            upload_scope = CallerScope(
                user_id=SEED_USERS[org_id][0][0], org_id=org_id, project_id=project_id
            )
            artifact = insert_artifact(
                session,
                upload_scope,
                project_id=project_id,
                filename=filename,
                content=(SAMPLE_DATA / filename).read_text(),
            )
            # insert_artifact writes its own artifact.upload audit entry in the
            # same transaction; there is deliberately no second one here.
            assert artifact.artifact_id is not None


if __name__ == "__main__":
    truncate_all()
    seed_sample_data()
    print("sample data seeded")
