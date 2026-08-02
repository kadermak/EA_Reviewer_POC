"""Schema bootstrap: roles, tables, grants, policies, triggers — then verify.

Run order matters. Tables must exist before grants and policies; policies must
exist before the verification passes. The final step refuses to continue if
isolation is not intact, because a running service with broken RLS is worse than
an outage.

Usage (local dev):  python -m review_agent.data.provision --reset
"""

from __future__ import annotations

import argparse

from sqlalchemy import text

from review_agent.data import rls
from review_agent.data.checkpoint import create_checkpoint_tables
from review_agent.data.db import (
    dispose_engines,
    get_admin_engine,
    get_owner_engine,
    role_passwords,
)
from review_agent.data.models import Base


def bootstrap(reset: bool = False) -> None:
    """Create/refresh the whole isolation core. Idempotent."""
    with get_admin_engine().begin() as conn:
        rls.provision_roles(conn, role_passwords())

    owner = get_owner_engine()

    if reset:
        with owner.begin() as conn:
            conn.execute(text("DROP VIEW IF EXISTS audit_log_metadata"))
        Base.metadata.drop_all(owner)

    Base.metadata.create_all(owner)

    # LangGraph's tables must exist BEFORE apply_all brings them under RLS, and
    # the migration must run before any run writes a checkpoint (SET NOT NULL
    # requires an empty table). Provisioning is the only moment both hold.
    create_checkpoint_tables(owner)

    with owner.begin() as conn:
        # One canonical application of the whole isolation state — see
        # rls.apply_all. The test-suite repair path calls the same function, so
        # the two cannot drift apart.
        rls.apply_all(conn)

    # Boot-time assertion (design §2.5a): verify our own work, or refuse to run.
    with owner.begin() as conn:
        rls.verify_isolation_or_raise(conn)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset", action="store_true", help="drop and recreate all tables"
    )
    args = parser.parse_args()
    bootstrap(reset=args.reset)
    dispose_engines()
    print("isolation core provisioned and verified")


if __name__ == "__main__":
    main()
