"""Database connection and per-request session scoping.

Every DB session MUST be opened with the caller's org scope applied so that
row-level security (see rls.py) filters out other organisations' rows. Nothing
in the application should ever run an unscoped query against tenant tables.

This module and rls.py are the ONLY places permitted to name the session
variable `app.current_org` — enforced by a lint test in the red-team suite.

See docs/PHASE1_DESIGN.md §2.4 and §3.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import Session

from review_agent.data.rls import (
    ORG_GUC,
    ROLE_APP,
    ROLE_AUTH,
    ROLE_COMPLIANCE,
    ROLE_OWNER,
    RUNTIME_ROLES,
)
from review_agent.data.scope import CallerScope

DEFAULT_ADMIN_URL = "postgresql+psycopg://postgres:devpw@localhost:5433/review_agent"

_engines: dict[str, Engine] = {}


def role_passwords() -> dict[str, str]:
    """Per-role passwords from the environment, with a dev-only default."""
    return {
        role: os.environ.get(f"{role.upper()}_PASSWORD", "devpw")
        for role in RUNTIME_ROLES
    }


def admin_url() -> URL:
    """Superuser URL — provisioning and tests only. Never used at runtime."""
    return make_url(os.environ.get("ADMIN_DATABASE_URL", DEFAULT_ADMIN_URL))


def url_for_role(role: str) -> URL:
    """Connection URL for a role.

    An explicit <ROLE>_DATABASE_URL wins; otherwise the admin URL's host/database
    is reused with the role's own credentials, so local dev needs one env var.
    """
    explicit = os.environ.get(f"{role.upper()}_DATABASE_URL")
    if explicit:
        return make_url(explicit)
    if role == ROLE_APP and os.environ.get("DATABASE_URL"):
        return make_url(os.environ["DATABASE_URL"])
    return admin_url().set(username=role, password=role_passwords()[role])


def _engine_for(role: str, **kwargs) -> Engine:
    key = f"{role}:{sorted(kwargs.items())}"
    if key not in _engines:
        _engines[key] = create_engine(url_for_role(role), future=True, **kwargs)
    return _engines[key]


def get_engine(**kwargs) -> Engine:
    """The runtime engine: unprivileged `review_app`, subject to every policy."""
    return _engine_for(ROLE_APP, **kwargs)


def get_owner_engine(**kwargs) -> Engine:
    """Migrations and seeding only. FORCE RLS binds this role too."""
    return _engine_for(ROLE_OWNER, **kwargs)


def get_auth_engine(**kwargs) -> Engine:
    """Scope resolution only — can read `users` and nothing else."""
    return _engine_for(ROLE_AUTH, **kwargs)


def get_compliance_engine(**kwargs) -> Engine:
    """Offline cross-org audit METADATA only (design §1.10). Not held by the app."""
    return _engine_for(ROLE_COMPLIANCE, **kwargs)


def get_admin_engine(**kwargs) -> Engine:
    """Superuser engine for provisioning. Not part of any runtime path."""
    key = f"__admin__:{sorted(kwargs.items())}"
    if key not in _engines:
        _engines[key] = create_engine(admin_url(), future=True, **kwargs)
    return _engines[key]


def dispose_engines() -> None:
    """Drop all cached engines (used between test phases)."""
    for engine in _engines.values():
        engine.dispose()
    _engines.clear()


@contextmanager
def scoped_raw_connection(scope: CallerScope) -> Iterator[tuple[Session, object]]:
    """A scoped session PLUS the raw DBAPI connection underneath it.

    Exists for one caller: LangGraph's PostgresSaver, which must write on a
    connection that already has the org variable set. Handing it this connection
    is what makes checkpoint writes inherit the caller's scope — and it keeps the
    session variable inside the data layer, where the confinement lint requires
    it to stay.

    A pool with a `configure` hook was tried instead and REJECTED: psycopg_pool's
    configure fires on connection CREATION, not per checkout, so a checkout
    intending one tenant silently reused the previous tenant's scope.
    """
    with scoped_session(scope) as session:
        yield session, session.connection().connection.driver_connection


@contextmanager
def scoped_session(scope: CallerScope, engine: Engine | None = None) -> Iterator[Session]:
    """Yield a DB session bound to the caller's org scope.

    Takes a CallerScope, never a bare org_id string: a `str` parameter here is
    exactly the shape that eventually receives request.json["org_id"]. The org_id
    must come from the resolved auth context (scope.py), never from user input or
    uploaded file content.

    SET LOCAL, inside an explicit transaction — never plain SET. With a connection
    pool a plain SET persists on the pooled connection after the request ends, so
    the next request to borrow it inherits the previous tenant's scope. That is
    the single most likely way this design gets breached in production;
    test_rls_prevents_cross_tenant_rows covers it.
    """
    if not isinstance(scope, CallerScope):
        raise TypeError(
            "scoped_session requires a CallerScope from resolve_scope(); "
            f"got {type(scope).__name__}. Passing a raw org_id is how untrusted "
            "input reaches the isolation boundary."
        )

    engine = engine if engine is not None else get_engine()
    with Session(engine) as session:
        with session.begin():
            # Bound as a parameter, never string-interpolated; is_local=true ties
            # it to this transaction so it cannot survive back into the pool.
            session.execute(
                text("SELECT set_config(:key, :value, true)"),
                {"key": ORG_GUC, "value": scope.org_id},
            )
            yield session
