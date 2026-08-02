"""Auth context and scope resolution.

Resolves the authenticated caller to their single organisation scope and makes it
available to db.scoped_session(). The scope is derived SERVER-SIDE from the session
/ SSO identity — never from request bodies, query params, or uploaded file content.

See docs/PHASE1_DESIGN.md §3.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text

from review_agent.audit import actions


class ScopeResolutionError(Exception):
    """The caller could not be resolved to exactly one active organisation.

    Always fatal for the request. There is no fallback org and no unscoped mode:
    a request that cannot be scoped is refused, never widened.
    """


@dataclass(frozen=True)
class CallerScope:
    """The resolved, trusted scope for a request.

    Frozen on purpose (design §3.2): once resolved, downstream code cannot mutate
    it. This is the ONLY type db.scoped_session() accepts — it deliberately does
    not take a bare org_id string, because a `str` parameter is the shape that
    eventually receives request.json["org_id"].
    """

    user_id: str
    org_id: str
    project_id: str | None = None


def resolve_scope(request) -> CallerScope:
    """Resolve the caller from the authenticated session to their org scope.

    `request` must carry claims already verified by the auth layer (signature,
    issuer, audience, expiry). We read ONLY the subject claim from it; the org is
    then looked up server-side. Nothing else on the request is consulted — not the
    body, not the query string, not headers the client controls.
    """
    claims = getattr(getattr(request, "state", None), "oidc_claims", None)
    if not claims or not claims.get("sub"):
        raise ScopeResolutionError("no verified OIDC subject on request")
    return resolve_scope_for_subject(claims["sub"])


def resolve_scope_for_subject(subject: str) -> CallerScope:
    """Map a verified OIDC subject to its organisation.

    Uses the narrowly-privileged `review_auth` role, which can do exactly one
    thing: execute resolve_user_scope(). It holds NO table privilege on `users`.
    This is the single circular-dependency point in the design (you need an org
    to scope a session, and this lookup is what produces the org); it is closed
    by making the permitted query shape STRUCTURAL rather than conventional —
    the earlier column grant let this role enumerate every org and its members.
    """
    from review_agent.data.db import get_auth_engine  # local: avoids import cycle

    # One function, one subject, at most one row. review_auth has no table
    # privilege on `users`, so there is no enumeration path to forget to avoid.
    sql = text("SELECT user_id, org_id FROM resolve_user_scope(:sub)")
    with get_auth_engine().connect() as conn:
        rows = conn.execute(sql, {"sub": subject}).fetchall()

    if len(rows) != 1:
        record_scope_denied(subject, "no single active user for subject")
        raise ScopeResolutionError(f"no single active user for subject {subject!r}")

    return CallerScope(user_id=rows[0].user_id, org_id=rows[0].org_id)


def record_scope_denied(subject: str, reason: str) -> None:
    """Audit a refused scope resolution.

    A denied caller has no tenant by definition, so the entry is recorded under
    the ORG_UNSCOPED sentinel — an org that owns nothing, which is why RLS
    confines these rows to a view containing no tenant data. No privilege change
    is needed: review_app already holds INSERT on audit_log, and setting the
    session variable to the sentinel satisfies the WITH CHECK policy.

    BEST EFFORT, deliberately. Everywhere else in this codebase an operation
    that cannot write its audit entry fails (the writes share a transaction).
    Here the "operation" is a REJECTION that has already succeeded: access is
    denied whether or not the record lands, and raising a second, different
    error would obscure the authentication failure the caller actually needs to
    see. So the audit failure is swallowed and the ScopeResolutionError stands.
    """
    from review_agent.data.db import scoped_session
    from review_agent.data.repository import record_audit

    try:
        audit_scope = CallerScope(user_id=subject, org_id=actions.ORG_UNSCOPED)
        with scoped_session(audit_scope) as session:
            record_audit(
                session,
                audit_scope,
                action=actions.SCOPE_DENIED,
                detail={"subject": subject, "reason": reason},
            )
    except Exception:  # noqa: BLE001 — never mask the auth failure
        pass
