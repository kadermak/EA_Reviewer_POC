"""DEV-ONLY reviewer UI server. NOT a production entry point.

Forges a verified OIDC subject so the reference page can be clicked through
locally. It mirrors tests/test_reviewer_api.py::client_for exactly: the
subject-injecting middleware is defined HERE, outside src/, so the production app
(`build_app`, run as `uvicorn review_agent.api.app:app`) carries no dev branch
and no test-only path. `resolve_scope` still runs in full — claims -> subject ->
SECURITY DEFINER lookup -> CallerScope — so the real scoping path is exercised;
only the "who verified the subject" step is faked.

SECURITY: this bypasses authentication. It binds to 127.0.0.1 ONLY and must never
be exposed. If it were reachable off-host, anyone could assume any tenant just by
passing --subject. There is intentionally no host flag.

Run:
    pip install -e ".[dev]"        # provides uvicorn (dev extra) + the test deps
    python dev_ui.py --subject user-a@org-a
    # open http://127.0.0.1:8000/?run=<run_id printed by the orchestration CLI>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fastapi import Request
from fastapi.responses import HTMLResponse

from review_agent.api.app import build_app
from review_agent.orchestration.run import _load_dotenv

REVIEWER_HTML = Path(__file__).resolve().parent / "ui" / "reviewer.html"
SUBMIT_HTML = Path(__file__).resolve().parent / "ui" / "submit.html"


def build_dev_app(subject: str):
    """The production app, plus a dev-only subject and a static page route."""
    app = build_app()

    @app.middleware("http")
    async def _inject_subject(request: Request, call_next):
        # Identical to the test middleware. Nothing in src/ ever sets this.
        request.state.oidc_claims = {"sub": subject}
        return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    def _reviewer_page():
        # Served same-origin so the page's relative fetch("/reviews/...") calls
        # reach the API with no CORS. Read-and-return avoids a StaticFiles
        # dependency (aiofiles), which is not installed.
        return REVIEWER_HTML.read_text(encoding="utf-8")

    @app.get("/submit", response_class=HTMLResponse)
    def _submit_page():
        return SUBMIT_HTML.read_text(encoding="utf-8")

    @app.get("/whoami")
    def _whoami():
        # The identity stamp for the submit page. Resolved SERVER-SIDE from the
        # forged subject — the org is shown, never asked. Lives HERE, not in
        # api/app.py, because the API layer is lint-forbidden from naming a tenant
        # identifier; this is dev scaffolding and may.
        #
        # Shows the org's DISPLAY name (from the DB, self-scoped — a caller sees
        # only their own organisations row), not the internal org_id. The company
        # name is deployment-level context, read from the sample data.
        from sqlalchemy import text as _sql
        from review_agent.data.db import scoped_session
        from review_agent.data.scope import resolve_scope_for_subject

        scope = resolve_scope_for_subject(subject)
        with scoped_session(scope) as session:
            org_name = session.execute(_sql("SELECT name FROM organisations")).scalar()
        return {
            "subject": scope.user_id,
            "org": org_name or scope.org_id,
            "company": _company_name(),
        }

    return app


def _company_name() -> str:
    """Deployment/company display name, from the sample data (not in the DB)."""
    import json
    data = json.loads(
        (Path(__file__).resolve().parent / "sample-data"
         / "mock_organisations.json").read_text()
    )
    return data.get("company", "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="user-a@org-a",
                        help="the tenant identity to forge (default org-a)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # The UI server makes NO model calls, so it needs no API key. It only loads
    # .env so DATABASE_URL / passwords resolve the same way the rest of the app's
    # entry point does.
    _load_dotenv()

    try:
        import uvicorn
    except ModuleNotFoundError:
        print("uvicorn is not installed. It is in the DEV extra (only this shim "
              "needs it).\n  pip install -e \".[dev]\"", file=sys.stderr)
        return 2

    print(f"DEV UI: forging subject {args.subject!r}. Binding 127.0.0.1 ONLY — "
          "do not expose (it bypasses auth).")
    print(f"  Generate a run first:  python -m review_agent.orchestration.run "
          f"--subject {args.subject}")
    print(f"  Then open:             http://127.0.0.1:{args.port}/?run=<run_id>")
    uvicorn.run(build_dev_app(args.subject), host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
