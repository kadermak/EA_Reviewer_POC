"""FastAPI application — the reviewer surface.

Endpoints (POC):
  GET  /queue                    - runs awaiting this caller's review
  GET  /reviews/{run_id}         - one review: artifact, flags, findings
  POST /reviews/{run_id}/decide  - record reviewer decisions
  POST /reviews/{run_id}/complete- finalise, once required decisions are in

FOUR CONSTRAINTS, all of which live HERE rather than in the front end
--------------------------------------------------------------------
1. NO TENANT IDENTIFIER ON THE WIRE. No request model or path parameter names an
   org. Scope is resolved server-side from the verified session (data/scope.py).
   If it cannot be parsed off the wire it cannot be trusted by accident.

2. NO SECOND READ PATH. Handlers contain no SQL. Every read goes through the
   scoped repository functions, so RLS applies before a filter does. This is the
   layer the constraint belongs to — the UI just makes HTTP calls, so a rule
   aimed at UI code would be unenforceable while appearing enforced.

3. NO AGGREGATE VERDICT. No route returns "compliant", a ratio, or a score. An
   aggregate is the artefact that gets screenshotted into a steering deck, and at
   that moment the tool has certified something. Counts of findings NEEDING
   ATTENTION are permitted; denominators are not.

4. THE API MAY NOT CHANGE A VERDICT. /decide writes `reviewer_action` and
   nothing else. BUG-16 reaching the last layer: the verdict is what the SAO
   rules on, so no code between the agent and the human may edit it.

There is deliberately no free-text field anywhere: "write me some code" has
nowhere to be typed, which is the primary control for out-of-scope requests and
is stronger than any classifier.
"""

# NOTE: no `from __future__ import annotations` here, deliberately. FastAPI
# resolves dependency and body types from the annotations at import time;
# postponed (string) annotations make it unable to resolve `Request`, at which
# point it silently reclassifies the dependency as a QUERY PARAMETER and every
# route 422s. Failing loudly in a test is why that was caught rather than
# shipped.
from typing import Annotated

from pydantic import BaseModel, Field

from review_agent.data import repository
from review_agent.data.db import scoped_session
from review_agent.data.scope import CallerScope, ScopeResolutionError, resolve_scope

from fastapi import Body, Depends, FastAPI, HTTPException, Request


class DecisionsRequest(BaseModel):
    """Reviewer decisions. Note what is NOT here: no org, no verdict, no severity.

    A field absent from the request model is a field the client cannot influence.
    """

    decisions: dict[str, str] = Field(
        ..., description="finding_id -> one of accepted | overridden | waived"
    )


def caller_scope(request: Request) -> CallerScope:
    """Resolve the caller server-side. The ONLY source of tenancy in this layer."""
    try:
        return resolve_scope(request)
    except ScopeResolutionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def build_app():
    app = FastAPI(title="Architecture Review Agent (POC)")
    Scope = Annotated[CallerScope, Depends(caller_scope)]

    @app.get("/queue")
    def queue(scope: Scope):
        """Runs awaiting this caller's review. RLS decides what is in the list."""
        with scoped_session(scope) as session:
            return {"runs": repository.list_review_queue(session, scope)}

    @app.get("/reviews/{run_id}")
    def review(run_id: str, scope: Scope):
        """One review. Passes are included IN FULL — shown, not hidden.

        They are individually overridable; they simply do not block completion
        (PHASE3_DESIGN §4.5).
        """
        with scoped_session(scope) as session:
            found = repository.load_review(session, scope, run_id)
        if found is None:
            # Not visible in this scope. Same answer as "does not exist" — the
            # 404 does not confirm another tenant's run.
            raise HTTPException(status_code=404, detail="review not found")
        return found

    @app.post("/reviews/{run_id}/decide")
    def decide(run_id: str, body: DecisionsRequest, scope: Scope):
        """Write reviewer_action. Nothing else about a finding may change."""
        with scoped_session(scope) as session:
            try:
                updated = repository.apply_reviewer_decisions(
                    session, scope, run_id, body.decisions
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"updated": updated}

    @app.post("/reviews/{run_id}/complete")
    def complete(run_id: str, scope: Scope):
        """Finalise, if every finding that ASSERTS something has been decided."""
        with scoped_session(scope) as session:
            outstanding = repository.findings_awaiting_decision(session, scope, run_id)
        if outstanding:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "findings still require a decision",
                    "awaiting_decision": outstanding,
                },
            )
        with scoped_session(scope) as session:
            repository.mark_run_completed(session, scope, run_id)
        return {"run_id": run_id, "status": "completed"}

    # --- submitter surface (FBR-4). Demo scaffolding, but held to every rule
    # above: scope from the session, no SQL in the handler, no aggregate, no
    # free-text field. The artifact body is the existing untrusted-content path
    # (sanitised at prompt construction, flagged by the input guard), not a new
    # instruction channel.

    @app.get("/projects/mine")
    def my_projects(scope: Scope):
        """The scoped project list for the submitter's picker — the caller's OWN.

        Named `/projects/mine`, never a bare `/projects`: the latter reads as "list
        all projects" (an enumeration endpoint), the same reason the reviewer
        surface is `/queue` not `/reviews`. This returns only what RLS already
        made visible, so a wrong-org submission is not policed — the id needed for
        it is simply not in this list (FBR-2).
        """
        with scoped_session(scope) as session:
            return {"projects": repository.list_projects(session, scope)}

    @app.post("/projects/{project_id}/submit")
    def submit(
        project_id: str,
        filename: str,
        content: Annotated[bytes, Body(media_type="text/plain")],
        scope: Scope,
    ):
        """Ingest an uploaded design and run it to a DRAFT the submitter can see.

        The artifact arrives as the raw request body (Markdown/plaintext only) —
        the reviewed object, handled by the existing sanitisation path, not a
        typed instruction field. `project_id` is a picked value; RLS + the
        consistency trigger reject one the caller cannot see. The review parks in
        `draft` (not the SAO queue) until the submitter sends it.
        """
        from review_agent.ingestion.extract import SUPPORTED_SUFFIXES
        from review_agent.orchestration.graph import start_review

        suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported format {suffix or filename!r}; "
                       "Markdown and plaintext only",
            )
        try:
            text_content = content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="artifact is not valid UTF-8")

        with scoped_session(scope) as session:
            artifact = repository.insert_artifact(
                session, scope, project_id=project_id,
                filename=filename, content=text_content,
            )
            artifact_id = str(artifact.artifact_id)

        try:
            run_id, _ = start_review(scope, artifact_id, hold_status="draft")
        except Exception as exc:  # noqa: BLE001 — a bad project/file is a 400, not a 500
            raise HTTPException(
                status_code=400, detail=f"review could not start: {exc}"
            ) from exc
        return {"run_id": run_id, "status": "draft"}

    @app.post("/reviews/{run_id}/send")
    def send(run_id: str, scope: Scope):
        """Move a DRAFT run into the SAO queue (draft -> awaiting_review)."""
        with scoped_session(scope) as session:
            try:
                repository.mark_run_sent(session, scope, run_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"run_id": run_id, "status": "awaiting_review"}

    return app


app = build_app()
