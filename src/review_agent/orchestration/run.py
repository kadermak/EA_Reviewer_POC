"""Run ONE review end to end, against the live model.

    python -m review_agent.orchestration.run --subject user-a@org-a

Deliberately thin: it resolves a scope, picks a visible artifact, starts the
graph, and prints what came back. Everything it touches is the ordinary path —
no test hooks, no stub provider — because the point of this entry point is to
exercise `models/providers/anthropic.py`, which nothing else does.

The graph pauses at the human interrupt, so this ends with findings persisted as
`pending` and a run awaiting the SAO. That is the whole review path short of a
human decision.
"""

from __future__ import annotations

import argparse
import os
import sys

from sqlalchemy import text

from review_agent.data.db import scoped_session
from review_agent.data.scope import ScopeResolutionError, resolve_scope_for_subject
from review_agent.models import client
from review_agent.models.types import ModelRole


def _load_dotenv(path: str = ".env") -> None:
    """Read .env if present. Nothing else in the codebase does this.

    Kept here rather than in the library: a module that silently reads a file
    from the working directory is a surprise, and the isolation gate must not
    acquire a credential just by importing something.
    """
    if not os.path.exists(path):
        return
    seen: dict[str, int] = {}
    for number, line in enumerate(open(path, encoding="utf-8"), start=1):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        # A DUPLICATE KEY IS AN ERROR, not a last-one-wins or first-one-wins.
        # .env is normally made by copying .env.example, so a real value and the
        # template's placeholder very easily coexist — and then which one takes
        # effect is decided by LINE ORDER. Reordering the file would silently
        # swap the credential. Neither precedence rule is defensible enough to
        # pick, so the ambiguity is refused instead of resolved.
        if key in seen:
            raise SystemExit(
                f"{path} sets {key} twice (lines {seen[key]} and {number}). "
                "Which one takes effect would depend on line order. Delete one "
                "— most likely the placeholder copied from .env.example."
            )
        seen[key] = number
        os.environ.setdefault(key, value.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="user-a@org-a",
                        help="verified OIDC subject to run as")
    parser.add_argument("--artifact-id", default=None,
                        help="defaults to the caller's first visible artifact")
    args = parser.parse_args()

    _load_dotenv()
    credential = (os.environ.get("ANTHROPIC_API_KEY")
                  or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    if not credential:
        print("No ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN. Add one to .env.",
              file=sys.stderr)
        return 2
    # "is it set" was a fail-open check: .env.example ships `sk-...`, which is a
    # perfectly non-empty string. Copying the template and forgetting to fill it
    # in therefore passed the preflight and failed at the provider instead —
    # reaching the network to discover a local configuration mistake.
    if credential.strip() in ("sk-...", "") or credential.startswith("sk-..."):
        print("The API key in .env is still the .env.example placeholder "
              "(sk-...). Replace it with a real key.", file=sys.stderr)
        return 2

    try:
        scope = resolve_scope_for_subject(args.subject)
    except ScopeResolutionError as exc:
        print(f"scope resolution refused: {exc}", file=sys.stderr)
        return 3

    artifact_id = args.artifact_id
    if artifact_id is None:
        with scoped_session(scope) as session:
            artifact_id = session.execute(
                text("SELECT artifact_id FROM artifacts ORDER BY uploaded_at LIMIT 1")
            ).scalar()
        if artifact_id is None:
            print("no artifact visible in this scope; run the seeder first",
                  file=sys.stderr)
            return 4

    # Print the resolved scope as a whole rather than reaching for its org field:
    # nothing in orchestration/ may NAME a tenant identifier (the BUG-2 lint), and
    # that rule is worth more than a tidier debug line. It has caught real drift.
    print(f"scope={scope}")
    print(f"artifact={artifact_id}")
    print(f"judgment model={client.resolve_model(ModelRole.JUDGMENT)}")

    # Imported late: this pulls in langgraph, and importing it before the checks
    # above would make a missing credential look like a slow start.
    from review_agent.orchestration.graph import start_review

    run_id, interrupted = start_review(scope, str(artifact_id))
    print(f"run={run_id} interrupted={'yes' if interrupted else 'no'}")

    with scoped_session(scope) as session:
        status = session.execute(
            text("SELECT status FROM review_runs WHERE run_id=:r"), {"r": run_id}
        ).scalar()
        # Filtered by artifact. Without this the report showed every finding
        # visible in the scope — which on a database with any prior state means
        # duplicate rule_ids and, confusingly, the SAME rule at two severities
        # (two rulebook versions), making a correct 14-finding run look broken.
        # `findings` has no run_id: it is keyed to the artifact, so re-reviewing
        # one APPENDS rather than replaces. See §3e.
        rows = session.execute(
            text(
                "SELECT rule_id, verdict, severity, evidence FROM findings "
                "WHERE artifact_id = :a ORDER BY rule_id"
            ),
            {"a": artifact_id},
        ).all()
        calls = session.execute(
            text("SELECT detail FROM audit_log WHERE action='model.call'")
        ).scalars().all()

    print(f"status={status}  findings={len(rows)}")
    for rule_id, verdict, severity, evidence in rows:
        quote = (evidence or "")[:70].replace("\n", " ")
        print(f"  {rule_id:<11} {verdict:<8} {severity:<8} {quote}")

    failed = False
    for call in calls:
        usage = call.get("usage", {})
        print(f"  model={call.get('model_id')} in={usage.get('input_tokens')} "
              f"out={usage.get('output_tokens')} stop={call.get('stop_reason')}")
        if call.get("error"):
            # Printed because a failed review is otherwise a short, quiet,
            # successful-looking run: status=failed, findings=0, exit 0.
            print(f"    error: {call['error']}")
            failed = True

    # A non-zero exit so a scripted run cannot mistake a provider outage for a
    # review that found nothing.
    return 5 if failed or status == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
