"""Golden-set evaluation (Phase 4, task 14).

    python -m evals.run_evals

Runs each sample artifact through the REAL graph — scoped ingestion, guardrails,
persistence — and scores the PERSISTED findings against the matching
golden_*.json answer key. It is a report, not a test: no thresholds, no pass/fail
bar, and it never runs in CI (PHASE4_DESIGN.md §1.6).

WHAT THIS MEASURES, AND WHAT IT CANNOT
--------------------------------------
n=2. Every number here comes from two hand-written documents containing known
defects, scored against a key written by the same author. One missed finding
moves recall by 10-17 points. `n=2` is printed next to every metric because a
percentage invites a confidence this sample does not support.

The golden files are PARTIAL (6 of 14 rules for org-a, 8 for org-b), so a `fail`
on an unmentioned rule is not necessarily wrong — it may be a real defect the
key's author did not plant. Those are counted separately as `unscoreable` rather
than folded into precision, which would score the agent against the key's
completeness instead of the artifact's content.

`unclear` on a must_catch rule is a MISS (§1.2): the SAO's exposure is identical
whether the agent said "no violation" or "cannot tell". Reported separately
because the remedy differs — a fail→pass error is a judgement error, a
fail→unclear error is usually an evidence problem, and the second is not fixable
by prompting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from sqlalchemy import text

SAMPLE = Path(__file__).resolve().parents[1] / "sample-data"

CASES = [
    ("user-a@org-a", "proj-a1", "artifact_org-a_proj-a1.md",
     "golden_org-a_proj-a1.json"),
    ("user-b@org-b", "proj-b1", "artifact_org-b_proj-b1.md",
     "golden_org-b_proj-b1.json"),
    # Added for the Phase 4 demo — SIMULATED, varied across rule categories and
    # fail/pass/unclear mixes; scoped to the existing mock orgs, no new tenants.
    ("user-a@org-a", "proj-a1", "artifact_org-a_proj-a1_atlas.md",
     "golden_org-a_proj-a1_atlas.json"),        # CLEAN — measures false positives
    ("user-a@org-a", "proj-a1", "artifact_org-a_proj-a1_lakehouse.md",
     "golden_org-a_proj-a1_lakehouse.json"),    # Data / Observability / Technology
    ("user-b@org-b", "proj-b1", "artifact_org-b_proj-b1_portal.md",
     "golden_org-b_proj-b1_portal.json"),       # Identity / Integration
    ("user-b@org-b", "proj-b1", "artifact_org-b_proj-b1_sketch.md",
     "golden_org-b_proj-b1_sketch.json"),       # unclear-heavy draft
]

N = len(CASES)


def load_golden(name: str) -> dict:
    return json.loads((SAMPLE / name).read_text())


# --- scoring -----------------------------------------------------------------

def score(agent_findings: list[dict], golden: dict) -> dict:
    """Score persisted findings against one answer key.

    `agent_findings` are rows read back from the DATABASE, not in-memory
    results: the eval measures what a reviewer would actually be shown.
    """
    expected = {f["rule_id"]: f for f in golden["expected_findings"]}
    actual = {f["rule_id"]: f for f in agent_findings}

    must_catch = [r for r, f in expected.items() if f["must_catch"]]

    caught, missed_as_pass, missed_as_unclear, absent = [], [], [], []
    for rule_id in must_catch:
        got = actual.get(rule_id)
        if got is None:
            # No verdict at all — distinct from a wrong one. It means the review
            # did not cover the rule, which validation is supposed to prevent,
            # so it points at the harness or the contract rather than judgement.
            absent.append(rule_id)
        elif got["verdict"] == expected[rule_id]["verdict"]:
            caught.append(rule_id)
        elif got["verdict"] == "unclear":
            missed_as_unclear.append(rule_id)
        else:
            missed_as_pass.append(rule_id)

    # Agreement ONLY over rules the key mentions. See the module docstring.
    mentioned_agreements = sum(
        1 for rule_id, exp in expected.items()
        if rule_id in actual and actual[rule_id]["verdict"] == exp["verdict"]
    )
    # A `fail` on a rule the key does not mention is NOT scored as a false
    # positive — the key is partial (§1.1), so it may be a real defect the
    # author did not plant. It is routed to the SAO as an ADJUDICATION ITEM
    # (§1.1a): the reviewer rules it a real defect (→ a new expected finding) or
    # a false positive (→ a documented pass case). Either way it is the first
    # genuine increment to the golden set — the mechanism by which it stops
    # being two hand-built artifacts. Carrying the evidence makes each one
    # actionable without re-running.
    unscoreable_fails = [
        {"rule_id": rule_id, "severity": f.get("severity"),
         "evidence": f.get("evidence", "")}
        for rule_id, f in actual.items()
        if f["verdict"] == "fail" and rule_id not in expected
    ]

    # Severity is joined from the rulebook by CODE, never authored by the model.
    # Checked here because the golden key is an INDEPENDENT record of it, so a
    # mismatch means the join is wrong — a code defect, not a judgement error.
    severity_mismatches = [
        rule_id for rule_id, exp in expected.items()
        if exp.get("severity") and rule_id in actual
        and actual[rule_id]["severity"] != exp["severity"]
    ]

    return {
        "must_catch_total": len(must_catch),
        "must_catch_caught": len(caught),
        "missed_as_pass": missed_as_pass,
        "missed_as_unclear": missed_as_unclear,
        "missed_absent": absent,
        "mentioned_total": len(expected),
        "mentioned_agreements": mentioned_agreements,
        "unscoreable_fails": unscoreable_fails,
        "severity_mismatches": severity_mismatches,
    }


# --- running one case --------------------------------------------------------

def run_case(subject: str, project_id: str, artifact_name: str) -> dict:
    """Upload and review one artifact through the real scoped path.

    Runs AS A REAL TENANT: every read goes through scoped_session, so the eval
    cannot assemble its numbers by reading across orgs. If it ever did, the run
    would produce wrong results rather than silently better ones.
    """
    from review_agent.data.db import scoped_session
    from review_agent.data.repository import insert_artifact
    from review_agent.data.scope import resolve_scope_for_subject
    from review_agent.orchestration.graph import start_review

    scope = resolve_scope_for_subject(subject)
    content = (SAMPLE / artifact_name).read_text()

    with scoped_session(scope) as session:
        artifact = insert_artifact(
            session, scope, project_id=project_id,
            filename=artifact_name, content=content,
        )
        artifact_id = str(artifact.artifact_id)

    started = time.monotonic()
    run_id, _ = start_review(scope, artifact_id)
    elapsed = time.monotonic() - started

    with scoped_session(scope) as session:
        status = session.execute(
            text("SELECT status FROM review_runs WHERE run_id=:r"), {"r": run_id}
        ).scalar()
        # Findings for THIS run — not the artifact's history (PHASE3 §3e(b)).
        findings = [
            {"rule_id": r, "verdict": v, "severity": s, "evidence": e}
            for r, v, s, e in session.execute(
                text("SELECT rule_id, verdict, severity, evidence FROM findings "
                     "WHERE run_id=:r ORDER BY rule_id"),
                {"r": run_id},
            )
        ]
        # Cost and retry causes come from the AUDIT LOG, not from in-memory
        # results — the same source an operator has after the fact, which is
        # also the only way to notice if the trail has stopped recording.
        calls = session.execute(
            text("SELECT detail FROM audit_log WHERE action='model.call'")
        ).scalars().all()
        completed = session.execute(
            text("SELECT detail FROM audit_log WHERE action='review.completed'")
        ).scalars().all()
        rejected = session.execute(
            text("SELECT detail FROM audit_log WHERE action='review.rejected'")
        ).scalars().all()

    retries: list[list[str]] = []
    for detail in completed:
        retries.extend(detail.get("validation_retries") or [])
    for detail in rejected:
        # A rejected review's errors are a retry cause too — the same failure
        # that simply ran out of attempts.
        if detail.get("validation_errors"):
            retries.append(list(detail["validation_errors"]))

    return {
        "run_id": run_id,
        "status": status,
        "findings": findings,
        "elapsed_s": elapsed,
        "input_tokens": sum(c.get("usage", {}).get("input_tokens", 0) for c in calls),
        "output_tokens": sum(c.get("usage", {}).get("output_tokens", 0) for c in calls),
        "model_calls": len(calls),
        "retries": retries,
        "rejected": [d.get("reject_reason") for d in rejected],
    }


# --- reporting ---------------------------------------------------------------

def _print_case(subject: str, artifact_name: str, outcome: dict, scored: dict) -> None:
    print(f"\n{'=' * 74}")
    print(f"{artifact_name}   ({subject})   1 of n={N}")
    print("=" * 74)
    print(f"  status        {outcome['status']}")
    print(f"  model calls   {outcome['model_calls']}   "
          f"tokens in/out {outcome['input_tokens']}/{outcome['output_tokens']}   "
          f"{outcome['elapsed_s']:.1f}s")

    if outcome["status"] != "awaiting_review":
        # A rejected review scores ZERO, and saying so plainly matters more than
        # any metric: the SAO saw nothing at all.
        print("\n  REVIEW PRODUCED NO FINDINGS — this case contributes zero recall.")
        for reason in outcome["rejected"]:
            print(f"    reject reason: {reason}")

    mc_total, mc_caught = scored["must_catch_total"], scored["must_catch_caught"]
    print(f"\n  must_catch recall     {mc_caught}/{mc_total}   (1 document)")
    for label, key in (
        ("missed, called pass", "missed_as_pass"),
        ("missed, called unclear", "missed_as_unclear"),
        ("missed, no verdict", "missed_absent"),
    ):
        if scored[key]:
            print(f"    {label:<24} {', '.join(scored[key])}")

    print(f"  agreement on mentioned rules          "
          f"{scored['mentioned_agreements']}/{scored['mentioned_total']}")

    adjudication = scored["unscoreable_fails"]
    print(f"  fails on rules NOT in the key         {len(adjudication)}"
          "   -> SAO adjudication (not scored as false positives)")
    for item in adjudication:
        quote = (item["evidence"] or "").replace("\n", " ")[:80]
        print(f"      {item['rule_id']:<11} {item['severity'] or '?':<8} \"{quote}\"")

    if scored["severity_mismatches"]:
        print(f"  !! SEVERITY MISMATCH   {', '.join(scored['severity_mismatches'])}"
              "   (code defect — severity is joined from the rulebook)")

    if outcome["retries"]:
        print(f"\n  validation retries    {len(outcome['retries'])} "
              "(cause shown — a rate without causes is unactionable)")
        for attempt in outcome["retries"]:
            for err in attempt[:4]:
                print(f"    - {err[:100]}")


# --- derived metrics (reporting only; score() is unchanged) ------------------
# These read score()'s output and the run's findings. They add nothing to what is
# measured — they only shape the requested precision / FPR views for the report.

SIMULATED_LABEL = (
    "SIMULATED evaluation data — a demonstration of the evaluation METHOD, not a "
    "measurement of real-world quality. Real quality needs the SAO's real past "
    "reviews as the answer key; these documents and keys were hand-authored for "
    "the demo."
)

OUT_DIR = Path(__file__).resolve().parent


def _precision_over_mentioned(findings: list[dict], golden: dict):
    """Of the agent's FAIL verdicts on rules the key MENTIONS, how many match a
    golden fail. Unmentioned fails are excluded (they are adjudication items, not
    false positives — §1.1). Returns (matched, total) or None if no such fails."""
    expected = {f["rule_id"]: f for f in golden["expected_findings"]}
    fails = [f for f in findings
             if f["verdict"] == "fail" and f["rule_id"] in expected]
    if not fails:
        return None
    matched = sum(1 for f in fails if expected[f["rule_id"]]["verdict"] == "fail")
    return matched, len(fails)


def _false_positive_rate(findings: list[dict], golden: dict):
    """Only meaningful on a CLEAN design (no expected fails): every fail the agent
    produces is a false positive. Returns (fails, findings_total) or None."""
    if any(f["verdict"] == "fail" for f in golden["expected_findings"]):
        return None
    return sum(1 for f in findings if f["verdict"] == "fail"), len(findings)


def _score_one_run(outcome: dict, golden: dict) -> dict:
    """One run's numbers: score() plus the derived precision / FPR views."""
    scored = score(outcome["findings"], golden)
    return {
        "run_id": outcome["run_id"],
        "status": outcome["status"],
        "must_catch_caught": scored["must_catch_caught"],
        "must_catch_total": scored["must_catch_total"],
        "missed_as_pass": scored["missed_as_pass"],
        "missed_as_unclear": scored["missed_as_unclear"],
        "precision_mentioned": _precision_over_mentioned(outcome["findings"], golden),
        "false_positive_rate": _false_positive_rate(outcome["findings"], golden),
        "unscoreable_fails": scored["unscoreable_fails"],
        "severity_mismatches": scored["severity_mismatches"],
        "n_findings": len(outcome["findings"]),
        "n_fails": sum(1 for f in outcome["findings"] if f["verdict"] == "fail"),
        "input_tokens": outcome["input_tokens"],
        "output_tokens": outcome["output_tokens"],
        "model_calls": outcome["model_calls"],
        "validation_retries": len(outcome["retries"]),
    }


def _spread(values: list) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"per_run": values, "min": None, "max": None, "varies": False}
    return {"per_run": values, "min": min(vals), "max": max(vals),
            "varies": min(vals) != max(vals)}


def _run_document(case, golden: dict, repeat: int, seed_fn) -> dict:
    """Run one document `repeat` times, each on a freshly seeded database so
    per-run token counts and findings do not bleed across runs."""
    subject, project_id, artifact_name, _ = case
    runs = []
    for i in range(repeat):
        seed_fn()  # truncate + reseed: isolate this run's audit + findings
        outcome = run_case(subject, project_id, artifact_name)
        runs.append(_score_one_run(outcome, golden))
        r = runs[-1]
        print(f"    run {i + 1}/{repeat}: status={r['status']:<15} "
              f"must_catch {r['must_catch_caught']}/{r['must_catch_total']}  "
              f"fails={r['n_fails']}  unscoreable={len(r['unscoreable_fails'])}")
    return {
        "artifact": artifact_name,
        "subject": subject,
        "org_id": golden["meta"].get("org_id"),
        "design_class": golden["meta"].get("design_class", "mixed"),
        "must_catch_total": runs[0]["must_catch_total"],
        "runs": runs,
        "recall_variation": _spread([r["must_catch_caught"] for r in runs]),
        "fails_variation": _spread([r["n_fails"] for r in runs]),
        "unscoreable_union": sorted({u["rule_id"]
                                     for r in runs for u in r["unscoreable_fails"]}),
    }


def _write_results_json(documents: list[dict], repeat: int) -> Path:
    import datetime
    from review_agent.rules.loader import load_rulebook
    rb = load_rulebook()
    payload = {
        "meta": {
            "simulated": True,
            "label": SIMULATED_LABEL,
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "rulebook_version": rb.version,
            "rulebook_sha256": rb.sha256,
            "documents": len(documents),
            "runs_per_document": repeat,
        },
        "documents": documents,
    }
    path = OUT_DIR / "results.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def _fmt_frac(pair):
    return "n/a" if pair is None else f"{pair[0]}/{pair[1]}"


def _write_results_md(documents: list[dict], repeat: int) -> Path:
    lines = [
        "# Evaluation results — SIMULATED",
        "",
        f"> **{SIMULATED_LABEL}**",
        "",
        "This shows the evaluation **method** running on hand-authored documents "
        "and answer keys. It is not a verdict on the agent's real-world accuracy. "
        f"Each document was run **{repeat} times**; run-to-run variation is shown, "
        "not smoothed — the variation is real signal, not noise to average away.",
        "",
        "## Per document",
        "",
        "| Document | Design class | must_catch recall (per run) | fails (per run) "
        "| precision¹ (last run) | FPR² | unscoreable³ |",
        "|---|---|---|---|---|---|---|",
    ]
    for d in documents:
        recall = "  ".join(f"{r['must_catch_caught']}/{r['must_catch_total']}"
                           for r in d["runs"])
        fails = "  ".join(str(r["n_fails"]) for r in d["runs"])
        last = d["runs"][-1]
        prec = _fmt_frac(last["precision_mentioned"])
        fpr = _fmt_frac(last["false_positive_rate"]) if last["false_positive_rate"] \
            else "—"
        unsc = ", ".join(d["unscoreable_union"]) or "none"
        star = " ⚠" if d["recall_variation"]["varies"] else ""
        lines.append(
            f"| `{d['artifact']}` | {d['design_class']} | {recall}{star} | {fails} "
            f"| {prec} | {fpr} | {unsc} |")
    lines += [
        "",
        "¹ precision over rules the key mentions — of the agent's fails on mentioned "
        "rules, the fraction that match a golden fail. Fails on UNmentioned rules "
        "are not counted here; they are adjudication items (below).",
        "² false-positive rate — only defined for the CLEAN design (`atlas`), where "
        "every fail is a false positive. `—` elsewhere.",
        "³ unscoreable fails — fails on rules the answer key does not mention "
        "(union across runs). Not scored; routed to the SAO to rule on.",
        "⚠ = must_catch recall varied between runs — same input, different verdicts.",
        "",
        "## Adjudication list (unscoreable fails)",
        "",
        "Each is a fail the agent produced on a rule the answer key does not "
        "mention. The SAO rules each a real defect (→ new expected finding) or a "
        "false positive (→ documented pass case). This is how a real golden set "
        "would grow beyond the hand-authored one.",
        "",
    ]
    any_adj = False
    for d in documents:
        items = {u["rule_id"]: u for r in d["runs"] for u in r["unscoreable_fails"]}
        if not items:
            continue
        any_adj = True
        lines.append(f"- **{d['artifact']}**")
        for rid, u in sorted(items.items()):
            quote = (u.get("evidence") or "").replace("\n", " ")[:90]
            lines.append(f"    - {rid} ({u.get('severity', '?')}): \"{quote}\"")
    if not any_adj:
        lines.append("_None this run._")
    lines += [
        "",
        "## What this does and does not show",
        "",
        "- **Shows:** the method works end to end — real ingestion, guardrails, the "
        "live model, persisted findings scored against an independent key; and that "
        "the numbers move between identical runs, which is why a single run is not a "
        "measurement.",
        "- **Does NOT show:** real-world accuracy. The documents are synthetic and "
        "the keys were written by the same author. A high score here can mean the "
        "prompt has learned these documents, not the standards. Real measurement "
        "needs the SAO's real past reviews as the key.",
        "- No threshold, no pass/fail bar, no aggregate 'grade' — deliberately.",
        "",
    ]
    path = OUT_DIR / "results.md"
    path.write_text("\n".join(lines))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true",
                        help="re-provision before running (drops and recreates)")
    parser.add_argument("--repeat", type=int, default=3,
                        help="runs per document, to expose run-to-run variation")
    args = parser.parse_args()

    from review_agent.orchestration.run import _load_dotenv

    _load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("No ANTHROPIC_API_KEY. Evals make real model calls.", file=sys.stderr)
        return 2

    from review_agent.data.provision import bootstrap
    from review_agent.data.seed import seed_sample_data, truncate_all

    if args.reset:
        bootstrap(reset=True)

    def seed_fn():
        truncate_all()
        seed_sample_data()

    print(SIMULATED_LABEL)
    print(f"\n{N} documents × {args.repeat} runs = {N * args.repeat} reviews "
          "(each a live model call).\n")

    documents = []
    for case in CASES:
        golden = load_golden(case[3])
        print(f"{case[2]}  [{golden['meta'].get('design_class', 'mixed')}]")
        documents.append(_run_document(case, golden, args.repeat, seed_fn))

    json_path = _write_results_json(documents, args.repeat)
    md_path = _write_results_md(documents, args.repeat)

    # Compact aggregate to the terminal; the files carry the detail.
    print(f"\n{'=' * 74}\nAGGREGATE  (SIMULATED — method demo, not a quality measure)")
    print("=" * 74)
    total_mc = sum(d["must_catch_total"] * args.repeat for d in documents)
    caught_mc = sum(r["must_catch_caught"] for d in documents for r in d["runs"])
    print(f"  must_catch recall (all runs)   {caught_mc}/{total_mc}")
    varied = [d["artifact"] for d in documents if d["recall_variation"]["varies"]]
    print(f"  documents whose recall VARIED between runs: "
          f"{', '.join(varied) if varied else 'none'}")
    clean = next((d for d in documents if d["design_class"] == "clean"), None)
    if clean:
        fprs = [r["false_positive_rate"] for r in clean["runs"]]
        print(f"  false positives on the CLEAN design (fails/run): "
              f"{[fp[0] for fp in fprs if fp]}")
    total_adj = sum(len(d["unscoreable_union"]) for d in documents)
    print(f"  adjudication items (unscoreable fails): {total_adj}")
    print(f"\n  wrote {json_path.relative_to(OUT_DIR.parent)}  and  "
          f"{md_path.relative_to(OUT_DIR.parent)}")
    print("  No threshold, no grade. Variation shown, not smoothed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
