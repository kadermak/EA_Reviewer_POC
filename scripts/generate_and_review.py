"""Autonomous generate -> review -> revise loop (Task 2).

    python scripts/generate_and_review.py
    python scripts/generate_and_review.py \
        --requirements "design a customer portal that stores personal data and exposes a public API" \
        --max-iterations 3

Wires two INDEPENDENT agents against the live models:

  * design_generator  (DEFAULT model) drafts from REQUIREMENTS. It never sees the
    EA rulebook — the only feedback it gets is the critic's fail findings.
  * conformance_agent (JUDGMENT model) is the critic. It loads the rulebook the
    generator cannot see and judges the draft against it.

Each round: generate/revise -> review -> if any `fail` findings remain, format
them as plain revision text and hand them back. Stops after N rounds or when no
fails remain. There is no persistence, no database, and no orchestration graph —
just the two model-facing functions, so the reasoning loop is visible end to end.

This is a demonstration harness, not part of the reviewed system: it prints to
stdout and writes nothing.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from collections import Counter

# Make the script self-sufficient: run it directly from the repo without relying
# on the editable install being importable from the current working directory.
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from review_agent.agents import conformance_agent, design_generator  # noqa: E402
from review_agent.rules.loader import load_rulebook  # noqa: E402

SAMPLE_REQUIREMENT = (
    "design a customer portal that stores personal data and exposes a public API"
)

_MARK = {"fail": "X", "pass": "+", "unclear": "?"}


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env reader so the live models have a credential.

    Deliberately tiny and local — the loop needs no database, so it does not pull
    in the orchestration entry point just to read a file. A duplicate key is
    refused rather than silently resolved by line order.
    """
    if not os.path.exists(path):
        return
    seen: dict[str, int] = {}
    with open(path, encoding="utf-8") as handle:
        for number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key in seen:
                raise SystemExit(
                    f"{path} sets {key} twice (lines {seen[key]} and {number}). "
                    "Delete one — most likely the .env.example placeholder."
                )
            seen[key] = number
            os.environ.setdefault(key, value.strip())


def _preflight_credential() -> str | None:
    credential = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get(
        "ANTHROPIC_AUTH_TOKEN"
    )
    if not credential or credential.strip() in ("", "sk-...") or credential.startswith(
        "sk-..."
    ):
        return (
            "No usable ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN found (the "
            ".env.example placeholder does not count). Add a real key to .env."
        )
    return None


def _format_feedback(fails) -> str:
    """Turn fail findings into revision text.

    Only fields a `Finding` carries cross into the generator: rule_id, severity,
    the reviewer's reasoning, and the quoted evidence. Rule text from the
    catalogue never does — that boundary is what keeps the generator working from
    the critic's judgement rather than from the rulebook itself.
    """
    lines = []
    for finding in fails:
        evidence = " ".join((finding.evidence or "").split())
        if len(evidence) > 160:
            evidence = evidence[:160] + "..."
        line = (
            f"- [{finding.rule_id}] (severity: {finding.severity}) "
            f"{finding.reasoning.strip()}"
        )
        if evidence:
            line += f'\n  Flagged text in your draft: "{evidence}"'
        lines.append(line)
    return "\n".join(lines)


def _print_findings(result) -> list:
    counts = Counter(f.verdict for f in result.findings)
    fails = [f for f in result.findings if f.verdict == "fail"]
    print(
        f"  review: {len(result.findings)} findings  "
        f"pass={counts['pass']} fail={counts['fail']} unclear={counts['unclear']}"
    )
    for finding in sorted(result.findings, key=lambda f: f.rule_id):
        reasoning = " ".join(finding.reasoning.split())[:96]
        print(
            f"    {_MARK.get(finding.verdict, '?')} {finding.rule_id:<11} "
            f"{finding.verdict:<8} {finding.severity:<8} {reasoning}"
        )
    return fails


def run_loop(requirements: str, max_iterations: int = 3) -> int:
    # The CRITIC's rulebook. It is loaded HERE and passed only to the conformance
    # agent; the generator is never handed it.
    rulebook = load_rulebook()
    print(f"requirements: {requirements}")
    print(
        f"rulebook: {rulebook.name} v{rulebook.version} "
        f"({len(rulebook.rules)} rules)  |  generator does NOT receive it\n"
    )

    feedback: str | None = None
    prior_design: str | None = None

    for iteration in range(1, max_iterations + 1):
        print("=" * 78)
        print(f"ITERATION {iteration}/{max_iterations}")
        print("=" * 78)

        gen = design_generator.generate_design(
            requirements, feedback=feedback, prior_design=prior_design
        )
        if not gen.ok:
            print(f"generation failed: {gen.reject_reason}")
            return 2
        design = gen.markdown

        print("\n--- draft design ---\n")
        print(design)
        print("\n--- conformance review ---")

        result = conformance_agent.review(design, rulebook)
        if not result.accepted:
            # The critic produced no usable findings (a refusal or repeated
            # invalid output). Nothing to feed back, so the loop stops honestly.
            print(f"  review produced no findings: {result.reject_reason}")
            return 3

        fails = _print_findings(result)
        print()

        if not fails:
            print(
                f"CONVERGED after {iteration} iteration(s): no fail findings "
                "remain (unclear items are submission gaps, not violations)."
            )
            return 0

        if iteration == max_iterations:
            print(
                f"STOPPED at the {max_iterations}-iteration limit with "
                f"{len(fails)} fail finding(s) still open."
            )
            return 1

        # Hand the fails back and go round again.
        feedback = _format_feedback(fails)
        prior_design = design
        print(f"-> feeding {len(fails)} fail finding(s) back for revision\n")

    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", default=SAMPLE_REQUIREMENT)
    parser.add_argument("--max-iterations", type=int, default=3)
    args = parser.parse_args()

    _load_dotenv()
    problem = _preflight_credential()
    if problem:
        print(problem, file=sys.stderr)
        return 2

    return run_loop(args.requirements, max_iterations=args.max_iterations)


if __name__ == "__main__":
    raise SystemExit(main())
