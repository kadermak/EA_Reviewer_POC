# Reviewer UI

**Status: the API surface is complete and property-tested; the front end is a
minimal reference page, not the planned React/Next.js app.**

That is a deliberate deviation from the stated tech decision, flagged rather than
made silently. The reasoning:

- Every constraint in PHASE3_DESIGN §4 lives in `api/` — no tenant identifier on
  the wire, no second read path, no aggregate verdict, no verdict mutation. The
  UI calls HTTP; a rule aimed at UI code would be unenforceable while appearing
  enforced.
- A Next.js app adds a Node toolchain to a repo whose entire test story is one
  `pytest` invocation against a real PostgreSQL. The POC's value is the isolation
  proof, and a second build system does not add to it.

`reviewer.html` is a single self-contained page that exercises the real endpoints.
Replacing it with a React/Next.js front end requires no API change — which is the
property that made deferring it safe.

## What the front end must preserve

These are decisions, not styling preferences (PHASE3_DESIGN §4.0, §4.4, §4.5):

1. **No aggregate verdict.** No score, ratio, or pass rate — not even "8/14".
   A count of findings *needing attention* is permitted; a denominator is not.
2. **Passes are shown in full and individually overridable, but never block.**
   Requiring a click on each produces rubber-stamping, and an incorrect pass is
   caught by nobody whether or not the click happened.
3. **`unclear` is an open gap**, counted separately, never folded into "passing",
   with its severity labelled *potential* exposure.
4. **Injection flags are document-level**, phrased as an observation with a next
   step, and cannot be dismissed or acknowledged.

The reference page now DEMONSTRATES 1–3 rather than merely permitting them, after
a local run showed it violating their spirit:

- **Severity is suppressed on passes.** `EA-SEC-01 · PASS · CRITICAL` read as a
  critical *problem*; severity is a rule property, not a verdict, and on a pass it
  carries no exposure. It is shown on fail (real) and unclear (`potential …`).
- **Passes are overridable but not presented as awaiting action.** They show a
  single *Override this pass*, not the fail's Accept/Override/Waive triad and not
  a `pending` marker — the rubber-stamping shape the fail/unclear-only design
  avoids. A pass never gates completion.
- **The "N still need your decision" count and the *decision required* badges are
  derived from ONE predicate** (`decision_required && action === "pending"`), so
  the header and the badges cannot show different numbers as decisions are made.
