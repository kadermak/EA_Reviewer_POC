# Evaluation results — SIMULATED

> **SIMULATED evaluation data — a demonstration of the evaluation METHOD, not a measurement of real-world quality. Real quality needs the SAO's real past reviews as the answer key; these documents and keys were hand-authored for the demo.**

This shows the evaluation **method** running on hand-authored documents and answer keys. It is not a verdict on the agent's real-world accuracy. Each document was run **3 times**; run-to-run variation is shown, not smoothed — the variation is real signal, not noise to average away.

## Per document

| Document | Design class | must_catch recall (per run) | fails (per run) | precision¹ (last run) | FPR² | unscoreable³ |
|---|---|---|---|---|---|---|
| `artifact_org-a_proj-a1.md` | mixed | 4/4  4/4  4/4 | 5  4  5 | 4/4 | — | EA-DAT-01 |
| `artifact_org-b_proj-b1.md` | mixed | 5/6  6/6  6/6 ⚠ | 5  6  6 | 6/6 | — | none |
| `artifact_org-a_proj-a1_atlas.md` | clean | 6/7  6/7  7/7 ⚠ | 0  0  0 | n/a | 0/14 | none |
| `artifact_org-a_proj-a1_lakehouse.md` | fails-data-obs-tech | 4/4  4/4  4/4 | 4  4  4 | 4/4 | — | none |
| `artifact_org-b_proj-b1_portal.md` | fails-identity-integration | 4/4  4/4  4/4 | 4  4  4 | 4/4 | — | none |
| `artifact_org-b_proj-b1_sketch.md` | unclear-heavy | 4/4  3/4  3/4 ⚠ | 1  2  2 | 1/2 | — | none |

¹ precision over rules the key mentions — of the agent's fails on mentioned rules, the fraction that match a golden fail. Fails on UNmentioned rules are not counted here; they are adjudication items (below).
² false-positive rate — only defined for the CLEAN design (`atlas`), where every fail is a false positive. `—` elsewhere.
³ unscoreable fails — fails on rules the answer key does not mention (union across runs). Not scored; routed to the SAO to rule on.
⚠ = must_catch recall varied between runs — same input, different verdicts.

## Adjudication list (unscoreable fails)

Each is a fail the agent produced on a rule the answer key does not mention. The SAO rules each a real defect (→ new expected finding) or a false positive (→ documented pass case). This is how a real golden set would grow beyond the hand-authored one.

- **artifact_org-a_proj-a1.md**
    - EA-DAT-01 (medium): "reads directly from the Payment Orchestrator's   database to build finance reports."

## What this does and does not show

- **Shows:** the method works end to end — real ingestion, guardrails, the live model, persisted findings scored against an independent key; and that the numbers move between identical runs, which is why a single run is not a measurement.
- **Does NOT show:** real-world accuracy. The documents are synthetic and the keys were written by the same author. A high score here can mean the prompt has learned these documents, not the standards. Real measurement needs the SAO's real past reviews as the key.
- No threshold, no pass/fail bar, no aggregate 'grade' — deliberately.
