# Enterprise Architecture Review Agent (POC)

An advisory tool that reviews a project's architecture design against enterprise
architecture (EA) standards and flags gaps for a human reviewer in the Solution
Architect Office (SAO). It advises; a human always signs off.

> This is a proof of concept. It ships with **simulated** standards and mock
> organisations so it can be built and tested before real data is available.

## Status
POC. One review domain (architecture conformance), two mock organisations, full
tenant-isolation + guardrail + audit machinery.

## Requirements
- Python 3.11+
- PostgreSQL 15+ (row-level security is central to the design)
- Node 18+ (for the reviewer UI)

## Quick start (developers)
```bash
# 1. Install (Python 3.11+)
pip install -e ".[dev]"

# 2. Bring up PostgreSQL 15+ — the red-team suite requires a real one and will
#    FAIL rather than skip without it (row-level security is a Postgres feature,
#    so a suite that skips could report success on a machine with no database).
docker run -d --name ra-postgres -e POSTGRES_PASSWORD=devpw \
  -e POSTGRES_DB=review_agent -p 5433:5432 postgres:16
cp .env.example .env

# 3. Provision roles, schema, RLS policies, grants and triggers (idempotent).
#    This refuses to finish if isolation is not verifiably intact.
python -m review_agent.data.provision --reset

# 4. Seed the two mock organisations + sample data
python -m review_agent.data.seed

# 5. Run the tests — the isolation red-team suite is the gate
pytest tests/test_isolation_redteam.py -v
```

## The one thing to understand
Tenant isolation is enforced at the **database** (row-level security on `org_id`),
never by asking the model to behave. Organisation A can never read Organisation B's
data because it is never fetched. `tests/test_isolation_redteam.py` proves this.

## Build order (phase-1 first)
See `docs`-level checklist below. Isolation core first, agent second, guardrails and
flow third, evaluation last.

### Phase 1 — Foundation & isolation core (prove before anything else) — DONE
Design: [`docs/PHASE1_DESIGN.md`](docs/PHASE1_DESIGN.md).
1. ✅ Repo scaffold (this).
2. ✅ PostgreSQL tenant schema (`src/review_agent/data/models.py`).
3. ✅ Row-level security (`src/review_agent/data/rls.py`) — the crux.
4. ✅ Auth context + scope resolution (`src/review_agent/data/scope.py`).
5. ✅ Isolation red-team tests (`tests/test_isolation_redteam.py`) — THE gate.

The suite makes **zero model calls** and connects as the unprivileged `review_app`
role (asserted before any test runs). Nothing in it depends on an LLM behaving
correctly — where that dependency could have crept in, it is documented as a
design bug and removed: see §5 of the design doc.

### Phase 2 — One working review path — DONE
Design: [`docs/PHASE2_DESIGN.md`](docs/PHASE2_DESIGN.md).
Built in the order **9 → 7 → 6 → 8**: the model wrapper must exist before
anything calls a model, or provider-specific calls leak into the agents and
swappability is lost.
9. ✅ Model abstraction (`src/review_agent/models/`) — one call surface, role
   routing, provider confined to a single adapter (lint-enforced).
7. ✅ Rulebook loader (`src/review_agent/rules/loader.py`) — full rulebook in
   context, no retrieval, version **and** hash.
6. ✅ Ingestion (`src/review_agent/ingestion/`) — extract; sanitise at
   prompt-construction, with raw text stored for audit.
8. ✅ Conformance agent (`src/review_agent/agents/conformance_agent.py`) — one
   verdict per rule, strict schema, whole-review rejection on invalid output.

Phase 2 produces **well-formed** findings. How good they are is Phase 4's
question, measured against the golden keys — no prompt tuning before then.

### Phase 3 — Guardrails, flow & human review — DONE
Design: [`docs/PHASE3_DESIGN.md`](docs/PHASE3_DESIGN.md). Built 12 -> 11 -> 10 -> 13.
12. ✅ LangGraph orchestration (`src/review_agent/orchestration/graph.py`) —
    checkpointer under RLS, bounded transaction segments, scope never checkpointed.
11. ✅ Output review (`src/review_agent/guardrails/output_review.py`) — may block
    or redact text; never alters a verdict. Enforced at the write, not the caller.
10. ✅ Input guardrail (`src/review_agent/guardrails/input_guard.py`) — deterministic
    checks only; the route shape remains the primary out-of-scope control.
13. ✅ Reviewer surface (`src/review_agent/api/app.py`, `ui/`) — four scoped
    routes, no aggregate verdict, decisions required only on `fail`/`unclear`.
    Front end is a reference page, not Next.js — see `ui/README.md`.

### Phase 4 — Prove & assess
14. Golden-set evaluation (`evals/run_evals.py`).
15. Write-up & go/no-go.

## Extending later (e.g. data risk register review)
Adding a new review domain is a normal code change, not a config trick:
- new agent module in `src/review_agent/agents/` (e.g. `data_risk_agent.py`)
- new criteria as **data** in `sample-data/` (e.g. `data_risk_rules.json`)
- register it in `src/review_agent/orchestration/graph.py`
- add evals + tests
Logic goes in code; rules go in data. Nothing here is a "skill" file.

## Repo layout

```
architecture-review-agent/
├── src/review_agent/
│   ├── data/            # isolation core: RLS policies, ORM models, provisioning, scoped repository + audit
│   ├── models/          # provider-neutral model client
│   │   └── providers/   # anthropic.py, ollama.py adapters
│   ├── agents/          # review logic (conformance, design generator)
│   ├── guardrails/      # input/output safety checks
│   ├── ingestion/       # artifact sanitisation (prompt-injection defence)
│   ├── orchestration/   # LangGraph review flow + run.py entry point
│   ├── rules/           # rulebook loader
│   ├── audit/           # append-only audit trail
│   ├── api/             # FastAPI reviewer/submitter endpoints
│   └── findings.py      # the finding contract
├── sample-data/         # EA rulebook, approved-tech catalogue, mock orgs, artifacts, golden keys
├── tests/               # red-team isolation suite + agent/loop tests
├── evals/               # golden-set quality evaluation
├── scripts/             # generate_and_review.py autonomous loop
├── ui/                  # reviewer.html, submit.html (reference pages)
├── docs/                # design docs (PHASE1–4), ARCHITECTURE, MIGRATIONS, …
├── dev_ui.py            # local dev server for the UI
└── .env.example         # config template
```

`src/review_agent/data/` is the isolation core (build and prove it first);
`agents/` holds review logic, `guardrails/` the safety checks, `sample-data/` the
fixtures, `evals/` measures quality, and `tests/` proves it.

## Documentation
Deeper design docs live in [`docs/`](docs/): start with
[`ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the system overview,
[`POC_VS_PRODUCTION.md`](docs/POC_VS_PRODUCTION.md) for the design decisions and
what a prod build would change, the phase docs (`PHASE1`–`PHASE4_DESIGN.md`) for
the rationale behind each layer, and [`MIGRATIONS.md`](docs/MIGRATIONS.md) for
schema changes.
