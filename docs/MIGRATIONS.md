# Schema migrations

This project provisions schema with `Base.metadata.create_all` (see
`data/provision.py`). **`create_all` only creates missing tables — it never adds
columns to a table that already exists.** So a new column on an existing table is
invisible until either:

- the database is rebuilt: `python -m review_agent.data.provision --reset` (this
  **drops all data** — fine for dev, never for a database you want to keep), or
- the column is added manually with `ALTER TABLE` (preserves data — use this for
  any populated database).

There is no migration framework; record each column-adding change here with its
`ALTER` so a populated deployment can be brought forward without a reset.

## Advisory model reasoning (findings.reasoning, review_runs.thinking_trace)

Adds the model's per-rule rationale and the whole-review reasoning trace (both
advisory, unvalidated, shown collapsed in the UI). Safe to apply online — both
are additive and non-blocking; `reasoning` defaults to `''` so existing rows and
any insert path that omits it stay valid.

```sql
ALTER TABLE findings     ADD COLUMN reasoning      text NOT NULL DEFAULT '';
ALTER TABLE review_runs  ADD COLUMN thinking_trace text;            -- nullable
```

Notes:
- No backfill: rows written before this change carry `reasoning = ''` (the UI
  simply shows no reasoning block for them) and `thinking_trace = NULL`.
- No RLS/policy/grant change is needed — `findings` and `review_runs` are already
  tenant-scoped, and adding a non-tenant column does not touch the isolation
  controls (verified: `provision --reset` re-runs the drift checks clean).
