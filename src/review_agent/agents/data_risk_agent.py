"""FUTURE: Data risk register review agent (NOT in the POC).

Placeholder to show where a new review domain plugs in. Adding this domain is a
normal code change:
  - implement review() below (the LOGIC)
  - add its criteria as DATA in sample-data/data_risk_rules.json
  - register it in orchestration/graph.py
  - add evals + tests
Logic lives here in code; the criteria live in the data file.
"""

# TODO(future): implement risk-register review analogous to conformance_agent.


def review(artifact: dict, risk_rules: list[dict]) -> list[dict]:
    """Return structured data-risk findings. TODO(future)."""
    raise NotImplementedError
