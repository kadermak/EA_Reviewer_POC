"""Load the standards rulebook(s) into memory for a review.

For the POC there is NO retrieval / vector search: the rule set is small, so the
FULL rulebook is loaded into the agent's context each review. Rules are DATA
(sample-data/*.json), kept separate from tenant artifacts. Adding a domain (e.g.
data risk) means loading a different rules file — not changing this loader's shape.

Full-context loading is a CORRECTNESS decision before it is a cost one:
retrieval can silently omit a rule, and "the agent never checked EA-SEC-01" is
indistinguishable to the reviewer from "the agent checked it and it passed". For
a compliance tool that is the worst available failure mode. See design §2.1.

The rulebook is GLOBAL. It is never stored in Postgres, never travels through a
scoped session, and no rule has an org_id — which is what keeps the Phase 1
drift check able to say "every table is tenant-scoped" with no exemptions.

See docs/PHASE2_DESIGN.md §2.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

SAMPLE_DATA = Path(__file__).resolve().parents[3] / "sample-data"

REQUIRED_RULE_FIELDS = ("id", "category", "statement", "severity", "check_hint")


class RulebookError(Exception):
    """The rulebook is malformed. Always fatal at load time.

    A bad rulebook is a configuration error: it must stop the process at
    startup, not surface hours later as a strange finding. Same posture as the
    Phase 1 verify_isolation_or_raise() — refuse to start rather than run wrong.
    """


@dataclass(frozen=True)
class Rule:
    id: str
    category: str
    statement: str
    severity: str
    check_hint: str
    rationale: str | None = None
    source_ref: str | None = None


@dataclass(frozen=True)
class Rulebook:
    """The full, validated rule set for one review domain."""

    name: str
    version: str
    sha256: str
    rules: tuple[Rule, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(rule.id for rule in self.rules)

    @property
    def by_id(self) -> dict[str, Rule]:
        return {rule.id: rule for rule in self.rules}

    def severity_for(self, rule_id: str) -> str:
        """Severity is a property of the RULE, not of the model's judgement.

        The model never emits severity (design §4.2, BUG-9) — it is joined here.
        Verified against the sample data: all 10 golden `fail` findings carry the
        severity their rule carries, and no golden `pass` finding carries one.
        """
        return self.by_id[rule_id].severity

    def as_prompt_json(self) -> str:
        """Deterministic serialisation for the prompt.

        Sorted keys and fixed separators, so the same rulebook always renders
        byte-identically. Non-deterministic JSON is the classic silent
        cache invalidator — this costs nothing now and is the prerequisite for
        prompt caching once the real catalogue clears the 4096-token minimum.
        """
        payload = [
            {
                "id": r.id,
                "category": r.category,
                "statement": r.statement,
                "check_hint": r.check_hint,
            }
            for r in self.rules
        ]
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), indent=1)


def _validate(raw: dict, name: str) -> None:
    meta = raw.get("rulebook_meta")
    if not isinstance(meta, dict) or not meta.get("version"):
        raise RulebookError(f"{name}: missing rulebook_meta.version")

    rules = raw.get("rules")
    if not isinstance(rules, list) or not rules:
        raise RulebookError(f"{name}: rulebook contains no rules")

    scale = set(meta.get("severity_scale") or ())
    seen: set[str] = set()
    for index, rule in enumerate(rules):
        missing = [f for f in REQUIRED_RULE_FIELDS if not rule.get(f)]
        if missing:
            raise RulebookError(
                f"{name}: rule #{index} ({rule.get('id', '<no id>')}) "
                f"is missing {missing}"
            )
        if rule["id"] in seen:
            raise RulebookError(f"{name}: duplicate rule id {rule['id']!r}")
        seen.add(rule["id"])
        if scale and rule["severity"] not in scale:
            raise RulebookError(
                f"{name}: rule {rule['id']} has severity {rule['severity']!r}, "
                f"which is not in rulebook_meta.severity_scale {sorted(scale)}"
            )


@lru_cache(maxsize=8)
def load_rulebook(name: str = "ea_standards.json") -> Rulebook:
    """Load, validate and hash a rulebook by filename.

    `name` is a deployment/config value. It is NEVER derived from an uploaded
    artifact, a request field, or model output (design BUG-14): an artifact that
    could select its own rulebook could select an empty one and pass everything.

    Cached per process — the SAO changes WHAT is checked by editing the JSON and
    redeploying. Hot reload is a later concern.
    """
    path = SAMPLE_DATA / name
    if path.name != name or not path.is_file():
        raise RulebookError(f"no such rulebook: {name!r}")

    text = path.read_text()
    raw = json.loads(text)
    _validate(raw, name)

    return Rulebook(
        name=name,
        version=raw["rulebook_meta"]["version"],
        # Hash the file bytes, not the parsed form: VERSION ALONE IS NOT ENOUGH.
        # The SAO edits this file to change what is checked — that is the whole
        # point of rules-as-data — and nothing forces a version bump. Two reviews
        # could both claim "0.1-sample" against materially different rule text.
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        rules=tuple(
            Rule(
                id=r["id"],
                category=r["category"],
                statement=r["statement"],
                severity=r["severity"],
                check_hint=r["check_hint"],
                rationale=r.get("rationale"),
                source_ref=r.get("source_ref"),
            )
            for r in raw["rules"]
        ),
    )
