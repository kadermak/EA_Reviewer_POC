"""Extract text (and later, diagram content) from uploaded design artifacts.

Output is clean text. It is NOT scope-tagged here: org_id and project_id are
stamped from CallerScope at insert (data/repository.insert_artifact), and this
module never sees a scope at all.

THE RULE THIS MODULE EXISTS NOT TO BREAK: nothing here parses the document for
identifiers. `artifact_org-a_proj-a1.md` states its own org and project in its
header, and an extractor that already parses document structure is exactly where
"we could just auto-detect the project" becomes tempting. The attacker writes the
header. Parsed metadata may be shown to a human as UNVERIFIED and clearly
labelled — it is never written to a scope column.

See docs/PHASE2_DESIGN.md §3.1-3.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Phase 2 ships text formats only. DOCX/PDF/vision are additive implementations
# behind this same signature — deliberately not pulling a large extraction
# dependency in to read Markdown.
SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}


class ExtractionError(Exception):
    """The file could not be turned into reviewable text."""


@dataclass(frozen=True)
class ExtractedDocument:
    text: str
    source_format: str
    warnings: tuple[str, ...] = field(default_factory=tuple)


def extract(file_path: str | Path) -> ExtractedDocument:
    """Return reviewable text from an uploaded artifact.

    The seam for later formats: a DOCX or PDF extractor is a new branch here
    returning the same type. Ingestion, the agent, and storage do not change.
    """
    path = Path(file_path)
    if not path.is_file():
        raise ExtractionError(f"no such artifact file: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ExtractionError(
            f"unsupported artifact format {suffix!r}. Phase 2 handles "
            f"{sorted(SUPPORTED_SUFFIXES)}; DOCX/PDF/diagram extraction is a "
            "later addition behind this same function."
        )

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ExtractionError(f"{path.name} is not valid UTF-8 text: {exc}") from exc

    warnings: list[str] = []
    if not text.strip():
        warnings.append("artifact is empty")

    return ExtractedDocument(
        text=text, source_format=suffix.lstrip("."), warnings=tuple(warnings)
    )
