"""Single model-call abstraction so the provider is swappable.

ALL model calls go through here. Business logic (agents, guardrails) must never
call a provider SDK directly. Switching Claude <-> Gemini should be a change in
the provider adapter (or a config value), not across the codebase.

Roles map to models via env (see .env.example):
  default   -> everyday agent work        (claude-sonnet-5)
  judgment  -> conformance + output review (claude-opus-4-8)
  classify  -> cheap classification        (claude-haiku-4-5)

This module has NO data-layer import: no CallerScope, no session, no tenant
identifier. It returns a ModelCallRecord and the caller persists it inside its
own scoped session (design §1.6). A model wrapper that needed a tenant scope in
order to log would be a wrapper that could get the tenant wrong.

See docs/PHASE2_DESIGN.md §1.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Protocol

from review_agent.models.types import (
    STOP_REASON_ERRORED,
    ModelBadRequest,
    ModelCallRecord,
    ModelError,
    ModelRequest,
    ModelResponse,
    ModelRole,
    Usage,
)

# Defaults match .env.example. Exact ids, never date-suffixed.
DEFAULT_MODELS: dict[ModelRole, str] = {
    ModelRole.DEFAULT: "claude-sonnet-5",
    ModelRole.JUDGMENT: "claude-opus-4-8",
    ModelRole.CLASSIFY: "claude-haiku-4-5",
}

ROLE_ENV_VARS: dict[ModelRole, str] = {
    ModelRole.DEFAULT: "MODEL_DEFAULT",
    ModelRole.JUDGMENT: "MODEL_JUDGMENT",
    ModelRole.CLASSIFY: "MODEL_CLASSIFY",
}


class Provider(Protocol):
    """What a provider adapter must implement. A Gemini adapter satisfies this too."""

    name: str

    def complete(
        self, model_id: str, request: ModelRequest, prompt_sha256: str, role: str
    ) -> ModelResponse:
        ...


_provider: Provider | None = None


def resolve_model(role: ModelRole) -> str:
    """The model id for a role. Env wins; the table above is the fallback."""
    return os.environ.get(ROLE_ENV_VARS[role], DEFAULT_MODELS[role])


def get_provider() -> Provider:
    """Instantiate the configured provider adapter (cached)."""
    global _provider
    if _provider is None:
        name = os.environ.get("MODEL_PROVIDER", "anthropic").lower()
        if name == "anthropic":
            from review_agent.models.providers.anthropic import AnthropicProvider

            _provider = AnthropicProvider()
        elif name == "ollama":
            from review_agent.models.providers.ollama import OllamaProvider

            _provider = OllamaProvider()
        else:
            raise ModelBadRequest(
                f"unknown MODEL_PROVIDER {name!r}. Add an adapter under "
                "review_agent/models/providers/ satisfying the Provider protocol."
            )
    return _provider


def set_provider(provider: Provider | None) -> None:
    """Install a provider. Tests use this to inject a deterministic stub."""
    global _provider
    _provider = provider


def prompt_fingerprint(request: ModelRequest) -> str:
    """Stable hash of everything that shapes the generation.

    This is what makes a review reproducible: with the artifact row, the
    rulebook hash and this value, the exact input can be rebuilt and
    re-verified. Serialisation is deterministic (sorted keys) so the same prompt
    always hashes the same — the same property that would later make prompt
    caching work.
    """
    payload = {
        "system": request.system,
        "user_content": [
            {"label": part.label, "text": part.text} for part in request.user_content
        ],
        "schema": request.schema,
        "reasoning": request.reasoning.value,
        "effort": request.effort.value,
        "max_output_tokens": request.max_output_tokens,
        # The tools OFFERED shape the generation as much as the schema does, so
        # they belong in the reproducibility hash. Only the neutral definition
        # (name/description/input_schema) is hashed — the handler is a callable,
        # not an input, and the tool RESULTS are outputs, not part of the request.
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in request.tools
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def call(role: ModelRole, request: ModelRequest) -> ModelResponse:
    """Call the model chosen for `role` and return a normalised response.

    The ONLY public entry point for model access in this codebase.
    """
    if not isinstance(role, ModelRole):
        raise ModelBadRequest(
            f"call() takes a ModelRole, not {type(role).__name__}. Passing a model "
            "id directly bypasses role routing and hides the model from the audit log."
        )
    model_id = resolve_model(role)
    fingerprint = prompt_fingerprint(request)
    try:
        return get_provider().complete(
            model_id=model_id,
            request=request,
            prompt_sha256=fingerprint,
            role=role.value,
        )
    except ModelError as exc:
        # Attach the record for the ATTEMPT, then re-raise unchanged. This is
        # NOT error handling — the caller still decides what a failure means.
        # It exists because only this function knows the resolved model id and
        # the prompt fingerprint, and a caller that had to reconstruct them
        # would eventually reconstruct one of them wrongly, or skip the audit
        # entry entirely because assembling it was work. The request left our
        # infrastructure whether or not a response came back.
        exc.call_record = ModelCallRecord(
            purpose=request.purpose,
            role=role.value,
            model_id=model_id,
            stop_reason=STOP_REASON_ERRORED,
            usage=Usage().as_dict(),
            prompt_sha256=fingerprint,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
