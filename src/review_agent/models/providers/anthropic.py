"""Anthropic provider adapter — the ONLY file permitted to import `anthropic`.

Everything provider-specific lives here: parameter names, structured-output
shape, thinking configuration, exception types, stop-reason strings. A lint in
tests/test_conformance.py asserts no other module under src/ imports the SDK,
which is what makes "the provider is swappable" a testable claim rather than an
aspiration.

Adding a Gemini adapter means writing a sibling module that satisfies the same
Provider protocol. No business logic changes.
"""

from __future__ import annotations

import json
import os

import anthropic

from review_agent.models.types import (
    ModelAuthError,
    ModelBadRequest,
    ModelCallRecord,
    ModelRateLimited,
    ModelRequest,
    ModelResponse,
    ModelTransportError,
    ModelUnavailable,
    Reasoning,
    StopReason,
    Usage,
)

# Anthropic's stop_reason strings -> our neutral enum.
_STOP_REASONS = {
    "end_turn": StopReason.COMPLETE,
    "stop_sequence": StopReason.COMPLETE,
    "max_tokens": StopReason.TRUNCATED,
    "refusal": StopReason.REFUSED,
    "tool_use": StopReason.TOOL_USE,
}

# The SDK already retries 429/5xx with exponential backoff. Set it explicitly so
# the value is visible rather than inherited, and do NOT wrap another retry loop
# around it — two nested backoffs turn a rate-limit blip into a multi-minute stall.
MAX_RETRIES = 2

# Bound on the tool-use turns in one generation. It FAILS CLOSED: a model that
# keeps calling tools past this yields a final message still on `tool_use`, which
# `_normalise` maps to StopReason.TOOL_USE with no structured output, and the
# caller rejects the review rather than spending unbounded turns. The bound only
# needs to exceed the handful of technology lookups a real review performs.
MAX_TOOL_ITERATIONS = 6


def _usage_of(message) -> Usage:
    """Extract token usage from one SDK message, tolerant of missing fields."""
    u = getattr(message, "usage", None)
    return Usage(
        input_tokens=getattr(u, "input_tokens", 0) or 0,
        output_tokens=getattr(u, "output_tokens", 0) or 0,
        cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
    )


def _add_usage(a: Usage, b: Usage) -> Usage:
    """Field-wise sum, so a multi-turn tool loop reports its TOTAL spend."""
    return Usage(
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        cache_read_input_tokens=a.cache_read_input_tokens + b.cache_read_input_tokens,
        cache_creation_input_tokens=(
            a.cache_creation_input_tokens + b.cache_creation_input_tokens
        ),
    )


class AnthropicProvider:
    """Implements the Provider protocol against the Messages API."""

    name = "anthropic"

    def __init__(self, client: "anthropic.Anthropic | None" = None) -> None:
        self._client = client

    def _get_client(self):
        if self._client is None:
            self._client = anthropic.Anthropic(max_retries=MAX_RETRIES)
        return self._client

    def complete(
        self, model_id: str, request: ModelRequest, prompt_sha256: str, role: str
    ) -> ModelResponse:
        # `messages` is built ONCE and grows in place across tool turns, so the
        # kwargs dict below keeps pointing at the same list — each follow-up call
        # sees the full conversation (the assistant's tool_use plus our
        # tool_result). The untrusted artifact is still only ever the first user
        # turn; nothing the model or a tool produces re-enters the system prompt.
        messages: list[dict] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": part.text}
                    for part in request.user_content
                ],
            }
        ]
        kwargs = {
            "model": model_id,
            "max_tokens": request.max_output_tokens,
            "system": request.system,
            "messages": messages,
        }

        # `effort` lives INSIDE output_config, not top-level.
        output_config: dict = {"effort": request.effort.value}
        if request.schema is not None:
            # The canonical structured-output parameter. (`output_format` is
            # deprecated API-wide.) The schema must set additionalProperties:
            # false and list required — the caller builds it that way. Structured
            # output coexists with tools: the model calls tools first, then emits
            # the schema-constrained JSON on its final (end_turn) turn.
            output_config["format"] = {
                "type": "json_schema",
                "schema": request.schema,
            }
        kwargs["output_config"] = output_config

        if request.reasoning is Reasoning.ADAPTIVE:
            # Set EXPLICITLY. On Opus 4.8 omitting `thinking` runs WITHOUT
            # thinking — adaptive is not the default. A wrapper that expressed
            # "adaptive" by omitting the parameter would silently disable
            # reasoning on the judgment model.
            kwargs["thinking"] = {"type": "adaptive"}

        tools_by_name = {t.name: t for t in request.tools}
        if request.tools:
            # Only the neutral shape crosses into the SDK. The handler stays on
            # our side of the boundary — the provider executes it but never needs
            # to know what it computes.
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in request.tools
            ]

        # Deliberately absent: temperature / top_p / top_k. Opus 4.8 rejects them.

        message = self._create(kwargs)
        # Every tool-use turn is a billed generation. Accumulate their usage so
        # the audited record reflects the WHOLE spend, not just the final turn —
        # otherwise a rollback could not un-spend tokens the log never recorded.
        prior_usage = Usage()
        iterations = 0
        while (
            tools_by_name
            and getattr(message, "stop_reason", None) == "tool_use"
            and iterations < MAX_TOOL_ITERATIONS
        ):
            iterations += 1
            prior_usage = _add_usage(prior_usage, _usage_of(message))
            messages.append({"role": "assistant", "content": message.content})
            messages.append(
                {"role": "user", "content": self._run_tools(message, tools_by_name)}
            )
            message = self._create(kwargs)

        return self._normalise(
            message, model_id, request, prompt_sha256, role, prior_usage
        )

    def _create(self, kwargs: dict):
        """One Messages API call, with every provider error translated.

        Shared by the first turn and every tool follow-up so a failure MID-LOOP
        is audited and surfaced exactly like a failure on the first turn — a
        tool round-trip does not open a hole where a provider exception escapes
        untranslated.
        """
        try:
            return self._get_client().messages.create(**kwargs)
        except anthropic.AuthenticationError as exc:
            raise ModelAuthError(str(exc)) from exc
        except anthropic.PermissionDeniedError as exc:
            raise ModelAuthError(str(exc)) from exc
        except anthropic.RateLimitError as exc:
            raise ModelRateLimited(str(exc)) from exc
        except anthropic.BadRequestError as exc:
            raise ModelBadRequest(str(exc)) from exc
        except anthropic.NotFoundError as exc:
            raise ModelBadRequest(f"unknown model or endpoint: {exc}") from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise ModelUnavailable(str(exc)) from exc
            raise ModelBadRequest(str(exc)) from exc
        except anthropic.APIConnectionError as exc:
            raise ModelTransportError(str(exc)) from exc

    @staticmethod
    def _run_tools(message, tools_by_name: dict) -> list[dict]:
        """Execute every tool_use block in `message`, return the tool_result blocks.

        A tool_use naming a tool we did not offer is reported back as an error
        result, not raised — the model can recover, and a fabricated tool name
        (the same shape as a fabricated rule id) must not crash the review.
        """
        results: list[dict] = []
        for block in message.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            tool = tools_by_name.get(block.name)
            if tool is None:
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"error: unknown tool {block.name!r}",
                        "is_error": True,
                    }
                )
                continue
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": tool.handler(dict(block.input or {})),
                }
            )
        return results

    @staticmethod
    def _normalise(
        message,
        model_id: str,
        request: ModelRequest,
        prompt_sha256: str,
        role: str,
        prior_usage: Usage = Usage(),
    ) -> ModelResponse:
        # FAIL CLOSED on an unrecognised value. Defaulting to COMPLETE meant a
        # stop reason the provider added after this dict was written would be
        # read as a finished generation, and whatever partial content came
        # with it would be validated as a complete review. That is the worst
        # available failure: a quietly incomplete review, indistinguishable
        # from a good one, where an error would have been obvious.
        stop_reason = _STOP_REASONS.get(message.stop_reason, StopReason.UNKNOWN)

        text = None
        for block in message.content:
            if getattr(block, "type", None) == "text":
                text = block.text
                break

        # The model's extended-thinking blocks (adaptive reasoning is ON for the
        # judgment role). Captured verbatim so the caller can persist it as
        # advisory context; the agent never reads it and it never re-enters a
        # prompt. On a tool-using generation this is the FINAL turn's thinking —
        # the reasoning that produced the findings. Multiple blocks are joined.
        thinking = "\n\n".join(
            block.thinking
            for block in message.content
            if getattr(block, "type", None) == "thinking"
            and getattr(block, "thinking", None)
        ) or None

        # Only attempt a parse when we asked for structured output AND the
        # generation actually completed. A truncated response is invalid JSON by
        # definition; parsing it would raise inside the adapter and disguise a
        # TRUNCATED stop reason as a parse bug.
        structured = None
        if request.schema is not None and stop_reason is StopReason.COMPLETE and text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                structured = None
            else:
                structured = parsed if isinstance(parsed, dict) else {"value": parsed}

        # The FINAL turn's usage plus every tool-use turn that preceded it.
        usage = _add_usage(prior_usage, _usage_of(message))
        request_id = getattr(message, "_request_id", None)

        return ModelResponse(
            text=text,
            structured=structured,
            stop_reason=stop_reason,
            model_id=model_id,
            usage=usage,
            request_id=request_id,
            call_record=ModelCallRecord(
                purpose=request.purpose,
                role=role,
                model_id=model_id,
                stop_reason=stop_reason.value,
                # The RAW provider string as well as our mapping of it. The
                # mapping is code that can be wrong; the raw value is the
                # evidence, and it is what tells an operator that an
                # unrecognised reason arrived rather than leaving them to
                # infer it from a mapped value. (Phase 1 BUG-7: evidence,
                # not testimony.)
                provider_stop_reason=message.stop_reason,
                usage=usage.as_dict(),
                prompt_sha256=prompt_sha256,
                request_id=request_id,
                thinking=thinking,
            ),
            raw=message,
        )


def known_model_ids(client=None) -> set[str]:
    """Model ids the configured account can reach. Used for boot-time validation."""
    client = client or anthropic.Anthropic(max_retries=MAX_RETRIES)
    return {m.id for m in client.models.list()}


def credentials_present() -> bool:
    """Cheap check used to decide whether live-model tests can run at all."""
    return bool(
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    )
