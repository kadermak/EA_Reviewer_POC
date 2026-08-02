"""Ollama provider adapter — local models via the /api/chat endpoint.

The sibling of the Anthropic adapter. It is the ONLY place that knows Ollama's
request/response shape, and it maps everything onto the neutral ModelResponse so
no business logic changes. Selected by MODEL_PROVIDER=ollama (see client.py).

  Endpoint: POST {OLLAMA_HOST}/api/chat  (default http://localhost:11434),
            stream disabled so one JSON object comes back.
  Model:    OLLAMA_MODEL. Roles are NOT mapped to different models here — one
            local model serves every role. The role-resolved id client.call
            passes in (a Claude id) is deliberately ignored; the audit record
            names the Ollama model actually used.

TOOL USE. If the local model supports native tool calling (qwen2:7b does on
Ollama >= 0.5) tools are negotiated natively: the model emits `tool_calls`, this
adapter runs the handlers and feeds the results back as `tool` messages, bounded
by MAX_TOOL_ITERATIONS. If the model does NOT support tools (Ollama replies "does
not support tools"), the adapter FALLS BACK to describing the tools in the prompt
and parsing a JSON tool-call out of the model's plain text. The text fallback is
best-effort: a small local model may ignore the protocol or emit a malformed
call, in which case the tool is simply not invoked and the finding is judged on
the model's own knowledge — lower quality, and documented as such.

STRUCTURED OUTPUT. Ollama's `format=<json schema>` constrains generation to the
schema's SHAPE, not its TRUTH (real rule ids, quoted evidence, full coverage).
That is sufficient because the conformance agent's validate() runs the SAME
semantic checks regardless of provider: schema-valid-but-wrong Ollama output is
rejected and retried exactly as Claude's would be. The residual gap — a smaller
model returning schema-valid but low-quality findings — surfaces as a validation
rejection or as poor recall against the golden set, never as a silent bad pass.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from review_agent.models.types import (
    ModelBadRequest,
    ModelCallRecord,
    ModelRequest,
    ModelTransportError,
    ModelUnavailable,
    ModelResponse,
    StopReason,
    Usage,
)

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1"
MAX_TOOL_ITERATIONS = 6
# Local generation on CPU/Metal is far slower than a hosted API; give it room.
_TIMEOUT_SECONDS = 900

# Ollama's done_reason -> our neutral enum. Unmapped values FAIL CLOSED to
# UNKNOWN (same posture as the Anthropic adapter): a reason we do not recognise
# must not be read as a finished generation.
_STOP_REASONS = {
    "stop": StopReason.COMPLETE,
    "length": StopReason.TRUNCATED,
}


def _post(path: str, payload: dict) -> dict:
    """POST JSON to the Ollama server and return the decoded reply.

    Uses only the standard library so adding local inference pulls in no new
    dependency. Provider errors are translated to the neutral ModelError family
    the business logic already catches.
    """
    host = os.environ.get("OLLAMA_HOST", DEFAULT_HOST).rstrip("/")
    url = host + path
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        if exc.code == 404:
            raise ModelBadRequest(
                f"Ollama 404 — is the model pulled? `ollama pull <model>`: {detail}"
            ) from exc
        if exc.code >= 500:
            raise ModelUnavailable(f"Ollama {exc.code}: {detail}") from exc
        raise ModelBadRequest(f"Ollama {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ModelTransportError(
            f"cannot reach Ollama at {url} ({exc.reason}). Is `ollama serve` running?"
        ) from exc

    obj = json.loads(body)
    # Ollama also reports some failures as HTTP 200 with an {"error": ...} body.
    if isinstance(obj, dict) and obj.get("error"):
        raise ModelBadRequest(f"Ollama error: {obj['error']}")
    return obj


class OllamaProvider:
    """Implements the Provider protocol against a local Ollama server."""

    name = "ollama"

    def complete(
        self, model_id: str, request: ModelRequest, prompt_sha256: str, role: str
    ) -> ModelResponse:
        model = os.environ.get("OLLAMA_MODEL") or DEFAULT_MODEL

        messages: list[dict] = [{"role": "system", "content": request.system}]
        messages.append(
            {
                "role": "user",
                "content": "\n\n".join(part.text for part in request.user_content),
            }
        )

        # Phase 1 (only when tools are offered): let the model call tools, and
        # accumulate the results into the conversation.
        if request.tools:
            messages = self._negotiate_tools(model, messages, request.tools)

        # Phase 2: the final generation. Structured output is requested HERE,
        # separately from tool negotiation, because a tool-calling turn does not
        # emit the schema — mixing the two makes both unreliable.
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": self._options(request),
        }
        if request.schema is not None:
            payload["format"] = request.schema
        data = _post("/api/chat", payload)
        return self._normalise(data, model, request, prompt_sha256, role)

    @staticmethod
    def _options(request: ModelRequest) -> dict:
        # num_ctx MUST hold system+rulebook+artifact (plus any tool turns) or the
        # prompt is silently truncated and the review is judged on a partial
        # document — the worst failure mode, so it is generous and overridable.
        return {
            "num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "8192")),
            "num_predict": request.max_output_tokens,
            # temperature 0 for reproducible structured output. (The neutral
            # surface forbids this knob for Anthropic because Opus rejects it;
            # that is a provider quirk, not a rule against determinism, and a
            # local model benefits from it.)
            "temperature": 0,
        }

    def _negotiate_tools(self, model: str, messages: list[dict], tools) -> list[dict]:
        """Run the tool exchange, returning messages enriched with tool results."""
        tools_by_name = {t.name: t for t in tools}
        native = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]
        try:
            data = _post(
                "/api/chat",
                {
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "tools": native,
                    "options": {"num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "8192"))},
                },
            )
        except ModelBadRequest as exc:
            if "does not support tools" in str(exc).lower():
                # The model cannot do native tool calls — describe them in the
                # prompt and parse a tool-call out of its text instead.
                return self._text_tool_loop(model, messages, tools_by_name)
            raise

        iterations = 0
        while iterations < MAX_TOOL_ITERATIONS:
            msg = data.get("message", {}) or {}
            calls = msg.get("tool_calls") or []
            if not calls:
                break
            iterations += 1
            messages.append(
                {"role": "assistant", "content": msg.get("content", "") or "", "tool_calls": calls}
            )
            for call in calls:
                fn = call.get("function", {}) or {}
                name = fn.get("name", "")
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tool = tools_by_name.get(name)
                result = (
                    tool.handler(dict(args))
                    if tool is not None
                    else f"error: unknown tool {name!r}"
                )
                messages.append({"role": "tool", "content": result})
            data = _post(
                "/api/chat",
                {
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "tools": native,
                    "options": {"num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "8192"))},
                },
            )
        return messages

    def _text_tool_loop(
        self, model: str, messages: list[dict], tools_by_name: dict
    ) -> list[dict]:
        """Fallback for models without native tools: prompt + parse a JSON call."""
        spec = [
            "You have tools. To call one, reply with ONLY this JSON and nothing "
            'else: {"tool_call": {"name": "<tool>", "arguments": { ... }}}',
            "When you no longer need a tool, stop and answer normally.",
            "Available tools:",
        ]
        for tool in tools_by_name.values():
            spec.append(
                f"- {tool.name}: {tool.description} "
                f"arguments: {json.dumps(tool.input_schema)}"
            )
        guide = {"role": "system", "content": "\n".join(spec)}
        convo = [messages[0], guide] + messages[1:]

        iterations = 0
        while iterations < MAX_TOOL_ITERATIONS:
            data = _post(
                "/api/chat",
                {"model": model, "messages": convo, "stream": False,
                 "options": {"num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "8192"))}},
            )
            content = (data.get("message", {}) or {}).get("content", "") or ""
            parsed = self._parse_text_tool_call(content)
            if parsed is None:
                break
            name, args = parsed
            tool = tools_by_name.get(name)
            result = (
                tool.handler(dict(args)) if tool is not None
                else f"error: unknown tool {name!r}"
            )
            convo.append({"role": "assistant", "content": content})
            convo.append({"role": "user", "content": f"Tool result for {name}: {result}"})
            iterations += 1

        # Drop the injected guide so the final structured call runs against the
        # same system prompt as every other request; the tool turns are kept.
        return [m for m in convo if m is not guide]

    @staticmethod
    def _parse_text_tool_call(content: str):
        """Extract the first {"tool_call": {...}} object from free text, or None."""
        if "tool_call" not in content:
            return None
        start = content.find("{")
        while start != -1:
            depth = 0
            for i in range(start, len(content)):
                if content[i] == "{":
                    depth += 1
                elif content[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(content[start : i + 1])
                        except json.JSONDecodeError:
                            break
                        call = obj.get("tool_call") if isinstance(obj, dict) else None
                        if isinstance(call, dict) and call.get("name"):
                            args = call.get("arguments") or {}
                            return call["name"], (args if isinstance(args, dict) else {})
                        return None
            start = content.find("{", start + 1)
        return None

    @staticmethod
    def _normalise(
        data: dict, model: str, request: ModelRequest, prompt_sha256: str, role: str
    ) -> ModelResponse:
        done_reason = data.get("done_reason")
        stop_reason = (
            _STOP_REASONS.get(done_reason, StopReason.UNKNOWN)
            if done_reason is not None
            else StopReason.UNKNOWN
        )

        msg = data.get("message", {}) or {}
        text = msg.get("content")

        # Parse only on a completed structured request. A truncated body is
        # invalid JSON by definition; parsing it would disguise TRUNCATED.
        structured = None
        if request.schema is not None and stop_reason is StopReason.COMPLETE and text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                structured = None
            else:
                structured = parsed if isinstance(parsed, dict) else {"value": parsed}

        usage = Usage(
            input_tokens=data.get("prompt_eval_count", 0) or 0,
            output_tokens=data.get("eval_count", 0) or 0,
        )
        return ModelResponse(
            text=text,
            structured=structured,
            stop_reason=stop_reason,
            model_id=model,
            usage=usage,
            call_record=ModelCallRecord(
                purpose=request.purpose,
                role=role,
                model_id=model,
                stop_reason=stop_reason.value,
                provider_stop_reason=done_reason,
                usage=usage.as_dict(),
                prompt_sha256=prompt_sha256,
            ),
            raw=data,
        )
