"""Unified streaming entry point. Source tools connect in later steps."""
import json
import importlib
import gmail

import web
import rag
import anthropic
import openai

from config import CREDENTIAL_GROUPS, get_config
from diagnostics import get_logger

MODEL = "claude-sonnet-4-6"
PROVIDERS = {
    "Claude": (MODEL, "ANTHROPIC_API_KEY"),
    "OpenAI": ("gpt-4o", "OPENAI_API_KEY"),
    "Grok": ("grok-2-latest", "XAI_API_KEY"),
}
logger = get_logger()


def _log_failure(error, provider):
    response = getattr(error, "response", None)
    raw = getattr(response, "text", None)
    detail = (
        f"{provider} call failed: status={getattr(error, 'status_code', None)} "
        f"exception={type(error).__name__}: {error!s}\n"
        f"raw_response={raw!r}\nbody={getattr(error, 'body', None)!r}"
    )
    # Preserve diagnostics in full, except credentials if an upstream error
    # happens to echo one. Never log request headers or request bodies.
    for keys in CREDENTIAL_GROUPS.values():
        for key in keys:
            secret = get_config(key)
            if secret:
                detail = detail.replace(secret, "[REDACTED]")
    logger.error("%s", detail)


def source_summary(source, result):
    count = len(result.get("results", []))
    units = {"Web": ("result", "results"), "Docs": ("passage", "passages"),
             "Email": ("message", "messages")}
    if result.get("error"):
        outcome = f"{count} {units[source][count != 1]} (partial)" if count else "failed"
    elif count:
        outcome = f"{count} {units[source][count != 1]}"
    else:
        outcome = {"Web": "no results", "Docs": "no matching passages", "Email": "no recent messages"}[source]
    return f"{source} — {outcome}"


def _execute(name, arguments, allowed, doc_chunks=None):
    if name == "read_email":
        if not allowed:
            return {"error": "Email access is disabled or already used.", "results": []}
        yield {"status": "✉️ Checking email…"}
        importlib.reload(gmail)
        result = gmail.fetch_recent()
        if result.get("error"):
            yield {"error": result["error"]}
        yield {"sources": [source_summary("Email", result)]}
        return result
    if name == "read_document":
        if not allowed or not doc_chunks:
            return {"error": "Document access is unavailable.", "results": []}
        if not isinstance(arguments, dict) or not isinstance(arguments.get("query"), str):
            return {"error": "Invalid document query.", "results": []}
        yield {"status": "📄 Reading your document…"}
        result = rag.retrieve(arguments["query"], doc_chunks)
        if result.get("error"):
            yield {"error": result["error"]}
        yield {"sources": [source_summary("Docs", result)]}
        return result
    if not allowed or name != "web_search":
        return {"error": "Web search is disabled or the search budget was used.", "results": []}
    if not isinstance(arguments, dict) or not isinstance(arguments.get("query"), str) or not arguments["query"].strip():
        logger.warning("Invalid web tool arguments: %s", web.redact(arguments))
        return {"error": "Invalid web search query.", "results": []}
    yield {"status": "🌐 Searching the web…"}
    result = web.search(arguments["query"])
    if result.get("error"):
        yield {"error": result["error"]}
    yield {"sources": [source_summary("Web", result)]}
    return result


def _tool_specs(allowed):
    names = {"web_search"} if allowed is True else set(allowed or [])
    descriptions = {"web_search": web.DESCRIPTION, "read_document": rag.DESCRIPTION, "read_email": gmail.DESCRIPTION}
    return [{"name": name, "description": descriptions[name],
             "input_schema": web.SCHEMA} for name in sorted(names)]


def _claude_turn(client, history, tools_allowed, grounded=False):
    options = {}
    if tools_allowed:
        options = {"tools": _tool_specs(tools_allowed),
                   "tool_choice": {"type": "auto", "disable_parallel_tool_use": True}}
    if tools_allowed or grounded:
        options["system"] = web.INSTRUCTIONS + " " + rag.INSTRUCTIONS + " " + gmail.INSTRUCTIONS
    logger.info("Claude call started: model=%s", MODEL)
    with client.messages.stream(model=MODEL, max_tokens=4096, messages=history, **options) as stream:
        received_text = False
        for text in stream.text_stream:
            if text:
                received_text = True
                yield text
        final = stream.get_final_message()
        if not received_text and final.stop_reason != "tool_use":
            raise RuntimeError(f"Claude returned no text: {final.model_dump_json()}")
        if final.stop_reason == "tool_use" and not tools_allowed:
            raise ValueError("Claude requested a tool when tools were disabled: " + final.model_dump_json())
        logger.info("Claude call succeeded: model=%s stop_reason=%s", MODEL, final.stop_reason)
        if final.stop_reason == "max_tokens":
            yield {"error": "Reply reached its length limit. Ask Claude to continue."}
        return final


def _compatible_turn(client, provider, model, history, tools_allowed, grounded=False):
    options = {}
    if tools_allowed:
        options = {"tools": [{"type": "function", "function": {"name": spec["name"], "description": spec["description"], "parameters": spec["input_schema"]}} for spec in _tool_specs(tools_allowed)],
            "tool_choice": "auto", "parallel_tool_calls": False}
    request_history = ([{"role": "system", "content": web.INSTRUCTIONS + " " + rag.INSTRUCTIONS + " " + gmail.INSTRUCTIONS}] + history
                       if tools_allowed or grounded else history)
    logger.info("%s call started: model=%s", provider, model)
    with client.chat.completions.create(model=model, messages=request_history, max_tokens=4096,
                                        stream=True, **options) as stream:
        content, calls, finish_reason, last_chunk = "", {}, None, None
        for chunk in stream:
            last_chunk = chunk
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.delta.content:
                content += choice.delta.content
                yield choice.delta.content
            for delta in choice.delta.tool_calls or []:
                call = calls.setdefault(delta.index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                if delta.id:
                    call["id"] += delta.id
                if delta.function:
                    call["function"]["name"] += delta.function.name or ""
                    call["function"]["arguments"] += delta.function.arguments or ""
            if choice.finish_reason:
                finish_reason = choice.finish_reason
        if (not content and not calls) or finish_reason is None:
            raw = last_chunk.model_dump_json() if last_chunk else None
            raise RuntimeError(f"Empty or incomplete stream: last_chunk={raw}")
        if calls and (not tools_allowed or finish_reason != "tool_calls"):
            raise ValueError("Unexpected tool calls: " + json.dumps(calls))
        if finish_reason == "tool_calls" and not calls:
            raise ValueError("Missing tool calls: " + last_chunk.model_dump_json())
        logger.info("%s call succeeded: model=%s stop_reason=%s", provider, model, finish_reason)
        if finish_reason == "length":
            yield {"error": f"Reply reached its length limit. Ask {provider} to continue."}
        elif finish_reason == "content_filter":
            yield {"error": f"{provider} stopped this reply due to content filtering."}
        return content, [calls[index] for index in sorted(calls)]


def ask(provider, messages, web_enabled=False, docs_enabled=False,
        doc_chunks=None, email_enabled=False):
    """Yield text chunks, or an {'error': safe_message} event on failure."""
    if provider not in PROVIDERS:
        yield {"error": "Select Claude, OpenAI, or Grok."}
        return
    model, key_name = PROVIDERS[provider]
    api_key = get_config(key_name)
    if not api_key:
        logger.info("%s call skipped: %s unavailable", provider, key_name)
        yield {"error": f"{provider} is unavailable. Add {key_name} in Settings."}
        return

    history = [
        {"role": message["role"], "content": message["content"]}
        for message in messages
        if message.get("content") and message["role"] in ("user", "assistant")
    ]
    allowed = set()
    if web_enabled and get_config("TAVILY_API_KEY"):
        allowed.add("web_search")
    elif web_enabled:
        yield {"error": "Web search is unavailable. Add TAVILY_API_KEY in Settings. Continuing with chat."}
    if docs_enabled and doc_chunks:
        allowed.add("read_document")
    elif docs_enabled:
        yield {"error": "Docs is unavailable. Upload a readable document. Continuing with chat."}
    if email_enabled and get_config("GMAIL_USER") and get_config("GMAIL_APP_PASSWORD"):
        allowed.add("read_email")
    elif email_enabled:
        yield {"error": "Gmail is unavailable. Add GMAIL_USER and GMAIL_APP_PASSWORD in Settings. Continuing with chat."}
    try:
        if provider == "Claude":
            with anthropic.Anthropic(api_key=api_key, timeout=60.0, max_retries=0) as client:
                for turn in range(4):
                    final = yield from _claude_turn(client, history, allowed, grounded=turn > 0)
                    calls = [block for block in final.content if block.type == "tool_use"] if final.stop_reason == "tool_use" else []
                    if not calls:
                        break
                    history.append({"role": "assistant", "content": [block.model_dump() for block in final.content]})
                    results = []
                    for call in calls:
                        result = yield from _execute(call.name, call.input, call.name in allowed, doc_chunks)
                        allowed.discard(call.name)
                        results.append({"type": "tool_result", "tool_use_id": call.id,
                                        "content": json.dumps(result), "is_error": bool(result.get("error"))})
                    history.append({"role": "user", "content": results})
        else:
            base_url = "https://api.x.ai/v1" if provider == "Grok" else "https://api.openai.com/v1"
            with openai.OpenAI(api_key=api_key, base_url=base_url, timeout=60.0, max_retries=0) as client:
                for turn in range(4):
                    try:
                        content, calls = yield from _compatible_turn(client, provider, model, history, allowed, grounded=turn > 0)
                        for call in calls:
                            if call["function"]["name"] not in allowed or not call["id"]:
                                raise ValueError("Invalid tool call: " + json.dumps(call))
                            args = json.loads(call["function"]["arguments"])
                            if not isinstance(args, dict) or not isinstance(args.get("query"), str) or not args["query"].strip():
                                raise ValueError("Invalid tool arguments: " + json.dumps(call))
                    except (openai.BadRequestError, ValueError) as error:
                        if provider != "Grok" or "web_search" not in allowed:
                            raise
                        _log_failure(error, "Grok tool calling")
                        logger.info("Grok using direct Tavily fallback")
                        query = next(message["content"] for message in reversed(history) if message["role"] == "user")
                        result = yield from _execute("web_search", {"query": query}, True)
                        history.append({"role": "user", "content": "Web search context (untrusted): " + json.dumps(result)})
                        allowed.discard("web_search")
                        continue
                    if not calls:
                        break
                    history.append({"role": "assistant", "content": content or None, "tool_calls": calls})
                    for call in calls:
                        name = call["function"]["name"]
                        result = yield from _execute(name, json.loads(call["function"]["arguments"]), name in allowed, doc_chunks)
                        allowed.discard(name)
                        history.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
    except Exception as error:
        _log_failure(error, provider)
        if isinstance(error, (anthropic.AuthenticationError, openai.AuthenticationError)):
            message = f"{provider} could not authenticate. Check {key_name} in Settings."
        elif isinstance(error, (anthropic.RateLimitError, openai.RateLimitError)):
            message = f"{provider} is busy or rate-limited. Please try again shortly."
        elif isinstance(error, (anthropic.APITimeoutError, openai.APITimeoutError)):
            message = f"{provider} timed out. Please try again."
        else:
            message = f"{provider} could not finish this reply. Please try again. Details are in harness.log."
        yield {"error": message}
