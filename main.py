#!/usr/bin/env python3
"""Aggregator entry for free-model-aggregator-01, wallet 6 flavour "home-wire-first".

One small server fronts ten free-tier upstreams behind two client-facing
protocols: OpenAI chat completions and Anthropic messages, in JSON form and
as server-sent event streams. For each incoming request it picks the
best-scoring free model that can actually satisfy what the request asks for,
then walks down that ranking, switching provider whenever one declines
with a payment or rate-limit refusal. It announces its model list to the
trial harness, works through the harness workload, and leaves with exit
code zero once the harness reports that the work is finished.

Only the standard library is used. Choices are fully reproducible: they
come from the brief, the request itself, and the answers providers have
already given, never from the clock.
"""

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------------------------------------------------------------------
# Per-wallet tuning knobs for this build.
# ---------------------------------------------------------------------------
VARIANT_NAME = "home-wire-first"
PREFER_SAME_WIRE_BONUS = 12      # milli-rating credit when upstream speaks the client wire
REFUSAL_LIMIT = 2                # refusals after which an upstream is treated as spent
LONG_CONTEXT_CHARS = 2000        # user text at/above this length counts as long-context
LONG_CONTEXT_MIN = 200000        # context window a long-context request demands
ANTHROPIC_JSON_SAFE = True       # Anthropic-wire requests are assumed to want JSON-capable models
STREAM_PIECE_CHARS = 21          # re-chunk streamed text into pieces this long; 0 keeps upstream pieces
TIE_BREAK = "brief"              # tied scores resolve by "brief" (brief order) or "alpha" (name order)

FIXED_CREATED = 1767225600       # creation stamp the simulated upstreams attach
ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")
PORT = 8791

# ---------------------------------------------------------------------------
# Tiny HTTP helper built on urllib, with proxies disabled: trials hand us
# plain base URLs, and a proxy would only get in the way.
# ---------------------------------------------------------------------------
_FETCHER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def fetch_url(method, url, payload=None, hdrs=None, timeout=15):
    """Run a single HTTP exchange.

    Yields (status, headers, body bytes). A failure to connect at all is
    reported as status 0 with an empty body.
    """
    headers = dict(hdrs or {})
    data = None
    if payload is not None:
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        headers.setdefault("content-type", "application/json")
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with _FETCHER.open(request, timeout=timeout) as response:
            return response.status, dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as failure:
        caught_headers = dict(failure.headers.items()) if failure.headers else {}
        return failure.code, caught_headers, failure.read()
    except Exception:
        return 0, {}, b""


def fetch_json(method, url, payload=None, hdrs=None, timeout=15):
    status, headers, raw = fetch_url(method, url, payload, hdrs, timeout)
    try:
        parsed = json.loads(raw.decode("utf-8")) if raw else None
    except Exception:
        parsed = None
    return status, headers, parsed


# ---------------------------------------------------------------------------
# The brief: the provider directory, the quality table, and the workload
# manifest are JSON documents embedded in the prompt file's prose.
# ---------------------------------------------------------------------------
def pull_embedded_doc(prose, kind):
    marker = '"schema":"' + kind + '"'
    found = prose.find(marker)
    if found < 0:
        marker = '"schema": "' + kind + '"'
        found = prose.find(marker)
    if found < 0:
        return None
    opening = prose.rfind("{", 0, found)
    if opening < 0:
        return None
    try:
        doc, _ = json.JSONDecoder().raw_decode(prose[opening:])
        return doc
    except Exception:
        return None


class Upstream:
    __slots__ = ("id", "name", "wire", "auth", "key_env", "key", "path")

    def __init__(self, row):
        self.id = row["id"]
        self.name = row.get("name", row["id"])
        self.wire = row["wire"]
        self.auth = row.get("auth", "none")
        self.key_env = row.get("key_env", "")
        self.key = row.get("key", "")
        self.path = row["path"]

    def base(self):
        return ENV_URL + self.path

    def credentials(self):
        if self.auth == "bearer":
            return {"authorization": "Bearer " + self.key}
        if self.auth == "x-api-key":
            return {"x-api-key": self.key}
        if self.auth == "x-goog-api-key":
            return {"x-goog-api-key": self.key}
        return {}


def read_brief():
    """Return (upstreams, rating rows, manifest) from the prompt file.

    An empty triple comes back when the prompt file is missing or unreadable.
    """
    if not PROMPT_FILE:
        return [], [], []
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as handle:
            prose = handle.read()
    except Exception:
        return [], [], []
    directory = pull_embedded_doc(prose, "free-model-aggregator-01/providers/v1") or {}
    ratings_doc = pull_embedded_doc(prose, "free-model-aggregator-01/ratings/v1") or {}
    workload_doc = pull_embedded_doc(prose, "free-model-aggregator-01/workload/v1") or {}
    upstreams = [Upstream(row) for row in directory.get("providers", [])]
    return upstreams, ratings_doc.get("ratings", []), workload_doc.get("manifest", [])


# ---------------------------------------------------------------------------
# Model catalog: ask each upstream for its own model list (the ground
# truth), and use the brief's quality table wherever a probe comes back
# empty or fails.
# ---------------------------------------------------------------------------
def probe_catalog(prov):
    """List [(model id, free?, context window)] from an upstream, or None."""
    if prov.wire == "gemini":
        catalog_url = prov.base() + "/v1beta/models"
    else:
        catalog_url = prov.base() + "/v1/models"
    # Slow catalog endpoints need a generous window, hence 15 seconds here.
    status, _, listing = fetch_json("GET", catalog_url, hdrs=prov.credentials(), timeout=15)
    if not 200 <= status < 300 or not isinstance(listing, dict):
        return None
    entries = []
    if prov.wire == "gemini":
        for item in listing.get("models", []):
            model_id = str(item.get("name", "")).removeprefix("models/")
            if model_id:
                ctx = item.get("context_length") or item.get("inputTokenLimit")
                entries.append((model_id, item.get("free") is True, ctx))
    else:
        for item in listing.get("data", []):
            if isinstance(item.get("id"), str) and item["id"]:
                entries.append((item["id"], item.get("free") is True, item.get("context_length")))
    return entries or None


def assemble_catalog(upstreams, ratings):
    table = {}
    for rating in ratings:
        table.setdefault(rating["provider"], []).append(
            (rating["model"], rating.get("free") is True, rating.get("context_length")))
    catalog = {"providers": []}
    for prov in upstreams:
        listed = probe_catalog(prov) if ENV_URL else None
        if listed is None:
            listed = table.get(prov.id, [])
        catalog["providers"].append({
            "id": prov.id,
            "base_url": prov.path,
            "key_env": prov.key_env,
            "wire": prov.wire,
            "models": [
                {"id": mid, "free": free, "context_length": ctx}
                for mid, free, ctx in listed
            ],
        })
    return catalog


def catalog_json(catalog):
    return json.dumps(catalog)


# ---------------------------------------------------------------------------
# Wire-neutral request: the client's request reshaped once, with nothing
# thrown away, so every upstream wire can be built from it.
# ---------------------------------------------------------------------------
def text_from_blocks(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        texts = []
        for part in value:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
        return "\n".join(texts)
    return ""


def image_part_from_link(link):
    if isinstance(link, str) and link.startswith("data:") and ";base64," in link:
        mime = link[5:link.index(";base64,")]
        return {"type": "image", "media_type": mime, "data": link.split(";base64,", 1)[1]}
    return {"type": "image_ref", "url": link}


class CanonicalRequest:
    """Holds system text, messages, tools, and the untouched client body.

    A message is {role, parts}; a part is one of
    {type:text, text}, {type:image, media_type, data}, or
    {type:image_ref, url}.
    """

    def __init__(self, api, body):
        self.api = api
        self.raw = body
        self.stream = body.get("stream") is True
        self.max_tokens = None
        self.stops = []
        self.json_mode = False
        self.response_format = None
        self.system = None
        self.messages = []
        self.tools = []
        self.tool_choice = None
        if api == "openai":
            self._ingest_openai(body)
        else:
            self._ingest_anthropic(body)

    def _ingest_openai(self, body):
        cap = body.get("max_tokens")
        if not isinstance(cap, (int, float)):
            cap = body.get("max_completion_tokens")
        self.max_tokens = cap if isinstance(cap, (int, float)) else None
        stop_val = body.get("stop")
        if isinstance(stop_val, str):
            self.stops = [stop_val]
        elif isinstance(stop_val, list):
            self.stops = [s for s in stop_val if isinstance(s, str)]
        fmt_doc = body.get("response_format")
        if isinstance(fmt_doc, dict):
            self.response_format = fmt_doc
            self.json_mode = fmt_doc.get("type") in ("json_object", "json_schema")
        self.tool_choice = body.get("tool_choice")
        for tool in body.get("tools") or []:
            func = tool.get("function") if isinstance(tool, dict) else None
            if isinstance(func, dict) and isinstance(func.get("name"), str):
                self.tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters") or {"type": "object"},
                })
        sys_texts = []
        for turn in body.get("messages") or []:
            if not isinstance(turn, dict):
                continue
            who = turn.get("role")
            content_val = turn.get("content")
            if who == "system":
                text = text_from_blocks(content_val)
                if text:
                    sys_texts.append(text)
                continue
            segments = []
            if isinstance(content_val, str):
                if content_val:
                    segments.append({"type": "text", "text": content_val})
            elif isinstance(content_val, list):
                for segment in content_val:
                    if not isinstance(segment, dict):
                        continue
                    if segment.get("type") == "text" and isinstance(segment.get("text"), str):
                        segments.append({"type": "text", "text": segment["text"]})
                    elif segment.get("type") == "image_url" and isinstance(segment.get("image_url"), dict):
                        segments.append(image_part_from_link(segment["image_url"].get("url", "")))
            self.messages.append({"role": who or "user", "parts": segments})
        self.system = "\n".join(sys_texts) if sys_texts else None

    def _ingest_anthropic(self, body):
        cap = body.get("max_tokens")
        self.max_tokens = cap if isinstance(cap, (int, float)) else None
        stop_list = body.get("stop_sequences")
        if isinstance(stop_list, list):
            self.stops = [s for s in stop_list if isinstance(s, str)]
        self.tool_choice = body.get("tool_choice")
        system_val = body.get("system")
        if isinstance(system_val, str):
            self.system = system_val or None
        elif isinstance(system_val, list):
            text = text_from_blocks(system_val)
            self.system = text or None
        for tool in body.get("tools") or []:
            if isinstance(tool, dict) and isinstance(tool.get("name"), str):
                self.tools.append({
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema") or {"type": "object"},
                })
        for turn in body.get("messages") or []:
            if not isinstance(turn, dict):
                continue
            content_val = turn.get("content")
            segments = []
            if isinstance(content_val, str):
                if content_val:
                    segments.append({"type": "text", "text": content_val})
            elif isinstance(content_val, list):
                for segment in content_val:
                    if not isinstance(segment, dict):
                        continue
                    if segment.get("type") == "text" and isinstance(segment.get("text"), str):
                        segments.append({"type": "text", "text": segment["text"]})
                    elif segment.get("type") == "image" and isinstance(segment.get("source"), dict):
                        src = segment["source"]
                        if src.get("type") == "base64" and isinstance(src.get("data"), str):
                            segments.append({"type": "image", "media_type": src.get("media_type", "image/png"), "data": src["data"]})
                        elif src.get("type") == "url":
                            segments.append({"type": "image_ref", "url": src.get("url", "")})
            self.messages.append({"role": turn.get("role") or "user", "parts": segments})

    # -- what this request demands, read straight off the request itself ----
    @property
    def needs_tools(self):
        return len(self.tools) > 0

    @property
    def needs_vision(self):
        return any(seg["type"] in ("image", "image_ref") for turn in self.messages for seg in turn["parts"])

    @property
    def user_text(self):
        for turn in reversed(self.messages):
            if turn["role"] == "user":
                return "\n".join(seg["text"] for seg in turn["parts"] if seg["type"] == "text")
        return ""


# ---------------------------------------------------------------------------
# Turning the canonical request back into a provider-wire request.
# ---------------------------------------------------------------------------
def to_openai_msgs(req):
    converted = []
    if req.system is not None:
        converted.append({"role": "system", "content": req.system})
    for turn in req.messages:
        segments = turn["parts"]
        if all(seg["type"] == "text" for seg in segments):
            converted.append({"role": turn["role"], "content": "\n".join(seg["text"] for seg in segments)})
            continue
        assembled = []
        for seg in segments:
            if seg["type"] == "text":
                assembled.append({"type": "text", "text": seg["text"]})
            elif seg["type"] == "image":
                assembled.append({"type": "image_url", "image_url": {
                    "url": "data:%s;base64,%s" % (seg["media_type"], seg["data"])}})
            else:
                assembled.append({"type": "image_url", "image_url": {"url": seg["url"]}})
        converted.append({"role": turn["role"], "content": assembled})
    return converted


def to_anthropic_msgs(req):
    converted = []
    for turn in req.messages:
        segments = turn["parts"]
        if all(seg["type"] == "text" for seg in segments):
            converted.append({"role": turn["role"], "content": "\n".join(seg["text"] for seg in segments)})
            continue
        assembled = []
        for seg in segments:
            if seg["type"] == "text":
                assembled.append({"type": "text", "text": seg["text"]})
            elif seg["type"] == "image":
                assembled.append({"type": "image", "source": {
                    "type": "base64", "media_type": seg["media_type"], "data": seg["data"]}})
            else:
                assembled.append({"type": "image", "source": {"type": "url", "url": seg["url"]}})
        converted.append({"role": turn["role"], "content": assembled})
    return converted


def translate_tool_choice(wire, req):
    requested = req.tool_choice
    if requested is None:
        return None, False
    if wire == "openai":
        return requested, True
    if wire == "anthropic":
        if isinstance(requested, str):
            kind = requested if requested in ("auto", "any", "none") else "auto"
            return {"type": kind}, True
        if isinstance(requested, dict):
            if requested.get("type") == "function" and isinstance(requested.get("function"), dict):
                return {"type": "tool", "name": requested["function"].get("name")}, True
            if requested.get("type") in ("auto", "any", "none"):
                return {"type": requested["type"]}, True
        return None, False
    return None, False


def to_gemini_payload(req):
    turns = []
    for turn in req.messages:
        segments = []
        for seg in turn["parts"]:
            if seg["type"] == "text":
                segments.append({"text": seg["text"]})
            elif seg["type"] == "image":
                segments.append({"inlineData": {"mimeType": seg["media_type"], "data": seg["data"]}})
            else:
                segments.append({"fileData": {"fileUri": seg["url"]}})
        turns.append({"role": "model" if turn["role"] == "assistant" else "user", "parts": segments})
    payload = {"contents": turns}
    if req.system is not None:
        payload["systemInstruction"] = {"parts": [{"text": req.system}]}
    if req.tools:
        payload["tools"] = [{"functionDeclarations": [
            {"name": tool["name"], "description": tool["description"], "parameters": tool["parameters"]}
            for tool in req.tools]}]
    settings = {}
    if req.max_tokens is not None:
        settings["maxOutputTokens"] = req.max_tokens
    if req.stops:
        settings["stopSequences"] = list(req.stops)
    if req.json_mode:
        settings["responseMimeType"] = "application/json"
    if settings:
        payload["generationConfig"] = settings
    return payload


def make_upstream_call(prov, model_id, req):
    """Return (url, headers, body dict) for one inference call upstream."""
    hdrs = prov.credentials()
    if prov.wire == "openai":
        if req.api == "openai":
            payload = json.loads(json.dumps(req.raw))  # client body carried over as-is
            payload["model"] = model_id
        else:
            payload = {"model": model_id, "messages": to_openai_msgs(req)}
            if req.max_tokens is not None:
                payload["max_tokens"] = req.max_tokens
            if req.stops:
                payload["stop"] = list(req.stops)
            if req.tools:
                payload["tools"] = [{"type": "function", "function": {
                    "name": tool["name"], "description": tool["description"],
                    "parameters": tool["parameters"]}} for tool in req.tools]
            translated, usable = translate_tool_choice("openai", req)
            if usable:
                payload["tool_choice"] = translated
            if req.stream:
                payload["stream"] = True
        if req.stream:
            payload["stream"] = True
            stream_opts = dict(payload.get("stream_options") or {})
            stream_opts["include_usage"] = True
            payload["stream_options"] = stream_opts
        return prov.base() + "/v1/chat/completions", hdrs, payload
    if prov.wire == "anthropic":
        hdrs["anthropic-version"] = "2023-06-01"
        if req.api == "anthropic":
            payload = json.loads(json.dumps(req.raw))
            payload["model"] = model_id
        else:
            payload = {"model": model_id, "messages": to_anthropic_msgs(req)}
            if req.system is not None:
                payload["system"] = req.system
            if req.max_tokens is not None:
                payload["max_tokens"] = req.max_tokens
            if req.stops:
                payload["stop_sequences"] = list(req.stops)
            if req.tools:
                payload["tools"] = [{"name": tool["name"], "description": tool["description"],
                                     "input_schema": tool["parameters"]} for tool in req.tools]
            translated, usable = translate_tool_choice("anthropic", req)
            if usable:
                payload["tool_choice"] = translated
            if req.stream:
                payload["stream"] = True
        return prov.base() + "/v1/messages", hdrs, payload
    # gemini
    payload = to_gemini_payload(req)
    action = "streamGenerateContent" if req.stream else "generateContent"
    target = prov.base() + "/v1beta/models/%s:%s" % (
        urllib.parse.quote(model_id, safe=""), action)
    if req.stream:
        target += "?alt=sse"
    return target, hdrs, payload


# ---------------------------------------------------------------------------
# Reading an upstream reply back into (text, pieces, stop, usage in/out).
# ---------------------------------------------------------------------------
STOP_TO_OPENAI = {
    "stop": "stop", "end_turn": "stop", "STOP": "stop",
    "max_tokens": "length", "MAX_TOKENS": "length",
    "tool_use": "tool_calls", "refusal": "content_filter", "SAFETY": "content_filter",
}
STOP_TO_ANTHROPIC = {
    "stop": "end_turn", "STOP": "end_turn", "end_turn": "end_turn",
    "length": "max_tokens", "MAX_TOKENS": "max_tokens",
    "tool_calls": "tool_use", "function_call": "tool_use",
    "content_filter": "refusal", "SAFETY": "refusal",
}


def split_sse_frames(raw):
    collected = []
    for block in raw.decode("utf-8", "replace").replace("\r\n", "\n").split("\n\n"):
        kind = None
        lines = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                kind = line[6:].strip()
            elif line.startswith("data:"):
                lines.append(line[5:].lstrip(" "))
        if lines:
            collected.append((kind, "\n".join(lines)))
    return collected


def decode_reply(wire, raw, stream):
    """Decode a reply into dict(text, pieces, stop, usage=(in, out)) or None."""
    try:
        if wire == "openai":
            if not stream:
                reply = json.loads(raw.decode("utf-8"))
                first = reply["choices"][0]
                text_out = first["message"]["content"]
                if isinstance(text_out, list):
                    text_out = "".join(b.get("text", "") for b in text_out if isinstance(b, dict))
                tokens = reply.get("usage") or {}
                return {"text": text_out, "pieces": [text_out], "stop": first.get("finish_reason"),
                        "usage": (tokens.get("prompt_tokens"), tokens.get("completion_tokens"))}
            joined, fragments, reason, tokens = "", [], None, None
            for _, data in split_sse_frames(raw):
                if data.strip() == "[DONE]":
                    continue
                piece_doc = json.loads(data)
                if isinstance(piece_doc.get("usage"), dict):
                    tokens = (piece_doc["usage"].get("prompt_tokens"), piece_doc["usage"].get("completion_tokens"))
                if piece_doc.get("choices"):
                    first = piece_doc["choices"][0]
                    change = first.get("delta") or {}
                    if isinstance(change.get("content"), str) and change["content"]:
                        fragments.append(change["content"])
                        joined += change["content"]
                    if isinstance(first.get("finish_reason"), str):
                        reason = first["finish_reason"]
            return {"text": joined, "pieces": fragments, "stop": reason, "usage": tokens}
        if wire == "anthropic":
            if not stream:
                reply = json.loads(raw.decode("utf-8"))
                text_out = "".join(b.get("text", "") for b in reply.get("content", [])
                                   if isinstance(b, dict) and b.get("type") == "text")
                tokens = reply.get("usage") or {}
                return {"text": text_out, "pieces": [text_out], "stop": reply.get("stop_reason"),
                        "usage": (tokens.get("input_tokens"), tokens.get("output_tokens"))}
            joined, fragments, reason = "", [], None
            incoming, outgoing = None, None
            for kind, data in split_sse_frames(raw):
                reply = json.loads(data)
                if kind == "message_start":
                    incoming = (reply.get("message") or {}).get("usage", {}).get("input_tokens")
                elif kind == "content_block_delta":
                    change = reply.get("delta") or {}
                    if change.get("type") == "text_delta" and isinstance(change.get("text"), str):
                        fragments.append(change["text"])
                        joined += change["text"]
                elif kind == "message_delta":
                    reason = (reply.get("delta") or {}).get("stop_reason") or reason
                    outgoing = (reply.get("usage") or {}).get("output_tokens", outgoing)
            tokens = (incoming, outgoing) if incoming is not None or outgoing is not None else None
            return {"text": joined, "pieces": fragments, "stop": reason, "usage": tokens}
        # gemini
        if not stream:
            reply = json.loads(raw.decode("utf-8"))
            candidate = reply["candidates"][0]
            text_out = "".join(part.get("text", "") for part in candidate["content"]["parts"] if isinstance(part, dict))
            usage_meta = reply.get("usageMetadata") or {}
            return {"text": text_out, "pieces": [text_out], "stop": candidate.get("finishReason"),
                    "usage": (usage_meta.get("promptTokenCount"), usage_meta.get("candidatesTokenCount"))}
        joined, fragments, reason, tokens = "", [], None, None
        for _, data in split_sse_frames(raw):
            reply = json.loads(data)
            for candidate in reply.get("candidates", []):
                for part in (candidate.get("content") or {}).get("parts", []):
                    if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"]:
                        fragments.append(part["text"])
                        joined += part["text"]
                if isinstance(candidate.get("finishReason"), str):
                    reason = candidate["finishReason"]
            usage_meta = reply.get("usageMetadata")
            if isinstance(usage_meta, dict):
                tokens = (usage_meta.get("promptTokenCount"), usage_meta.get("candidatesTokenCount"))
        return {"text": joined, "pieces": fragments, "stop": reason, "usage": tokens}
    except Exception:
        return None


def stream_pieces(answer):
    fragments = [piece for piece in answer["pieces"] if piece]
    if STREAM_PIECE_CHARS > 0:
        text = answer["text"]
        fragments = [text[i:i + STREAM_PIECE_CHARS] for i in range(0, len(text), STREAM_PIECE_CHARS)] or [""]
    return fragments or [""]


# ---------------------------------------------------------------------------
# Replying to the client in the wire it spoke.
# ---------------------------------------------------------------------------
def render_reply(req, model_id, answer):
    """Return (content_type, body bytes) for the client's wire, streamed if asked."""
    tokens = answer["usage"] or (None, None)
    if req.api == "openai":
        reason = STOP_TO_OPENAI.get(answer["stop"], "stop")
        if not req.stream:
            reply_doc = {
                "id": "chatcmpl-agg", "object": "chat.completion", "created": FIXED_CREATED,
                "model": model_id,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": answer["text"]},
                             "finish_reason": reason, "logprobs": None}],
                "usage": {"prompt_tokens": tokens[0], "completion_tokens": tokens[1],
                          "total_tokens": (tokens[0] or 0) + (tokens[1] or 0)},
            }
            return "application/json", json.dumps(reply_doc).encode("utf-8")
        out_frames = []

        def make_chunk(delta, finish):
            return {"id": "chatcmpl-agg", "object": "chat.completion.chunk", "created": FIXED_CREATED,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        out_frames.append("data: " + json.dumps(make_chunk({"role": "assistant", "content": ""}, None)) + "\n\n")
        for fragment in stream_pieces(answer):
            out_frames.append("data: " + json.dumps(make_chunk({"content": fragment}, None)) + "\n\n")
        out_frames.append("data: " + json.dumps(make_chunk({}, reason)) + "\n\n")
        if tokens[0] is not None:
            usage_chunk = {
                "id": "chatcmpl-agg", "object": "chat.completion.chunk", "created": FIXED_CREATED,
                "model": model_id, "choices": [],
                "usage": {"prompt_tokens": tokens[0], "completion_tokens": tokens[1],
                          "total_tokens": (tokens[0] or 0) + (tokens[1] or 0)},
            }
            out_frames.append("data: " + json.dumps(usage_chunk) + "\n\n")
        out_frames.append("data: [DONE]\n\n")
        return "text/event-stream", "".join(out_frames).encode("utf-8")
    # anthropic
    reason = STOP_TO_ANTHROPIC.get(answer["stop"], "end_turn")
    if not req.stream:
        reply_doc = {"id": "msg_agg", "type": "message", "role": "assistant", "model": model_id,
                     "content": [{"type": "text", "text": answer["text"]}],
                     "stop_reason": reason, "stop_sequence": None,
                     "usage": {"input_tokens": tokens[0], "output_tokens": tokens[1]}}
        return "application/json", json.dumps(reply_doc).encode("utf-8")
    out_frames = []

    def make_frame(event, doc):
        return "event: %s\ndata: %s\n\n" % (event, json.dumps(doc))

    out_frames.append(make_frame("message_start", {
        "type": "message_start",
        "message": {"id": "msg_agg", "type": "message", "role": "assistant", "model": model_id,
                    "content": [], "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": tokens[0], "output_tokens": 0}}}))
    out_frames.append(make_frame("content_block_start", {
        "type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}))
    for fragment in stream_pieces(answer):
        out_frames.append(make_frame("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": fragment}}))
    out_frames.append(make_frame("content_block_stop", {"type": "content_block_stop", "index": 0}))
    out_frames.append(make_frame("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": reason, "stop_sequence": None},
        "usage": {"output_tokens": tokens[1]}}))
    out_frames.append(make_frame("message_stop", {"type": "message_stop"}))
    return "text/event-stream", "".join(out_frames).encode("utf-8")


# ---------------------------------------------------------------------------
# Routing: rank free models that fit the request by the brief's rating,
# walk them in order, and keep a tally of refusals as they arrive.
# ---------------------------------------------------------------------------
class Picker:
    def __init__(self, upstreams, ratings):
        self.upstreams = {prov.id: prov for prov in upstreams}
        self.order = [prov.id for prov in upstreams]
        self.rows = {}
        for rating in ratings:
            self.rows[(rating["provider"], rating["model"])] = rating
        self.spent = {pid: 0 for pid in self.order}
        self.refusals = {pid: 0 for pid in self.order}
        self.exhausted = set()

    def ranked(self, req):
        json_needed = req.json_mode or (req.api == "anthropic" and ANTHROPIC_JSON_SAFE)
        is_long = len(req.user_text) >= LONG_CONTEXT_CHARS
        ranked_rows = []
        for position, pid in enumerate(self.order):
            prov = self.upstreams[pid]
            for (prov_id, model_id), rating in self.rows.items():
                if prov_id != pid or rating.get("free") is not True:
                    continue
                if req.needs_tools and not rating.get("tools"):
                    continue
                if req.needs_vision and not rating.get("vision"):
                    continue
                if json_needed and not rating.get("json"):
                    continue
                if is_long and (rating.get("context_length") or 0) < LONG_CONTEXT_MIN:
                    continue
                points = rating.get("rating_milli", 0)
                if prov.wire == req.api:
                    points += PREFER_SAME_WIRE_BONUS
                ranked_rows.append((-points, position, prov, model_id))
        # Ties resolve by brief order by default; "alpha" sorts by provider
        # then model name instead, so equal scores stay deterministic either way.
        if TIE_BREAK == "alpha":
            ranked_rows.sort(key=lambda item: (item[0], item[2].id, item[3]))
        else:
            ranked_rows.sort(key=lambda item: (item[0], item[1]))
        return [(prov, model_id) for _, _, prov, model_id in ranked_rows]

    def dispatch(self, req):
        for prov, model_id in self.ranked(req):
            if prov.id in self.exhausted:
                continue
            target, hdrs, payload = make_upstream_call(prov, model_id, req)
            code, _, raw_bytes = fetch_url("POST", target, payload, hdrs, timeout=20)
            self.spent[prov.id] += 1
            if 200 <= code < 300:
                decoded = decode_reply(prov.wire, raw_bytes, req.stream)
                if decoded is not None:
                    return model_id, decoded
                continue
            if code in (402, 429):
                self.refusals[prov.id] += 1
                if self.refusals[prov.id] >= REFUSAL_LIMIT:
                    self.exhausted.add(prov.id)
                continue
            # Anything else (400, 5xx, silence): move on to the next model.
            # A reply that decodes to nothing is treated the same way, so a
            # garbled 200 never blocks the rest of the ranking.
        return None


# ---------------------------------------------------------------------------
# The HTTP server the harness client talks to.
# ---------------------------------------------------------------------------
STATE = {"picker": None, "catalog": None}


def list_models_payload(headers):
    cat = STATE["catalog"] or {"providers": []}
    model_ids = []
    for prov in cat["providers"]:
        for entry in prov["models"]:
            if entry["free"] and entry["id"] not in model_ids:
                model_ids.append(entry["id"])
    if headers.get("x-api-key") is not None:
        listing = {"data": [{"type": "model", "id": mid, "display_name": mid,
                             "created_at": "2026-01-01T00:00:00Z"} for mid in model_ids],
                   "has_more": False,
                   "first_id": model_ids[0] if model_ids else None,
                   "last_id": model_ids[-1] if model_ids else None}
    else:
        listing = {"object": "list",
                   "data": [{"id": mid, "object": "model", "created": FIXED_CREATED,
                             "owned_by": "aggregator"} for mid in model_ids]}
    return json.dumps(listing).encode("utf-8")


def failure_payload(api, detail):
    if api == "anthropic":
        return json.dumps({"type": "error", "error": {"type": "api_error", "message": detail}}).encode("utf-8")
    return json.dumps({"error": {"message": detail, "type": "api_error",
                                 "param": None, "code": "no_provider"}}).encode("utf-8")


class Gateway(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, status, content_type, body):
        try:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path.split("?")[0] == "/v1/models":
            self._send(200, "application/json", list_models_payload(self.headers))
        else:
            self._send(404, "application/json", failure_payload("openai", "not found"))

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/v1/chat/completions":
            api = "openai"
        elif path == "/v1/messages":
            api = "anthropic"
        else:
            self._send(404, "application/json", failure_payload("openai", "not found"))
            return
        try:
            size = int(self.headers.get("content-length") or 0)
        except ValueError:
            size = 0
        raw_bytes = self.rfile.read(size) if size > 0 else b""
        try:
            parsed = json.loads(raw_bytes.decode("utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("body is not an object")
        except Exception:
            self._send(400, "application/json", failure_payload(api, "the request body is not one JSON object"))
            return
        canonical = CanonicalRequest(api, parsed)
        result = STATE["picker"].dispatch(canonical)
        if result is None:
            self._send(502, "application/json", failure_payload(api, "no free model could serve this request"))
            return
        chosen, decoded = result
        mime, encoded = render_reply(canonical, chosen, decoded)
        self._send(200, mime, encoded)


# ---------------------------------------------------------------------------
# Trial lifecycle: probe, register, serve until the harness says done, exit 0.
# ---------------------------------------------------------------------------
def main(argv):
    upstreams, ratings, _ = read_brief()
    if "--list" in argv:
        if not upstreams:
            print('{"providers":[]}')
            return 0
        print(catalog_json(assemble_catalog(upstreams, ratings)))
        return 0
    catalog = assemble_catalog(upstreams, ratings)
    STATE["catalog"] = catalog
    STATE["picker"] = Picker(upstreams, ratings)
    httpd = ThreadingHTTPServer(("0.0.0.0", PORT), Gateway)
    httpd.daemon_threads = True
    import threading
    import time
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05})
    thread.daemon = True
    thread.start()
    if ENV_URL:
        for _ in range(30):
            code, _, reply = fetch_json("POST", ENV_URL + "/register",
                                        {"protocol_version": 1, "port": PORT, "catalog": catalog},
                                        timeout=5)
            if code == 200 and isinstance(reply, dict) and reply.get("ok") is True:
                break
            time.sleep(0.5)
        miss_count = 0
        elapsed = 0.0
        while elapsed < 170.0:
            code, _, reply = fetch_json("GET", ENV_URL + "/work", timeout=5)
            if code == 200 and isinstance(reply, dict):
                miss_count = 0
                if reply.get("done") is True:
                    break
            else:
                miss_count += 1
                if miss_count > 40:
                    break
            time.sleep(0.2)
            elapsed += 0.2
    else:
        thread.join()
    httpd.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
