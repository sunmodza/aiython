"""Capability calls through provider SDKs, loaded only when needed."""
from __future__ import annotations

import base64
from contextlib import ExitStack
import io
import ipaddress
import json
import mimetypes
from pathlib import Path
import re
import socket
import time
from urllib.parse import urlsplit
import uuid
import wave

import httpx

from .assets import Asset, Audio, Image, Video
from .capabilities import CapabilityError, CapabilityResult, Embeddings, InvocationError, canonical
from .models import ConfigError
from .providers import failure_reason, remaining, route_settings, sdk
from .stats import provider_request_progress, provider_request_started, record_token_usage


def plain(value):
    return value.model_dump(exclude_none=False) if hasattr(value, "model_dump") else value


def text_of(document):
    if isinstance(document, str):
        return document
    if isinstance(document, dict) and isinstance(document.get("text"), str):
        return document["text"]
    raise CapabilityError("Expected text or a document/search hit containing text")


def public_context(value):
    if type(value) is dict:
        return {key: public_context(item) for key, item in value.items() if key != "object"}
    if type(value) in (list, tuple):
        return [public_context(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise CapabilityError("Reasoning context must contain text or JSON records")


def data_uri(asset: Asset) -> str:
    return f"data:{asset.mime_type};base64,{base64.b64encode(asset.path.read_bytes()).decode()}"


def _public_https(url: str) -> bool:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        return False
    try:
        addresses = socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
        return bool(addresses) and all(ipaddress.ip_address(item[4][0]).is_global for item in addresses)
    except (OSError, ValueError):
        return False


def download_image(url: str) -> bytes:
    """Fetch a provider image URL without redirects, credentials or private hosts."""
    if not _public_https(url):
        raise InvocationError("Untrusted image download URL", accepted=True)
    try:
        with httpx.Client(follow_redirects=False, timeout=30) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                chunks, size = [], 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > 32 * 1024 * 1024:
                        raise InvocationError("Image response is too large", accepted=True)
                    chunks.append(chunk)
                return b"".join(chunks)
    except httpx.HTTPError:
        raise InvocationError("Image download failed", accepted=True) from None


def _openrouter_video_policy_error(exc: Exception) -> str | None:
    """Translate structured account-policy exclusions without exposing response bodies."""
    error = getattr(getattr(exc, "data", None), "error", None)
    metadata = getattr(error, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    reasons = metadata.get("ineligibility_reasons")
    if not isinstance(reasons, list):
        return None
    if any(isinstance(item, dict) and str(item.get("reason", "")).startswith("zdr-violation")
           for item in reasons):
        return ("OpenRouter video generation is blocked by Zero Data Retention (ZDR) settings. "
                "Video jobs require temporary retention. Change the account privacy setting at "
                "https://openrouter.ai/settings/privacy or choose another video provider")
    return None


class LiteLLMAdapter:
    version = "3"

    def __init__(self, config, profile, name):
        self.config, self.profile, self.name = config, profile, name
        info = config.providers.get(name)
        if not info:
            raise ConfigError(f"Unknown LiteLLM route {name}")
        self.api_base = info.get("api_base")
        self.model = info["model"]

    def capabilities(self):
        return {"reasoning", "vision", "document_understanding", "embedding", "reranking",
                "speech_to_text", "text_to_speech", "image_generation", "image_editing", "video"}

    def _settings(self, model=None):
        return route_settings(self.config, {"provider": self.name, "model": model or self.model})

    def _call(self, method, model, **kwargs):
        settings = self._settings(model)
        sdk_model, api_base = model, settings["api_base"]
        if method in ("transcription", "speech") and model.startswith("openrouter/"):
            # OpenRouter exposes OpenAI-compatible audio endpoints, but this
            # LiteLLM release has no OpenRouter audio dispatcher. Keep the
            # OpenRouter slug as the upstream model through LiteLLM's OpenAI path.
            sdk_model = "openai/" + model.removeprefix("openrouter/")
            api_base = api_base or "https://openrouter.ai/api/v1"
            if method == "transcription":
                # LiteLLM's Whisper fallback otherwise asks for verbose_json,
                # which OpenRouter's transcription models can reject.
                kwargs.setdefault("response_format", "json")
        number = provider_request_started(f"{method} {model}")
        try:
            fn = getattr(sdk(), method)
            timeout = 600 if method.startswith("video_") else remaining(self.profile.timeout)
            result = fn(model=sdk_model, api_key=settings["api_key"], api_base=api_base,
                        timeout=timeout, max_retries=0, **kwargs)
            provider_request_progress(number, "response received")
            usage = result.get("usage") if isinstance(result, dict) else getattr(result, "usage", None)
            if hasattr(usage, "model_dump"):
                usage = usage.model_dump()
            record_token_usage(usage)
            return result
        except InvocationError:
            raise
        except Exception as exc:
            code = getattr(exc, "status_code", None)
            if type(exc).__name__ == "UnsupportedParamsError":
                parameter = re.search(r"Setting `([A-Za-z_][A-Za-z0-9_]*)`", str(exc))
                name = parameter.group(1) if parameter else "a request parameter"
                raise ConfigError(f"LiteLLM does not support {name} for {method} with model {model}") from None
            if isinstance(exc, ValueError) and "Unmapped provider passed" in str(exc):
                raise InvocationError(f"LiteLLM does not support {method} for this model provider",
                                      accepted=False) from None
            raise InvocationError(failure_reason(code) + (f" (HTTP {code})" if type(code) is int else ""),
                                  retryable=code == 429,
                                  accepted=code not in (400, 401, 403, 404, 422, 429)) from None

    def _openrouter_video(self, method, route_model, *, timeout=60, **kwargs):
        from openrouter import OpenRouter

        settings = self._settings(route_model)
        number = provider_request_started(f"video_{method} {route_model}")
        try:
            with OpenRouter(api_key=settings["api_key"],
                            server_url=settings["api_base"] or "https://openrouter.ai/api/v1",
                            retry_config=None) as client:
                operation = getattr(client.video_generation, method)
                response = operation(retries=None, timeout_ms=int(max(1, timeout) * 1000), **kwargs)
                if method == "get_video_content":
                    try:
                        result = (response.read(), response.headers.get("content-type", "video/mp4"))
                    finally:
                        response.close()
                else:
                    result = plain(response)
            provider_request_progress(number, "response received")
            return result
        except InvocationError:
            raise
        except Exception as exc:
            if method == "generate" and (policy_error := _openrouter_video_policy_error(exc)):
                raise ConfigError(policy_error) from None
            code = getattr(exc, "status_code", None)
            if type(code) is not int:
                response = getattr(exc, "raw_response", None) or getattr(exc, "http_res", None)
                code = getattr(response, "status_code", None)
            raise InvocationError(failure_reason(code) + (f" (HTTP {code})" if type(code) is int else ""),
                                  retryable=code == 429,
                                  accepted=code not in (400, 401, 402, 403, 404, 422, 429)) from None

    def _part(self, asset, model):
        if asset.mime_type.startswith("image/"):
            return {"type": "image_url", "image_url": {"url": data_uri(asset)}}
        if asset.path.stat().st_size > 10 * 1024 * 1024:
            if not model.startswith("gemini/"):
                raise CapabilityError("Large media input requires a provider with a file upload API")
            settings = self._settings(model)
            number = provider_request_started(f"create_file {model}")
            try:
                with asset.path.open("rb") as file:
                    created = sdk().create_file(file=file, purpose="messages", custom_llm_provider="gemini",
                                                api_key=settings["api_key"], api_base=settings["api_base"],
                                                max_retries=0, timeout=remaining(self.profile.timeout))
                provider_request_progress(number, "upload complete")
                identifier = plain(created)["id"]
                deadline = time.monotonic() + remaining(self.profile.timeout)
                while plain(created).get("status", "uploaded").lower() in ("uploaded", "processing", "pending"):
                    if time.monotonic() >= deadline:
                        raise InvocationError("Media processing timed out after upload", accepted=True)
                    time.sleep(2)
                    created = sdk().file_retrieve(identifier, custom_llm_provider="gemini",
                                                 api_key=settings["api_key"], api_base=settings["api_base"],
                                                 max_retries=0, timeout=min(30, max(1, deadline - time.monotonic())))
                if plain(created).get("status") != "processed":
                    raise InvocationError("Media processing failed after upload", accepted=True)
                return {"type": "file", "file": {"file_id": identifier}}
            except Exception:
                raise InvocationError("Media upload failed; acceptance is unknown", accepted=True) from None
        return {"type": "file", "file": {"filename": asset.path.name, "file_data": data_uri(asset)}}

    @staticmethod
    def artifact(context, data: bytes, mime: str, kind):
        if not isinstance(data, bytes) or not data:
            raise InvocationError("Provider returned empty media", accepted=True)
        if kind is Audio and mime.startswith(("audio/L16", "audio/pcm")):
            output = io.BytesIO()
            with wave.open(output, "wb") as file:
                file.setnchannels(1)
                file.setsampwidth(2)
                file.setframerate(24000)
                file.writeframes(data)
            data, mime = output.getvalue(), "audio/wav"
        suffix = mimetypes.guess_extension(mime.split(";")[0]) or ".bin"
        folder = context.store.root / "artifacts"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (uuid.uuid4().hex + suffix)
        with path.open("xb") as file:
            file.write(data)
        return kind(path, mime)

    def _completion(self, request, prompt, assets=()):
        content = [{"type": "text", "text": prompt}]
        content.extend(self._part(asset, request.model) for asset in assets)
        response = self._call("completion", request.model,
                              messages=[{"role": "user", "content": content}], stream=False)
        value = plain(response)
        message = value["choices"][0]["message"]
        output = message.get("content")
        if isinstance(output, list):
            output = "\n".join(part.get("text", "") for part in output if isinstance(part, dict))
        if not isinstance(output, str):
            raise InvocationError("Model returned no text", accepted=True)
        return output, value.get("usage") or {}

    def wait_video(self, identifier, context):
        record = context.store.job(identifier)
        if not record:
            raise CapabilityError("Unknown video job")
        if record.get("artifact") and Path(record["artifact"]).is_file():
            return Video(Path(record["artifact"]), "video/mp4")
        settings = self._settings(record["model"])
        provider = record["model"].partition("/")[0]
        deadline = time.monotonic() + 600
        while True:
            try:
                if provider == "openrouter":
                    result = self._openrouter_video("get_generation", record["model"], job_id=identifier,
                                                    timeout=min(60, max(1, deadline - time.monotonic())))
                else:
                    number = provider_request_started(f"video_status {record['model']}")
                    result = sdk().video_status(identifier, custom_llm_provider=provider,
                                                api_key=settings["api_key"],
                                                api_base=settings["api_base"], max_retries=0,
                                                timeout=min(60, max(1, deadline - time.monotonic())))
                    provider_request_progress(number, "response received")
                status = plain(result).get("status")
            except ConfigError:
                raise
            except Exception:
                raise InvocationError(f"Video job remains pending; resume operation {identifier}", accepted=True) from None
            if status in ("failed", "cancelled", "canceled"):
                context.store.job(identifier, {**record, "status": status})
                raise InvocationError(f"Video job {status}", accepted=True)
            if status in ("completed", "succeeded", "complete"):
                try:
                    if provider == "openrouter":
                        data, mime = self._openrouter_video("get_video_content", record["model"], job_id=identifier,
                                                            timeout=min(60, max(1, deadline - time.monotonic())))
                    else:
                        number = provider_request_started(f"video_content {record['model']}")
                        data = sdk().video_content(identifier, custom_llm_provider=provider,
                                                   api_key=settings["api_key"],
                                                   api_base=settings["api_base"], max_retries=0, timeout=60)
                        provider_request_progress(number, "response received")
                        mime = "video/mp4"
                    artifact = self.artifact(context, data, mime, Video)
                except ConfigError:
                    raise
                except Exception:
                    raise InvocationError(f"Video download failed; resume operation {identifier}", accepted=True) from None
                context.store.job(identifier, {**record, "status": "complete", "artifact": str(artifact.path)})
                return artifact
            if time.monotonic() >= deadline:
                raise InvocationError(f"Video job remains pending; resume operation {identifier}", accepted=True)
            time.sleep(5)

    def invoke(self, request, context):
        p, cap = request.params, request.capability
        try:
            if cap == "embedding":
                inputs = p["inputs"]
                if all(isinstance(item, str) for item in inputs):
                    response = plain(self._call("embedding", request.model, input=inputs,
                                                dimensions=p.get("dimensions"), task_type=p.get("task_type")))
                    rows = sorted(response["data"], key=lambda row: row["index"])
                    vectors = [row["embedding"] for row in rows]
                else:
                    vectors = []
                    for item in inputs:
                        content = item if isinstance(item, str) else [self._part(item, request.model)]
                        response = plain(self._call("embedding", request.model, input=[content],
                                                    dimensions=p.get("dimensions"), task_type=p.get("task_type")))
                        vectors.append(response["data"][0]["embedding"])
                if len(vectors) != len(inputs):
                    raise ValueError("Embedding count mismatch")
                space = canonical({"provider": self.api_base, "model": request.model,
                                   "revision": request.revision, "dimensions": len(vectors[0])})
                return CapabilityResult(Embeddings(vectors, space))
            if cap == "reranking":
                docs = p["documents"]
                response = plain(self._call("rerank", request.model, query=p["query"],
                                            documents=[text_of(item) for item in docs],
                                            top_n=p.get("limit", len(docs))))
                rows = response["results"]
                indices = [row["index"] for row in rows]
                if any(type(i) is not int or not 0 <= i < len(docs) for i in indices) or len(indices) != len(set(indices)):
                    raise ValueError("Invalid rerank indices")
                return CapabilityResult([docs[i] for i in indices], response.get("usage") or {})
            if cap == "video" and p["mode"] == "generate":
                images = p.get("assets", [])
                if len(images) > 1:
                    raise CapabilityError("Video generation accepts one starting image")
                if request.model.startswith("openrouter/"):
                    payload = {"model": request.model.removeprefix("openrouter/"), "prompt": p["prompt"]}
                    if images:
                        payload["frame_images"] = [{"type": "image_url", "frame_type": "first_frame",
                                                    "image_url": {"url": data_uri(images[0])}}]
                    response = self._openrouter_video("generate", request.model, **payload)
                else:
                    response = plain(self._call("video_generation", request.model, prompt=p["prompt"],
                                                input_reference=images[0].path if images else None))
                identifier = response.get("id")
                if not isinstance(identifier, str) or not identifier:
                    raise InvocationError("Video submission returned no job ID; acceptance is unknown", accepted=True)
                context.store.job(identifier, {"status": "pending", "model": request.model,
                                               "provider": request.provider, "provider_url": self.api_base})
                return CapabilityResult(self.wait_video(identifier, context))
            if cap == "speech_to_text":
                parts = []
                for asset in p["assets"]:
                    with asset.path.open("rb") as file:
                        result = plain(self._call("transcription", request.model, file=file, prompt=p.get("prompt")))
                    parts.append(result["text"])
                return CapabilityResult("\n".join(parts))
            if cap == "text_to_speech":
                response = self._call("speech", request.model, input=p["text"], voice=p.get("voice", "Kore"))
                data = response if isinstance(response, bytes) else getattr(response, "content", None)
                if data is None and callable(getattr(response, "read", None)):
                    data = response.read()
                mime = ("audio/wav" if data.startswith(b"RIFF") else
                        "audio/mpeg" if data.startswith((b"ID3", b"\xff\xfb")) else
                        "audio/pcm" if request.model.startswith("gemini/") else "audio/mpeg")
                return CapabilityResult(self.artifact(context, data, mime, Audio))
            if cap in ("image_generation", "image_editing"):
                if cap == "image_generation":
                    response = self._call("image_generation", request.model, prompt=p["prompt"])
                else:
                    with ExitStack() as stack:
                        images = [stack.enter_context(asset.path.open("rb")) for asset in p["assets"]]
                        response = self._call("image_edit", request.model,
                                              image=images[0] if len(images) == 1 else images,
                                              prompt=p["prompt"])
                row = plain(response)["data"][0]
                data = base64.b64decode(row["b64_json"], validate=True) if row.get("b64_json") else download_image(row["url"])
                return CapabilityResult(self.artifact(context, data, "image/png", Image))
            prompt = p.get("prompt", "")
            if cap == "reasoning" and "context" in p:
                prompt += "\nContext (cite source fields):\n" + canonical(public_context(p["context"]))
            if cap == "document_understanding" and not prompt:
                prompt = ("Extract document content as a JSON array of objects with text, "
                          "zero-based asset_index and one-based page number.")
            if "context" in p and cap != "reasoning":
                prompt += "\n" + canonical(public_context(p["context"]))
            text, usage = self._completion(request, prompt, p.get("assets", []))
            if cap == "document_understanding" and not p.get("prompt"):
                chunks = json.loads(text)
                if not isinstance(chunks, list) or not chunks:
                    raise ValueError("Document extraction returned no chunks")
                text = []
                for chunk in chunks:
                    index, page = chunk["asset_index"], chunk["page"]
                    if type(index) is not int or not 0 <= index < len(p["assets"]) or type(page) is not int or page < 1:
                        raise ValueError("Invalid document location")
                    text.append({"text": chunk["text"], "source": f"{p['assets'][index].path}:page {page}"})
            return CapabilityResult(text, usage)
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
            raise InvocationError("Malformed capability response", accepted=True) from None


def make_adapter(config, profile, name):
    return LiteLLMAdapter(config, profile, name)
