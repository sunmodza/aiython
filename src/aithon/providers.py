"""One model gateway backed by the LiteLLM Python SDK.

LiteLLM is imported only when an AI statement actually makes a model call.
The runtime owns retries, deadlines and redacted diagnostics.
"""
from __future__ import annotations

from contextvars import ContextVar
import os
import json
import time

from .models import ConfigError, ProviderError
from .stats import CURRENT_STATS, provider_request_progress, provider_request_started, record_token_usage, record_request_bytes


INVOCATION_DEADLINE: ContextVar[float | None] = ContextVar("aithon_invocation_deadline", default=None)


def sdk():
    # LiteLLM's development default loads an ambient .env at import time.
    # Aithon resolves its own project credentials and passes them explicitly.
    os.environ["LITELLM_MODE"] = "PRODUCTION"
    import litellm
    litellm.suppress_debug_info = True
    return litellm


def remaining(default: float) -> float:
    deadline = INVOCATION_DEADLINE.get()
    if deadline is None:
        return default
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise ProviderError("AI invocation exceeded its wall-clock deadline")
    return min(default, seconds)


def response_error(profile, reason, *, code=None):
    detail = {"reason": reason}
    if type(code) is int:
        detail["code"] = code
    stats = CURRENT_STATS.get()
    if stats is not None:
        stats.provider_errors.append(detail)
    suffix = f" (code {code})" if type(code) is int else ""
    raise ProviderError(f"Profile {profile.name!r}: {reason}{suffix}")


def failure_reason(code: int | None) -> str:
    if code == 401:
        return "authentication failed; check the configured API key"
    if code == 403:
        return "access denied; check API key permissions and model access"
    if code == 404:
        return "model or endpoint is unavailable"
    return "provider request failed"


def parse_completion(response, profile):
    result = response.model_dump(exclude_none=False) if hasattr(response, "model_dump") else response
    if not isinstance(result, dict):
        response_error(profile, "response root must be an object")
    record_token_usage(result.get("usage"))
    if result.get("error") is not None:
        response_error(profile, "provider returned an error envelope")
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        response_error(profile, "response has no completion choices")
    choice = choices[0]
    if choice.get("finish_reason") in ("length", "content_filter", "error") or choice.get("error") is not None:
        response_error(profile, "provider did not complete the tool response")
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("refusal"):
        response_error(profile, "completion has no usable assistant message")
    calls = message.get("tool_calls")
    if calls is not None and not isinstance(calls, list):
        response_error(profile, "message tool_calls must be an array")
    return {"role": "assistant", "content": message.get("content"),
            **({"tool_calls": calls} if calls else {})}


def route_settings(config, route):
    info = config.providers.get(route["provider"], {})
    key_name = info.get("api_key_env")
    key = os.environ.get(key_name, config.secrets.get(key_name)) if key_name else None
    if key_name and not key:
        raise ConfigError(f"Missing credential {key_name} for model {route['model']}")
    return {"model": route["model"], "api_key": key, "api_base": info.get("api_base")}


class LiteLLMProvider:
    def __init__(self, config, profile):
        self.config = config
        self.profile = profile

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        routes = self.profile.routes.get("reasoning", [])
        if not routes:
            raise ConfigError("Reasoning model is not configured")
        for index, route in enumerate(routes):
            settings = route_settings(self.config, route)
            number = provider_request_started(settings["model"])
            try:
                record_request_bytes(len(json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False).encode()))
                llm = sdk()
                result = llm.completion(
                    model=settings["model"], messages=messages, tools=tools,
                    tool_choice=self.profile.tool_choice if tools else "auto",
                    api_key=settings["api_key"], api_base=settings["api_base"],
                    timeout=remaining(self.profile.timeout), max_retries=0, stream=False,
                )
                provider_request_progress(number, "response received")
                return parse_completion(result, self.profile)
            except (ConfigError, ProviderError):
                raise
            except Exception as exc:
                code = getattr(exc, "status_code", None)
                if code == 429 and index + 1 < len(routes):
                    continue
                response_error(self.profile, failure_reason(code), code=code)
        raise ProviderError("No reasoning route succeeded")

    async def acomplete(self, messages: list[dict], tools: list[dict]) -> dict:
        routes = self.profile.routes.get("reasoning", [])
        if not routes:
            raise ConfigError("Reasoning model is not configured")
        for index, route in enumerate(routes):
            settings = route_settings(self.config, route)
            number = provider_request_started(settings["model"])
            try:
                record_request_bytes(len(json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False).encode()))
                llm = sdk()
                result = await llm.acompletion(
                    model=settings["model"], messages=messages, tools=tools,
                    tool_choice=self.profile.tool_choice if tools else "auto",
                    api_key=settings["api_key"], api_base=settings["api_base"],
                    timeout=remaining(self.profile.timeout), max_retries=0, stream=False,
                )
                provider_request_progress(number, "response received")
                return parse_completion(result, self.profile)
            except (ConfigError, ProviderError):
                raise
            except Exception as exc:
                code = getattr(exc, "status_code", None)
                if code == 429 and index + 1 < len(routes):
                    continue
                response_error(self.profile, failure_reason(code), code=code)
        raise ProviderError("No reasoning route succeeded")


def load_provider(config, profile):
    return LiteLLMProvider(config, profile)
