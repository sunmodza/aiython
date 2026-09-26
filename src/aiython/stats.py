"""Opt-in timing; no source, values, prompts or credentials are recorded."""
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
import json
import sys
import threading


@dataclass
class InvocationStats:
    filename: str
    line: int
    mode: str
    model_calls: int = 0
    provider_requests: int = 0
    tools: int = 0
    request_bytes: int | None = None
    request_token_estimates: list[int] = field(default_factory=list)
    context_build_seconds: float = 0
    context_bytes: dict[str, int] = field(default_factory=dict)
    provider_seconds: float = 0
    retry_backoff_seconds: float = 0
    runtime_seconds: float = 0
    invocation_seconds: float = 0
    capability_seconds: float = 0
    capability_provider_seconds: float = 0
    cache_seconds: float = 0
    sdk_load_seconds: float = 0
    empty_responses: int = 0
    invalid_batches: int = 0
    tool_failures: int = 0
    tool_counts: dict[str, int] = field(default_factory=dict)
    model_call_seconds: list[float] = field(default_factory=list)
    capability_events: list[dict] = field(default_factory=list)
    token_usage: dict[str, int] = field(default_factory=dict)
    provider_errors: list[dict] = field(default_factory=list)
    tool_errors: list[dict] = field(default_factory=list)

    def __post_init__(self):
        # Not a dataclass field: diagnostics serialize data, never this lock.
        self._lock = threading.Lock()

    def add_seconds(self, name, seconds):
        with self._lock:
            setattr(self, name, getattr(self, name) + seconds)


CURRENT_STATS: ContextVar[InvocationStats | None] = ContextVar("aiython_stats", default=None)


def provider_request_started(label: str) -> int | None:
    stats = CURRENT_STATS.get()
    if stats is None:
        return None
    with stats._lock:
        stats.provider_requests += 1
        number = stats.provider_requests
    print(f"aiython provider request {number}: sending {label}", file=sys.stderr, flush=True)
    return number


def provider_request_progress(number: int | None, stage: str):
    if number is not None:
        print(f"aiython provider request {number}: {stage}", file=sys.stderr, flush=True)


def record_request_bytes(size: int):
    stats = CURRENT_STATS.get()
    if stats is not None:
        with stats._lock:
            stats.request_bytes = (stats.request_bytes or 0) + size


def record_token_usage(usage):
    """Record only provider-reported counts, never response content."""
    stats = CURRENT_STATS.get()
    if stats is None or not isinstance(usage, dict):
        return
    counts = {name: usage.get(name) for name in ("prompt_tokens", "completion_tokens", "total_tokens")}
    for group, name in (("completion_tokens_details", "reasoning_tokens"),
                        ("prompt_tokens_details", "cache_write_tokens"),
                        ("prompt_tokens_details", "cached_tokens")):
        details = usage.get(group)
        if isinstance(details, dict):
            counts[name] = details.get(name)
    prompt, cached = counts.get('prompt_tokens'), counts.get('cached_tokens')
    if type(prompt) is int and type(cached) is int and 0 <= cached <= prompt:
        counts['uncached_prompt_tokens'] = prompt - cached
    with stats._lock:
        for name, count in counts.items():
            if type(count) is int and count >= 0:
                stats.token_usage[name] = stats.token_usage.get(name, 0) + count


class Stats:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.invocations: list[InvocationStats] = []
        self._lock = threading.Lock()
        self.parse_seconds = 0.0
        self.prepare_seconds = 0.0
        self.preparation_cache_hits = 0
        self.preparation_cache_misses = 0
        self.run = None

    def start(self, request, recovery):
        if not self.enabled:
            return None
        record = InvocationStats(request.span.filename, request.span.line,
                                 "recovery" if recovery else "syntax")
        with self._lock:
            self.invocations.append(record)
        return record

    def report(self):
        if self.enabled:
            with self._lock:
                records = [asdict(s) for s in self.invocations]
            print("aiython stats: " + json.dumps(records), file=sys.stderr)
            if self.run is not None:
                print("aiython run stats: " + json.dumps({**self.run,
                    'parse_seconds': self.parse_seconds, 'prepare_seconds': self.prepare_seconds,
                    'preparation_cache_hits': self.preparation_cache_hits,
                    'preparation_cache_misses': self.preparation_cache_misses}), file=sys.stderr)
