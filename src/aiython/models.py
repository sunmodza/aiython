from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol


class AiythonError(Exception):
    """An Aiython infrastructure error; never recursively recovered by AI."""


class ConfigError(AiythonError):
    pass


class ProviderError(AiythonError):
    pass


class DirectiveError(AiythonError):
    pass


@dataclass(frozen=True)
class ProfileConfig:
    name: str
    provider: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str | None = None
    prompt: str = ""
    timeout: float = 120
    max_rounds: int = 16
    tool_choice: str = "required"
    routes: dict[str, list[dict]] = field(default_factory=dict)
    permissions: tuple[str, ...] = ("read_asset", "read_code", "network", "generate_file", "write_filesystem", "execute_code")


@dataclass(frozen=True)
class ResolvedConfig:
    path: Path | None
    project_root: Path
    default_profile: str | None = None
    profiles: dict[str, ProfileConfig] = field(default_factory=dict)
    secrets: dict[str, str] = field(default_factory=dict, repr=False)
    cli_profile: str | None = None
    force_profile: str | None = None
    providers: dict[str, dict] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class DirectiveContext:
    profile: str | None = None
    prompts: tuple[str, ...] = ()
    capability: str | None = None
    provider: str | None = None

    def extend(self, other: DirectiveContext) -> DirectiveContext:
        return DirectiveContext(other.profile or self.profile, self.prompts + other.prompts,
                                other.capability or self.capability, other.provider or self.provider)


@dataclass(frozen=True)
class SourceSpan:
    filename: str
    line: int
    column: int
    end_line: int
    end_column: int


@dataclass
class AgentRequest:
    statement: str
    frame_code: str
    related_objects: dict[str, Any]
    frame_objects: dict[str, Any]
    span: SourceSpan
    profile: ProfileConfig
    prompts: tuple[str, ...]
    capability: str | None = field(default=None, kw_only=True)
    provider: str | None = field(default=None, kw_only=True)
    output_type: str | None = field(default=None, kw_only=True)
    requires_result: bool = field(default=False, kw_only=True)


@dataclass
class RecoveryRequest(AgentRequest):
    exception: Exception
    traceback: Any
    origin: SourceSpan
    attempt: int
    replacement_target: str | None = None
    retry_allowed: bool = field(default=True, kw_only=True)


@dataclass
class RecoveryDecision:
    action: Literal["complete", "retry", "reraise"]
    value: Any = None
    has_value: bool = False
    explanation: str = ""


class Agent(Protocol):
    def execute(self, request: AgentRequest, runtime: Any) -> Any: ...
    def recover(self, request: RecoveryRequest, runtime: Any) -> RecoveryDecision: ...


class Provider(Protocol):
    def complete(self, messages: list[dict], tools: list[dict]) -> dict: ...
