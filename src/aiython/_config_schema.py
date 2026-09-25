"""Configuration validation, loaded only when a project config is present."""
from __future__ import annotations

from urllib.parse import urlsplit
from pydantic import BaseModel, ConfigDict, Field, field_validator
from .config import ENV_NAME, PERMISSIONS


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ModelRoute(StrictModel):
    model: str
    api_base: str | None = None
    api_key_env: str | None = None
    revision: str = ""

    @field_validator("model")
    @classmethod
    def model_id(cls, value: str) -> str:
        if not value.strip() or any(ch.isspace() or ord(ch) < 32 for ch in value):
            raise ValueError("model must be a nonempty LiteLLM model ID without whitespace")
        return value

    @field_validator("api_base")
    @classmethod
    def endpoint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlsplit(value)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname or
                parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("api_base must be an HTTP(S) URL without credentials, query or fragment")
        return value

    @field_validator("api_key_env")
    @classmethod
    def key_name(cls, value: str | None) -> str | None:
        if value is not None and not ENV_NAME.fullmatch(value):
            raise ValueError("api_key_env must name an environment variable")
        return value


class ProfileSettings(StrictModel):
    model: str | dict | None = None
    api_base: str | None = None
    api_key_env: str | None = None
    prompt: str | None = None
    timeout: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    max_rounds: int | None = Field(default=None, ge=1)
    permissions: list[str] | None = None
    capabilities: dict[str, object] = Field(default_factory=dict)


class ProjectSettings(StrictModel):
    version: int
    model: str | dict
    api_base: str | None = None
    api_key_env: str | None = None
    prompt: str = ""
    timeout: float = Field(default=120, gt=0, allow_inf_nan=False)
    max_rounds: int = Field(default=16, ge=1)
    env_file: str | None = None
    permissions: list[str] = Field(default_factory=lambda: sorted(PERMISSIONS))
    capabilities: dict[str, object] = Field(default_factory=dict)
    profiles: dict[str, ProfileSettings] = Field(default_factory=dict)

    @field_validator("version")
    @classmethod
    def current_version(cls, value: int) -> int:
        if value != 3:
            raise ValueError("version = 3 is required")
        return value

