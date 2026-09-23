"""Project-scoped Aithon configuration and LiteLLM route normalization."""
from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .models import ConfigError, ProfileConfig, ResolvedConfig


ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
CAPABILITIES = frozenset((
    "reasoning", "vision", "document_understanding", "embedding", "reranking",
    "speech_to_text", "text_to_speech", "image_generation", "image_editing", "video",
))
PERMISSIONS = frozenset((
    "read_asset", "read_code", "network", "generate_file", "write_filesystem", "execute_code",
))


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


def route(value: object) -> ModelRoute:
    if isinstance(value, str):
        return ModelRoute(model=value)
    return ModelRoute.model_validate(value)


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


def discover(start: Path) -> Path | None:
    for directory in (start, *start.parents):
        candidate = directory / "aithon.toml"
        if candidate.is_file():
            return candidate
        if (directory / ".git").exists():
            break
    return None


def read_env(path: Path) -> dict[str, str]:
    """Parse a literal dotenv file without interpolation or process env changes."""
    try:
        values = dotenv_values(path, interpolate=False)
    except OSError as exc:
        raise ConfigError(f"Cannot read env_file ({type(exc).__name__})") from None
    if not path.is_file() or path.is_symlink() or any(not ENV_NAME.fullmatch(k) or v is None for k, v in values.items()):
        raise ConfigError("env_file requires literal KEY=VALUE entries")
    return {key: value for key, value in values.items() if value is not None}


def _profile(name: str, base: ProjectSettings, override: ProfileSettings | None,
             providers: dict[str, dict]) -> ProfileConfig:
    values = override or ProfileSettings()
    selected = route(values.model if values.model is not None else base.model)
    base_url = values.api_base if values.api_base is not None else base.api_base
    key_env = values.api_key_env if values.api_key_env is not None else base.api_key_env
    if base_url is not None or key_env is not None:
        selected = ModelRoute(model=selected.model, api_base=base_url or selected.api_base,
                              api_key_env=key_env or selected.api_key_env, revision=selected.revision)
    capabilities = dict(base.capabilities)
    if values.model is not None:
        capabilities["reasoning"] = selected.model_dump(exclude_none=True)
    capabilities.update(values.capabilities)
    capabilities.setdefault("reasoning", selected.model_dump(exclude_none=True))
    routes: dict[str, list[dict]] = {}
    for capability, entry in capabilities.items():
        normalized = capability.replace("-", "_")
        if normalized not in CAPABILITIES or normalized in routes:
            raise ConfigError(f"Unknown or repeated capability route: {capability}")
        if normalized == "video" and isinstance(entry, dict) and "model" not in entry:
            if set(entry) != {"understand", "generate"}:
                raise ConfigError("video route requires understand and generate models")
            choices = [(mode, item) for mode, item in entry.items()]
        else:
            choices = [(None, item) for item in (entry if isinstance(entry, list) else [entry])]
        if not choices:
            raise ConfigError(f"Empty route for {normalized}")
        routes[normalized] = []
        for index, (mode, item) in enumerate(choices):
            spec = route(item)
            alias = f"{name}:{normalized}:{index}"
            providers[alias] = {"model": spec.model, "api_base": spec.api_base,
                                "api_key_env": spec.api_key_env}
            current = {"provider": alias, "model": spec.model, "revision": spec.revision}
            if mode:
                current["modes"] = [mode]
            routes[normalized].append(current)
    permissions = values.permissions if values.permissions is not None else base.permissions
    if set(permissions) - PERMISSIONS or len(set(permissions)) != len(permissions):
        raise ConfigError("Invalid permissions")
    return ProfileConfig(name=name, provider="litellm", model=selected.model,
                         base_url=selected.api_base or "", api_key_env=selected.api_key_env,
                         prompt=values.prompt if values.prompt is not None else base.prompt,
                         timeout=values.timeout if values.timeout is not None else base.timeout,
                         max_rounds=values.max_rounds if values.max_rounds is not None else base.max_rounds,
                         routes=routes, permissions=tuple(permissions))


def resolve(script: Path, *, config_path: str | None = None,
            profile: str | None = None, force_profile: str | None = None) -> ResolvedConfig:
    script = script.resolve()
    if profile and force_profile:
        raise ConfigError("Use either --profile or --force-profile, not both")
    path = Path(config_path).resolve() if config_path else discover(script.parent)
    root = path.parent if path else script.parent
    if path is None:
        return ResolvedConfig(None, root, cli_profile=profile, force_profile=force_profile)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"Cannot read Aithon config: {path} ({type(exc).__name__})") from None
    if raw.get("version") != 3:
        raise ConfigError("aithon.toml version 1/2 is no longer supported; run 'aithon setup' for a version 3 example and migrate your model/routes")
    try:
        settings = ProjectSettings.model_validate(raw)
        providers: dict[str, dict] = {}
        profiles = {"default": _profile("default", settings, None, providers)}
        for name, override in settings.profiles.items():
            if name == "default" or not name.strip():
                raise ConfigError("Profile names must be nonempty and must not redefine default")
            profiles[name] = _profile(name, settings, override, providers)
    except ValidationError as exc:
        detail = exc.errors(include_url=False)[0]
        raise ConfigError(f"Invalid aithon.toml at {'.'.join(map(str, detail['loc']))}: {detail['msg']}") from None
    secrets = {}
    if settings.env_file:
        target = path.parent / settings.env_file
        if not target.resolve().is_relative_to(path.parent.resolve()):
            raise ConfigError("env_file must stay inside the project")
        secrets = read_env(target)
    return ResolvedConfig(path, root, "default", profiles, secrets, profile, force_profile, providers)


def credential(config: ResolvedConfig, profile: ProfileConfig) -> str | None:
    if not profile.api_key_env:
        return None
    return os.environ.get(profile.api_key_env, config.secrets.get(profile.api_key_env))


def describe(config: ResolvedConfig) -> dict:
    return {"config": str(config.path) if config.path else None,
            "project_root": str(config.project_root),
            "default_profile": config.force_profile or config.cli_profile or config.default_profile,
            "profiles": {name: {"model": profile.model,
                                "credential_configured": bool(credential(config, profile)),
                                "routes": profile.routes, "permissions": profile.permissions}
                         for name, profile in config.profiles.items()}}
