"""Create or update a project configuration without exposing credentials in TOML."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import getpass
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import tomllib
from urllib.parse import urlsplit

import httpx
import tomlkit
from pydantic import ValidationError

from .config import CAPABILITIES, ProjectSettings, discover, read_env, resolve
from .models import ConfigError


PROVIDERS = {
    "openrouter": ("openrouter", "OPENROUTER_API_KEY"),
    "openai": ("openai", "OPENAI_API_KEY"),
    "gemini": ("gemini", "GEMINI_API_KEY"),
    "custom": ("openai", None),
    "other": ("", None),
}
DEFAULT_ENV_FILE = ".aiython/credentials.env"
SETUP_CAPABILITIES = tuple(sorted(CAPABILITIES - {"reasoning"}))


@dataclass(frozen=True)
class ModelChoice:
    id: str
    name: str = ""


# OpenRouter publishes these input/output modalities for each model. A missing
# field means the catalog cannot establish compatibility, so keep manual entry.
OPENROUTER_MODALITIES = {
    ("reasoning", None): ("text", "text"),
    ("vision", None): ("image", "text"),
    ("document_understanding", None): ("file", "text"),
    ("embedding", None): ("text", "embeddings"),
    ("reranking", None): ("text", "rerank"),
    ("speech_to_text", None): ("audio", "transcription"),
    ("text_to_speech", None): ("text", "speech"),
    ("image_generation", None): ("text", "image"),
    ("image_editing", None): ("image", "image"),
    ("video", "understand"): ("video", "text"),
    ("video", "generate"): ("text", "video"),
}


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(prog="aiython setup", description="Create or update a project aiython.toml")
    command.add_argument("--provider", choices=PROVIDERS, help="Model service")
    command.add_argument("--model", help="Model ID; other settings are retained")
    command.add_argument("--capability", choices=SETUP_CAPABILITIES, help="Configure a capability route")
    command.add_argument("--profile", help="Profile whose capability route is configured (default: default)")
    command.add_argument("--understand-model", help="Video understanding model ID")
    command.add_argument("--generate-model", help="Video generation model ID")
    command.add_argument("--base-url", help="API base URL for a custom service")
    command.add_argument("--api-key-env", help="Environment variable that contains the API key")
    command.add_argument("--set-key", action="store_true", help="Securely prompt to save or replace the API key")
    command.add_argument("--check", action="store_true", help="Test the reasoning model with one billable tool call")
    command.add_argument("--path", type=Path, help="Exact project aiython.toml path")
    command.add_argument("--non-interactive", action="store_true", help="Use flags only; never prompt")
    return command


def _ask(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{label}{suffix}: ").strip()
    except EOFError:
        raise ConfigError(f"Missing {label.lower()}; pass it as a setup option") from None
    return answer or default or ""


def _choice(message: str, options: list[tuple[str, str]], default: str | None = None) -> str:
    from prompt_toolkit.shortcuts import choice
    return choice(message=message, options=options, default=default)


def _valid_name(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name))


def _valid_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        return (parsed.scheme in ("http", "https") and bool(parsed.hostname)
                and not any(char.isspace() for char in url)
                and not any((parsed.username, parsed.password, parsed.query, parsed.fragment)))
    except ValueError:
        return False


def _path(args) -> Path:
    path = args.path.absolute() if args.path is not None else (discover(Path.cwd()) or Path.cwd() / "aiython.toml")
    if path.name != "aiython.toml" or not path.parent.is_dir():
        raise ConfigError("Choose an existing project directory and a file named aiython.toml")
    if path.is_symlink():
        raise ConfigError("Cannot edit a symlinked aiython.toml")
    return path


def _model(document) -> str:
    entry = document.get("model")
    if isinstance(entry, str):
        return str(entry)
    if entry is not None and isinstance(entry.get("model"), str):
        return str(entry["model"])
    return ""


def _route_value(document, name: str) -> str:
    entry = document.get("model")
    value = document.get(name)
    if value is None and entry is not None and not isinstance(entry, str):
        value = entry.get(name)
    return str(value) if value is not None else ""


def _provider(model: str, base_url: str) -> str:
    if base_url:
        return "custom"
    prefix = model.split("/", 1)[0]
    return prefix if prefix in ("openrouter", "openai", "gemini") else "other"


def _model_id(provider: str, value: str) -> str:
    value = value.strip()
    if not value or any(char.isspace() or ord(char) < 32 for char in value):
        raise ConfigError("Model ID is required and cannot contain whitespace")
    prefix = PROVIDERS[provider][0]
    return prefix + "/" + value if prefix and not value.startswith(prefix + "/") else value


def _catalog_matches(provider: str, item: dict, capability: str, mode: str | None) -> bool:
    if provider == "openrouter":
        wanted = OPENROUTER_MODALITIES.get((capability, mode))
        if wanted is None:
            return True
        architecture = item.get("architecture")
        if not isinstance(architecture, dict):
            return True  # Older or incomplete catalogs cannot prove incompatibility.
        inputs = architecture.get("input_modalities")
        outputs = architecture.get("output_modalities")
        if isinstance(inputs, list) and wanted[0] not in inputs:
            return False
        if isinstance(outputs, list) and wanted[1] not in outputs:
            return False
        if capability in ("image_generation", "image_editing") and isinstance(outputs, list) and "text" not in outputs:
            # The pinned LiteLLM OpenRouter image adapter uses chat completions.
            # Image-only models require OpenRouter's dedicated image endpoint.
            return False
        parameters = item.get("supported_parameters")
        if capability == "reasoning" and isinstance(parameters, list) and "tools" not in parameters:
            return False
    elif provider == "gemini":
        methods = item.get("supportedGenerationMethods")
        if isinstance(methods, list):
            method = "embedContent" if capability == "embedding" else "generateContent" if capability == "reasoning" else None
            if method and method not in methods:
                return False
    return True


def _catalog(provider: str, base_url: str, key: str | None, *,
             capability: str = "reasoning", mode: str | None = None) -> list[ModelChoice]:
    """List plausible models for this route; failure never prevents manual entry."""
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    if provider == "openrouter":
        if capability == "video" and mode == "generate":
            url = "https://openrouter.ai/api/v1/videos/models"
            params = None
        else:
            url = "https://openrouter.ai/api/v1/models"
            params = {"output_modalities": "all"}
    elif provider == "openai" and key:
        url = "https://api.openai.com/v1/models"
        params = None
    elif provider == "gemini" and key:
        url = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000"
        headers = {"x-goog-api-key": key}
        params = None
    elif provider == "custom" and base_url:
        url = base_url.rstrip("/") + "/models"
        params = None
    else:
        return []
    try:
        response = httpx.get(url, headers=headers, params=params, timeout=5, follow_redirects=False)
        response.raise_for_status()
        payload = response.json()
        entries = payload.get("models" if provider == "gemini" else "data", [])
        choices = {}
        for item in entries:
            if not isinstance(item, dict) or not isinstance(item.get("id") or item.get("name"), str):
                continue
            if not _catalog_matches(provider, item, capability, mode):
                continue
            model = item.get("id") or item["name"].removeprefix("models/")
            model_id = _model_id(provider, model)
            display_name = item.get("displayName") if provider == "gemini" else item.get("name")
            choices[model_id] = ModelChoice(model_id, display_name if isinstance(display_name, str) else "")
        return sorted(choices.values(), key=lambda choice: choice.id)
    except (httpx.HTTPError, ValueError, TypeError, KeyError, AttributeError):
        return []


def _search_words(value: str) -> list[str]:
    return [word for word in re.split(r"[\s/._:-]+", value.casefold()) if word]


def _model_matches(choice: ModelChoice, query: str) -> bool:
    words = _search_words(choice.id + " " + choice.name)
    return all(any(word.startswith(term) for word in words) for term in _search_words(query))


def _choose_model(provider: str, current: str, base_url: str, key: str | None, *,
                  capability: str = "reasoning", mode: str | None = None) -> str:
    choices = _catalog(provider, base_url, key, capability=capability, mode=mode)
    if choices:
        from prompt_toolkit import prompt
        from prompt_toolkit.completion import Completer, Completion

        class ModelCompleter(Completer):
            def get_completions(self, document, complete_event):
                query = document.text_before_cursor.strip()
                for choice in choices:
                    if _model_matches(choice, query):
                        yield Completion(choice.id, start_position=-len(document.text_before_cursor),
                                         display_meta=choice.name)

        label = capability.replace("_", " ") + (f" ({mode})" if mode else "")
        print(f"Search {len(choices)} {label} models. Match words in the ID or name; Tab selects a suggestion. Exact IDs also work.")
        try:
            answer = prompt("Model ID: ", completer=ModelCompleter(),
                            complete_while_typing=True, placeholder=current or None).strip()
        except EOFError:
            raise ConfigError("Model ID is required; use --model MODEL_ID") from None
        return answer or current
    print("Model catalog unavailable; enter a model ID manually.")
    return _ask("Model ID", current or None)


def _current_key(path: Path, document, key_env: str) -> str | None:
    if not key_env:
        return None
    if key_env in os.environ:
        return os.environ[key_env] or None
    env_file = document.get("env_file")
    if env_file:
        candidate = path.parent / str(env_file)
        if candidate.resolve().is_relative_to(path.parent.resolve()) and candidate.is_file():
            return read_env(candidate).get(key_env)
    return None


def _effective_key(path: Path, document, key_env: str, new_key: str) -> str | None:
    if key_env in os.environ:
        return os.environ[key_env] or None
    return new_key or _current_key(path, document, key_env)


def _ignore_credentials(root: Path, relative_path: str) -> None:
    ignore = root / ".gitignore"
    if ignore.is_symlink():
        raise ConfigError("Cannot save an API key through a symlinked .gitignore")
    entry = ".aiython/" if relative_path.startswith(".aiython/") else "/" + relative_path
    try:
        source = ignore.read_text(encoding="utf-8") if ignore.exists() else ""
        if entry in {line.strip() for line in source.splitlines()}:
            return
        prefix = "\n" if source and not source.endswith("\n") else ""
        with ignore.open("a", encoding="utf-8") as file:
            file.write(prefix + entry + "\n")
    except OSError as exc:
        raise ConfigError(f"Cannot update .gitignore ({type(exc).__name__})") from None


def _write_keys(root: Path, relative_path: str, updates: dict[str, str]) -> None:
    target = root / relative_path
    if not target.resolve().is_relative_to(root.resolve()) or target.is_symlink():
        raise ConfigError("Cannot save an API key outside the project or through a symlink")
    parent = target.parent
    while parent != root.parent:
        if parent.is_symlink():
            raise ConfigError("Cannot save an API key through a symlinked directory")
        parent = parent.parent
    if target.exists() and not target.is_file():
        raise ConfigError("Credential path is not a file")
    values = read_env(target) if target.exists() else {}
    values.update(updates)
    try:
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _ignore_credentials(root, relative_path)
        descriptor, temp_name = tempfile.mkstemp(prefix=".credentials-", dir=target.parent)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                for item, value in values.items():
                    file.write(f"{item}={json.dumps(value, ensure_ascii=False)}\n")
            os.replace(temp_name, target)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
    except OSError as exc:
        raise ConfigError(f"Cannot save API key ({type(exc).__name__})") from None


def _write_key(root: Path, relative_path: str, name: str, key: str) -> None:
    _write_keys(root, relative_path, {name: key})


def _validated_content(document) -> str:
    content = tomlkit.dumps(document)
    try:
        ProjectSettings.model_validate(tomllib.loads(content))
    except ValidationError as exc:
        detail = exc.errors(include_url=False)[0]
        raise ConfigError(f"Invalid setup configuration at {'.'.join(map(str, detail['loc']))}: {detail['msg']}") from None
    return content


def _write_config(path: Path, content: str, *, existing: bool) -> None:
    try:
        descriptor, temp_name = tempfile.mkstemp(prefix=".aiython-config-", dir=path.parent)
        try:
            os.fchmod(descriptor, (path.stat().st_mode & 0o777) if existing else 0o644)
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                file.write(content)
            if path.is_symlink():
                raise ConfigError(f"{path} changed during setup; it was not edited")
            if existing:
                os.replace(temp_name, path)
            else:
                os.link(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
    except OSError as exc:
        raise ConfigError(f"Cannot save {path} ({type(exc).__name__})") from None


def _check_model(model: str, base_url: str, key_env: str, key: str | None) -> None:
    if key_env and not key:
        raise ConfigError(f"Missing credential {key_env}; set it or use aiython setup --set-key")
    from .providers import failure_reason, sdk
    print(f"Testing {model} with one tool call (may incur a small provider charge)...", flush=True)
    tool = {"type": "function", "function": {"name": "aiython_probe",
            "description": "Confirm tool calling is available", "parameters": {"type": "object",
            "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}}}
    try:
        response = sdk().completion(model=model, api_base=base_url or None, api_key=key,
                                    messages=[{"role": "user", "content": "Call aiython_probe with ok=true."}],
                                    tools=[tool], tool_choice="required", timeout=25,
                                    max_retries=0, stream=False)
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        reason = failure_reason(status) if isinstance(status, int) else type(exc).__name__
        raise ConfigError(f"Model test failed: {reason}" + (f" (HTTP {status})" if status else "")) from None
    result = response.model_dump(exclude_none=False) if hasattr(response, "model_dump") else response
    try:
        calls = result["choices"][0]["message"]["tool_calls"]
        if not any(call.get("function", {}).get("name") == "aiython_probe" for call in calls):
            raise ValueError
    except (KeyError, IndexError, TypeError, ValueError):
        raise ConfigError("Model responded, but did not call a tool; choose a tool-capable model") from None
    print("Connection and tool calling verified.")


def _route_fields(value) -> tuple[str, str, str]:
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, str):
        return value, "", ""
    if isinstance(value, dict):
        return value.get("model", ""), value.get("api_base", ""), value.get("api_key_env", "")
    return "", "", ""


def _capability_entry(document, profile: str, capability: str):
    raw = tomllib.loads(tomlkit.dumps(document))
    base = raw.get("capabilities", {})
    if profile == "default":
        return base.get(capability)
    override = raw.get("profiles", {}).get(profile, {}).get("capabilities", {})
    return override.get(capability, base.get(capability))


def _capability_section(document, profile: str):
    if profile == "default":
        parent = document
    else:
        parent = document["profiles"][profile]
    if "capabilities" not in parent:
        parent["capabilities"] = tomlkit.table()
    return parent["capabilities"]


def _strip_missing_hint(content: str, profile: str, capability: str) -> str:
    marker = f"# >>> Aiython missing route: profile={json.dumps(profile)} capability={capability}"
    start = content.find(marker)
    if start < 0:
        return content
    end = content.find("# <<< Aiython missing route", start)
    if end < 0:
        return content
    line_end = content.find("\n", end)
    return content[:start] + content[(line_end + 1) if line_end >= 0 else len(content):]


def _missing_hints(document) -> list[tuple[str, str]]:
    found = []
    marker = "# >>> Aiython missing route: profile="
    for line in tomlkit.dumps(document).splitlines():
        if not line.startswith(marker):
            continue
        profile_text, separator, capability = line[len(marker):].partition(" capability=")
        if not separator or capability not in SETUP_CAPABILITIES:
            continue
        try:
            profile = json.loads(profile_text)
        except ValueError:
            continue
        if isinstance(profile, str) and not _capability_entry(document, profile, capability):
            found.append((profile, capability))
    return found


def _select_capability_route(args, path: Path, document, profile_config, current, label: str,
                             specified_model: str | None, interactive: bool, new_keys: dict[str, str],
                             preferred_provider: str | None = None, *, capability: str,
                             mode: str | None = None):
    current_model, current_base, current_env = _route_fields(current)
    main_provider = _provider(profile_config.model, profile_config.base_url)
    old_provider = _provider(current_model, current_base) if current_model else (preferred_provider or main_provider)
    selected = args.provider or old_provider
    if specified_model and not args.provider and not current_base:
        prefix = specified_model.split("/", 1)[0]
        if prefix in ("openrouter", "openai", "gemini"):
            selected = prefix
        elif not current_model and "/" in specified_model:
            selected = "other"
    keep_model = bool(current_model and specified_model is None and
                      (args.set_key or args.api_key_env is not None or args.base_url is not None))
    if interactive and specified_model is None and not args.provider and not keep_model:
        selected = _choice(f"{label} provider", [(name, name.title()) for name in PROVIDERS],
                           default=selected if selected in PROVIDERS else "gemini")
    changed = selected != old_provider
    base_url = args.base_url if args.base_url is not None else (current_base if not changed else "")
    if selected == "custom" and not base_url:
        base_url = profile_config.base_url if main_provider == "custom" else ""
    if selected == "custom" and not base_url and interactive:
        base_url = _ask(f"{label} API base URL")
    if selected == "custom" and not base_url:
        raise ConfigError(f"{label} custom provider requires --base-url")
    if base_url and not _valid_url(base_url):
        raise ConfigError("API base URL must be HTTP(S) without credentials, query or fragment")
    key_env = args.api_key_env if args.api_key_env is not None else (
        current_env if current_env and not changed else PROVIDERS[selected][1])
    competing_names = {name for provider, (_, name) in PROVIDERS.items()
                       if provider != selected and name}
    if (args.api_key_env is None and not current_env and selected == main_provider and not changed
            and profile_config.api_key_env and profile_config.api_key_env not in competing_names):
        key_env = profile_config.api_key_env
    if key_env is None and interactive and selected in ("custom", "other"):
        key_env = _ask(f"{label} API key environment variable (blank for none)")
    key_env = key_env or ""
    if key_env and not _valid_name(key_env):
        raise ConfigError("API key environment variable must be a valid name")
    if selected in ("openrouter", "openai", "gemini") and not key_env:
        raise ConfigError(f"{selected} requires an API key environment variable")
    if args.set_key and not key_env:
        raise ConfigError("--set-key requires an API key environment variable")
    if interactive and key_env and key_env not in new_keys and (args.set_key or not _current_key(path, document, key_env)):
        entered = getpass.getpass(f"{label} API key (blank to keep current or set it later): ").strip()
        if entered:
            new_keys[key_env] = entered
    if specified_model is not None:
        model = _model_id(selected, specified_model)
    elif keep_model:
        model = current_model
    elif interactive:
        model = _model_id(selected, _choose_model(selected, current_model if not changed else "",
                                                   base_url, _effective_key(path, document, key_env,
                                                                            new_keys.get(key_env, "")),
                                                   capability=capability, mode=mode))
    elif current_model and not changed:
        model = current_model
    else:
        raise ConfigError(f"{label} model is required; pass its model option")
    route = tomlkit.inline_table()
    route["model"] = model
    if base_url:
        route["api_base"] = base_url
    if key_env:
        route["api_key_env"] = key_env
    if isinstance(current, dict) and model == current_model and current.get("revision"):
        route["revision"] = current["revision"]
    return route


def _configure_capability(args, path: Path, document, capability: str, profile: str,
                          interactive: bool) -> Path:
    if not path.exists():
        raise ConfigError("Create the main project configuration with 'aiython setup' before adding capabilities")
    if args.check:
        raise ConfigError("--check tests the reasoning model only; capability routes are not tested automatically")
    if capability != "video" and (args.understand_model or args.generate_model):
        raise ConfigError("--understand-model and --generate-model require --capability video")
    if capability == "video" and args.model:
        raise ConfigError("Video requires --understand-model and --generate-model, not --model")
    config = resolve(path.parent / "main.py", config_path=str(path))
    if profile not in config.profiles:
        raise ConfigError(f"Profile {profile!r} is not configured")
    current = _capability_entry(document, profile, capability)
    if isinstance(current, list) and not (args.model or args.understand_model or args.generate_model):
        raise ConfigError("This capability has fallback routes; pass a model to replace them or edit TOML directly")
    new_keys: dict[str, str] = {}
    if capability == "video":
        previous = current if isinstance(current, dict) and "model" not in current else {}
        choices = {}
        preferred_provider = None
        for mode, specified in (("understand", args.understand_model), ("generate", args.generate_model)):
            old = previous.get(mode)
            other_specified = args.generate_model if mode == "understand" else args.understand_model
            if specified is None and other_specified is not None and old is not None:
                choices[mode] = old
                if mode == "understand":
                    old_model, old_base, _ = _route_fields(old)
                    preferred_provider = _provider(old_model, old_base)
                continue
            choices[mode] = _select_capability_route(
                args, path, document, config.profiles[profile], old, f"Video {mode}", specified,
                interactive, new_keys, preferred_provider, capability=capability, mode=mode)
            if mode == "understand":
                preferred_provider = _provider(str(choices[mode]["model"]),
                                               str(choices[mode].get("api_base", "")))
        route = tomlkit.inline_table()
        for mode, choice in choices.items():
            route[mode] = choice
    else:
        route = _select_capability_route(args, path, document, config.profiles[profile], current,
                                         capability.replace("_", " ").title(), args.model, interactive, new_keys,
                                         capability=capability)
    if args.set_key and not new_keys:
        print("API key was not changed.")
        return path
    section = _capability_section(document, profile)
    section[capability] = route
    if new_keys and not document.get("env_file"):
        document["env_file"] = DEFAULT_ENV_FILE
    content = _strip_missing_hint(_validated_content(document), profile, capability)
    if new_keys:
        _write_keys(path.parent, str(document["env_file"]), new_keys)
    _write_config(path, content, existing=True)
    print(f"Updated {path}: {profile}.{capability}")
    for name in new_keys:
        if name in os.environ:
            print(f"{name} in the process environment takes precedence over the saved key.")
    return path


def setup(argv: list[str]) -> Path:
    args = parser().parse_args(argv)
    path = _path(args)
    existing = path.exists()
    interactive = not args.non_interactive and sys.stdin.isatty()
    if args.profile and not args.capability:
        raise ConfigError("--profile requires --capability")
    if (args.understand_model or args.generate_model) and args.capability != "video":
        raise ConfigError("Video model options require --capability video")
    changes = any((args.provider, args.model, args.base_url, args.api_key_env is not None, args.set_key))
    wizard = interactive and not changes and not args.check
    if args.set_key and not interactive:
        raise ConfigError("--set-key needs a terminal; for automation set the API key environment variable")
    if existing:
        resolve(path.parent / "main.py", config_path=str(path))
        try:
            document = tomlkit.parse(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ConfigError(f"Cannot edit {path} ({type(exc).__name__})") from None
    else:
        document = tomlkit.parse('version = 3\nmodel = ""\n\n# Optional capability routes:\n'
                                 '# [capabilities]\n# embedding = "openai/text-embedding-3-large"\n'
                                 '# vision = "gemini/gemini-2.5-flash"\n')
    if args.capability:
        try:
            return _configure_capability(args, path, document, args.capability, args.profile or "default", interactive)
        except (EOFError, KeyboardInterrupt):
            raise ConfigError("Setup cancelled; no changes saved") from None
    old_model = _model(document)
    old_base = _route_value(document, "api_base")
    old_provider = _provider(old_model, old_base)
    old_env = _route_value(document, "api_key_env")
    selected = args.provider or (old_provider if existing else "openrouter")
    if not args.provider and args.base_url:
        selected = "custom"
    elif not args.provider and args.model and not (existing and old_provider == "custom"):
        prefix = args.model.split("/", 1)[0]
        if prefix in ("openrouter", "openai", "gemini"):
            selected = prefix
        elif not existing and "/" in args.model:
            selected = "other"
    action = "configure"
    if existing and wizard:
        configured = bool(_current_key(path, document, old_env))
        source = "environment" if old_env in os.environ else "project file"
        hints = _missing_hints(document)
        print(f"Config: {path}\nProvider: {old_provider}\nModel: {old_model}\n"
              f"API key: {'configured (' + source + ')' if configured else 'missing'}")
        if hints:
            print(f"Suggested next: configure {hints[0][1]} for profile {hints[0][0]}")
        try:
            action = _choice("What would you like to do?", [
                ("model", "Search or change model"), ("provider", "Change provider"),
                ("capability", "Configure capabilities"), ("key", "Update API key"),
                ("check", "Test connection"), ("done", "Done")],
                default="capability" if hints else "model")
        except (EOFError, KeyboardInterrupt):
            raise ConfigError("Setup cancelled; no changes saved") from None
        if action == "done":
            return path
        if action == "check":
            _check_model(old_model, old_base, old_env, _current_key(path, document, old_env))
            return path
        if action == "capability":
            profiles = list(resolve(path.parent / "main.py", config_path=str(path)).profiles)
            hinted_profile = next((name for name, _ in hints if name in profiles), "default")
            profile = (_choice("Profile", [(name, name) for name in profiles], default=hinted_profile)
                       if len(profiles) > 1 else "default")
            while True:
                hinted = [cap for hint_profile, cap in _missing_hints(document) if hint_profile == profile]
                ordered = [*hinted, *(name for name in SETUP_CAPABILITIES if name not in hinted)]
                options = [(name, f"{name.replace('_', ' ').title()} "
                            f"[{'suggested' if name in hinted else 'configured' if _capability_entry(document, profile, name) else 'not configured'}]")
                           for name in ordered]
                capability = _choice("Capability", [*options, ("done", "Done")],
                                     default=hinted[0] if hinted else None)
                if capability == "done":
                    return path
                _configure_capability(args, path, document, capability, profile, interactive=True)
                document = tomlkit.parse(path.read_text(encoding="utf-8"))
        if action == "key":
            args.set_key = True
    elif existing and not changes and not args.check:
        print(f"Using existing project configuration {path}")
        return path
    elif existing and args.check and not changes:
        _check_model(old_model, old_base, old_env, _current_key(path, document, old_env))
        return path

    try:
        if interactive and (not existing or action == "provider") and not args.provider:
            selected = _choice("Provider", [(name, name.title()) for name in PROVIDERS],
                               default=old_provider if existing else "openrouter")
        provider_changed = existing and selected != old_provider
        base_url = args.base_url if args.base_url is not None else (old_base if not provider_changed else "")
        if selected == "custom" and not base_url and interactive:
            base_url = _ask("API base URL")
        if selected == "custom" and not base_url:
            raise ConfigError("Custom provider requires --base-url")
        if base_url and not _valid_url(base_url):
            raise ConfigError("API base URL must be HTTP(S) without credentials, query or fragment")
        key_env = args.api_key_env if args.api_key_env is not None else (
            PROVIDERS[selected][1] if provider_changed or not existing else old_env)
        if key_env is None and interactive and selected in ("custom", "other") and action != "key":
            key_env = _ask("API key environment variable (blank for none)")
        key_env = key_env or ""
        if key_env and not _valid_name(key_env):
            raise ConfigError("API key environment variable must be a valid name")
        if selected in ("openrouter", "openai", "gemini") and not key_env:
            raise ConfigError(f"{selected} requires an API key environment variable")
        new_key = ""
        if interactive and key_env and (args.set_key or not _current_key(path, document, key_env)):
            new_key = getpass.getpass("API key (blank to keep current or set it later): ").strip()
        if args.set_key and not new_key:
            print("API key was not changed.")
            return path
        needs_model = not existing or provider_changed or action == "model" or args.model is not None
        if needs_model:
            if args.model is not None:
                value = args.model
            elif interactive:
                value = _choose_model(selected, old_model if not provider_changed else "",
                                      base_url, _effective_key(path, document, key_env, new_key))
            else:
                raise ConfigError("Model ID is required; use --model MODEL_ID")
            model = _model_id(selected, value)
        else:
            model = old_model
        entry = document.get("model")
        if isinstance(entry, str):
            document["model"] = model
        else:
            entry["model"] = model
        if provider_changed and entry is not None and not isinstance(entry, str):
            entry.pop("api_base", None)
            entry.pop("api_key_env", None)
        if base_url:
            document["api_base"] = base_url
        else:
            document.pop("api_base", None)
        if key_env:
            document["api_key_env"] = key_env
        else:
            document.pop("api_key_env", None)
        if new_key and not document.get("env_file"):
            document["env_file"] = DEFAULT_ENV_FILE
        key = _effective_key(path, document, key_env, new_key)
        should_check = args.check
        if wizard and (not existing or action in ("model", "provider", "key")):
            if key_env and not key:
                print("Skipping the model test until an API key is configured.")
            else:
                should_check = _choice("Test the selected model now? (one small billable request)",
                                       [("yes", "Yes"), ("no", "Skip")], default="yes") == "yes"
        if should_check:
            try:
                _check_model(model, base_url, key_env, key)
            except ConfigError as exc:
                if not wizard:
                    raise
                print(exc)
                if _choice("Save settings without a successful test?", [("no", "No"), ("yes", "Yes")],
                           default="no") == "no":
                    print("No settings were changed.")
                    return path
        content = _validated_content(document)
        if new_key:
            _write_key(path.parent, str(document["env_file"]), key_env, new_key)
        _write_config(path, content, existing=existing)
    except (EOFError, KeyboardInterrupt):
        raise ConfigError("Setup cancelled; no changes saved") from None
    print(f"{'Updated' if existing else 'Created'} {path}")
    if new_key:
        print(f"Saved API key in {document['env_file']} (mode 0600); added it to .gitignore.")
        if key_env in os.environ:
            print(f"{key_env} in the process environment takes precedence over the saved key.")
    elif key_env and not key:
        print(f"Set {key_env} or run 'aiython setup --set-key' before running an AI statement.")
    print("Run: aiython --stats path/to/your_script.py")
    return path
