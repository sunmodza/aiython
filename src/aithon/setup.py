"""Create or update a project configuration without exposing credentials in TOML."""
from __future__ import annotations

import argparse
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

from .config import ProjectSettings, discover, read_env, resolve
from .models import ConfigError


PROVIDERS = {
    "openrouter": ("openrouter", "OPENROUTER_API_KEY"),
    "openai": ("openai", "OPENAI_API_KEY"),
    "gemini": ("gemini", "GEMINI_API_KEY"),
    "custom": ("openai", None),
    "other": ("", None),
}
DEFAULT_ENV_FILE = ".aithon/credentials.env"


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(prog="aithon setup", description="Create or update a project aithon.toml")
    command.add_argument("--provider", choices=PROVIDERS, help="Model service")
    command.add_argument("--model", help="Model ID; other settings are retained")
    command.add_argument("--base-url", help="API base URL for a custom service")
    command.add_argument("--api-key-env", help="Environment variable that contains the API key")
    command.add_argument("--set-key", action="store_true", help="Securely prompt to save or replace the API key")
    command.add_argument("--check", action="store_true", help="Test the model with one billable tool call")
    command.add_argument("--path", type=Path, help="Exact project aithon.toml path")
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
    path = args.path.absolute() if args.path is not None else (discover(Path.cwd()) or Path.cwd() / "aithon.toml")
    if path.name != "aithon.toml" or not path.parent.is_dir():
        raise ConfigError("Choose an existing project directory and a file named aithon.toml")
    if path.is_symlink():
        raise ConfigError("Cannot edit a symlinked aithon.toml")
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


def _catalog(provider: str, base_url: str, key: str | None) -> list[str]:
    """List models for setup only; failure never prevents manual model entry."""
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    if provider == "openrouter":
        url = "https://openrouter.ai/api/v1/models"
    elif provider == "openai" and key:
        url = "https://api.openai.com/v1/models"
    elif provider == "gemini" and key:
        url = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000"
        headers = {"x-goog-api-key": key}
    elif provider == "custom" and base_url:
        url = base_url.rstrip("/") + "/models"
    else:
        return []
    try:
        response = httpx.get(url, headers=headers, timeout=5, follow_redirects=False)
        response.raise_for_status()
        payload = response.json()
        entries = payload.get("models" if provider == "gemini" else "data", [])
        names = []
        for item in entries:
            if not isinstance(item, dict) or not isinstance(item.get("id") or item.get("name"), str):
                continue
            if provider == "gemini" and "generateContent" not in item.get("supportedGenerationMethods", []):
                continue
            name = item.get("id") or item["name"].removeprefix("models/")
            names.append(_model_id(provider, name))
        return sorted(set(names))
    except (httpx.HTTPError, ValueError, TypeError, KeyError, AttributeError):
        return []


def _choose_model(provider: str, current: str, base_url: str, key: str | None) -> str:
    names = _catalog(provider, base_url, key)
    if names:
        from prompt_toolkit import prompt
        from prompt_toolkit.completion import FuzzyWordCompleter
        print(f"Search {len(names)} available models. Type to filter; Tab selects a suggestion. Exact IDs also work.")
        try:
            answer = prompt("Model ID: ", completer=FuzzyWordCompleter(names, WORD=True),
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
    entry = ".aithon/" if relative_path.startswith(".aithon/") else "/" + relative_path
    try:
        source = ignore.read_text(encoding="utf-8") if ignore.exists() else ""
        if entry in {line.strip() for line in source.splitlines()}:
            return
        prefix = "\n" if source and not source.endswith("\n") else ""
        with ignore.open("a", encoding="utf-8") as file:
            file.write(prefix + entry + "\n")
    except OSError as exc:
        raise ConfigError(f"Cannot update .gitignore ({type(exc).__name__})") from None


def _write_key(root: Path, relative_path: str, name: str, key: str) -> None:
    target = root / relative_path
    if not target.resolve().is_relative_to(root.resolve()) or target.is_symlink():
        raise ConfigError("Cannot save an API key outside the project or through a symlink")
    for parent in (target.parent, *target.parent.parents):
        if parent == root.parent:
            break
        if parent.is_symlink():
            raise ConfigError("Cannot save an API key through a symlinked directory")
    if target.exists() and not target.is_file():
        raise ConfigError("Credential path is not a file")
    values = read_env(target) if target.exists() else {}
    values[name] = key
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
        descriptor, temp_name = tempfile.mkstemp(prefix=".aithon-config-", dir=path.parent)
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
        raise ConfigError(f"Missing credential {key_env}; set it or use aithon setup --set-key")
    from .providers import failure_reason, sdk
    print(f"Testing {model} with one tool call (may incur a small provider charge)...", flush=True)
    tool = {"type": "function", "function": {"name": "aithon_probe",
            "description": "Confirm tool calling is available", "parameters": {"type": "object",
            "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}}}
    try:
        response = sdk().completion(model=model, api_base=base_url or None, api_key=key,
                                    messages=[{"role": "user", "content": "Call aithon_probe with ok=true."}],
                                    tools=[tool], tool_choice="required", timeout=25,
                                    max_retries=0, stream=False)
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        reason = failure_reason(status) if isinstance(status, int) else type(exc).__name__
        raise ConfigError(f"Model test failed: {reason}" + (f" (HTTP {status})" if status else "")) from None
    result = response.model_dump(exclude_none=False) if hasattr(response, "model_dump") else response
    try:
        calls = result["choices"][0]["message"]["tool_calls"]
        if not any(call.get("function", {}).get("name") == "aithon_probe" for call in calls):
            raise ValueError
    except (KeyError, IndexError, TypeError, ValueError):
        raise ConfigError("Model responded, but did not call a tool; choose a tool-capable model") from None
    print("Connection and tool calling verified.")


def setup(argv: list[str]) -> Path:
    args = parser().parse_args(argv)
    path = _path(args)
    existing = path.exists()
    interactive = not args.non_interactive and sys.stdin.isatty()
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
        print(f"Config: {path}\nProvider: {old_provider}\nModel: {old_model}\n"
              f"API key: {'configured (' + source + ')' if configured else 'missing'}")
        try:
            action = _choice("What would you like to do?", [
                ("model", "Search or change model"), ("provider", "Change provider"),
                ("key", "Update API key"), ("check", "Test connection"), ("done", "Done")],
                default="model")
        except (EOFError, KeyboardInterrupt):
            raise ConfigError("Setup cancelled; no changes saved") from None
        if action == "done":
            return path
        if action == "check":
            _check_model(old_model, old_base, old_env, _current_key(path, document, old_env))
            return path
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
        if not model:
            raise ConfigError("Model ID is required; use --model MODEL_ID")
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
        print(f"Set {key_env} or run 'aithon setup --set-key' before running an AI statement.")
    print("Run: aithon --stats path/to/your_script.py")
    return path
