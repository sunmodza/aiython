"""Create a project configuration without exposing credentials in TOML."""
from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit

from .config import discover, resolve
from .models import ConfigError


PROVIDERS = {
    "openrouter": ("openrouter", None, "OPENROUTER_API_KEY"),
    "openai": ("openai", None, "OPENAI_API_KEY"),
    "gemini": ("gemini", None, "GEMINI_API_KEY"),
    "custom": ("openai", None, None),
}


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(prog="aithon setup", description="Create a project aithon.toml")
    command.add_argument("--provider", choices=PROVIDERS, help="Model service (default: openrouter)")
    command.add_argument("--model", help="Exact model ID supported by the selected service")
    command.add_argument("--base-url", help="API base URL; required for a custom service")
    command.add_argument("--api-key-env", help="Environment variable that contains the API key")
    command.add_argument("--path", type=Path, default=Path("aithon.toml"), help="Project aithon.toml path")
    command.add_argument("--non-interactive", action="store_true", help="Use flags only; never prompt")
    return command


def _ask(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{label}{suffix}: ").strip()
    except EOFError:
        raise ConfigError(f"Missing {label.lower()}; pass it as a setup option") from None
    return answer or default or ""


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


def _ignore_credentials(root: Path) -> None:
    ignore = root / ".gitignore"
    if ignore.is_symlink():
        raise ConfigError("Cannot save an API key through a symlinked .gitignore")
    try:
        source = ignore.read_text(encoding="utf-8") if ignore.exists() else ""
        if ".aithon/" in {line.strip().removeprefix("/") for line in source.splitlines()}:
            return
        prefix = "\n" if source and not source.endswith("\n") else ""
        with ignore.open("a", encoding="utf-8") as file:
            file.write(prefix + ".aithon/\n")
    except OSError as exc:
        raise ConfigError(f"Cannot update .gitignore ({type(exc).__name__})") from None


def _save_key(root: Path, name: str, key: str) -> Path:
    folder = root / ".aithon"
    if folder.is_symlink():
        raise ConfigError("Cannot save an API key through a symlinked .aithon directory")
    if folder.exists() and not folder.is_dir():
        raise ConfigError("Cannot save an API key: .aithon is not a directory")
    try:
        folder.mkdir(mode=0o700, exist_ok=True)
        path = folder / "credentials.env"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(f"{name}={json.dumps(key, ensure_ascii=False)}\n")
        return path
    except FileExistsError:
        raise ConfigError(".aithon/credentials.env already exists; it was not changed") from None
    except OSError as exc:
        raise ConfigError(f"Cannot save API key ({type(exc).__name__})") from None


def setup(argv: list[str]) -> Path:
    args = parser().parse_args(argv)
    interactive = not args.non_interactive and sys.stdin.isatty()
    explicit_path = any(item == "--path" or item.startswith("--path=") for item in argv)
    path = args.path.absolute()
    if path.name != "aithon.toml" or not path.parent.is_dir():
        raise ConfigError("Choose an existing project directory and a file named aithon.toml")
    if path.exists() or path.is_symlink():
        parent = discover(path.parent.parent)
        inherited = f" It shadows {parent}; remove this file to use the parent configuration." if parent else ""
        raise ConfigError(f"{path} already exists; setup will not overwrite it.{inherited}")
    parent = discover(path.parent.parent)
    if parent and not explicit_path:
        if args.provider or args.model or args.base_url or args.api_key_env:
            raise ConfigError(f"Project already uses {parent}; edit that config or pass --path to create a separate one")
        print(f"Using existing project configuration {parent}")
        return parent

    selected = args.provider or (_ask("Provider (openrouter/openai/gemini/custom)", "openrouter")
                                 if interactive else "openrouter")
    if selected not in PROVIDERS:
        raise ConfigError("Provider must be openrouter, openai, gemini or custom")
    prefix, default_url, default_env = PROVIDERS[selected]
    model = (args.model or (_ask("Model ID") if interactive else "")).strip()
    if not model or any(char.isspace() or ord(char) < 32 for char in model):
        raise ConfigError("Model ID is required; use aithon setup --model MODEL_ID")
    base_url = args.base_url or default_url or (_ask("API base URL") if interactive and selected == "custom" else "")
    if selected == "custom" and not base_url:
        raise ConfigError("Custom provider requires --base-url")
    if base_url and not _valid_url(base_url):
        raise ConfigError("API base URL must be HTTP(S) without credentials, query or fragment")
    key_env = args.api_key_env
    if key_env is None:
        key_env = (_ask("API key environment variable (blank for none)") if interactive and selected == "custom"
                   else default_env)
    if selected != "custom" and not key_env:
        raise ConfigError(f"{selected} requires an API key environment variable")
    if key_env and not _valid_name(key_env):
        raise ConfigError("API key environment variable must be a valid name")

    key = ""
    if interactive and key_env and not os.environ.get(key_env):
        try:
            key = getpass.getpass("API key (blank to set the environment variable later): ").strip()
        except EOFError:
            key = ""
    if key:
        if (path.parent / ".aithon" / "credentials.env").exists():
            raise ConfigError(".aithon/credentials.env already exists; it was not changed")
        _ignore_credentials(path.parent)
        key_path = _save_key(path.parent, key_env, key)
    else:
        key_path = None

    model_id = model if model.startswith(prefix + "/") else prefix + "/" + model
    lines = ['version = 3', f'model = {json.dumps(model_id, ensure_ascii=False)}']
    if key_path:
        lines.append('env_file = ".aithon/credentials.env"')
    if base_url:
        lines.append(f'api_base = {json.dumps(base_url, ensure_ascii=False)}')
    if key_env:
        lines.append(f'api_key_env = {json.dumps(key_env)}')
    lines += ['', '# Optional capability routes:', '# [capabilities]',
              '# embedding = "openai/text-embedding-3-large"',
              '# vision = "gemini/gemini-2.5-flash"', '']
    created = False
    try:
        with path.open("x", encoding="utf-8") as file:
            created = True
            file.write("\n".join(lines))
        resolve(path.parent / "main.py", config_path=str(path))
    except FileExistsError:
        if key_path:
            key_path.unlink()
        raise ConfigError(f"{path} already exists; setup will not overwrite it") from None
    except Exception:
        if created:
            path.unlink()
        if key_path:
            key_path.unlink()
        raise

    print(f"Created {path}")
    if key_path:
        print("Saved API key in .aithon/credentials.env (mode 0600); added .aithon/ to .gitignore.")
    elif key_env and not os.environ.get(key_env):
        print(f"Set {key_env} in your environment before running an AI statement.")
    print("Run: aithon --stats path/to/your_script.py")
    return path
