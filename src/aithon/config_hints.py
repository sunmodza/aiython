"""Append inert configuration examples when a capability route is missing."""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import sys
import tomllib

from .models import ProfileConfig, ResolvedConfig


MODELS = {
    "embedding": "openai/YOUR_EMBEDDING_MODEL",
    "reranking": "cohere/YOUR_RERANK_MODEL",
    "document_understanding": "gemini/YOUR_MULTIMODAL_MODEL",
    "vision": "gemini/YOUR_MULTIMODAL_MODEL",
    "speech_to_text": "openai/YOUR_TRANSCRIPTION_MODEL",
    "text_to_speech": "openai/YOUR_TTS_MODEL",
    "image_generation": "openai/YOUR_IMAGE_MODEL",
    "image_editing": "openai/YOUR_IMAGE_MODEL",
}


def example(profile: ProfileConfig, capability: str) -> str:
    section = "capabilities" if profile.name == "default" else f"profiles.{json.dumps(profile.name)}.capabilities"
    marker = f"# >>> Aithon missing route: profile={json.dumps(profile.name)} capability={capability}"
    route = ("video = { understand = \"gemini/YOUR_VIDEO_UNDERSTANDING_MODEL\", "
             "generate = \"gemini/YOUR_VIDEO_GENERATION_MODEL\" }") if capability == "video" else (
                 f"{capability} = {json.dumps(MODELS.get(capability, 'openai/YOUR_MODEL'))}")
    return (f"{marker}\n"
            f"# Add this line to [{section}], creating the section if necessary:\n"
            f"# [{section}]\n# {route}\n"
            "# Replace the placeholder with a LiteLLM model ID and configure its credential.\n"
            "# <<< Aithon missing route\n")


def append_missing_route_example(config: ResolvedConfig, profile: ProfileConfig,
                                 capability: str) -> Path | None:
    path = config.path
    if path is None or path.is_symlink() or path.parent.resolve() != config.project_root.resolve():
        return None
    try:
        source = path.read_text(encoding="utf-8")
        if tomllib.loads(source).get("version") != 3 or not stat.S_ISREG(path.stat().st_mode):
            return None
        marker = f"# >>> Aithon missing route: profile={json.dumps(profile.name)} capability={capability}"
        if marker in source:
            return path
        flags = os.O_WRONLY | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "a", encoding="utf-8") as file:
            file.write(("\n" if source and not source.endswith("\n") else "") + "\n" + example(profile, capability))
        print(f"aithon: added commented {capability} route example to {path}", file=sys.stderr)
        return path
    except (OSError, ValueError, TypeError, UnicodeError):
        return None
