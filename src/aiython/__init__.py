"""Python execution with explicit, project-scoped AI configuration."""

def main() -> None:
    from .cli import main as cli_main
    cli_main()

from .assets import Asset, Image, Audio, Video, Document, VectorIndex

def __getattr__(name):
    if name == "send_a2a":
        from .a2a import send_a2a
        return send_a2a
    if name in {"group", "join", "worker_entry", "current"}:
        from . import collaboration
        return getattr(collaboration, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["main", "Asset", "Image", "Audio", "Video", "Document", "VectorIndex",
           "group", "join", "worker_entry", "current", "send_a2a"]
