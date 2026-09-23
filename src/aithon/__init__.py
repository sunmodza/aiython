"""Python execution with explicit, project-scoped AI configuration."""

def main() -> None:
    from .cli import main as cli_main
    cli_main()

from .assets import Asset, Image, Audio, Video, Document, VectorIndex

__all__ = ["main", "Asset", "Image", "Audio", "Video", "Document", "VectorIndex"]
