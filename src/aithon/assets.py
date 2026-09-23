"""Lazy local assets. Identity handles remain owned by the live RuntimeBridge."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import mimetypes
import os
from pathlib import Path
import shutil


@dataclass(frozen=True)
class Asset:
    path: Path
    mime_type: str | None = None

    def __post_init__(self):
        object.__setattr__(self, 'path', Path(self.path))
        if self.mime_type is None:
            object.__setattr__(self, 'mime_type', mimetypes.guess_type(self.path)[0] or 'application/octet-stream')

    def __fspath__(self):
        return os.fspath(self.path)

    def save(self, destination):
        destination = Path(destination)
        with self.path.open('rb') as source, destination.open('xb') as target:
            shutil.copyfileobj(source, target)
        return type(self)(destination, self.mime_type)

    def content_hash(self):
        with self.path.open('rb') as source:
            return hashlib.file_digest(source, 'sha256').hexdigest()


class Image(Asset):
    pass


class Audio(Asset):
    pass


class Video(Asset):
    pass


class Document(Asset):
    pass


@dataclass(frozen=True)
class VectorIndex:
    path: Path
    name: str
    space: str
    dimensions: int
    _objects: tuple = field(default=(), repr=False, compare=False)


ASSET_TYPES = {cls.__name__: cls for cls in (Asset, Image, Audio, Video, Document)}
