"""Raw file storage, content-addressed.

The SHA-256 of the bytes *is* the address. That is what makes re-uploading the same document a
no-op instead of a duplicate knowledge graph (``docs/DECISIONS.md`` D6).
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Final

from indra.core.config import Settings
from indra.core.exceptions import BlobStoreError
from indra.core.logging import get_logger

logger = get_logger(__name__)

_UNSAFE: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_NAME: Final[int] = 96


def safe_filename(name: str) -> str:
    """Reduce an arbitrary upload name to something safe to place on disk.

    Windows-hostile characters, path separators and traversal sequences all removed. The content
    hash carries identity, so mangling the display name costs nothing.
    """
    cleaned = _UNSAFE.sub("_", Path(name).name).strip("._") or "upload"
    return cleaned[:_MAX_NAME]


class FileBlobStore:
    """Content-addressed storage on the local filesystem."""

    name = "blobs:file"
    backend = "file"

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def _target(self, content_hash: str, filename: str) -> Path:
        # Two-level fan-out keeps any single directory from growing past a few hundred entries.
        shard = self._root / content_hash[:2] / content_hash[2:4]
        return shard / f"{content_hash[:16]}_{safe_filename(filename)}"

    async def put(self, content: bytes, *, filename: str, content_hash: str) -> str:
        existing = await self.exists(content_hash)
        if existing is not None:
            logger.debug("blob already stored", extra={"content_hash": content_hash[:12]})
            return existing
        target = self._target(content_hash, filename)
        try:
            await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)
            await asyncio.to_thread(target.write_bytes, content)
        except OSError as exc:
            raise BlobStoreError(
                f"Could not write uploaded file to {target}. Check disk space and permissions.",
                context={"filename": filename, "bytes": len(content)},
                cause=exc,
            ) from exc
        return target.as_posix()

    async def get(self, uri: str) -> bytes:
        path = Path(uri)
        try:
            return await asyncio.to_thread(path.read_bytes)
        except OSError as exc:
            raise BlobStoreError(
                f"Stored file is missing or unreadable: {uri}",
                context={"uri": uri},
                cause=exc,
            ) from exc

    async def exists(self, content_hash: str) -> str | None:
        shard = self._root / content_hash[:2] / content_hash[2:4]
        if not shard.is_dir():
            return None
        prefix = content_hash[:16]
        for candidate in shard.iterdir():
            if candidate.name.startswith(prefix):
                return candidate.as_posix()
        return None

    async def path_for(self, uri: str) -> Path:
        path = Path(uri)
        if not path.exists():
            raise BlobStoreError(f"Stored file is missing: {uri}", context={"uri": uri})
        return path

    async def health(self) -> dict[str, Any]:
        writable = self._root.is_dir()
        return {
            "ok": writable,
            "backend": "file",
            "detail": f"root={self._root}" if writable else f"root not writable: {self._root}",
        }

    async def close(self) -> None:
        return None


class MemoryBlobStore:
    """Blobs in a dict. Used by tests and by ``STORAGE_BACKEND=memory``."""

    name = "blobs:memory"
    backend = "memory"

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}
        self._by_hash: dict[str, str] = {}
        self._materialised: dict[str, Path] = {}

    async def put(self, content: bytes, *, filename: str, content_hash: str) -> str:
        if content_hash in self._by_hash:
            return self._by_hash[content_hash]
        uri = f"memory://{content_hash[:16]}/{safe_filename(filename)}"
        self._data[uri] = content
        self._by_hash[content_hash] = uri
        return uri

    async def get(self, uri: str) -> bytes:
        if uri not in self._data:
            raise BlobStoreError(f"No blob at {uri}", context={"uri": uri})
        return self._data[uri]

    async def exists(self, content_hash: str) -> str | None:
        return self._by_hash.get(content_hash)

    async def path_for(self, uri: str) -> Path:
        """Materialise to a temp file — parsers need a real path (pdfplumber, cv2, tesseract)."""
        if uri in self._materialised and self._materialised[uri].exists():
            return self._materialised[uri]
        import tempfile

        content = await self.get(uri)
        suffix = Path(uri).suffix or ".bin"
        handle = tempfile.NamedTemporaryFile(prefix="indra_", suffix=suffix, delete=False)
        try:
            await asyncio.to_thread(handle.write, content)
        finally:
            handle.close()
        path = Path(handle.name)
        self._materialised[uri] = path
        return path

    async def health(self) -> dict[str, Any]:
        return {"ok": True, "backend": "memory", "detail": f"{len(self._data)} blobs held in process"}

    async def close(self) -> None:
        for path in self._materialised.values():
            try:
                path.unlink(missing_ok=True)
            except OSError:  # pragma: no cover - best effort cleanup
                logger.debug("could not remove temp blob", extra={"path": str(path)})


def build_blob_store(settings: Settings, *, in_memory: bool) -> FileBlobStore | MemoryBlobStore:
    return MemoryBlobStore() if in_memory else FileBlobStore(settings.raw_dir)


__all__ = ["FileBlobStore", "MemoryBlobStore", "build_blob_store", "safe_filename"]
