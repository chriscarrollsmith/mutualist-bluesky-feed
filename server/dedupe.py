"""Collapse word-for-word copies of a post already in the feed index.

The fingerprint is a hash of the post text. Whitespace is collapsed and
compatibility characters are folded; case and punctuation stay intact.
Empty text has no fingerprint, so image-only posts stay independent.

The process keeps a set of hashes already stored on ``Post.content_hash``.
That set avoids a database lookup on every firehose post. The unique index
on the column is the backstop when two copies arrive together.
"""

from __future__ import annotations

import hashlib
import threading
import unicodedata
from collections.abc import Iterable

from server import config
from server.database import Post

# Invisible characters sometimes inserted by copy-paste. Zero-width joiner
# stays so emoji sequences keep their identity.
_INVISIBLE = dict.fromkeys(map(ord, '\u200b\ufeff\u2060'))

_lock = threading.Lock()
_seen: set[str] = set()
_loaded = False


def content_fingerprint(text: str) -> str | None:
    """Return a stable hash of word-for-word post text, or None when empty."""
    normalized = unicodedata.normalize('NFKC', text).translate(_INVISIBLE)
    collapsed = ' '.join(normalized.split())
    if not collapsed:
        return None
    return hashlib.sha256(collapsed.encode('utf-8')).hexdigest()


def is_indexed_duplicate(fingerprint: str | None) -> bool:
    """True when an indexed post already has this fingerprint."""
    if not fingerprint or not config.DEDUPE_IDENTICAL_POSTS:
        return False
    with _lock:
        _ensure_loaded()
        return fingerprint in _seen


def remember_fingerprint(fingerprint: str | None) -> None:
    """Record a hash that was just stored on an indexed post."""
    if not fingerprint:
        return
    with _lock:
        _ensure_loaded()
        _seen.add(fingerprint)


def forget_fingerprints(fingerprints: Iterable[str | None]) -> None:
    """Drop hashes whose rows have been removed.

    A cache that has not been loaded yet reloads from the database, which
    already reflects the delete.
    """
    with _lock:
        if not _loaded:
            return
        for fingerprint in fingerprints:
            if fingerprint:
                _seen.discard(fingerprint)


def clear_seen_fingerprints() -> None:
    """Drop the process cache so the next check reloads from the database."""
    global _loaded
    with _lock:
        _seen.clear()
        _loaded = False


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    rows = Post.select(Post.content_hash).where(Post.content_hash.is_null(False))
    _seen.update(row.content_hash for row in rows if row.content_hash)
    _loaded = True
