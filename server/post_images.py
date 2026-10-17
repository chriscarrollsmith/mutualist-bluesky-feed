"""JPEG thumbnails of Bluesky embeds for the decision model.

Jetstream records store blob refs, not bytes, and the Decisions API accepts
images only as base64 data URLs. Regex candidates that carry images are
downloaded from the public CDN thumbnail preset before classification. A
failed download omits that image; the text decision still runs.
"""

from __future__ import annotations

import base64
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any, cast

from server.logger import logger

MAX_IMAGES = 4
MAX_IMAGE_BYTES = 1_000_000
FETCH_TIMEOUT_S = 2.5
_CDN = 'https://cdn.bsky.app/img/feed_thumbnail/plain/{did}/{cid}@jpeg'

_DID = re.compile(r'^did:[a-z0-9]+:[a-zA-Z0-9._:%-]{1,200}$')
_CID = re.compile(r'^[A-Za-z0-9]{8,128}$')
_AT_URI_DID = re.compile(r'^at://(did:[a-z0-9]+:[a-zA-Z0-9._:%-]{1,200})/')
_CDN_URL = re.compile(
    r'^https://cdn\.bsky\.app/img/[a-z0-9_]+/plain/'
    r'(did:[a-z0-9]+:[a-zA-Z0-9._:%-]{1,200})/'
    r'([A-Za-z0-9]{8,128})(?:@[a-z0-9]+)?$'
)

ThumbnailFetch = Callable[[str, str], bytes | None]


def images_enabled() -> bool:
    raw = os.environ.get('DECISION_IMAGES', 'true').strip().lower()
    return raw in {'1', 'true', 't', 'yes', 'y'}


def thumbnail_url(did: str, cid: str) -> str:
    return _CDN.format(did=did, cid=cid)


def image_refs(embed: object | None, *, author_did: str | None) -> list[tuple[str, str]]:
    """Return ``(did, cid)`` pairs for images on this embed, at most four."""
    refs: list[tuple[str, str]] = []
    _walk(embed, author_did, refs, 0)
    return refs


def image_data_urls(
    embed: object | None,
    *,
    author_did: str | None,
    fetch: ThumbnailFetch | None = None,
) -> list[str]:
    """Download embed thumbnails and return JPEG data URLs.

    ``fetch`` replaces the CDN download in tests. It receives ``(did, cid)``.
    """
    if not images_enabled():
        return []
    refs = image_refs(embed, author_did=author_did)
    if not refs:
        return []
    load = fetch or _fetch_thumbnail
    urls: list[str] = []
    for did, cid in refs:
        blob = load(did, cid)
        if not blob:
            continue
        encoded = base64.b64encode(blob).decode('ascii')
        urls.append(f'data:image/jpeg;base64,{encoded}')
    if urls:
        logger.info('decider images=%d', len(urls))
    else:
        logger.warning('decider image fetch failed count=%d', len(refs))
    return urls


def _walk(node: object, did: str | None, refs: list[tuple[str, str]], depth: int) -> None:
    if depth > 4 or len(refs) >= MAX_IMAGES or not isinstance(node, dict):
        return
    images = node.get('images')
    if isinstance(images, list):
        for image in images:
            _add_image_item(refs, image, did)
            if len(refs) >= MAX_IMAGES:
                return
    external = node.get('external')
    if isinstance(external, dict):
        _consider(refs, external.get('thumb'), did)
    media = node.get('media')
    if isinstance(media, dict):
        _walk(media, did, refs, depth + 1)
    quoted = _quoted_embed(node.get('record'))
    if quoted is not None:
        payload, embed = quoted
        quoted_did = _did_from_uri(payload.get('uri')) or did
        _walk(embed, quoted_did, refs, depth + 1)


def _quoted_embed(record: object) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not isinstance(record, dict):
        return None
    inner = record.get('record')
    payload = cast(dict[str, Any], inner if isinstance(inner, dict) else record)
    for key in ('value', 'record'):
        nested = payload.get(key)
        if not isinstance(nested, dict):
            continue
        embed = nested.get('embed')
        if isinstance(embed, dict):
            return payload, cast(dict[str, Any], embed)
    embed = payload.get('embed')
    if isinstance(embed, dict):
        return payload, cast(dict[str, Any], embed)
    return None


def _add_image_item(refs: list[tuple[str, str]], image: object, did: str | None) -> None:
    if not isinstance(image, dict):
        return
    before = len(refs)
    for key in ('image', 'thumb', 'fullsize'):
        value = image.get(key)
        if not value:
            continue
        _consider(refs, value, did)
        if len(refs) > before:
            return


def _consider(refs: list[tuple[str, str]], value: object, did: str | None) -> None:
    if isinstance(value, str):
        parsed = _ref_from_cdn_url(value)
        if parsed is not None:
            _add(refs, parsed[0], parsed[1])
        return
    cid = _cid_from_blob(value)
    if cid:
        _add(refs, did, cid)


def _add(refs: list[tuple[str, str]], did: str | None, cid: str | None) -> None:
    if not did or not cid or not _DID.match(did) or not _CID.match(cid):
        return
    pair = (did, cid)
    if pair not in refs:
        refs.append(pair)


def _cid_from_blob(blob: object) -> str | None:
    if not isinstance(blob, dict):
        return None
    ref = blob.get('ref')
    if isinstance(ref, str):
        return ref
    if isinstance(ref, dict):
        link = ref.get('$link')
        if isinstance(link, str):
            return link
    link = blob.get('$link')
    return link if isinstance(link, str) else None


def _did_from_uri(uri: object) -> str | None:
    if not isinstance(uri, str):
        return None
    match = _AT_URI_DID.match(uri)
    return match.group(1) if match else None


def _ref_from_cdn_url(url: str) -> tuple[str, str] | None:
    match = _CDN_URL.match(url)
    if match is None:
        return None
    return match.group(1), match.group(2)


def _fetch_thumbnail(did: str, cid: str) -> bytes | None:
    url = thumbnail_url(did, cid)
    request = urllib.request.Request(
        url,
        headers={
            'Accept': 'image/jpeg',
            'User-Agent': 'mutualist-bluesky-feed-decider/1.0',
        },
        method='GET',
    )
    try:
        with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_S) as response:
            blob = response.read(MAX_IMAGE_BYTES + 1)
    except urllib.error.URLError, TimeoutError, OSError:
        logger.debug('decider image fetch failed cid=%s', cid)
        return None
    if len(blob) < 32 or len(blob) > MAX_IMAGE_BYTES or not blob.startswith(b'\xff\xd8'):
        logger.debug('decider image unusable cid=%s bytes=%d', cid, len(blob))
        return None
    return blob
