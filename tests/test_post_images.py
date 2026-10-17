"""Image refs and thumbnail downloads for the decider. No network."""

from __future__ import annotations

import pytest
from server.post_images import image_data_urls, image_refs, thumbnail_url

AUTHOR = 'did:plc:abc123'
QUOTED = 'did:plc:quoted9'
CID = 'bafkreihdwdcefgh4dqkjv67uzcmw7ojee6xedzdetojuzjevtenxquvyku'
CID_2 = 'bafkreibbafkreibbafkreibbafkreibbafkreibbafkreibbafkrei'


def _blob(cid: str) -> dict[str, object]:
    return {'$type': 'blob', 'ref': {'$link': cid}, 'mimeType': 'image/jpeg', 'size': 20}


def test_image_refs_from_jetstream_images_embed() -> None:
    embed = {'$type': 'app.bsky.embed.images', 'images': [{'alt': 'poster', 'image': _blob(CID)}]}
    assert image_refs(embed, author_did=AUTHOR) == [(AUTHOR, CID)]
    assert thumbnail_url(AUTHOR, CID) == (
        f'https://cdn.bsky.app/img/feed_thumbnail/plain/{AUTHOR}/{CID}@jpeg'
    )


def test_image_refs_from_record_with_media_and_quote() -> None:
    embed = {
        '$type': 'app.bsky.embed.recordWithMedia',
        'media': {'images': [{'image': _blob(CID)}]},
        'record': {
            'record': {
                'uri': f'at://{QUOTED}/app.bsky.feed.post/abc',
                'value': {'embed': {'images': [{'image': _blob(CID_2)}]}},
            }
        },
    }
    assert image_refs(embed, author_did=AUTHOR) == [(AUTHOR, CID), (QUOTED, CID_2)]


def test_image_refs_from_view_thumb_url_ignores_foreign_hosts() -> None:
    thumb = f'https://cdn.bsky.app/img/feed_fullsize/plain/{QUOTED}/{CID}@jpeg'
    embed = {
        '$type': 'app.bsky.embed.images#view',
        'images': [{'thumb': thumb, 'fullsize': thumb, 'alt': 'poster'}],
    }
    assert image_refs(embed, author_did=AUTHOR) == [(QUOTED, CID)]
    rejected = {
        'images': [
            {'thumb': 'https://evil.example/img.jpg'},
            {'image': _blob('../etc/passwd')},
        ]
    }
    assert image_refs(rejected, author_did=AUTHOR) == []


def test_external_thumb_blob() -> None:
    embed = {'external': {'uri': 'https://example.com', 'thumb': _blob(CID)}}
    assert image_refs(embed, author_did=AUTHOR) == [(AUTHOR, CID)]


def test_image_refs_caps_at_four() -> None:
    embed = {'images': [{'image': _blob(f'{CID}{index}')} for index in range(6)]}
    assert len(image_refs(embed, author_did=AUTHOR)) == 4


def test_image_data_urls_encodes_jpeg_and_skips_failures() -> None:
    embed = {'images': [{'image': _blob(CID)}, {'image': _blob(CID_2)}]}
    jpeg = b'\xff\xd8' + b'x' * 40

    def fetch(did: str, cid: str) -> bytes | None:
        assert did == AUTHOR
        return jpeg if cid == CID else None

    urls = image_data_urls(embed, author_did=AUTHOR, fetch=fetch)
    assert len(urls) == 1
    assert urls[0].startswith('data:image/jpeg;base64,')


def test_image_data_urls_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('DECISION_IMAGES', 'false')
    embed = {'images': [{'image': _blob(CID)}]}

    def boom(did: str, cid: str) -> bytes | None:
        del did, cid
        raise AssertionError('fetched while disabled')

    assert image_data_urls(embed, author_did=AUTHOR, fetch=boom) == []


def test_thumbnail_download_uses_cdn(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    jpeg = b'\xff\xd8' + b'x' * 40

    class _Response:
        def read(self, limit: int) -> bytes:
            assert limit > len(jpeg)
            return jpeg

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

    def fake_urlopen(request, timeout: float) -> _Response:
        captured['url'] = request.full_url
        captured['timeout'] = timeout
        return _Response()

    monkeypatch.setattr('server.post_images.urllib.request.urlopen', fake_urlopen)
    embed = {'images': [{'image': _blob(CID)}]}
    urls = image_data_urls(embed, author_did=AUTHOR)
    assert urls[0].startswith('data:image/jpeg;base64,')
    assert captured['url'] == thumbnail_url(AUTHOR, CID)
    assert captured['timeout'] == 2.5
