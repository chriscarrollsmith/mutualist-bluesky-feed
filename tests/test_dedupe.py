"""Word-for-word duplicate posts collapse to the earliest indexed copy."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any

import peewee
import pytest
from server.database import Post, SubscriptionState, db
from server.dedupe import clear_seen_fingerprints, content_fingerprint, is_indexed_duplicate


@pytest.fixture()
def isolated_db() -> Any:
    """Rebuild post tables so each test starts empty."""
    db.connect(reuse_if_open=True)
    db.drop_tables([Post, SubscriptionState], safe=True)
    db.create_tables([Post, SubscriptionState])
    clear_seen_fingerprints()
    yield
    Post.delete().execute()
    clear_seen_fingerprints()


_TEXT = 'Neighborhood mutual credit clearing is open tonight for members.'


def _create_event(
    *,
    author: str,
    text: str,
    uri_key: str,
    alt: str | None = None,
) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat().replace('+00:00', 'Z')
    record: dict[str, Any] = {
        '$type': 'app.bsky.feed.post',
        'text': text,
        'createdAt': now,
    }
    if alt is not None:
        record['embed'] = {'images': [{'alt': alt}]}
    return {
        'operation': 'create',
        'uri': f'at://{author}/app.bsky.feed.post/{uri_key}',
        'cid': f'bafy{uri_key}',
        'author': author,
        'record': record,
    }


def test_content_fingerprint_collapses_whitespace_and_keeps_case() -> None:
    assert (
        content_fingerprint('  mutual   credit\n') == hashlib.sha256(b'mutual credit').hexdigest()
    )
    assert content_fingerprint('Mutual credit') != content_fingerprint('mutual credit')
    assert content_fingerprint('mutual credit.') != content_fingerprint('mutual credit')
    assert content_fingerprint('mutual\u200b credit') == content_fingerprint('mutual credit')
    assert content_fingerprint('ﬁle sharing') == content_fingerprint('file sharing')
    assert content_fingerprint('   \n') is None
    assert content_fingerprint('') is None


def test_identical_text_from_another_account_is_not_indexed(isolated_db: None) -> None:
    from server.indexer import handle_event

    first = _create_event(
        author='did:plc:dedupeauthor0000000000001',
        text=_TEXT,
        uri_key='first',
    )
    second = _create_event(
        author='did:plc:dedupeauthor0000000000002',
        text=_TEXT,
        uri_key='second',
        alt='A different flyer for the same announcement.',
    )
    handle_event(first)
    handle_event(second)

    rows = list(Post.select())
    assert len(rows) == 1
    assert rows[0].uri == first['uri']
    assert rows[0].content_hash == content_fingerprint(_TEXT)


def test_same_author_repeat_is_not_indexed(isolated_db: None) -> None:
    from server.indexer import handle_event

    author = 'did:plc:dedupeauthor0000000000001'
    handle_event(_create_event(author=author, text=_TEXT, uri_key='once'))
    handle_event(_create_event(author=author, text=f'  {_TEXT}\n', uri_key='twice'))
    assert Post.select().count() == 1


def test_case_and_punctuation_differences_both_stay(isolated_db: None) -> None:
    from server.indexer import handle_event

    author = 'did:plc:dedupeauthor0000000000001'
    handle_event(_create_event(author=author, text=_TEXT, uri_key='exact'))
    handle_event(_create_event(author=author, text=_TEXT.lower(), uri_key='lower'))
    handle_event(_create_event(author=author, text=f'{_TEXT}!', uri_key='bang'))
    assert Post.select().count() == 3


def test_empty_text_posts_are_not_collapsed(
    isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from server import config
    from server.indexer import handle_event

    authors = (
        'did:plc:dedupeempty000000000000001',
        'did:plc:dedupeempty000000000000002',
    )
    monkeypatch.setattr(config, 'ALLOWLIST_DIDS', set(authors))
    handle_event(_create_event(author=authors[0], text='', uri_key='blank'))
    handle_event(_create_event(author=authors[1], text='   \n', uri_key='space'))
    rows = list(Post.select())
    assert len(rows) == 2
    assert all(row.content_hash is None for row in rows)


def test_duplicate_skips_the_matcher(isolated_db: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from server.indexer import handle_event
    from server.matcher import match_post

    calls = {'n': 0}
    real = match_post

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls['n'] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr('server.indexer.match_post', counting)
    author_a = 'did:plc:dedupeauthor0000000000001'
    author_b = 'did:plc:dedupeauthor0000000000002'
    handle_event(_create_event(author=author_a, text=_TEXT, uri_key='kept'))
    handle_event(_create_event(author=author_b, text=_TEXT, uri_key='copy'))
    assert calls['n'] == 1
    assert Post.select().count() == 1


def test_delete_allows_the_same_text_to_be_indexed_again(isolated_db: None) -> None:
    from server.indexer import handle_event

    first = _create_event(
        author='did:plc:dedupeauthor0000000000001',
        text=_TEXT,
        uri_key='first',
    )
    second = _create_event(
        author='did:plc:dedupeauthor0000000000002',
        text=_TEXT,
        uri_key='second',
    )
    handle_event(first)
    handle_event({'operation': 'delete', 'uri': first['uri']})
    handle_event(second)
    rows = list(Post.select())
    assert len(rows) == 1
    assert rows[0].uri == second['uri']


def test_prune_forgets_the_fingerprint(isolated_db: None) -> None:
    from server.database import prune_old_posts, utc_now
    from server.indexer import handle_event

    first = _create_event(
        author='did:plc:dedupeauthor0000000000001',
        text=_TEXT,
        uri_key='old',
    )
    handle_event(first)
    row = Post.get(Post.uri == first['uri'])
    row.indexed_at = utc_now() - timedelta(days=30)
    row.save()
    assert prune_old_posts() == 1

    second = _create_event(
        author='did:plc:dedupeauthor0000000000002',
        text=_TEXT,
        uri_key='fresh',
    )
    handle_event(second)
    rows = list(Post.select())
    assert len(rows) == 1
    assert rows[0].uri == second['uri']


def test_purge_forgets_blocklisted_fingerprint(isolated_db: None) -> None:
    from server import config
    from server.indexer import handle_event, purge_blocklisted_posts

    assert config.BLOCKLIST_DIDS
    blocked = sorted(config.BLOCKLIST_DIDS)[0]
    digest = content_fingerprint(_TEXT)
    assert digest is not None
    Post.create(
        uri=f'at://{blocked}/app.bsky.feed.post/blocked',
        cid='bafyblocked',
        author_did=blocked,
        content_hash=digest,
        match_reason='strong_positive',
    )
    clear_seen_fingerprints()
    assert is_indexed_duplicate(digest)
    assert purge_blocklisted_posts() == 1
    assert not is_indexed_duplicate(digest)

    fresh = _create_event(
        author='did:plc:dedupeauthor0000000000001',
        text=_TEXT,
        uri_key='afterpurge',
    )
    handle_event(fresh)
    assert Post.get(Post.uri == fresh['uri']).content_hash == digest


def test_unique_index_blocks_a_stale_cache_race(isolated_db: None) -> None:
    from server.indexer import handle_event

    digest = content_fingerprint(_TEXT)
    assert digest is not None
    clear_seen_fingerprints()
    assert is_indexed_duplicate('0' * 64) is False
    Post.create(
        uri='at://did:plc:dedupeauthor0000000000001/app.bsky.feed.post/winner',
        cid='bafywinner',
        author_did='did:plc:dedupeauthor0000000000001',
        content_hash=digest,
        match_reason='strong_positive',
    )
    loser = _create_event(
        author='did:plc:dedupeauthor0000000000002',
        text=_TEXT,
        uri_key='loser',
    )
    handle_event(loser)
    assert Post.select().count() == 1
    assert Post.select().where(Post.uri == loser['uri']).count() == 0
    assert is_indexed_duplicate(digest)


def test_dedupe_disabled_keeps_every_copy(
    isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from server import config
    from server.indexer import handle_event

    monkeypatch.setattr(config, 'DEDUPE_IDENTICAL_POSTS', False)
    handle_event(
        _create_event(
            author='did:plc:dedupeauthor0000000000001',
            text=_TEXT,
            uri_key='one',
        )
    )
    handle_event(
        _create_event(
            author='did:plc:dedupeauthor0000000000002',
            text=_TEXT,
            uri_key='two',
        )
    )
    rows = list(Post.select())
    assert len(rows) == 2
    assert all(row.content_hash is None for row in rows)


def test_feed_skeleton_serves_the_first_copy(isolated_db: None) -> None:
    from server.algos.feed import handler
    from server.indexer import handle_event

    first = _create_event(
        author='did:plc:dedupeauthor0000000000001',
        text=_TEXT,
        uri_key='shown',
    )
    second = _create_event(
        author='did:plc:dedupeauthor0000000000002',
        text=_TEXT,
        uri_key='hidden',
    )
    other = _create_event(
        author='did:plc:dedupeauthor0000000000003',
        text='The community land trust closed on the parcel today.',
        uri_key='other',
    )
    handle_event(first)
    handle_event(second)
    handle_event(other)
    body = handler(None, 10)
    uris = [item['post'] for item in body['feed']]
    assert first['uri'] in uris
    assert second['uri'] not in uris
    assert other['uri'] in uris


def test_ensure_columns_adds_unique_content_hash(isolated_db: None) -> None:
    from server.database import _ensure_columns

    db.execute_sql('DROP INDEX IF EXISTS "post_content_hash"')
    db.execute_sql('ALTER TABLE "post" DROP COLUMN "content_hash"')
    _ensure_columns()

    columns = {row[1] for row in db.execute_sql('PRAGMA table_info("post")').fetchall()}
    assert 'content_hash' in columns
    indexes = {row[1]: row[2] for row in db.execute_sql('PRAGMA index_list("post")').fetchall()}
    assert indexes['post_content_hash'] == 1

    Post.create(uri='at://did/post/a', cid='a', content_hash=None)
    Post.create(uri='at://did/post/b', cid='b', content_hash=None)
    Post.create(uri='at://did/post/c', cid='c', content_hash='abc')
    with pytest.raises(peewee.IntegrityError):
        Post.create(uri='at://did/post/d', cid='d', content_hash='abc')
