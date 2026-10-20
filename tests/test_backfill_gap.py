"""Unit tests for gap backfill helpers (no network)."""

from __future__ import annotations

from datetime import UTC, datetime

from scripts.backfill_gap import (
    appview_post_to_event,
    created_at_from_post,
    extract_author_feed_posts,
    in_window,
    iter_author_posts,
    iter_search_posts,
    parse_iso_datetime,
)


def test_parse_iso_datetime_z_and_offset() -> None:
    assert parse_iso_datetime('2026-07-30T16:00:00Z') == datetime(2026, 7, 30, 16, 0, 0)
    assert parse_iso_datetime('2026-07-30T12:00:00-04:00') == datetime(2026, 7, 30, 16, 0, 0)


def test_in_window_inclusive() -> None:
    since = datetime(2026, 7, 30, 16, 0, 0)
    until = datetime(2026, 7, 31, 15, 0, 0)
    assert in_window(datetime(2026, 7, 30, 16, 0, 0), since=since, until=until)
    assert in_window(datetime(2026, 7, 31, 15, 0, 0), since=since, until=until)
    assert not in_window(datetime(2026, 7, 30, 15, 59, 59), since=since, until=until)
    assert not in_window(None, since=since, until=until)


def test_appview_post_to_event_sets_time_us_from_created_at() -> None:
    post = {
        'uri': 'at://did:plc:author/app.bsky.feed.post/abc',
        'cid': 'bafytest',
        'author': {'did': 'did:plc:author', 'handle': 'author.bsky.social'},
        'record': {
            'text': 'Hello from Troy NY',
            'createdAt': '2026-07-30T18:30:00.000Z',
            'langs': ['en'],
        },
    }
    event = appview_post_to_event(post)
    assert event is not None
    assert event['operation'] == 'create'
    assert event['uri'].endswith('/abc')
    assert event['author'] == 'did:plc:author'
    assert event['record']['text'] == 'Hello from Troy NY'
    # time_us must match createdAt so IGNORE_ARCHIVED_POSTS uses event time.
    expected_us = int(datetime(2026, 7, 30, 18, 30, 0, tzinfo=UTC).timestamp() * 1_000_000)
    assert event['time_us'] == expected_us


def test_appview_post_to_event_rejects_incomplete() -> None:
    assert appview_post_to_event({'uri': 'x'}) is None
    assert (
        appview_post_to_event(
            {
                'uri': 'at://did:plc:x/app.bsky.feed.post/1',
                'author': {'did': 'did:plc:x'},
                'record': {'text': 'no timestamp'},
            }
        )
        is None
    )


def test_extract_author_feed_skips_reposts() -> None:
    payload = {
        'feed': [
            {
                'post': {
                    'uri': 'own',
                    'author': {'did': 'did:plc:author', 'handle': 'example.bsky.social'},
                    'record': {'text': 'mine', 'createdAt': '2026-07-30T17:00:00Z'},
                }
            },
            {
                'reason': {'$type': 'app.bsky.feed.defs#reasonRepost'},
                'post': {
                    'uri': 'repost',
                    'author': {'did': 'did:plc:other', 'handle': 'other.bsky.social'},
                    'record': {'text': 'theirs', 'createdAt': '2026-07-30T17:00:00Z'},
                },
            },
        ]
    }
    posts = extract_author_feed_posts(payload, actor='did:plc:author')
    assert [p['uri'] for p in posts] == ['own']


def test_iter_author_posts_stops_before_since() -> None:
    since = datetime(2026, 7, 30, 16, 0, 0)
    until = datetime(2026, 7, 31, 12, 0, 0)
    pages = [
        {
            'feed': [
                {
                    'post': {
                        'uri': 'new',
                        'author': {'did': 'did:plc:a'},
                        'record': {
                            'text': 'in window',
                            'createdAt': '2026-07-30T20:00:00Z',
                        },
                    }
                },
                {
                    'post': {
                        'uri': 'old',
                        'author': {'did': 'did:plc:a'},
                        'record': {
                            'text': 'before window',
                            'createdAt': '2026-07-30T10:00:00Z',
                        },
                    }
                },
            ],
            'cursor': 'should-not-follow',
        }
    ]

    def fetcher(_url: str) -> dict:
        return pages.pop(0)

    uris = [
        p['uri']
        for p in iter_author_posts(
            'did:plc:a',
            since=since,
            until=until,
            api_host='https://api.bsky.app',
            fetcher=fetcher,
            pause_s=0,
        )
    ]
    assert uris == ['new']
    assert pages == []  # did not request the next page after hitting older posts


def test_iter_search_posts_filters_window() -> None:
    since = datetime(2026, 7, 30, 16, 0, 0)
    until = datetime(2026, 7, 31, 12, 0, 0)

    def fetcher(url: str) -> dict:
        assert 'sort=latest' in url
        assert 'since=' in url
        return {
            'posts': [
                {
                    'uri': 'in',
                    'author': {'did': 'did:plc:x'},
                    'record': {'text': 'anarchism', 'createdAt': '2026-07-30T18:00:00Z'},
                },
                {
                    'uri': 'out',
                    'author': {'did': 'did:plc:x'},
                    'record': {'text': 'anarchism', 'createdAt': '2026-07-29T18:00:00Z'},
                },
            ]
        }

    uris = [
        p['uri']
        for p in iter_search_posts(
            'anarchism',
            since=since,
            until=until,
            api_host='https://api.bsky.app',
            fetcher=fetcher,
            pause_s=0,
        )
    ]
    assert uris == ['in']


def test_created_at_from_post() -> None:
    assert created_at_from_post({'record': {'createdAt': '2026-07-30T01:02:03Z'}}) == datetime(
        2026, 7, 30, 1, 2, 3
    )
    assert created_at_from_post({'record': {}}) is None


def test_process_event_indexes_and_preserves_created_order(tmp_path) -> None:
    from scripts.backfill_gap import bind_database, process_event, uri_exists
    from server.database import Post

    db_path = tmp_path / 'backfill.db'
    bind_database(db_path)
    event = appview_post_to_event(
        {
            'uri': 'at://did:plc:author/app.bsky.feed.post/gap1',
            'cid': 'bafygap1',
            'author': {'did': 'did:plc:author'},
            'record': {
                'text': 'Mutual credit clearing tonight',
                'createdAt': '2026-07-30T18:30:00.000Z',
                'langs': ['en'],
            },
        }
    )
    assert event is not None
    assert process_event(event, dry_run=False, indexed_at_mode='created') == 'indexed'
    assert uri_exists(event['uri'])
    row = Post.get(Post.uri == event['uri'])
    assert row.created_at == datetime(2026, 7, 30, 18, 30, 0)
    assert row.indexed_at == datetime(2026, 7, 30, 18, 30, 0)
    assert process_event(event, dry_run=False, indexed_at_mode='created') == 'exists'


def test_jsonl_roundtrip(tmp_path) -> None:
    from scripts.backfill_gap import dump_posts_jsonl, load_posts_jsonl

    posts = [
        {
            'uri': 'at://did:plc:x/app.bsky.feed.post/1',
            'record': {'text': 'hi', 'createdAt': '2026-07-30T18:00:00Z'},
        }
    ]
    path = tmp_path / 'posts.jsonl'
    assert dump_posts_jsonl(path, posts) == 1
    assert load_posts_jsonl(path) == posts


def test_process_event_dry_run_does_not_write(tmp_path) -> None:
    from scripts.backfill_gap import bind_database, process_event, uri_exists

    bind_database(tmp_path / 'dry.db')
    event = appview_post_to_event(
        {
            'uri': 'at://did:plc:author/app.bsky.feed.post/dry1',
            'cid': 'bafydry1',
            'author': {'did': 'did:plc:author'},
            'record': {
                'text': 'Mutualist reading circle meetup',
                'createdAt': '2026-07-30T19:00:00.000Z',
                'langs': ['en'],
            },
        }
    )
    assert event is not None
    status = process_event(event, dry_run=True, indexed_at_mode='created')
    assert status == 'dry_run_match'
    assert not uri_exists(event['uri'])
