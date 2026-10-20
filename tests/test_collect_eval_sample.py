"""Unit tests for eval sampling helpers (no network)."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.collect_eval_sample import (
    build_parser,
    collect,
    dedupe_candidates,
    extract_posts_from_feed_payload,
    extract_posts_from_search_payload,
    load_list_file,
    looks_event_like,
    post_to_candidate,
)


def test_load_list_file_skips_comments(tmp_path: Path) -> None:
    path = tmp_path / 'handles.txt'
    path.write_text(
        '# comment\n\nexample.bsky.social\n# another\nother.example\n',
        encoding='utf-8',
    )
    assert load_list_file(path) == ['example.bsky.social', 'other.example']


def test_looks_event_like() -> None:
    assert looks_event_like('Show tonight at 8pm')
    assert looks_event_like('Tickets on sale for Saturday')
    assert not looks_event_like('Just another weather update')


def test_post_to_candidate_sets_strata() -> None:
    post = {
        'uri': 'at://did:plc:abc/app.bsky.feed.post/xyz',
        'record': {'text': 'Hello from the desk'},
        'author': {'handle': 'example.bsky.social', 'did': 'did:plc:abc'},
    }
    row = post_to_candidate(
        post,
        source='authors',
        signal='author',
        bucket='allowlisted_author',
        split='dev',
    )
    assert row['id'].endswith('/xyz')
    assert row['expected'] is None
    assert row['signal'] == 'author'
    assert row['bucket'] == 'allowlisted_author'
    assert 'source=authors' in row['note']


def test_extract_posts_helpers() -> None:
    feed_posts = extract_posts_from_feed_payload(
        {'feed': [{'post': {'uri': 'a', 'record': {'text': 'one'}}}, {'post': None}]}
    )
    assert [p['uri'] for p in feed_posts] == ['a']
    search_posts = extract_posts_from_search_payload(
        {'posts': [{'uri': 'b', 'record': {'text': 'two'}}, 'bad']}
    )
    assert [p['uri'] for p in search_posts] == ['b']


def test_extract_author_feed_skips_reposts_and_other_authors() -> None:
    payload = {
        'feed': [
            {
                'post': {
                    'uri': 'own',
                    'record': {'text': 'mine'},
                    'author': {'handle': 'example.bsky.social', 'did': 'did:plc:tu'},
                }
            },
            {
                'reason': {'$type': 'app.bsky.feed.defs#reasonRepost'},
                'post': {
                    'uri': 'repost',
                    'record': {'text': 'theirs'},
                    'author': {'handle': 'other.bsky.social', 'did': 'did:plc:other'},
                },
            },
            {
                'post': {
                    'uri': 'mismatch',
                    'record': {'text': 'wrong author'},
                    'author': {'handle': 'other.bsky.social', 'did': 'did:plc:other'},
                }
            },
        ]
    }
    posts = extract_posts_from_feed_payload(payload, actor='example.bsky.social')
    assert [p['uri'] for p in posts] == ['own']


def test_dedupe_candidates_skips_existing() -> None:
    rows = [
        {'id': 'a', 'text': '1'},
        {'id': 'b', 'text': '2'},
        {'id': 'a', 'text': 'dup'},
    ]
    out = dedupe_candidates(rows, skip_ids={'b'})
    assert [r['id'] for r in out] == ['a']


def test_collect_authors_and_near_miss_with_stub_fetcher(tmp_path: Path) -> None:
    cases_path = tmp_path / 'eval_cases.json'
    cases_path.write_text('[{"id": "skip-me", "text": "x", "expected": false}]', encoding='utf-8')

    def fetcher(url: str) -> dict:
        if 'getAuthorFeed' in url:
            return {
                'feed': [
                    {
                        'post': {
                            'uri': 'at://did:plc:1/app.bsky.feed.post/one',
                            'record': {'text': 'Desk notes with no topic phrase'},
                            'author': {'handle': 'example.bsky.social', 'did': 'did:plc:1'},
                        }
                    },
                    {
                        'post': {
                            'uri': 'skip-me',
                            'record': {'text': 'already labeled'},
                            'author': {'handle': 'example.bsky.social', 'did': 'did:plc:1'},
                        }
                    },
                ]
            }
        if 'searchPosts' in url:
            return {
                'posts': [
                    {
                        'uri': 'at://did:plc:2/app.bsky.feed.post/fp',
                        'record': {'text': 'News about anarcho-capitalism and Rothbard'},
                        'author': {'handle': 'chi.example', 'did': 'did:plc:2'},
                    }
                ]
            }
        raise AssertionError(url)

    authors_args = build_parser().parse_args(
        [
            '--skip-existing',
            str(cases_path),
            'authors',
            '--handle',
            'example.bsky.social',
            '--limit-per-author',
            '5',
        ]
    )
    author_rows = collect(authors_args, fetcher=fetcher)
    assert len(author_rows) == 1
    assert author_rows[0]['signal'] == 'author'
    assert author_rows[0]['bucket'] == 'allowlisted_author'

    near_args = build_parser().parse_args(
        [
            '--no-skip-existing',
            'near-miss',
            '--only-queries',
            '--query',
            'anarcho-capitalism',
            '--limit-per-query',
            '5',
        ]
    )
    near_rows = collect(near_args, fetcher=fetcher)
    assert len(near_rows) == 1
    assert near_rows[0]['bucket'] == 'skyfeed_fp'


def test_collect_events_filters_non_event_text() -> None:
    def fetcher(url: str) -> dict:
        return {
            'posts': [
                {
                    'uri': 'at://did:plc:3/app.bsky.feed.post/event',
                    'record': {'text': 'Teach-in tonight at the infoshop'},
                    'author': {'handle': 'venue.example', 'did': 'did:plc:3'},
                },
                {
                    'uri': 'at://did:plc:3/app.bsky.feed.post/plain',
                    'record': {'text': 'The infoshop is a nice building'},
                    'author': {'handle': 'venue.example', 'did': 'did:plc:3'},
                },
            ]
        }

    args = build_parser().parse_args(
        [
            '--no-skip-existing',
            'events',
            '--only-queries',
            '--query',
            'infoshop',
            '--require-event-cue',
        ]
    )
    rows = collect(args, fetcher=fetcher)
    assert [r['id'] for r in rows] == ['at://did:plc:3/app.bsky.feed.post/event']
    assert rows[0]['signal'] == 'event'


def test_collect_authors_requires_actors() -> None:
    args = build_parser().parse_args(['--no-skip-existing', 'authors'])
    with pytest.raises(ValueError, match='allowlist'):
        collect(args, fetcher=lambda url: {})


def test_cli_reports_usage_errors(capsys: pytest.CaptureFixture[str]) -> None:
    from scripts.collect_eval_sample import main

    assert main(['--no-skip-existing', 'authors']) == 1
    err = capsys.readouterr().err
    assert 'allowlist' in err
