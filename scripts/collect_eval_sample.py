#!/usr/bin/env python3
"""Sample Bluesky posts for hand-labeling into the matcher eval set.

This tool can pull:

- ``authors`` — posts from allowlisted or CLI handles/DIDs
- ``near-miss`` — search hits for ambiguous or false-friend phrases
- ``events`` — search hits for in-scope event announcements
- ``feed`` — posts from a custom feed URI
- ``search`` — free-form ``searchPosts`` queries

Output is JSON lines on stdout. Label ``expected`` (true/false), then merge with:

    uv run python scripts/append_eval_cases.py --input /tmp/labeled.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.allowlists import load_list_file  # noqa: E402

DEFAULT_API_HOST = 'https://api.bsky.app'
USER_AGENT = (
    'mutualist-bluesky-feed-eval/0.1 (+https://github.com/chriscarrollsmith/mutualist-bluesky-feed)'
)

NEAR_MISS_QUERIES = (
    'anarcho-capitalism',
    'ancap',
    'Sons of Anarchy',
    'Anarchy Online',
    'decentralized finance',
    'DeFi web3',
    'Mises Institute',
    'state of anarchy',
    'this is anarchy',
    'Rothbard anarchism',
    'anarcho-syndicalism',
    'Food Not Bombs',
    'CrimethInc',
    'communalism election',
    'social-ecological systems',
    'EU confederalism',
    'Rojava martyrs',
)

EVENT_QUERIES = (
    'mutualist reading',
    'mutualism meetup',
    'mutual credit workshop',
    'Proudhon reading circle',
    'Really Really Free Market',
    'occupancy and use teach-in',
    'agorism meetup',
    'C4SS event',
)

EVENT_CUE_RE = re.compile(
    r"""
    \b(?:
        tonight|tomorrow|this\s+weekend|this\s+saturday|this\s+sunday|
        doors(?:\s+at)?|tickets?|save\s+the\s+date|join\s+us|
        open\s+mic|festival|concert|show\s+starts|doors\s+open|
        \d{1,2}:\d{2}\s*(?:am|pm)|(?<!\d)\d{1,2}\s*(?:am|pm)\b|
        january|february|march|april|june|july|august|september|
        october|november|december
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

JsonObject = dict[str, Any]
Fetcher = Callable[[str], JsonObject]


def load_existing_ids(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, list):
        raise ValueError(f'{path} must contain a JSON array')
    return {str(case['id']) for case in payload if isinstance(case, dict) and 'id' in case}


def looks_event_like(text: str) -> bool:
    return bool(EVENT_CUE_RE.search(text or ''))


def post_to_candidate(
    post: JsonObject,
    *,
    source: str,
    signal: str,
    bucket: str,
    split: str,
) -> JsonObject:
    record = post.get('record') or {}
    author = post.get('author') or {}
    text = str(record.get('text') or '')
    langs = record.get('langs') if isinstance(record.get('langs'), list) else None
    candidate: dict[str, Any] = {
        'id': str(post.get('uri') or ''),
        'text': text,
        'author_handle': author.get('handle'),
        'author_did': author.get('did'),
        'expected': None,
        'signal': signal,
        'bucket': bucket,
        'split': split,
        'regression': True,
        'note': f'label me: true=keep, false=drop | source={source}',
    }
    if langs:
        candidate['langs'] = langs
    return candidate


def extract_posts_from_feed_payload(
    payload: JsonObject,
    *,
    actor: str | None = None,
    include_reposts: bool = False,
) -> list[JsonObject]:
    """Extract posts from getFeed / getAuthorFeed payloads.

    For author sampling, reposts are skipped by default and (when ``actor`` is
    set) only posts whose author handle/DID matches the requested actor are
    kept.
    """
    actor_key = (actor or '').strip().lower()
    posts: list[JsonObject] = []
    for item in payload.get('feed') or []:
        if not isinstance(item, dict):
            continue
        if not include_reposts and item.get('reason') is not None:
            continue
        post = item.get('post')
        if not isinstance(post, dict):
            continue
        if actor_key:
            author = post.get('author') or {}
            handle = str(author.get('handle') or '').lower()
            did = str(author.get('did') or '').lower()
            if actor_key not in {handle, did}:
                continue
        posts.append(post)
    return posts


def extract_posts_from_search_payload(payload: JsonObject) -> list[JsonObject]:
    posts: list[JsonObject] = []
    for post in payload.get('posts') or []:
        if isinstance(post, dict):
            posts.append(post)
    return posts


def default_fetcher(url: str, *, timeout: float = 30.0) -> JsonObject:
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def api_url(host: str, xrpc: str, params: dict[str, Any]) -> str:
    base = host.rstrip('/') + '/xrpc/' + xrpc.lstrip('/')
    return base + '?' + urllib.parse.urlencode(params)


def fetch_feed_posts(
    *,
    feed: str,
    limit: int,
    api_host: str,
    fetcher: Fetcher,
) -> list[JsonObject]:
    url = api_url(
        api_host,
        'app.bsky.feed.getFeed',
        {'feed': feed, 'limit': max(1, min(limit, 100))},
    )
    return extract_posts_from_feed_payload(fetcher(url))


def fetch_author_posts(
    *,
    actor: str,
    limit: int,
    api_host: str,
    fetcher: Fetcher,
) -> list[JsonObject]:
    url = api_url(
        api_host,
        'app.bsky.feed.getAuthorFeed',
        {
            'actor': actor,
            # Request extras so that after dropping reposts we still have enough
            # author-owned posts to satisfy limit_per_author.
            'limit': max(1, min(max(limit * 3, limit), 100)),
            'filter': 'posts_no_replies',
        },
    )
    posts = extract_posts_from_feed_payload(
        fetcher(url),
        actor=actor,
        include_reposts=False,
    )
    return posts[: max(1, limit)]


def fetch_search_posts(
    *,
    query: str,
    limit: int,
    api_host: str,
    fetcher: Fetcher,
) -> list[JsonObject]:
    url = api_url(
        api_host,
        'app.bsky.feed.searchPosts',
        {'q': query, 'limit': max(1, min(limit, 100))},
    )
    return extract_posts_from_search_payload(fetcher(url))


def dedupe_candidates(
    rows: Iterable[JsonObject],
    *,
    skip_ids: set[str],
    require_text: bool = True,
) -> list[JsonObject]:
    seen: set[str] = set(skip_ids)
    out: list[JsonObject] = []
    for row in rows:
        row_id = str(row.get('id') or '')
        if not row_id or row_id in seen:
            continue
        if require_text and not str(row.get('text') or '').strip():
            continue
        seen.add(row_id)
        out.append(row)
    return out


def collect_from_authors(
    actors: list[str],
    *,
    limit_per_author: int,
    split: str,
    api_host: str,
    fetcher: Fetcher,
) -> list[JsonObject]:
    rows: list[JsonObject] = []
    for actor in actors:
        try:
            posts = fetch_author_posts(
                actor=actor,
                limit=limit_per_author,
                api_host=api_host,
                fetcher=fetcher,
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f'warn: author fetch failed for {actor}: {exc}', file=sys.stderr)
            continue
        for post in posts:
            rows.append(
                post_to_candidate(
                    post,
                    source='authors',
                    signal='author',
                    bucket='allowlisted_author',
                    split=split,
                )
            )
    return rows


def collect_from_queries(
    queries: list[str],
    *,
    limit_per_query: int,
    source: str,
    signal: str,
    bucket: str,
    split: str,
    require_event_cue: bool,
    api_host: str,
    fetcher: Fetcher,
) -> list[JsonObject]:
    rows: list[JsonObject] = []
    for query in queries:
        try:
            posts = fetch_search_posts(
                query=query,
                limit=limit_per_query,
                api_host=api_host,
                fetcher=fetcher,
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f'warn: search failed for {query!r}: {exc}', file=sys.stderr)
            continue
        for post in posts:
            record = post.get('record') or {}
            text = str(record.get('text') or '')
            if require_event_cue and not looks_event_like(text):
                continue
            rows.append(
                post_to_candidate(
                    post,
                    source=source,
                    signal=signal,
                    bucket=bucket,
                    split=split,
                )
            )
    return rows


def resolve_actors(args: argparse.Namespace) -> list[str]:
    actors: list[str] = []
    if args.from_allowlist:
        actors.extend(load_list_file(ROOT / 'data' / 'allowlist_handles.txt'))
        actors.extend(load_list_file(ROOT / 'data' / 'allowlist_dids.txt'))
    actors.extend(args.handle or [])
    actors.extend(args.did or [])
    # Preserve order, drop duplicates / empties.
    seen: set[str] = set()
    ordered: list[str] = []
    for actor in actors:
        key = actor.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--api-host',
        default=os.environ.get('BSKY_API_HOST', DEFAULT_API_HOST),
        help=f'Bluesky HTTP API host (default: {DEFAULT_API_HOST})',
    )
    parser.add_argument(
        '--skip-existing',
        type=Path,
        nargs='?',
        const=ROOT / 'data' / 'eval_cases.json',
        default=ROOT / 'data' / 'eval_cases.json',
        help='JSON cases file whose ids should be skipped (default: data/eval_cases.json)',
    )
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Do not skip ids already present in the eval cases file',
    )
    parser.add_argument(
        '--split',
        choices=('dev', 'holdout'),
        default='dev',
        help='Suggested split field for emitted rows (default: dev)',
    )

    sub = parser.add_subparsers(dest='mode', required=True)

    feed_p = sub.add_parser('feed', help='Sample an existing custom feed URI')
    feed_p.add_argument(
        '--feed',
        default=os.environ.get('FEED_URI'),
        help='Feed URI (default: FEED_URI env)',
    )
    feed_p.add_argument('--limit', type=int, default=50)

    authors_p = sub.add_parser('authors', help='Sample posts from allowlisted or named accounts')
    authors_p.add_argument(
        '--from-allowlist',
        action='store_true',
        help='Include handles/DIDs from data/allowlist_*.txt',
    )
    authors_p.add_argument('--handle', action='append', default=[], help='Handle to sample')
    authors_p.add_argument('--did', action='append', default=[], help='DID to sample')
    authors_p.add_argument('--limit-per-author', type=int, default=25)

    near_p = sub.add_parser(
        'near-miss',
        help='Search for ambiguous or false-friend phrases',
    )
    near_p.add_argument(
        '--query',
        action='append',
        default=[],
        help='Extra search query (repeatable); defaults include known FP phrases',
    )
    near_p.add_argument('--limit-per-query', type=int, default=20)
    near_p.add_argument(
        '--only-queries',
        action='store_true',
        help='Use only --query values (skip built-in near-miss presets)',
    )

    events_p = sub.add_parser('events', help='Search for in-scope event announcements')
    events_p.add_argument(
        '--query',
        action='append',
        default=[],
        help='Extra search query (repeatable); defaults include in-scope event cues',
    )
    events_p.add_argument('--limit-per-query', type=int, default=20)
    events_p.add_argument(
        '--only-queries',
        action='store_true',
        help='Use only --query values (skip built-in event presets)',
    )
    events_p.add_argument(
        '--require-event-cue',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Keep only posts with date/time/ticket-like cues (default: true)',
    )

    search_p = sub.add_parser('search', help='Free-form searchPosts sampling')
    search_p.add_argument('--query', action='append', required=True, help='Search query')
    search_p.add_argument('--limit-per-query', type=int, default=25)
    search_p.add_argument(
        '--signal',
        default='text',
        choices=('text', 'author', 'event'),
        help='Suggested signal stratum',
    )
    search_p.add_argument('--bucket', default='unspecified', help='Suggested bucket stratum')
    search_p.add_argument(
        '--require-event-cue',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Keep only posts with date/time/ticket-like cues',
    )

    return parser


def collect(args: argparse.Namespace, *, fetcher: Fetcher | None = None) -> list[JsonObject]:
    fetch = fetcher or default_fetcher
    skip_ids = set() if args.no_skip_existing else load_existing_ids(args.skip_existing)
    rows: list[JsonObject]

    if args.mode == 'feed':
        if not args.feed:
            raise ValueError('Pass --feed or set FEED_URI')
        posts = fetch_feed_posts(
            feed=args.feed,
            limit=args.limit,
            api_host=args.api_host,
            fetcher=fetch,
        )
        rows = [
            post_to_candidate(
                post,
                source='feed',
                signal='text',
                bucket='skyfeed_fp',
                split=args.split,
            )
            for post in posts
        ]
    elif args.mode == 'authors':
        actors = resolve_actors(args)
        if not actors:
            raise ValueError('Pass --from-allowlist and/or --handle/--did')
        rows = collect_from_authors(
            actors,
            limit_per_author=args.limit_per_author,
            split=args.split,
            api_host=args.api_host,
            fetcher=fetch,
        )
    elif args.mode == 'near-miss':
        queries = list(args.query)
        if not args.only_queries:
            queries = list(NEAR_MISS_QUERIES) + queries
        if not queries:
            raise ValueError('No near-miss queries configured')
        rows = collect_from_queries(
            queries,
            limit_per_query=args.limit_per_query,
            source='near-miss',
            signal='text',
            bucket='skyfeed_fp',
            split=args.split,
            require_event_cue=False,
            api_host=args.api_host,
            fetcher=fetch,
        )
    elif args.mode == 'events':
        queries = list(args.query)
        if not args.only_queries:
            queries = list(EVENT_QUERIES) + queries
        if not queries:
            raise ValueError('No event queries configured')
        rows = collect_from_queries(
            queries,
            limit_per_query=args.limit_per_query,
            source='events',
            signal='event',
            bucket='in_scope_event',
            split=args.split,
            require_event_cue=args.require_event_cue,
            api_host=args.api_host,
            fetcher=fetch,
        )
    elif args.mode == 'search':
        rows = collect_from_queries(
            list(args.query),
            limit_per_query=args.limit_per_query,
            source='search',
            signal=args.signal,
            bucket=args.bucket,
            split=args.split,
            require_event_cue=args.require_event_cue,
            api_host=args.api_host,
            fetcher=fetch,
        )
    else:
        raise ValueError(f'Unknown mode: {args.mode}')

    return dedupe_candidates(rows, skip_ids=skip_ids)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        rows = collect(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    print(f'# emitted {len(rows)} candidate rows', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
