#!/usr/bin/env python3
"""Backfill indexed posts for a time gap via Bluesky AppView (no Jetstream).

Use after a live cursor skip, or whenever Jetstream missed a window. Pulls from
allowlist author timelines and/or ``searchPosts``, maps each hit into a
Jetstream-shaped event, and runs ``server.indexer.handle_event``.

Does **not** touch ``SubscriptionState``. Safe to re-run (URI ``on_conflict_ignore``).

Examples:

    # Dry-run the recent gap against a local DB copy
    uv run python scripts/backfill_gap.py \\
      --since 2026-07-30T16:00:00Z --until 2026-07-31T15:35:00Z \\
      --source both --database ./feed_database.db --dry-run

    # Write matches (on Fly: DATABASE_PATH=/data/feed_database.db)
    uv run python scripts/backfill_gap.py \\
      --since 2026-07-30T16:00:00Z --until 2026-07-31T15:35:00Z \\
      --source both --database /data/feed_database.db
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.allowlists import load_allowlist_dids  # noqa: E402

DEFAULT_API_HOST = 'https://api.bsky.app'
USER_AGENT = 'mutualist-bluesky-feed-backfill/0.1 (+https://github.com/chriscarrollsmith/mutualist-bluesky-feed)'

# High-precision mutualist queries for AppView search (latest sort).
# Kept in sync with server.fn_search.HIGH_PRECISION_QUERIES. The daily
# false-negative review refuses these: they mostly return posts the matcher
# already keeps.
DEFAULT_SEARCH_QUERIES = (
    'mutualism',
    'mutualist',
    'Proudhon',
    'mutual credit',
    'occupancy and use',
    'agorism',
    'free-market anti-capitalism',
    'left-wing market anarchism',
    'Kropotkin',
    'Benjamin Tucker',
    'Josiah Warren',
    'C4SS',
    '#mutualism',
)

JsonObject = dict[str, Any]
Fetcher = Callable[[str], JsonObject]


def parse_iso_datetime(value: str) -> datetime:
    """Parse CLI / Bluesky timestamps to naive UTC."""
    normalized = value.strip().replace('Z', '+00:00')
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def created_at_from_post(post: JsonObject) -> datetime | None:
    record = post.get('record') if isinstance(post.get('record'), dict) else {}
    raw = record.get('createdAt') if isinstance(record, dict) else None
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return parse_iso_datetime(raw)
    except ValueError:
        return None


def in_window(
    created_at: datetime | None,
    *,
    since: datetime,
    until: datetime,
) -> bool:
    if created_at is None:
        return False
    return since <= created_at <= until


def appview_post_to_event(post: JsonObject) -> dict[str, Any] | None:
    """Map an AppView post view to a Jetstream-shaped create event."""
    uri = post.get('uri')
    cid = post.get('cid')
    author = post.get('author') if isinstance(post.get('author'), dict) else {}
    author_did = author.get('did') if isinstance(author, dict) else None
    record = post.get('record') if isinstance(post.get('record'), dict) else None
    if not isinstance(uri, str) or not uri:
        return None
    if not isinstance(author_did, str) or not author_did:
        return None
    if not isinstance(record, dict):
        return None

    created_at = created_at_from_post(post)
    if created_at is None:
        return None
    time_us = int(created_at.replace(tzinfo=UTC).timestamp() * 1_000_000)
    return {
        'operation': 'create',
        'uri': uri,
        'cid': cid if isinstance(cid, str) else '',
        'author': author_did,
        'time_us': time_us,
        'record': record,
    }


def api_url(host: str, xrpc: str, params: dict[str, Any]) -> str:
    base = host.rstrip('/') + '/xrpc/' + xrpc.lstrip('/')
    return base + '?' + urllib.parse.urlencode(params)


def default_fetcher(url: str, *, timeout: float = 30.0) -> JsonObject:
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        if exc.code in {429, 500, 502, 503, 504}:
            time.sleep(1.5)
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                return json.loads(resp.read().decode('utf-8'))
        raise


def extract_author_feed_posts(payload: JsonObject, *, actor: str) -> list[JsonObject]:
    actor_key = actor.strip().lower()
    posts: list[JsonObject] = []
    for item in payload.get('feed') or []:
        if not isinstance(item, dict):
            continue
        if item.get('reason') is not None:
            continue
        post = item.get('post')
        if not isinstance(post, dict):
            continue
        author = post.get('author') or {}
        handle = str(author.get('handle') or '').lower()
        did = str(author.get('did') or '').lower()
        if actor_key not in {handle, did}:
            continue
        posts.append(post)
    return posts


def extract_search_posts(payload: JsonObject) -> list[JsonObject]:
    posts: list[JsonObject] = []
    for post in payload.get('posts') or []:
        if isinstance(post, dict):
            posts.append(post)
    return posts


def iter_author_posts(
    actor: str,
    *,
    since: datetime,
    until: datetime,
    api_host: str,
    fetcher: Fetcher,
    pause_s: float = 0.05,
) -> Iterator[JsonObject]:
    """Page getAuthorFeed newest-first until posts fall before ``since``."""
    cursor: str | None = None
    while True:
        params: dict[str, Any] = {
            'actor': actor,
            'limit': 100,
            'filter': 'posts_no_replies',
        }
        if cursor:
            params['cursor'] = cursor
        payload = fetcher(api_url(api_host, 'app.bsky.feed.getAuthorFeed', params))
        batch = extract_author_feed_posts(payload, actor=actor)
        if not batch:
            break

        stop = False
        for post in batch:
            created_at = created_at_from_post(post)
            if created_at is not None and created_at < since:
                stop = True
                break
            if in_window(created_at, since=since, until=until):
                yield post

        if stop:
            break
        cursor = payload.get('cursor')
        if not isinstance(cursor, str) or not cursor:
            break
        if pause_s:
            time.sleep(pause_s)


def iter_search_posts(
    query: str,
    *,
    since: datetime,
    until: datetime,
    api_host: str,
    fetcher: Fetcher,
    pause_s: float = 0.05,
    max_pages: int = 20,
) -> Iterator[JsonObject]:
    """Page searchPosts (latest) bounded by since/until query params + filter."""
    cursor: str | None = None
    since_param = since.replace(tzinfo=UTC).isoformat().replace('+00:00', 'Z')
    until_param = until.replace(tzinfo=UTC).isoformat().replace('+00:00', 'Z')
    for _ in range(max_pages):
        params: dict[str, Any] = {
            'q': query,
            'limit': 100,
            'sort': 'latest',
            'since': since_param,
            'until': until_param,
        }
        if cursor:
            params['cursor'] = cursor
        payload = fetcher(api_url(api_host, 'app.bsky.feed.searchPosts', params))
        batch = extract_search_posts(payload)
        if not batch:
            break

        stop = False
        yielded = 0
        for post in batch:
            created_at = created_at_from_post(post)
            if created_at is not None and created_at < since:
                stop = True
                break
            if in_window(created_at, since=since, until=until):
                yielded += 1
                yield post

        if stop or yielded == 0:
            # No in-window hits left (or only older rows); stop paging.
            if stop:
                break
            # If the page was entirely after ``until``, keep going; else stop.
            newest = created_at_from_post(batch[0])
            if newest is not None and newest > until:
                cursor = payload.get('cursor')
                if not isinstance(cursor, str) or not cursor:
                    break
                if pause_s:
                    time.sleep(pause_s)
                continue
            break

        cursor = payload.get('cursor')
        if not isinstance(cursor, str) or not cursor:
            break
        if pause_s:
            time.sleep(pause_s)


def bind_database(path: Path) -> None:
    """Point Peewee at ``path`` after config/dotenv have already loaded."""
    from server.database import AuthorDecision, AuthorLocalStats, Post, SubscriptionState, db

    if not db.is_closed():
        db.close()
    db.init(
        str(path),
        pragmas={
            'journal_mode': 'wal',
            'synchronous': 'normal',
        },
    )
    db.connect(reuse_if_open=True)
    db.create_tables([Post, SubscriptionState, AuthorLocalStats, AuthorDecision], safe=True)


def set_indexed_at_to_created(uri: str, created_at: datetime) -> None:
    from server.database import Post

    Post.update(indexed_at=created_at).where(Post.uri == uri).execute()


def uri_exists(uri: str) -> bool:
    from server.database import Post

    return Post.select().where(Post.uri == uri).exists()


def process_event(
    event: dict[str, Any],
    *,
    dry_run: bool,
    indexed_at_mode: str,
) -> str:
    """Return status: indexed | dry_run_match | dry_run_skip | exists | skipped."""
    from server import config
    from server.indexer import handle_event
    from server.matcher import extract_alt_text, match_post

    uri = str(event['uri'])
    if uri_exists(uri):
        return 'exists'

    if dry_run:
        record = event.get('record') or {}
        if config.IGNORE_REPLY_POSTS and record.get('reply'):
            return 'dry_run_skip'
        text = record.get('text') or ''
        embed = record.get('embed')
        alt_text = extract_alt_text(embed)
        langs = record.get('langs') or []
        if not isinstance(langs, list):
            langs = []
        result = match_post(
            text,
            alt_text=alt_text,
            langs=langs,
            author_did=event.get('author'),
            allowlist_dids=config.ALLOWLIST_DIDS,
            allowlist_handles=config.ALLOWLIST_HANDLES,
            blocklist_dids=config.BLOCKLIST_DIDS,
            blocklist_handles=config.BLOCKLIST_HANDLES,
            embed=embed,
            use_default_decider=True,
        )
        return 'dry_run_match' if result.matched else 'dry_run_skip'

    before = uri_exists(uri)
    handle_event(event)
    if before or not uri_exists(uri):
        return 'skipped'

    if indexed_at_mode == 'created':
        created_at = created_at_from_post({'record': event.get('record') or {}})
        if created_at is not None:
            set_indexed_at_to_created(uri, created_at)
    return 'indexed'


def collect_posts(
    *,
    source: str,
    since: datetime,
    until: datetime,
    api_host: str,
    fetcher: Fetcher,
    search_queries: tuple[str, ...],
    actors: Iterable[str],
) -> list[JsonObject]:
    by_uri: dict[str, JsonObject] = {}

    if source in {'authors', 'both'}:
        for actor in actors:
            try:
                for post in iter_author_posts(
                    actor,
                    since=since,
                    until=until,
                    api_host=api_host,
                    fetcher=fetcher,
                ):
                    uri = post.get('uri')
                    if isinstance(uri, str) and uri:
                        by_uri[uri] = post
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                print(f'warn: author fetch failed for {actor}: {exc}', file=sys.stderr)

    if source in {'search', 'both'}:
        for query in search_queries:
            try:
                for post in iter_search_posts(
                    query,
                    since=since,
                    until=until,
                    api_host=api_host,
                    fetcher=fetcher,
                ):
                    uri = post.get('uri')
                    if isinstance(uri, str) and uri:
                        by_uri.setdefault(uri, post)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                print(f'warn: search failed for {query!r}: {exc}', file=sys.stderr)

    return list(by_uri.values())


def load_posts_jsonl(path: Path) -> list[JsonObject]:
    posts: list[JsonObject] = []
    with path.open(encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{line_no}: invalid JSON') from exc
            if isinstance(row, dict):
                posts.append(row)
    return posts


def dump_posts_jsonl(path: Path, posts: Iterable[JsonObject]) -> int:
    count = 0
    with path.open('w', encoding='utf-8') as handle:
        for post in posts:
            handle.write(json.dumps(post, ensure_ascii=False) + '\n')
            count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--since',
        required=True,
        help='Inclusive window start (ISO-8601, e.g. 2026-07-30T16:00:00Z)',
    )
    parser.add_argument(
        '--until',
        required=True,
        help='Inclusive window end (ISO-8601)',
    )
    parser.add_argument(
        '--source',
        choices=('authors', 'search', 'both'),
        default='both',
        help='AppView source (default: both)',
    )
    parser.add_argument(
        '--database',
        type=Path,
        default=None,
        help='SQLite path (default: DATABASE_PATH / config)',
    )
    parser.add_argument(
        '--api-host',
        default=os.environ.get('BSKY_API_HOST', DEFAULT_API_HOST),
        help=f'AppView host (default: {DEFAULT_API_HOST})',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Fetch and match without writing to the database',
    )
    parser.add_argument(
        '--indexed-at',
        choices=('created', 'now'),
        default='created',
        help='indexed_at for newly inserted rows (default: created, for natural order)',
    )
    parser.add_argument(
        '--query',
        action='append',
        default=[],
        help='Extra searchPosts query (repeatable); replaces defaults if any given',
    )
    parser.add_argument(
        '--dump-jsonl',
        type=Path,
        default=None,
        help='Write fetched AppView posts as JSONL (for offline indexing elsewhere)',
    )
    parser.add_argument(
        '--from-jsonl',
        type=Path,
        default=None,
        help='Index posts from JSONL instead of calling AppView (skips --source fetch)',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    since = parse_iso_datetime(args.since)
    until = parse_iso_datetime(args.until)
    if until < since:
        print('error: --until must be >= --since', file=sys.stderr)
        return 2

    # Ensure config can import when run outside the feed server env.
    os.environ.setdefault('FEEDGEN_HOSTNAME', 'localhost')
    os.environ.setdefault('SERVICE_DID', 'did:web:localhost')
    os.environ.setdefault(
        'FEED_URI',
        'at://did:plc:test/app.bsky.feed.generator/test',
    )

    db_path = args.database
    if db_path is None:
        from server import config

        db_path = Path(config.DATABASE_PATH)
    bind_database(db_path)

    queries = tuple(args.query) if args.query else DEFAULT_SEARCH_QUERIES
    actors = sorted(load_allowlist_dids())

    print(
        f'backfill window {since.isoformat()}Z .. {until.isoformat()}Z '
        f'source={args.source} dry_run={args.dry_run} database={db_path}',
        file=sys.stderr,
    )

    if args.from_jsonl is not None:
        posts = load_posts_jsonl(args.from_jsonl)
        print(f'loaded jsonl: {len(posts)} from {args.from_jsonl}', file=sys.stderr)
    else:
        if args.source in {'authors', 'both'}:
            print(f'allowlist actors: {len(actors)}', file=sys.stderr)
        posts = collect_posts(
            source=args.source,
            since=since,
            until=until,
            api_host=args.api_host,
            fetcher=default_fetcher,
            search_queries=queries,
            actors=actors,
        )
        print(f'candidates: {len(posts)}', file=sys.stderr)
        if args.dump_jsonl is not None:
            n = dump_posts_jsonl(args.dump_jsonl, posts)
            print(f'wrote jsonl: {n} -> {args.dump_jsonl}', file=sys.stderr)

    counts = {
        'indexed': 0,
        'dry_run_match': 0,
        'dry_run_skip': 0,
        'exists': 0,
        'skipped': 0,
        'bad_payload': 0,
    }
    for post in posts:
        event = appview_post_to_event(post)
        if event is None:
            counts['bad_payload'] += 1
            continue
        if not in_window(
            created_at_from_post(post),
            since=since,
            until=until,
        ):
            continue
        status = process_event(
            event,
            dry_run=args.dry_run,
            indexed_at_mode=args.indexed_at,
        )
        counts[status] = counts.get(status, 0) + 1

    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
