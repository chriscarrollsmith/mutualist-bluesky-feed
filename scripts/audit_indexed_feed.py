#!/usr/bin/env python3
"""Rematch live feed / indexed SQLite posts against the current matcher.

Indexed rows store URI metadata only (no text). This tool hydrates records from
the Bluesky AppView, runs the regex floor with production allowlists/blocklists
(and optional soft priors from ``AuthorDecision``), and reports posts that
would no longer keep. The live decider is not called. Use ``--purge`` on a
database path to delete those URIs.

Examples:

    # What subscribers currently see (AppView getFeed)
    uv run python scripts/audit_indexed_feed.py --source feed \\
      --feed "$FEED_URI" --limit 100

    # Full indexed set from a local SQLite copy
    uv run python scripts/audit_indexed_feed.py --source db \\
      --database ./feed_database.db --apply-soft-priors

    # Delete would-drop rows from that database
    uv run python scripts/audit_indexed_feed.py --source db \\
      --database ./feed_database.db --apply-soft-priors --purge
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.allowlists import (  # noqa: E402
    load_allowlist_dids,
    load_allowlist_handles,
    load_blocklist_dids,
    load_blocklist_handles,
)
from server.matcher import extract_alt_text, match_post  # noqa: E402

DEFAULT_API_HOST = 'https://api.bsky.app'
USER_AGENT = 'mutualist-bluesky-feed-audit/0.1 (+https://github.com/chriscarrollsmith/mutualist-bluesky-feed)'
GET_POSTS_BATCH = 25


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {'1', 'true', 't', 'yes', 'y'}


# Mirror server.config defaults without requiring FEEDGEN_HOSTNAME / FEED_URI.
IGNORE_REPLY_POSTS = _env_bool('IGNORE_REPLY_POSTS', True)
SOFT_PRIOR_MIN_KEEPS = int(
    os.environ.get('SOFT_PRIOR_MIN_KEEPS', os.environ.get('SOFT_PRIOR_MIN_STRONG', '3'))
)
SOFT_PRIOR_WINDOW_DAYS = int(os.environ.get('SOFT_PRIOR_WINDOW_DAYS', '30'))
MUTED_KEYWORDS = tuple(
    part.strip().lower() for part in os.environ.get('MUTED_KEYWORDS', '').split(',') if part.strip()
)

JsonObject = dict[str, Any]
Fetcher = Callable[[str], JsonObject]


@dataclass(frozen=True)
class AuditRow:
    uri: str
    author_did: str | None
    author_handle: str | None
    text_preview: str
    indexed_reason: str | None
    matched: bool
    reason: str
    status: str  # ok | not_found | error


def api_url(host: str, xrpc: str, params: dict[str, Any]) -> str:
    base = host.rstrip('/') + '/xrpc/' + xrpc.lstrip('/')
    return base + '?' + urllib.parse.urlencode(params, doseq=True)


def default_fetcher(url: str, *, timeout: float = 30.0) -> JsonObject:
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def chunked(items: Sequence[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def load_uris_from_db(database: Path, *, limit: int | None = None) -> list[tuple[str, str | None]]:
    """Return (uri, indexed match_reason) pairs ordered by indexed_at desc."""
    con = sqlite3.connect(str(database))
    try:
        sql = 'SELECT uri, match_reason FROM post ORDER BY indexed_at DESC'
        if limit is not None:
            sql += f' LIMIT {int(limit)}'
        return [(str(uri), reason) for uri, reason in con.execute(sql)]
    finally:
        con.close()


def _parse_sqlite_ts(value: object) -> datetime | None:
    if isinstance(value, datetime):
        ts = value
    elif isinstance(value, str):
        try:
            ts = datetime.fromisoformat(value)
        except ValueError:
            return None
    else:
        return None
    if ts.tzinfo is not None:
        ts = ts.astimezone(UTC).replace(tzinfo=None)
    return ts


def load_soft_prior_dids_from_db(
    database: Path,
    *,
    min_keeps: int = SOFT_PRIOR_MIN_KEEPS,
    window_days: int = SOFT_PRIOR_WINDOW_DAYS,
    now: datetime | None = None,
) -> set[str]:
    """DIDs with enough recent decider keeps and no rejections in this DB."""
    moment = now if now is not None else datetime.now(UTC).replace(tzinfo=None)
    cutoff = moment - timedelta(days=window_days)
    con = sqlite3.connect(str(database))
    try:
        try:
            rows = con.execute('SELECT author_did, kept, decided_at FROM authordecision').fetchall()
        except sqlite3.OperationalError:
            return set()
    finally:
        con.close()

    keeps: dict[str, int] = {}
    rejects: dict[str, int] = {}
    for did, kept, decided_at in rows:
        if not did:
            continue
        ts = _parse_sqlite_ts(decided_at)
        if ts is None or ts < cutoff:
            continue
        key = str(did)
        if int(kept):
            keeps[key] = keeps.get(key, 0) + 1
        else:
            rejects[key] = rejects.get(key, 0) + 1
    return {did for did, count in keeps.items() if count >= min_keeps and rejects.get(did, 0) == 0}


def fetch_feed_uris(
    *,
    feed: str,
    limit: int,
    api_host: str,
    fetcher: Fetcher,
) -> list[tuple[str, str | None]]:
    """Page getFeed; indexed_reason is unknown from AppView alone."""
    out: list[tuple[str, str | None]] = []
    cursor: str | None = None
    while len(out) < limit:
        params: dict[str, Any] = {
            'feed': feed,
            'limit': max(1, min(100, limit - len(out))),
        }
        if cursor:
            params['cursor'] = cursor
        payload = fetcher(api_url(api_host, 'app.bsky.feed.getFeed', params))
        batch = 0
        for item in payload.get('feed') or []:
            if not isinstance(item, dict):
                continue
            post = item.get('post')
            if not isinstance(post, dict):
                continue
            uri = post.get('uri')
            if isinstance(uri, str) and uri:
                out.append((uri, None))
                batch += 1
                if len(out) >= limit:
                    break
        if len(out) >= limit:
            break
        cursor = payload.get('cursor')
        if not cursor or batch == 0:
            break
    return out


def fetch_posts_by_uri(
    uris: Sequence[str],
    *,
    api_host: str,
    fetcher: Fetcher,
    pause_s: float = 0.05,
) -> dict[str, JsonObject]:
    """Hydrate posts via getPosts (max 25 URIs per request)."""
    found: dict[str, JsonObject] = {}
    for batch in chunked(list(uris), GET_POSTS_BATCH):
        params = [('uris', uri) for uri in batch]
        url = (
            api_host.rstrip('/') + '/xrpc/app.bsky.feed.getPosts?' + urllib.parse.urlencode(params)
        )
        try:
            payload = fetcher(url)
        except urllib.error.HTTPError as exc:
            # Retry once on 429 / 5xx
            if exc.code in {429, 500, 502, 503, 504}:
                time.sleep(1.0)
                payload = fetcher(url)
            else:
                raise
        for post in payload.get('posts') or []:
            if isinstance(post, dict) and isinstance(post.get('uri'), str):
                found[post['uri']] = post
        if pause_s:
            time.sleep(pause_s)
    return found


def post_alt_text(post: JsonObject) -> str:
    record = post.get('record') if isinstance(post.get('record'), dict) else {}
    chunks = [
        extract_alt_text(record.get('embed') if isinstance(record, dict) else None),
        extract_alt_text(post.get('embed')),
    ]
    # AppView external view sometimes nests under embed.external only — already
    # covered by extract_alt_text; join unique non-empty chunks.
    seen: set[str] = set()
    ordered: list[str] = []
    for chunk in chunks:
        text = chunk.strip()
        if text and text not in seen:
            seen.add(text)
            ordered.append(text)
    return ' '.join(ordered)


def is_muted(text: str, alt_text: str) -> bool:
    if not MUTED_KEYWORDS:
        return False
    haystack = f'{text} {alt_text}'.lower()
    return any(keyword in haystack for keyword in MUTED_KEYWORDS)


def evaluate_post(
    post: JsonObject,
    *,
    allowlist_dids: set[str],
    allowlist_handles: set[str],
    soft_prior_dids: set[str],
    ignore_replies: bool = IGNORE_REPLY_POSTS,
    blocklist_dids: set[str] | None = None,
    blocklist_handles: set[str] | None = None,
) -> tuple[bool, str]:
    """Return (matched, reason) using the same gates as indexer.create."""
    record = post.get('record') if isinstance(post.get('record'), dict) else {}
    author = post.get('author') if isinstance(post.get('author'), dict) else {}
    if ignore_replies and isinstance(record, dict) and record.get('reply'):
        return False, 'reply_ignored'

    text = str(record.get('text') or '') if isinstance(record, dict) else ''
    alt_text = post_alt_text(post)
    if is_muted(text, alt_text):
        return False, 'muted'

    langs = record.get('langs') if isinstance(record, dict) else None
    if not isinstance(langs, list):
        langs = None

    result = match_post(
        text,
        alt_text=alt_text,
        langs=langs,
        author_did=author.get('did') if isinstance(author, dict) else None,
        author_handle=author.get('handle') if isinstance(author, dict) else None,
        allowlist_dids=allowlist_dids,
        allowlist_handles=allowlist_handles,
        blocklist_dids=blocklist_dids or set(),
        blocklist_handles=blocklist_handles or set(),
        soft_prior_dids=soft_prior_dids,
        embed=post.get('embed') if isinstance(post.get('embed'), dict) else None,
    )
    return result.matched, result.reason


def audit_uris(
    uri_rows: Sequence[tuple[str, str | None]],
    *,
    api_host: str,
    fetcher: Fetcher,
    allowlist_dids: set[str],
    allowlist_handles: set[str],
    soft_prior_dids: set[str],
    blocklist_dids: set[str] | None = None,
    blocklist_handles: set[str] | None = None,
) -> list[AuditRow]:
    uris = [uri for uri, _ in uri_rows]
    indexed = {uri: reason for uri, reason in uri_rows}
    hydrated = fetch_posts_by_uri(uris, api_host=api_host, fetcher=fetcher)
    rows: list[AuditRow] = []
    for uri in uris:
        post = hydrated.get(uri)
        if post is None:
            rows.append(
                AuditRow(
                    uri=uri,
                    author_did=None,
                    author_handle=None,
                    text_preview='',
                    indexed_reason=indexed.get(uri),
                    matched=False,
                    reason='not_found',
                    status='not_found',
                )
            )
            continue
        author_obj = post.get('author')
        author: JsonObject = author_obj if isinstance(author_obj, dict) else {}
        record_obj = post.get('record')
        record: JsonObject = record_obj if isinstance(record_obj, dict) else {}
        text = str(record.get('text') or '')
        matched, reason = evaluate_post(
            post,
            allowlist_dids=allowlist_dids,
            allowlist_handles=allowlist_handles,
            soft_prior_dids=soft_prior_dids,
            blocklist_dids=blocklist_dids,
            blocklist_handles=blocklist_handles,
        )
        did = author.get('did')
        handle = author.get('handle')
        rows.append(
            AuditRow(
                uri=uri,
                author_did=str(did) if did else None,
                author_handle=str(handle) if handle else None,
                text_preview=text.replace('\n', ' ')[:160],
                indexed_reason=indexed.get(uri),
                matched=matched,
                reason=reason,
                status='ok',
            )
        )
    return rows


def purge_uris(database: Path, uris: Sequence[str]) -> int:
    if not uris:
        return 0
    con = sqlite3.connect(str(database))
    try:
        cur = con.cursor()
        deleted = 0
        for batch in chunked(list(uris), 200):
            placeholders = ','.join('?' * len(batch))
            cur.execute(f'DELETE FROM post WHERE uri IN ({placeholders})', batch)
            deleted += cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else 0
        con.commit()
        # Checkpoint WAL so a subsequent sftp of the main file is consistent.
        cur.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        return deleted
    finally:
        con.close()


def print_report(rows: Sequence[AuditRow]) -> None:
    keeps = [r for r in rows if r.matched]
    drops = [r for r in rows if not r.matched]
    print(f'Audited {len(rows)} posts')
    print(f'Would keep: {len(keeps)}')
    print(f'Would drop: {len(drops)}')
    print()
    print('Drop reason counts:')
    for reason, n in Counter(r.reason for r in drops).most_common():
        print(f'  {n:3d}  {reason}')
    print()
    print('Keep reason counts:')
    for reason, n in Counter(r.reason for r in keeps).most_common():
        print(f'  {n:3d}  {reason}')
    if drops:
        print()
        print('--- Would-drop ---')
        for row in drops:
            handle = f'@{row.author_handle}' if row.author_handle else '(unknown)'
            indexed = row.indexed_reason or '?'
            print(f'- {handle} indexed={indexed} now={row.reason} {row.text_preview!r}')
            print(f'  {row.uri}')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--source',
        choices=('feed', 'db'),
        required=True,
        help='Audit AppView getFeed posts, or URIs from a SQLite database',
    )
    parser.add_argument(
        '--feed',
        default=os.environ.get('FEED_URI'),
        help='Feed AT URI for --source feed (default: FEED_URI)',
    )
    parser.add_argument(
        '--database',
        type=Path,
        help='SQLite path for --source db and/or --purge',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Max posts to audit (feed default: 100; db default: all)',
    )
    parser.add_argument(
        '--api-host',
        default=os.environ.get('BSKY_API_HOST', DEFAULT_API_HOST),
        help=f'Bluesky AppView host (default: {DEFAULT_API_HOST})',
    )
    parser.add_argument(
        '--apply-soft-priors',
        action='store_true',
        help='Load eligible soft-prior DIDs from --database authordecision rows',
    )
    parser.add_argument(
        '--jsonl',
        type=Path,
        help='Write per-post audit rows as JSON lines',
    )
    parser.add_argument(
        '--purge',
        action='store_true',
        help='Delete would-drop URIs from --database (requires --source db)',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='With --purge, report deletes without writing',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    allowlist_dids = load_allowlist_dids()
    allowlist_handles = load_allowlist_handles()
    blocklist_dids = load_blocklist_dids()
    blocklist_handles = load_blocklist_handles()

    if args.source == 'feed':
        if not args.feed:
            print('error: --feed or FEED_URI is required for --source feed', file=sys.stderr)
            return 2
        limit = 100 if args.limit is None else args.limit
        uri_rows = fetch_feed_uris(
            feed=args.feed,
            limit=limit,
            api_host=args.api_host,
            fetcher=default_fetcher,
        )
    else:
        if not args.database:
            print('error: --database is required for --source db', file=sys.stderr)
            return 2
        if not args.database.is_file():
            print(f'error: database not found: {args.database}', file=sys.stderr)
            return 2
        uri_rows = load_uris_from_db(args.database, limit=args.limit)

    soft_prior_dids: set[str] = set()
    if args.apply_soft_priors:
        if not args.database:
            print('error: --apply-soft-priors requires --database', file=sys.stderr)
            return 2
        soft_prior_dids = load_soft_prior_dids_from_db(args.database)

    rows = audit_uris(
        uri_rows,
        api_host=args.api_host,
        fetcher=default_fetcher,
        allowlist_dids=allowlist_dids,
        allowlist_handles=allowlist_handles,
        soft_prior_dids=soft_prior_dids,
        blocklist_dids=blocklist_dids,
        blocklist_handles=blocklist_handles,
    )
    print_report(rows)

    if args.jsonl:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl.open('w', encoding='utf-8') as fh:
            for row in rows:
                fh.write(json.dumps(asdict(row), ensure_ascii=False) + '\n')
        print(f'\nWrote {len(rows)} rows to {args.jsonl}')

    drops = [r.uri for r in rows if not r.matched]
    if args.purge:
        if args.source != 'db' or not args.database:
            print('error: --purge requires --source db and --database', file=sys.stderr)
            return 2
        if args.dry_run:
            print(f'\nDry-run: would delete {len(drops)} posts from {args.database}')
            return 0
        deleted = purge_uris(args.database, drops)
        print(f'\nDeleted {deleted} posts from {args.database}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
