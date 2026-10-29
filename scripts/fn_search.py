#!/usr/bin/env python3
"""Plan, fetch, and record the daily false-negative AppView search.

Do not pass the high-precision backfill presets (``mutualism``, ``Proudhon``,
and the rest of ``DEFAULT_SEARCH_QUERIES``). The planner reads the strategy catalog
and the running log, then returns a slate that exploits queries which have already
surfaced false negatives and explores underused families.

``search`` fetches one unauthenticated page per query. It does not send
``cursor``. A leading ``#`` is stripped before the request. This command does
not record a result.

Examples:

    uv run python scripts/fn_search.py summary
    uv run python scripts/fn_search.py plan
    uv run python scripts/fn_search.py search --plan
    uv run python scripts/fn_search.py record \\
      --query "tool library" --family practice \\
      --hits 4 --false-negatives 1 \\
      --false-negative-id at://did:plc:example/app.bsky.feed.post/abc
    uv run python scripts/fn_search.py record \\
      --query "mutual aid" --family bare_term --status blocked
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
from datetime import UTC, datetime, timedelta
from pathlib import Path
from random import Random
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.appview_search import (  # noqa: E402
    DEFAULT_API_HOST,
    FN_SEARCH_USER_AGENT,
    fetch_json,
    search_posts_params,
    search_posts_url,
    search_query_text,
)
from server.fn_search import (  # noqa: E402
    DEFAULT_CATALOG_PATH,
    DEFAULT_LOG_PATH,
    SearchLogEntry,
    SearchStrategy,
    _reject_high_precision,
    load_catalog,
    load_log,
    parse_timestamp,
    plan_report,
    record_search,
    summary_report,
)
from server.matcher import extract_alt_text  # noqa: E402


def _daily_seed(now: datetime) -> int:
    return int(now.strftime('%Y%m%d'))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument('--log', type=Path, default=DEFAULT_LOG_PATH)
    sub = parser.add_subparsers(dest='command', required=True)

    plan = sub.add_parser('plan', help="Print today's exploit/explore query slate")
    plan.add_argument('--count', type=int, default=8, help='Queries to plan (default: 8)')
    plan.add_argument(
        '--seed',
        type=int,
        default=None,
        help='RNG seed (default: UTC date, so a rerun the same day is stable)',
    )
    plan.add_argument(
        '--now',
        default=None,
        help='Override the clock (ISO-8601) for the recency penalty and default seed',
    )

    sub.add_parser('summary', help='Print per-query false-negative totals and family entropy')

    record = sub.add_parser('record', help='Append one query result to the running log')
    record.add_argument('--query', required=True)
    record.add_argument('--family', required=True)
    record.add_argument('--status', choices=('ok', 'blocked'), default='ok')
    record.add_argument('--hits', type=int, default=0)
    record.add_argument('--false-negatives', type=int, default=0)
    record.add_argument(
        '--false-negative-id',
        action='append',
        default=[],
        help='URI of a post this query surfaced that the feed should have included',
    )
    record.add_argument('--notes', default='')
    record.add_argument('--used-at', default=None, help='ISO-8601 timestamp (default: now)')
    record.add_argument(
        '--no-remember',
        action='store_true',
        help='Do not add a new successful query to the strategy catalog',
    )

    search = sub.add_parser(
        'search',
        help='Fetch one unauthenticated searchPosts page per query',
    )
    search.add_argument(
        '--query',
        action='append',
        default=[],
        help='Query to fetch (repeatable). A leading # is stripped.',
    )
    search.add_argument(
        '--plan',
        action='store_true',
        help="Fetch today's planned slate instead of --query",
    )
    search.add_argument('--count', type=int, default=8, help='Slate size when using --plan')
    search.add_argument('--seed', type=int, default=None, help='RNG seed for --plan')
    search.add_argument('--now', default=None, help='ISO-8601 clock for the window and --plan')
    search.add_argument('--since', default=None, help='Window start (ISO-8601, default: 24h ago)')
    search.add_argument('--until', default=None, help='Window end (ISO-8601, default: now)')
    search.add_argument('--limit', type=int, default=100, help='Page size, 1 to 100')
    search.add_argument('--api-host', default=DEFAULT_API_HOST)
    return parser


def _compact_post(post: dict[str, Any]) -> dict[str, Any]:
    record_raw = post.get('record')
    author_raw = post.get('author')
    record = record_raw if isinstance(record_raw, dict) else {}
    author = author_raw if isinstance(author_raw, dict) else {}
    embed = record.get('embed')
    if embed is None:
        embed = post.get('embed')
    text = record.get('text')
    created_at = record.get('createdAt')
    return {
        'uri': post.get('uri'),
        'did': author.get('did'),
        'handle': author.get('handle'),
        'created_at': created_at if isinstance(created_at, str) else None,
        'text': text if isinstance(text, str) else '',
        'reply': bool(record.get('reply')),
        'alt': extract_alt_text(embed),
    }


def fetch_query_page(
    query: str,
    *,
    since: datetime,
    until: datetime,
    limit: int,
    api_host: str,
    family: str | None = None,
    role: str | None = None,
) -> dict[str, Any]:
    """Fetch the first ``searchPosts`` page. Never sends ``cursor``."""
    sent = search_query_text(query)
    _reject_high_precision(sent)
    params = search_posts_params(
        sent,
        since=since,
        until=until,
        limit=limit,
        cursor=None,
        access_token=None,
    )
    url = search_posts_url(api_host, params)
    base = {
        'query': query,
        'query_sent': sent,
        'stripped_hash': sent != ' '.join(query.split()),
        'family': family,
        'role': role,
        'auth': 'none',
        'pages': 1,
        'paging': 'first-page',
    }
    try:
        payload = fetch_json(url, user_agent=FN_SEARCH_USER_AGENT)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='replace').replace('\n', ' ').strip()[:160]
        return {
            **base,
            'status': 'blocked',
            'http': exc.code,
            'pages': 0,
            'fetched': 0,
            'truncated': False,
            'error': f'HTTP {exc.code} {detail}'.strip(),
            'posts': [],
        }
    except urllib.error.URLError as exc:
        return {
            **base,
            'status': 'blocked',
            'http': None,
            'pages': 0,
            'fetched': 0,
            'truncated': False,
            'error': str(exc.reason),
            'posts': [],
        }

    posts_raw = payload.get('posts')
    raw_posts = posts_raw if isinstance(posts_raw, list) else []
    posts = [_compact_post(post) for post in raw_posts if isinstance(post, dict)]
    cursor = payload.get('cursor')
    return {
        **base,
        'status': 'ok',
        'http': 200,
        'fetched': len(posts),
        'truncated': isinstance(cursor, str) and bool(cursor),
        'error': None,
        'posts': posts,
    }


def _planned_search_rows(
    catalog: tuple[SearchStrategy, ...],
    log: tuple[SearchLogEntry, ...],
    *,
    count: int,
    seed: int | None,
    now: datetime,
) -> list[dict[str, str | None]]:
    report = plan_report(
        catalog,
        log,
        slate_size=count,
        rng=Random(_daily_seed(now) if seed is None else seed),
        now=now,
    )
    rows: list[dict[str, str | None]] = []
    for item in report['queries']:
        rows.append(
            {
                'query': str(item['query']),
                'family': str(item['family']) if item.get('family') else None,
                'role': str(item['role']) if item.get('role') else None,
            }
        )
    return rows


def _search_command(
    args: argparse.Namespace,
    catalog: tuple[SearchStrategy, ...],
    log: tuple[SearchLogEntry, ...],
) -> int:
    if args.plan and args.query:
        raise ValueError('pass --plan or --query, not both')
    now = parse_timestamp(args.now) if args.now else datetime.now(UTC)
    until = parse_timestamp(args.until) if args.until else now
    since = parse_timestamp(args.since) if args.since else until - timedelta(hours=24)
    if until < since:
        raise ValueError('--until must be >= --since')
    if args.plan:
        rows = _planned_search_rows(catalog, log, count=args.count, seed=args.seed, now=now)
    elif args.query:
        rows = [{'query': query, 'family': None, 'role': None} for query in args.query]
    else:
        raise ValueError('pass --plan or at least one --query')

    for row in rows:
        result = fetch_query_page(
            str(row['query']),
            since=since,
            until=until,
            limit=args.limit,
            api_host=args.api_host,
            family=row['family'],
            role=row['role'],
        )
        print(json.dumps(result, ensure_ascii=False))
        print(
            f'{result["query_sent"]}: {result["status"]} fetched={result["fetched"]} '
            f'truncated={result["truncated"]}',
            file=sys.stderr,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        catalog = load_catalog(args.catalog)
        log = load_log(args.log)
        if args.command == 'summary':
            print(json.dumps(summary_report(catalog, log), indent=2, ensure_ascii=False))
            return 0
        if args.command == 'plan':
            now = parse_timestamp(args.now) if args.now else datetime.now(UTC)
            seed = _daily_seed(now) if args.seed is None else args.seed
            report = plan_report(
                catalog,
                log,
                slate_size=args.count,
                rng=Random(seed),
                now=now,
            )
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0
        if args.command == 'search':
            return _search_command(args, catalog, log)
        entry, remembered = record_search(
            log_path=args.log,
            catalog_path=args.catalog,
            query=args.query,
            family=args.family,
            status=args.status,
            hits=args.hits,
            false_negatives=args.false_negatives,
            false_negative_ids=args.false_negative_id,
            notes=args.notes,
            used_at=parse_timestamp(args.used_at) if args.used_at else None,
            remember=not args.no_remember,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2

    print(
        json.dumps(
            {'recorded': entry.to_json(), 'catalog_updated': remembered},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
