#!/usr/bin/env python3
"""Plan and record the daily false-negative AppView search.

Do not pass the high-precision backfill presets (``mutualism``, ``Proudhon``,
and the rest of ``DEFAULT_SEARCH_QUERIES``). The planner reads the strategy catalog
and the running log, then returns a slate that exploits queries which have already
surfaced false negatives and explores underused families.

Examples:

    uv run python scripts/fn_search.py summary
    uv run python scripts/fn_search.py plan
    uv run python scripts/fn_search.py record \\
      --query "time bank" --family practice \\
      --hits 4 --false-negatives 1 \\
      --false-negative-id at://did:plc:example/app.bsky.feed.post/abc
    uv run python scripts/fn_search.py record \\
      --query "mutual aid" --family bare_term --status blocked
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from random import Random

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.fn_search import (  # noqa: E402
    DEFAULT_CATALOG_PATH,
    DEFAULT_LOG_PATH,
    load_catalog,
    load_log,
    parse_timestamp,
    plan_report,
    record_search,
    summary_report,
)


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
    return parser


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
