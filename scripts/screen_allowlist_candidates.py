#!/usr/bin/env python3
"""Screen Bluesky accounts for mutualist allowlist fitness.

Fetches recent author-feed samples and flags volume / templated-slop risks so
selection passes can keep a high signal/noise ratio.

Examples:

    uv run python scripts/screen_allowlist_candidates.py \\
        c4ssdotorg.bsky.social kevincarson1.bsky.social
    uv run python scripts/screen_allowlist_candidates.py --from-file /tmp/candidates.txt
    uv run python scripts/screen_allowlist_candidates.py --from-allowlist --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.allowlists import load_list_file  # noqa: E402

DEFAULT_API_HOST = 'https://api.bsky.app'
USER_AGENT = 'mutualist-bluesky-feed-screen/0.1 (+https://github.com/chriscarrollsmith/mutualist-bluesky-feed)'

# Soft thresholds for always-keep allowlisting (not matcher hard rules).
HIGH_VOLUME_PPD = 20.0
ELEVATED_VOLUME_PPD = 8.0

SLOP_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ('promo', re.compile(r'click here|link in bio|use code|%\s*off|shop now|limited time', re.I)),
    (
        'pr_speak',
        re.compile(
            r"we(?:'re| are) (?:excited|thrilled|proud) to (?:announce|share)",
            re.I,
        ),
    ),
    (
        'bizspam',
        re.compile(
            r'call (?:us )?today|free (?:estimate|quote)|'
            r'(?:hvac|roofing|plumbing|windows?).{0,60}serving|'
            r'company (?:address|phone|email)',
            re.I,
        ),
    ),
)

BOT_RE = re.compile(r'(?i)\bbot\b|unofficial bot|not monitored|brid\.gy|\brss\b|autopost')


JsonObject = dict[str, Any]


@dataclass
class ScreenResult:
    handle: str
    did: str | None = None
    display_name: str | None = None
    description: str | None = None
    followers: int | None = None
    posts_count: int | None = None
    sample_n: int = 0
    recent_7d: int = 0
    recent_30d: int = 0
    ppd_7d: float = 0.0
    ppd_30d: float = 0.0
    unique_prefix_ratio: float = 1.0
    slop_hits: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    recommendation: str = 'review'
    sample_texts: list[str] = field(default_factory=list)
    error: str | None = None


def _get_json(url: str, *, timeout: float) -> JsonObject:
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode('utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'expected JSON object from {url}')
    return payload


def fetch_profile(actor: str, *, api_host: str, timeout: float) -> JsonObject:
    url = (
        api_host.rstrip('/')
        + '/xrpc/app.bsky.actor.getProfile?'
        + urllib.parse.urlencode({'actor': actor})
    )
    return _get_json(url, timeout=timeout)


def fetch_author_feed(
    actor: str,
    *,
    api_host: str,
    limit: int,
    timeout: float,
) -> JsonObject:
    url = (
        api_host.rstrip('/')
        + '/xrpc/app.bsky.feed.getAuthorFeed?'
        + urllib.parse.urlencode({'actor': actor, 'limit': limit})
    )
    return _get_json(url, timeout=timeout)


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return None


def recommend(
    *,
    flags: list[str],
) -> str:
    """Map screen flags to a coarse allowlist recommendation."""
    if any(f.startswith('high_volume') or f in {'bot_or_auto', 'fetch_error'} for f in flags):
        return 'reject'
    if any(f.startswith('elevated_volume') or f.startswith('slop:') for f in flags):
        return 'review'
    if 'inactive_or_empty' in flags:
        return 'skip_empty'
    if not flags:
        return 'likely_ok'
    return 'review'


def collect_flags(
    *,
    handle: str,
    display_name: str | None,
    description: str,
    ppd_7d: float,
    ppd_30d: float,
    unique_prefix_ratio: float,
    sample_n: int,
    sample_texts: list[str],
) -> tuple[list[str], list[str]]:
    """Return (flags, slop_hits) for an account sample."""
    flags: list[str] = []
    slop_hits: list[str] = []
    blob = f'{handle} {display_name or ""} {description}'
    if BOT_RE.search(blob):
        flags.append('bot_or_auto')

    joined = '\n'.join(sample_texts).lower()
    for name, pattern in SLOP_PATTERNS:
        if pattern.search(joined) or pattern.search(description):
            slop_hits.append(name)
    if slop_hits:
        flags.append('slop:' + ','.join(slop_hits))

    max_ppd = max(ppd_7d, ppd_30d)
    if max_ppd >= HIGH_VOLUME_PPD:
        flags.append(f'high_volume:{max_ppd}/d')
    elif max_ppd >= ELEVATED_VOLUME_PPD:
        flags.append(f'elevated_volume:{max_ppd}/d')

    if unique_prefix_ratio < 0.45 and sample_n >= 8:
        flags.append(f'low_diversity:{unique_prefix_ratio}')

    if sample_n == 0:
        flags.append('inactive_or_empty')
    return flags, slop_hits


def screen_actor(
    actor: str,
    *,
    api_host: str,
    limit: int = 50,
    timeout: float = 30.0,
) -> ScreenResult:
    result = ScreenResult(handle=actor)
    try:
        profile = fetch_profile(actor, api_host=api_host, timeout=timeout)
        feed = fetch_author_feed(actor, api_host=api_host, limit=limit, timeout=timeout)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as exc:
        result.error = str(exc)
        result.flags.append('fetch_error')
        result.recommendation = recommend(flags=result.flags)
        return result

    handle = str(profile.get('handle') or actor)
    did = profile.get('did')
    result.handle = handle
    result.did = did if isinstance(did, str) else None
    result.display_name = (
        profile.get('displayName') if isinstance(profile.get('displayName'), str) else None
    )
    raw_desc = profile.get('description')
    desc = raw_desc if isinstance(raw_desc, str) else ''
    result.description = desc
    result.followers = (
        profile.get('followersCount') if isinstance(profile.get('followersCount'), int) else None
    )
    result.posts_count = (
        profile.get('postsCount') if isinstance(profile.get('postsCount'), int) else None
    )

    now = datetime.now(UTC)
    own_posts: list[tuple[datetime | None, bool, str]] = []
    for item in feed.get('feed') or []:
        if not isinstance(item, dict):
            continue
        post = item.get('post') or {}
        if not isinstance(post, dict):
            continue
        author = post.get('author') or {}
        if not isinstance(author, dict) or author.get('did') != did:
            continue
        record = post.get('record') or {}
        if not isinstance(record, dict):
            continue
        created = _parse_ts(
            record.get('createdAt') if isinstance(record.get('createdAt'), str) else None
        )
        text = str(record.get('text') or '')
        own_posts.append((created, bool(record.get('reply')), text))

    result.sample_n = len(own_posts)
    recent7 = [p for p in own_posts if p[0] and p[0] >= now - timedelta(days=7)]
    recent30 = [p for p in own_posts if p[0] and p[0] >= now - timedelta(days=30)]
    result.recent_7d = len(recent7)
    result.recent_30d = len(recent30)
    result.ppd_7d = round(len(recent7) / 7.0, 2)
    result.ppd_30d = round(len(recent30) / 30.0, 2)

    originals = [t for _, reply, t in own_posts if not reply and t.strip()]
    result.sample_texts = originals[:5]
    prefixes = [t.strip()[:40].lower() for t in originals]
    result.unique_prefix_ratio = round(len(set(prefixes)) / max(len(prefixes), 1), 2)

    flags, slop_hits = collect_flags(
        handle=handle,
        display_name=result.display_name,
        description=desc,
        ppd_7d=result.ppd_7d,
        ppd_30d=result.ppd_30d,
        unique_prefix_ratio=result.unique_prefix_ratio,
        sample_n=result.sample_n,
        sample_texts=originals,
    )
    result.flags = flags
    result.slop_hits = slop_hits
    result.recommendation = recommend(flags=flags)
    return result


def _load_actors(args: argparse.Namespace) -> list[str]:
    actors: list[str] = []
    actors.extend(args.actors)
    if args.from_file:
        actors.extend(load_list_file(args.from_file))
    if args.from_allowlist:
        actors.extend(load_list_file(ROOT / 'data' / 'allowlist_handles.txt'))
    # Preserve order, drop blanks/dupes
    seen: set[str] = set()
    ordered: list[str] = []
    for actor in actors:
        key = actor.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(actor.strip())
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('actors', nargs='*', help='Handles or DIDs to screen')
    parser.add_argument('--from-file', type=Path, help='Path to handle list (one per line)')
    parser.add_argument(
        '--from-allowlist',
        action='store_true',
        help='Include data/allowlist_handles.txt',
    )
    parser.add_argument('--api-host', default=DEFAULT_API_HOST)
    parser.add_argument('--limit', type=int, default=50, help='Author-feed sample size')
    parser.add_argument('--json', action='store_true', help='Emit JSON lines')
    args = parser.parse_args()

    actors = _load_actors(args)
    if not actors:
        print('pass handles and/or --from-file / --from-allowlist', file=sys.stderr)
        return 2

    exit_code = 0
    for actor in actors:
        result = screen_actor(actor, api_host=args.api_host, limit=args.limit)
        if result.error or result.recommendation == 'reject':
            exit_code = 1
        if args.json:
            print(json.dumps(asdict(result), ensure_ascii=False))
            continue
        flags = ','.join(result.flags) if result.flags else '-'
        print(
            f'{result.recommendation:12} {result.handle:40} '
            f'ppd7={result.ppd_7d:<5} ppd30={result.ppd_30d:<5} '
            f'fol={result.followers or 0:<6} flags={flags}'
        )
        if result.display_name or result.description:
            print(f'             {result.display_name or ""} — {(result.description or "")[:120]}')
        for text in result.sample_texts[:2]:
            print(f'             · {text[:120].replace(chr(10), " ")}')
        if result.error:
            print(f'             error: {result.error}', file=sys.stderr)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
