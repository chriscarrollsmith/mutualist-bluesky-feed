"""Plan and log AppView queries for the daily false-negative review.

The feed database only stores matcher keeps, and those rows expire. This module
is the measurement record for the other side: which ``searchPosts`` queries
were tried, and how many mutualist posts each one surfaced that the matcher
had dropped.

Planning does not use the high-precision backfill presets. Those mostly return
posts the matcher already keeps. A day's slate mixes

- **exploit** — queries that have already surfaced false negatives, weighted by
  how many, with a penalty for repeating one that ran in the last day and a half
- **explore** — queries from underused families in the strategy catalog, so the
  distribution over families does not collapse and untried queries stay in rotation

False-negative counts are recorded after review. A blocked ``searchPosts`` call
is logged so the attempt is not forgotten, and it is left out of the yield and
entropy statistics.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from random import Random
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG_PATH = ROOT / 'data' / 'fn_search_strategies.json'
DEFAULT_LOG_PATH = ROOT / 'data' / 'fn_search_log.jsonl'

# Exact queries from scripts/backfill_gap.DEFAULT_SEARCH_QUERIES. Kept here so
# planning does not import the backfill script. Tests assert the two lists match.
HIGH_PRECISION_QUERIES = frozenset(
    {
        'mutualism',
        'mutualist',
        'proudhon',
        'mutual credit',
        'occupancy and use',
        'agorism',
        'free-market anti-capitalism',
        'left-wing market anarchism',
        'kropotkin',
        'benjamin tucker',
        'josiah warren',
        'c4ss',
        '#mutualism',
    }
)

# Share of the slate spent on new or underused strategies.
# Low family entropy pushes toward the ceiling; a uniform history sits on the floor.
EXPLORE_FLOOR = 0.35
EXPLORE_CEILING = 0.80
RECENCY_WINDOW = timedelta(hours=36)
RECENCY_WEIGHT = 0.25

STATUS_OK = 'ok'
STATUS_BLOCKED = 'blocked'


def normalize_query(query: str) -> str:
    return ' '.join(query.split()).casefold()


def _utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def format_timestamp(dt: datetime) -> str:
    return _utc(dt).strftime('%Y-%m-%dT%H:%M:%SZ')


def parse_timestamp(value: str) -> datetime:
    normalized = value.strip().replace('Z', '+00:00')
    parsed = datetime.fromisoformat(normalized)
    return _utc(parsed)


@dataclass(frozen=True)
class SearchStrategy:
    query: str
    family: str

    @property
    def key(self) -> str:
        return normalize_query(self.query)


@dataclass(frozen=True)
class SearchLogEntry:
    used_at: datetime
    query: str
    family: str
    status: str
    hits: int
    false_negatives: int
    false_negative_ids: tuple[str, ...] = ()
    notes: str = ''

    @property
    def key(self) -> str:
        return normalize_query(self.query)

    def to_json(self) -> dict[str, Any]:
        return {
            'used_at': format_timestamp(self.used_at),
            'query': self.query,
            'family': self.family,
            'status': self.status,
            'hits': self.hits,
            'false_negatives': self.false_negatives,
            'false_negative_ids': list(self.false_negative_ids),
            'notes': self.notes,
        }


@dataclass(frozen=True)
class QueryStats:
    query: str
    family: str
    uses: int
    false_negatives: int
    last_used: datetime

    @property
    def key(self) -> str:
        return normalize_query(self.query)


@dataclass(frozen=True)
class PlannedQuery:
    query: str
    family: str
    role: str
    historical_false_negatives: int
    uses: int

    def to_json(self) -> dict[str, Any]:
        return {
            'query': self.query,
            'family': self.family,
            'role': self.role,
            'historical_false_negatives': self.historical_false_negatives,
            'uses': self.uses,
        }


def _reject_high_precision(query: str) -> None:
    if normalize_query(query) in HIGH_PRECISION_QUERIES:
        raise ValueError(
            f'{query!r} is a high-precision backfill preset. Daily false-negative '
            'review does not use those queries.'
        )


def is_hashtag_strategy(query: str, family: str = '') -> bool:
    """AppView searchPosts returns HTTP 400 for a query that starts with #."""
    return family.strip().casefold() == 'hashtag' or query.lstrip().startswith('#')


def _reject_hashtag(query: str, family: str = '') -> None:
    if not is_hashtag_strategy(query, family):
        return
    if family.strip().casefold() == 'hashtag':
        raise ValueError(
            'hashtag is not a search family. AppView searchPosts returns '
            'HTTP 400 InvalidRequest for a query that starts with #.'
        )
    raise ValueError(
        f'{query!r} is not an AppView searchPosts strategy. '
        'A query that starts with # returns HTTP 400 InvalidRequest.'
    )


def load_catalog(path: Path | None = None) -> tuple[SearchStrategy, ...]:
    catalog_path = path or DEFAULT_CATALOG_PATH
    payload = json.loads(catalog_path.read_text(encoding='utf-8'))
    families = payload.get('families')
    if not isinstance(families, list) or not families:
        raise ValueError(f'{catalog_path} has no strategy families')

    strategies: list[SearchStrategy] = []
    seen: set[str] = set()
    for family in families:
        family_id = str(family.get('id') or '').strip()
        if not family_id:
            raise ValueError(f'{catalog_path} contains a family with no id')
        _reject_hashtag('', family_id)
        queries = family.get('queries')
        if not isinstance(queries, list) or not queries:
            raise ValueError(f'family {family_id!r} has no queries')
        for query in queries:
            text = str(query).strip()
            if not text:
                raise ValueError(f'family {family_id!r} contains an empty query')
            _reject_high_precision(text)
            _reject_hashtag(text, family_id)
            key = normalize_query(text)
            if key in seen:
                raise ValueError(f'duplicate strategy query {text!r}')
            seen.add(key)
            strategies.append(SearchStrategy(query=text, family=family_id))
    return tuple(strategies)


def _entry_from_json(payload: Mapping[str, Any], *, line_no: int) -> SearchLogEntry:
    try:
        used_at = parse_timestamp(str(payload['used_at']))
        query = str(payload['query']).strip()
        family = str(payload['family']).strip()
        status = str(payload.get('status') or STATUS_OK)
        hits = int(payload['hits'])
        false_negatives = int(payload['false_negatives'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'invalid false-negative search log line {line_no}: {exc}') from exc

    if not query or not family:
        raise ValueError(f'line {line_no}: query and family are required')
    if status not in {STATUS_OK, STATUS_BLOCKED}:
        raise ValueError(f'line {line_no}: status must be ok or blocked')
    if hits < 0 or false_negatives < 0:
        raise ValueError(f'line {line_no}: hits and false_negatives must be >= 0')
    if status == STATUS_BLOCKED and (hits or false_negatives):
        raise ValueError(f'line {line_no}: a blocked search has no hits or false negatives')
    if status == STATUS_OK and false_negatives > hits:
        raise ValueError(f'line {line_no}: false_negatives cannot exceed hits')

    raw_ids = payload.get('false_negative_ids') or []
    if not isinstance(raw_ids, list):
        raise ValueError(f'line {line_no}: false_negative_ids must be a list')
    ids = tuple(str(item).strip() for item in raw_ids)
    if any(not item for item in ids):
        raise ValueError(f'line {line_no}: false_negative_ids contains an empty id')
    if ids and len(ids) != false_negatives:
        raise ValueError(
            f'line {line_no}: false_negative_ids has {len(ids)} entries '
            f'for false_negatives={false_negatives}'
        )
    notes = str(payload.get('notes') or '')
    return SearchLogEntry(
        used_at=used_at,
        query=query,
        family=family,
        status=status,
        hits=hits,
        false_negatives=false_negatives,
        false_negative_ids=ids,
        notes=notes,
    )


def load_log(path: Path | None = None) -> tuple[SearchLogEntry, ...]:
    log_path = path or DEFAULT_LOG_PATH
    if not log_path.is_file():
        return ()
    entries: list[SearchLogEntry] = []
    for line_no, line in enumerate(log_path.read_text(encoding='utf-8').splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith('#'):
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f'invalid JSON on line {line_no} of {log_path}: {exc}') from exc
        if not isinstance(payload, dict):
            raise ValueError(f'line {line_no} of {log_path} must be a JSON object')
        entries.append(_entry_from_json(payload, line_no=line_no))
    return tuple(entries)


def append_log(path: Path, entry: SearchLogEntry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(entry.to_json(), ensure_ascii=False) + '\n')


def successful_uses(log: Sequence[SearchLogEntry]) -> tuple[SearchLogEntry, ...]:
    """Completed searches. Blocked calls stay in the file and out of the stats."""
    return tuple(entry for entry in log if entry.status == STATUS_OK)


def query_stats(log: Sequence[SearchLogEntry]) -> dict[str, QueryStats]:
    stats: dict[str, QueryStats] = {}
    for entry in successful_uses(log):
        current = stats.get(entry.key)
        if current is None:
            stats[entry.key] = QueryStats(
                query=entry.query,
                family=entry.family,
                uses=1,
                false_negatives=entry.false_negatives,
                last_used=entry.used_at,
            )
            continue
        last_used = entry.used_at if entry.used_at >= current.last_used else current.last_used
        stats[entry.key] = QueryStats(
            query=current.query,
            family=entry.family,
            uses=current.uses + 1,
            false_negatives=current.false_negatives + entry.false_negatives,
            last_used=last_used,
        )
    return stats


def family_counts(log: Sequence[SearchLogEntry]) -> Counter[str]:
    return Counter(entry.family for entry in successful_uses(log))


def shannon_entropy(counts: Mapping[str, int]) -> float:
    total = sum(count for count in counts.values() if count > 0)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy


def normalized_entropy(counts: Mapping[str, int], support: int) -> float:
    """Shannon entropy divided by the entropy of a uniform distribution over ``support``."""
    if support <= 1:
        return 0.0
    return shannon_entropy(counts) / math.log(support)


def explore_slot_count(
    slate_size: int,
    log: Sequence[SearchLogEntry],
    *,
    family_support: int,
) -> int:
    """How many of today's queries should be exploration rather than exploitation."""
    if slate_size <= 0:
        return 0
    ok = successful_uses(log)
    has_yield = any(entry.false_negatives > 0 for entry in ok)
    if not has_yield:
        return slate_size

    diversity = min(1.0, normalized_entropy(family_counts(ok), family_support))
    explore_fraction = EXPLORE_FLOOR + (1.0 - diversity) * (EXPLORE_CEILING - EXPLORE_FLOOR)
    if slate_size == 1:
        return 1 if explore_fraction >= 0.5 else 0

    explore_n = int(round(slate_size * explore_fraction))
    return min(slate_size - 1, max(1, explore_n))


def _weighted_sample(
    rng: Random,
    items: Sequence[Any],
    weights: Sequence[float],
    count: int,
) -> list[Any]:
    pool = [[item, weight] for item, weight in zip(items, weights, strict=True) if weight > 0]
    picked: list[Any] = []
    for _ in range(count):
        if not pool:
            break
        total = sum(weight for _, weight in pool)
        if total <= 0:
            break
        threshold = rng.random() * total
        cursor = 0.0
        index = len(pool) - 1
        for candidate, (_, weight) in enumerate(pool):
            cursor += weight
            if cursor >= threshold:
                index = candidate
                break
        picked.append(pool.pop(index)[0])
    return picked


def _exploit_weight(stats: QueryStats, *, now: datetime) -> float:
    if stats.false_negatives <= 0:
        return 0.0
    weight = float(stats.false_negatives)
    if now - _utc(stats.last_used) < RECENCY_WINDOW:
        weight *= RECENCY_WEIGHT
    return weight


def _pick_exploit(
    stats_by_key: Mapping[str, QueryStats],
    *,
    count: int,
    rng: Random,
    now: datetime,
    chosen: set[str],
) -> list[PlannedQuery]:
    candidates = [
        item
        for item in stats_by_key.values()
        if item.key not in chosen
        and item.key not in HIGH_PRECISION_QUERIES
        and not is_hashtag_strategy(item.query, item.family)
    ]
    weights = [_exploit_weight(item, now=now) for item in candidates]
    picked = _weighted_sample(rng, candidates, weights, count)
    chosen.update(item.key for item in picked)
    return [
        PlannedQuery(
            query=item.query,
            family=item.family,
            role='exploit',
            historical_false_negatives=item.false_negatives,
            uses=item.uses,
        )
        for item in picked
    ]


def _uses(stats_by_key: Mapping[str, QueryStats], key: str) -> int:
    found = stats_by_key.get(key)
    return 0 if found is None else found.uses


def _pick_explore(
    catalog: Sequence[SearchStrategy],
    stats_by_key: Mapping[str, QueryStats],
    *,
    count: int,
    rng: Random,
    counts: Counter[str],
    chosen: set[str],
) -> list[PlannedQuery]:
    remaining = [
        item
        for item in catalog
        if item.key not in chosen and not is_hashtag_strategy(item.query, item.family)
    ]
    picked: list[PlannedQuery] = []
    for _ in range(count):
        by_family: dict[str, list[SearchStrategy]] = {}
        for item in remaining:
            by_family.setdefault(item.family, []).append(item)
        if not by_family:
            break
        rarest_count = min(counts[family] for family in by_family)
        rarest = sorted(family for family in by_family if counts[family] == rarest_count)
        family = rarest[0] if len(rarest) == 1 else rng.choice(rarest)
        options = by_family[family]
        fewest_uses = min(_uses(stats_by_key, item.key) for item in options)
        least_used = [item for item in options if _uses(stats_by_key, item.key) == fewest_uses]
        least_used.sort(key=lambda item: item.query)
        strategy = least_used[0] if len(least_used) == 1 else rng.choice(least_used)
        historical = stats_by_key.get(strategy.key)
        picked.append(
            PlannedQuery(
                query=strategy.query,
                family=strategy.family,
                role='explore',
                historical_false_negatives=0 if historical is None else historical.false_negatives,
                uses=0 if historical is None else historical.uses,
            )
        )
        remaining = [item for item in remaining if item.key != strategy.key]
        counts[family] += 1
    return picked


def plan_daily_queries(
    catalog: Sequence[SearchStrategy],
    log: Sequence[SearchLogEntry],
    *,
    slate_size: int,
    rng: Random,
    now: datetime | None = None,
) -> list[PlannedQuery]:
    """Choose today's queries. Exploration grows when family entropy is low."""
    if slate_size <= 0:
        return []
    moment = _utc(now or datetime.now(UTC))
    families = {item.family for item in catalog}
    families.update(entry.family for entry in successful_uses(log))
    explore_n = explore_slot_count(slate_size, log, family_support=len(families))
    exploit_n = slate_size - explore_n

    stats_by_key = query_stats(log)
    counts = family_counts(log)
    chosen: set[str] = set()
    exploit = _pick_exploit(
        stats_by_key,
        count=exploit_n,
        rng=rng,
        now=moment,
        chosen=chosen,
    )
    for item in exploit:
        counts[item.family] += 1
    explore = _pick_explore(
        catalog,
        stats_by_key,
        count=explore_n + (exploit_n - len(exploit)),
        rng=rng,
        counts=counts,
        chosen=chosen,
    )
    return exploit + explore


def plan_report(
    catalog: Sequence[SearchStrategy],
    log: Sequence[SearchLogEntry],
    *,
    slate_size: int,
    rng: Random,
    now: datetime | None = None,
) -> dict[str, Any]:
    families = {item.family for item in catalog}
    families.update(entry.family for entry in successful_uses(log))
    counts = family_counts(log)
    diversity = min(1.0, normalized_entropy(counts, len(families)))
    queries = plan_daily_queries(
        catalog,
        log,
        slate_size=slate_size,
        rng=rng,
        now=now,
    )
    explore_n = sum(1 for item in queries if item.role == 'explore')
    fraction = 0.0 if slate_size <= 0 else explore_n / slate_size
    return {
        'normalized_entropy': round(diversity, 4),
        'explore_fraction': round(fraction, 4),
        'family_uses': dict(sorted(counts.items())),
        'queries': [item.to_json() for item in queries],
    }


def summary_report(
    catalog: Sequence[SearchStrategy],
    log: Sequence[SearchLogEntry],
) -> dict[str, Any]:
    families = {item.family for item in catalog}
    families.update(entry.family for entry in successful_uses(log))
    counts = family_counts(log)
    rows = sorted(
        query_stats(log).values(),
        key=lambda item: (-item.false_negatives, -item.uses, item.query),
    )
    return {
        'logged_uses': len(log),
        'completed_uses': len(successful_uses(log)),
        'normalized_entropy': round(min(1.0, normalized_entropy(counts, len(families))), 4),
        'family_uses': dict(sorted(counts.items())),
        'queries': [
            {
                'query': item.query,
                'family': item.family,
                'uses': item.uses,
                'false_negatives': item.false_negatives,
                'last_used': format_timestamp(item.last_used),
            }
            for item in rows
        ],
    }


def _catalog_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'{path} must contain a JSON object')
    return payload


def remember_strategy(catalog_path: Path, strategy: SearchStrategy) -> bool:
    """Append ``strategy`` to the catalog if that query is not already listed.

    Returns True when the file changed.
    """
    _reject_high_precision(strategy.query)
    _reject_hashtag(strategy.query, strategy.family)
    payload = _catalog_payload(catalog_path)
    families = payload.get('families')
    if not isinstance(families, list):
        raise ValueError(f'{catalog_path} has no strategy families')

    existing = load_catalog(catalog_path)
    if any(item.key == strategy.key for item in existing):
        return False

    family_id = strategy.family.strip()
    query = ' '.join(strategy.query.split())
    for family in families:
        if str(family.get('id') or '').strip() == family_id:
            queries = family.setdefault('queries', [])
            if not isinstance(queries, list):
                raise ValueError(f'family {family_id!r} queries must be a list')
            queries.append(query)
            break
    else:
        families.append({'id': family_id, 'queries': [query]})

    catalog_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    return True


def record_search(
    *,
    log_path: Path,
    catalog_path: Path,
    query: str,
    family: str,
    status: str,
    hits: int,
    false_negatives: int,
    false_negative_ids: Sequence[str] = (),
    notes: str = '',
    used_at: datetime | None = None,
    remember: bool = True,
) -> tuple[SearchLogEntry, bool]:
    """Append one review result. New queries that find misses join the catalog."""
    text = ' '.join(query.split())
    family_id = family.strip()
    if not text or not family_id:
        raise ValueError('query and family are required')
    _reject_high_precision(text)
    _reject_hashtag(text, family_id)

    catalog = load_catalog(catalog_path)
    known = next((item for item in catalog if item.key == normalize_query(text)), None)
    if known is not None and known.family != family_id:
        raise ValueError(f'{text!r} is already in family {known.family!r}, not {family_id!r}')

    entry = SearchLogEntry(
        used_at=_utc(used_at or datetime.now(UTC)),
        query=known.query if known is not None else text,
        family=family_id,
        status=status,
        hits=hits,
        false_negatives=false_negatives,
        false_negative_ids=tuple(false_negative_ids),
        notes=notes,
    )
    # Validate with the same rules as load, before appending.
    _entry_from_json(entry.to_json(), line_no=1)
    append_log(log_path, entry)

    remembered = False
    if remember and status == STATUS_OK and false_negatives > 0 and known is None:
        remembered = remember_strategy(
            catalog_path,
            SearchStrategy(query=entry.query, family=entry.family),
        )
    return entry, remembered
