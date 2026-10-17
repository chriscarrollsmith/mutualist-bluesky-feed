"""Planner and log for the daily false-negative search."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from random import Random

import pytest
from scripts.backfill_gap import DEFAULT_SEARCH_QUERIES
from scripts.fn_search import main as fn_search_main
from server.fn_search import (
    HIGH_PRECISION_QUERIES,
    STATUS_BLOCKED,
    STATUS_OK,
    SearchLogEntry,
    SearchStrategy,
    explore_slot_count,
    load_catalog,
    load_log,
    normalized_entropy,
    plan_daily_queries,
    record_search,
    shannon_entropy,
)
from server.matcher import match_post

NOW = datetime(2026, 10, 9, 14, 0, tzinfo=UTC)
CATALOG = (
    SearchStrategy('anarchism', 'bare_term'),
    SearchStrategy('mutual aid', 'bare_term'),
    SearchStrategy('co-op', 'bare_term'),
    SearchStrategy('time bank', 'practice'),
    SearchStrategy('community land trust', 'practice'),
    SearchStrategy('tool library', 'practice'),
    SearchStrategy('Lysander Spooner', 'figure'),
    SearchStrategy('Markets Not Capitalism', 'figure'),
    SearchStrategy('What is Property', 'figure'),
    SearchStrategy('counter-economics', 'current'),
    SearchStrategy('labor for labor', 'current'),
    SearchStrategy('four monopolies', 'current'),
)


def _entry(
    query: str,
    family: str,
    *,
    false_negatives: int = 0,
    hits: int | None = None,
    used_at: datetime = NOW - timedelta(days=3),
    status: str = STATUS_OK,
) -> SearchLogEntry:
    resolved_hits = false_negatives if hits is None else hits
    return SearchLogEntry(
        used_at=used_at,
        query=query,
        family=family,
        status=status,
        hits=resolved_hits,
        false_negatives=false_negatives,
    )


def _uniform_log() -> list[SearchLogEntry]:
    entries: list[SearchLogEntry] = []
    for strategy in CATALOG:
        entries.append(_entry(strategy.query, strategy.family, false_negatives=1, hits=4))
    return entries


def test_checked_in_catalog_excludes_backfill_presets() -> None:
    assert HIGH_PRECISION_QUERIES == frozenset(query.casefold() for query in DEFAULT_SEARCH_QUERIES)
    catalog = load_catalog()
    assert catalog
    families = {item.family for item in catalog}
    assert families >= {'bare_term', 'practice', 'figure', 'current', 'event_phrase', 'hashtag'}
    assert all(item.key not in HIGH_PRECISION_QUERIES for item in catalog)
    assert load_log() == ()


def test_catalog_queries_are_not_already_strong_keeps() -> None:
    """A query the matcher keeps on its own teaches the planner that nothing was missed."""
    for strategy in load_catalog():
        result = match_post(strategy.query)
        assert result.matched is False, (
            f'{strategy.query!r} ({strategy.family}) already keeps: {result.reason}'
        )


def test_catalog_rejects_high_precision_preset(tmp_path: Path) -> None:
    path = tmp_path / 'strategies.json'
    path.write_text(
        json.dumps(
            {
                'version': 1,
                'families': [{'id': 'bare_term', 'queries': ['Proudhon']}],
            }
        ),
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='high-precision'):
        load_catalog(path)


def test_empty_log_is_all_exploration() -> None:
    plan = plan_daily_queries(CATALOG, (), slate_size=8, rng=Random(0), now=NOW)
    assert len(plan) == 8
    assert {item.role for item in plan} == {'explore'}
    assert len({item.query for item in plan}) == 8
    assert all(item.query.casefold() not in HIGH_PRECISION_QUERIES for item in plan)


def test_same_seed_replans_the_same_slate() -> None:
    log = _uniform_log()
    first = plan_daily_queries(CATALOG, log, slate_size=8, rng=Random(7), now=NOW)
    second = plan_daily_queries(CATALOG, log, slate_size=8, rng=Random(7), now=NOW)
    assert first == second


def test_low_entropy_explores_more_than_a_uniform_history() -> None:
    peaked = [
        _entry(
            'anarchism',
            'bare_term',
            false_negatives=2,
            hits=5,
            used_at=NOW - timedelta(days=4),
        )
        for _ in range(12)
    ]
    peaked_explore = explore_slot_count(8, peaked, family_support=4)
    uniform_explore = explore_slot_count(8, _uniform_log(), family_support=4)
    assert peaked_explore > uniform_explore
    assert shannon_entropy({'bare_term': 12}) < shannon_entropy(
        {'bare_term': 3, 'practice': 3, 'figure': 3, 'current': 3}
    )


def test_peaked_history_explores_unused_families_first() -> None:
    log = [
        _entry(
            'anarchism',
            'bare_term',
            false_negatives=3,
            hits=6,
            used_at=NOW - timedelta(days=4),
        )
    ]
    plan = plan_daily_queries(CATALOG, log, slate_size=8, rng=Random(1), now=NOW)
    explore = [item for item in plan if item.role == 'explore']
    assert explore
    assert explore[0].family != 'bare_term'
    assert all(item.family != 'bare_term' for item in explore[:3])


def test_only_successful_query_is_exploited() -> None:
    log = [
        _entry(
            'time bank',
            'practice',
            false_negatives=4,
            hits=10,
            used_at=NOW - timedelta(days=5),
        )
    ]
    plan = plan_daily_queries(CATALOG, log, slate_size=8, rng=Random(3), now=NOW)
    exploit = [item for item in plan if item.role == 'exploit']
    assert [item.query for item in exploit] == ['time bank']
    assert exploit[0].historical_false_negatives == 4


def test_exploit_prefers_older_equal_yield_over_a_recent_repeat() -> None:
    log = [
        _entry(
            'time bank',
            'practice',
            false_negatives=4,
            hits=8,
            used_at=NOW - timedelta(hours=2),
        ),
        _entry(
            'community land trust',
            'practice',
            false_negatives=4,
            hits=8,
            used_at=NOW - timedelta(days=10),
        ),
    ]
    # Completed searches in the other families raise entropy without adding yield,
    # so the single exploit slot is contested only by the two practice queries.
    for strategy in CATALOG:
        if strategy.family == 'practice':
            continue
        log.append(_entry(strategy.query, strategy.family, false_negatives=0, hits=3))

    older_wins = 0
    for seed in range(200):
        plan = plan_daily_queries(CATALOG, log, slate_size=2, rng=Random(seed), now=NOW)
        exploit = [item.query for item in plan if item.role == 'exploit']
        assert len(exploit) == 1
        if exploit[0] == 'community land trust':
            older_wins += 1
    assert older_wins > 140


def test_logged_query_outside_the_catalog_can_be_exploited() -> None:
    log = [
        _entry(
            'infoshop',
            'practice',
            false_negatives=2,
            hits=2,
            used_at=NOW - timedelta(days=5),
        )
    ]
    plan = plan_daily_queries(CATALOG, log, slate_size=4, rng=Random(0), now=NOW)
    assert any(item.query == 'infoshop' and item.role == 'exploit' for item in plan)


def test_blocked_search_does_not_create_yield() -> None:
    log = [
        _entry('anarchism', 'bare_term', status=STATUS_BLOCKED, hits=0, false_negatives=0),
    ]
    plan = plan_daily_queries(CATALOG, log, slate_size=4, rng=Random(0), now=NOW)
    assert {item.role for item in plan} == {'explore'}
    assert normalized_entropy({}, 4) == 0.0


def test_record_roundtrip_and_remember_new_success(tmp_path: Path) -> None:
    catalog_path = tmp_path / 'strategies.json'
    log_path = tmp_path / 'fn_search_log.jsonl'
    catalog_path.write_text(
        json.dumps({'version': 1, 'families': [{'id': 'bare_term', 'queries': ['anarchism']}]})
        + '\n',
        encoding='utf-8',
    )
    entry, remembered = record_search(
        log_path=log_path,
        catalog_path=catalog_path,
        query='time bank',
        family='practice',
        status=STATUS_OK,
        hits=5,
        false_negatives=2,
        false_negative_ids=(
            'at://did:plc:example/app.bsky.feed.post/1',
            'at://did:plc:example/app.bsky.feed.post/2',
        ),
        used_at=NOW,
    )
    assert remembered is True
    assert entry.false_negatives == 2
    loaded = load_log(log_path)
    assert loaded[0].query == 'time bank'
    families = json.loads(catalog_path.read_text(encoding='utf-8'))['families']
    assert 'time bank' in families[1]['queries']

    again, remembered_again = record_search(
        log_path=log_path,
        catalog_path=catalog_path,
        query='time bank',
        family='practice',
        status=STATUS_OK,
        hits=1,
        false_negatives=0,
        used_at=NOW,
    )
    assert remembered_again is False
    assert again.query == 'time bank'
    assert len(load_log(log_path)) == 2


def test_record_keeps_catalog_casing_and_refuses_a_different_family(tmp_path: Path) -> None:
    catalog_path = tmp_path / 'strategies.json'
    catalog_path.write_text(
        json.dumps({'version': 1, 'families': [{'id': 'bare_term', 'queries': ['mutual aid']}]})
        + '\n',
        encoding='utf-8',
    )
    entry, remembered = record_search(
        log_path=tmp_path / 'log.jsonl',
        catalog_path=catalog_path,
        query='Mutual Aid',
        family='bare_term',
        status=STATUS_OK,
        hits=2,
        false_negatives=1,
        false_negative_ids=('at://did:plc:example/app.bsky.feed.post/abc',),
        used_at=NOW,
    )
    assert remembered is False
    assert entry.query == 'mutual aid'
    with pytest.raises(ValueError, match='already in family'):
        record_search(
            log_path=tmp_path / 'log.jsonl',
            catalog_path=catalog_path,
            query='mutual aid',
            family='practice',
            status=STATUS_OK,
            hits=1,
            false_negatives=0,
            used_at=NOW,
        )


def test_record_refuses_high_precision_preset(tmp_path: Path) -> None:
    catalog_path = tmp_path / 'strategies.json'
    catalog_path.write_text(
        json.dumps({'version': 1, 'families': [{'id': 'bare_term', 'queries': ['anarchism']}]})
        + '\n',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='high-precision'):
        record_search(
            log_path=tmp_path / 'log.jsonl',
            catalog_path=catalog_path,
            query='Proudhon',
            family='figure',
            status=STATUS_OK,
            hits=3,
            false_negatives=1,
        )


def test_cli_plan_and_blocked_record(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    catalog_path = tmp_path / 'strategies.json'
    log_path = tmp_path / 'log.jsonl'
    catalog_path.write_text(
        json.dumps(
            {
                'version': 1,
                'families': [
                    {'id': 'bare_term', 'queries': ['anarchism', 'mutual aid', 'co-op']},
                    {'id': 'figure', 'queries': ['Lysander Spooner', 'What is Property']},
                ],
            }
        )
        + '\n',
        encoding='utf-8',
    )
    code = fn_search_main(
        [
            '--catalog',
            str(catalog_path),
            '--log',
            str(log_path),
            'record',
            '--query',
            'anarchism',
            '--family',
            'bare_term',
            '--status',
            'blocked',
        ]
    )
    assert code == 0
    assert load_log(log_path)[0].status == STATUS_BLOCKED
    recorded = json.loads(capsys.readouterr().out)
    assert recorded['catalog_updated'] is False

    summary_code = fn_search_main(
        ['--catalog', str(catalog_path), '--log', str(log_path), 'summary']
    )
    assert summary_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary['logged_uses'] == 1
    assert summary['completed_uses'] == 0
    assert summary['queries'] == []

    plan_code = fn_search_main(
        [
            '--catalog',
            str(catalog_path),
            '--log',
            str(log_path),
            'plan',
            '--count',
            '3',
            '--seed',
            '1',
            '--now',
            '2026-10-09T14:00:00Z',
        ]
    )
    assert plan_code == 0
    plan = json.loads(capsys.readouterr().out)
    assert len(plan['queries']) == 3
    assert {item['role'] for item in plan['queries']} == {'explore'}
