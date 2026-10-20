"""Unit tests for stratified eval reporting helpers."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.eval_filter import Confusion, evaluate_cases

CASES_PATH = Path(__file__).resolve().parents[1] / 'data' / 'eval_cases.json'


def test_confusion_metrics() -> None:
    c = Confusion()
    c.add(predicted=True, expected=True)
    c.add(predicted=True, expected=False)
    c.add(predicted=False, expected=False)
    c.add(predicted=False, expected=True)
    assert c.tp == 1 and c.fp == 1 and c.tn == 1 and c.fn == 1
    assert c.precision == 0.5
    assert c.recall == 0.5
    assert c.f1 == 0.5
    assert c.accuracy == 0.5


def test_evaluate_cases_stratifies_and_respects_gaps() -> None:
    cases = [
        {
            'id': 'tp-text',
            'text': 'Hello from the mutual credit clearing',
            'expected': True,
            'signal': 'text',
            'bucket': 'strong_positive',
            'split': 'dev',
            'regression': True,
        },
        {
            'id': 'tp-author',
            'text': 'Newsletter is out.',
            'author_handle': 'allowlisted.bsky.social',
            'expected': True,
            'signal': 'author',
            'bucket': 'allowlisted_author',
            'split': 'holdout',
            'regression': True,
        },
        {
            'id': 'gap-author',
            'text': 'Open mic tonight.',
            'author_handle': 'not-allowlisted.bsky.social',
            'expected': True,
            'signal': 'author',
            'bucket': 'allowlisted_author',
            'split': 'dev',
            'regression': False,
        },
    ]
    report = evaluate_cases(cases, allowlist_handles={'allowlisted.bsky.social'})
    assert report.overall.tp == 2
    assert report.overall.fn == 1
    assert report.by_split['dev'].tp == 1
    assert report.by_split['holdout'].tp == 1
    assert report.by_signal['author'].tp == 1
    assert report.by_signal['author'].fn == 1
    assert report.by_bucket['allowlisted_author'].fn == 1
    assert len(report.failures) == 1
    assert len(report.regression_failures) == 0


def test_eval_cases_have_required_strata() -> None:
    cases = json.loads(CASES_PATH.read_text(encoding='utf-8'))
    assert cases, 'eval_cases.json must not be empty'

    required = {'signal', 'bucket', 'split'}
    signals = set()
    buckets = set()
    splits = set()
    for case in cases:
        missing = required - case.keys()
        assert not missing, f'{case["id"]} missing fields: {sorted(missing)}'
        assert case['split'] in {'dev', 'holdout'}, case['id']
        assert case['signal'] in {'text', 'author', 'event'}, case['id']
        signals.add(case['signal'])
        buckets.add(case['bucket'])
        splits.add(case['split'])

    assert 'author' in signals
    assert 'event' in signals
    assert 'author_blocklist' in buckets
    assert 'author_soft_prior' in buckets
    assert 'in_scope_event' in buckets
    assert 'ambiguous_classifier' in buckets
    assert 'solicit_reject' in buckets
    assert splits == {'dev', 'holdout'}

    soft_prior_cases = [c for c in cases if c.get('bucket') == 'author_soft_prior']
    assert soft_prior_cases, 'expected soft-prior eval cases'
    assert any(c.get('soft_prior') is True and c.get('expected') is True for c in soft_prior_cases)

    clf_cases = [c for c in cases if c.get('bucket') == 'ambiguous_classifier']
    assert any(c.get('expected') is True and c.get('split') == 'holdout' for c in clf_cases)
    assert any(c.get('expected') is False for c in clf_cases)

    solicit_cases = [c for c in cases if c.get('bucket') == 'solicit_reject']
    assert solicit_cases, 'expected solicit_reject eval cases'
    assert all(c.get('expected') is False for c in solicit_cases)

    author_cases = [c for c in cases if c.get('signal') == 'author']
    fn_event = [
        c for c in cases if c.get('bucket') == 'in_scope_event' and c.get('expected') is True
    ]
    assert len(author_cases) >= 3
    assert len(fn_event) >= 2
    assert any(c.get('bucket') == 'author_soft_prior' for c in author_cases)
    assert any(c.get('bucket') == 'author_blocklist' for c in author_cases)
    assert any(c.get('split') == 'holdout' for c in fn_event)
