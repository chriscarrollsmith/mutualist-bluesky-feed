"""Unit tests for appending labeled samples into eval_cases.json."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.append_eval_cases import main, merge_cases, normalize_case, read_jsonl


def test_normalize_case_skips_unlabeled() -> None:
    assert (
        normalize_case(
            {'id': 'x', 'text': 'hi', 'expected': None},
            default_signal='text',
            default_bucket='unspecified',
            default_split='dev',
            default_regression=True,
        )
        is None
    )


def test_normalize_case_applies_defaults_and_optional_fields() -> None:
    case = normalize_case(
        {
            'id': 'tp-allowlisted-author',
            'text': 'Newsletter is out.',
            'expected': True,
            'author_handle': 'example.bsky.social',
            'note': 'no topic phrase',
            'langs': ['en'],
            'soft_prior': True,
        },
        default_signal='author',
        default_bucket='allowlisted_author',
        default_split='holdout',
        default_regression=False,
    )
    assert case == {
        'id': 'tp-allowlisted-author',
        'text': 'Newsletter is out.',
        'expected': True,
        'signal': 'author',
        'bucket': 'allowlisted_author',
        'split': 'holdout',
        'regression': False,
        'author_handle': 'example.bsky.social',
        'note': 'no topic phrase',
        'langs': ['en'],
        'soft_prior': True,
    }


def test_merge_cases_adds_new_and_skips_duplicates() -> None:
    existing = [
        {
            'id': 'already',
            'text': 'old',
            'expected': False,
            'signal': 'text',
            'bucket': 'skyfeed_fp',
            'split': 'dev',
            'regression': True,
        }
    ]
    incoming = [
        {'id': 'already', 'text': 'old', 'expected': False},
        {'id': 'new-one', 'text': 'Tonight at the infoshop', 'expected': True, 'signal': 'event'},
        {'id': 'unlabeled', 'text': 'maybe', 'expected': None},
    ]
    merged, stats = merge_cases(
        existing,
        incoming,
        default_signal='text',
        default_bucket='in_scope_event',
        default_split='dev',
        default_regression=True,
    )
    assert stats == {'added': 1, 'skipped_unlabeled': 1, 'skipped_existing': 1}
    assert len(merged) == 2
    assert merged[1]['id'] == 'new-one'
    assert merged[1]['bucket'] == 'in_scope_event'


def test_read_jsonl(tmp_path: Path) -> None:
    path = tmp_path / 'rows.jsonl'
    path.write_text(
        '# comment\n{"id": "a", "text": "t", "expected": true}\n\n',
        encoding='utf-8',
    )
    rows = read_jsonl(path)
    assert rows[0]['id'] == 'a'


def test_main_dry_run_and_write(tmp_path: Path) -> None:
    cases = tmp_path / 'eval_cases.json'
    cases.write_text('[]\n', encoding='utf-8')
    labeled = tmp_path / 'labeled.jsonl'
    labeled.write_text(
        json.dumps(
            {
                'id': 'fp-new',
                'text': 'Ancap news',
                'expected': False,
                'signal': 'text',
                'bucket': 'skyfeed_fp',
                'split': 'dev',
            }
        )
        + '\n',
        encoding='utf-8',
    )

    assert main(['--cases', str(cases), '--input', str(labeled), '--dry-run']) == 0
    assert json.loads(cases.read_text(encoding='utf-8')) == []

    assert main(['--cases', str(cases), '--input', str(labeled)]) == 0
    written = json.loads(cases.read_text(encoding='utf-8'))
    assert len(written) == 1
    assert written[0]['id'] == 'fp-new'

    # Second append is a no-op for the same id.
    assert main(['--cases', str(cases), '--input', str(labeled)]) == 0
    assert len(json.loads(cases.read_text(encoding='utf-8'))) == 1


def test_normalize_rejects_bad_expected() -> None:
    with pytest.raises(ValueError, match='boolean'):
        normalize_case(
            {'id': 'x', 'text': 't', 'expected': 'true'},
            default_signal='text',
            default_bucket='unspecified',
            default_split='dev',
            default_regression=True,
        )
