"""Unit tests for offline LLM label bootstrap (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.llm_label_judge import (
    SYSTEM_PROMPT,
    apply_proposal,
    judge_rows,
    main,
    parse_judge_response,
)


def test_prompt_excludes_biological_mutualism() -> None:
    note = 'biological mutualism (symbiosis between species)'
    assert note in SYSTEM_PROMPT.lower()


def test_parse_judge_response_plain_and_fenced() -> None:
    plain = parse_judge_response('{"expected": false, "rationale": "ancap", "confidence": "high"}')
    assert plain == {
        'expected': False,
        'rationale': 'ancap',
        'confidence': 'high',
    }
    fenced = parse_judge_response(
        '```json\n{"expected": true, "rationale": "mutual aid", "confidence": "MEDIUM"}\n```'
    )
    assert fenced['expected'] is True
    assert fenced['confidence'] == 'medium'


def test_apply_proposal_marks_human_confirm() -> None:
    row = {
        'id': 'x',
        'text': 'Tonight at the infoshop',
        'expected': None,
        'note': 'label me',
    }
    out = apply_proposal(
        row,
        {'expected': True, 'rationale': 'anarchist venue', 'confidence': 'high'},
    )
    assert out['expected'] is True
    assert out['proposed_by'] == 'llm'
    assert out['needs_human_confirm'] is True
    assert 'anarchist venue' in out['note']


def test_judge_rows_skips_labeled_and_uses_injected_judge() -> None:
    rows = [
        {'id': 'a', 'text': 'Ancap tip', 'expected': None},
        {'id': 'b', 'text': 'already', 'expected': False},
    ]

    def judge_fp(_row: dict) -> dict:
        return {
            'expected': False,
            'rationale': 'false friend',
            'confidence': 'high',
        }

    out, stats = judge_rows(rows, judge=judge_fp)
    assert stats == {
        'proposed': 1,
        'skipped_labeled': 1,
        'skipped_empty': 0,
        'errors': 0,
    }
    assert len(out) == 1
    assert out[0]['id'] == 'a'
    assert out[0]['expected'] is False


def test_main_with_injected_judge(tmp_path: Path) -> None:
    inp = tmp_path / 'in.jsonl'
    out = tmp_path / 'out.jsonl'
    inp.write_text(
        json.dumps({'id': 'c1', 'text': 'Hello from the mutual aid fridge', 'expected': None})
        + '\n',
        encoding='utf-8',
    )

    def judge(_row: dict) -> dict:
        return {'expected': True, 'rationale': 'local', 'confidence': 'medium'}

    assert main(['--input', str(inp), '--output', str(out)], judge=judge) == 0
    written = [json.loads(line) for line in out.read_text(encoding='utf-8').splitlines()]
    assert written[0]['expected'] is True
    assert written[0]['needs_human_confirm'] is True


def test_main_requires_api_key_without_injected_judge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    inp = tmp_path / 'in.jsonl'
    inp.write_text(
        json.dumps({'id': 'c1', 'text': 'x', 'expected': None}) + '\n',
        encoding='utf-8',
    )
    assert main(['--input', str(inp), '--api-key', '']) == 1


def test_resolve_api_defaults_prefers_deepseek(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.llm_label_judge import (
        DEFAULT_DEEPSEEK_API_URL,
        DEFAULT_DEEPSEEK_MODEL,
        resolve_api_defaults,
    )

    monkeypatch.setenv('DEEPSEEK_API_KEY', 'ds-test')
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    key, model, url = resolve_api_defaults()
    assert key == 'ds-test'
    assert model == DEFAULT_DEEPSEEK_MODEL
    assert url == DEFAULT_DEEPSEEK_API_URL
