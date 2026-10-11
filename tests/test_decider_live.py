"""Opt-in live checks for the Perplexity decider.

Run with ``uv run pytest -m live``. The default suite skips this module.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from server.decider import OpenRouterDecider
from server.matcher import match_post

pytestmark = pytest.mark.live

_MUTUALIST = (
    'Neighborhood mutual credit clearing this month: we price shared tools at cost, '
    'occupancy-and-use rather than landlord title. Proudhonian mutualism in practice.'
)
_HOSTILE = (
    'Proudhonian mutualism is a socialist plot. Breitbart is right about the '
    'dangers of a massive socialist welfare state — vote them out.'
)


def _live_decider() -> OpenRouterDecider:
    load_dotenv(Path(__file__).resolve().parents[1] / '.env')
    api_key = os.environ.get('OPENROUTER_API_KEY', '').strip()
    if not api_key:
        pytest.skip('OPENROUTER_API_KEY is not set')
    return OpenRouterDecider(api_key=api_key, timeout_s=30.0)


def test_live_decider_keeps_mutualist_post_and_rejects_hostile_one() -> None:
    decider = _live_decider()

    kept = decider.decide(_MUTUALIST)
    assert kept is not None, 'mutualist post was unscored'
    assert kept.keep, f'expected keep, noul={kept.score:.3f}'

    rejected = decider.decide(_HOSTILE)
    assert rejected is not None, 'hostile post was unscored'
    assert not rejected.keep, f'expected reject, noul={rejected.score:.3f}'


def test_live_match_post_filters_regex_candidates() -> None:
    decider = _live_decider()

    keep = match_post(_MUTUALIST, decider=decider)
    assert keep.matched, f'expected decision_keep, reason={keep.reason}'
    assert keep.reason.startswith('decision_keep:')

    drop = match_post(_HOSTILE, decider=decider)
    assert not drop.matched, f'expected decision_reject, reason={drop.reason}'
    assert drop.reason.startswith('decision_reject:')
