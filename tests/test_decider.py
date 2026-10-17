"""Unit tests for the OpenRouter decision-model client (no network)."""

from __future__ import annotations

import pytest
from server.decider import (
    _KEEP_QUESTION,
    Decision,
    FakeDecider,
    OpenRouterDecider,
    decision_from_payload,
)
from server.matcher import match_post


def test_prompt_excludes_biological_mutualism() -> None:
    keep = _KEEP_QUESTION['keep']
    note = 'biological mutualism (symbiosis between species)'
    assert note in keep['instructions'].lower()
    assert 'out of scope' in keep['instructions'].lower()
    assert note in keep['criteria']['false'].lower()


def test_decision_from_payload_thresholds_noul() -> None:
    payload = {'answers': {'keep': {'type': 'noul', 'noul': 0.62}}}
    assert decision_from_payload(payload, keep_min=0.5) == Decision(True, 0.62)
    assert decision_from_payload(payload, keep_min=0.7) == Decision(False, 0.62)


def test_decision_from_payload_rejects_unusable_body() -> None:
    with pytest.raises(ValueError, match='noul'):
        decision_from_payload(
            {'answers': {'keep': {'type': 'choice', 'choice': 'yes'}}},
            keep_min=0.5,
        )


def test_openrouter_decider_uses_injected_judge() -> None:
    seen: list[str] = []

    def judge(text: str) -> dict[str, object]:
        seen.append(text)
        return {'answers': {'keep': {'type': 'noul', 'noul': 0.2}}}

    decider = OpenRouterDecider(api_key='test', keep_min=0.5, _judge=judge)
    decision = decider.decide('Mutual credit for neighbors.')
    assert decision == Decision(False, 0.2)
    assert seen == ['Mutual credit for neighbors.']
    assert decider.decide('Mutual credit for neighbors.') == Decision(False, 0.2)
    assert len(seen) == 1


def test_decider_rejects_strong_positive() -> None:
    result = match_post(
        'Mutual credit clearing beats landlord title.',
        decider=FakeDecider(keep=False),
    )
    assert result.matched is False
    assert result.reason.startswith('decision_reject:strong_positive:')


def test_decider_keeps_strong_positive() -> None:
    result = match_post(
        'Mutual credit clearing beats landlord title.',
        decider=FakeDecider(keep=True, score=0.88),
    )
    assert result.matched is True
    assert result.reason == 'decision_keep:strong_positive:0.880'


def test_ambiguous_leftover_skips_decider() -> None:
    result = match_post(
        'Our co-op shares tools citywide and refuses landlords.',
        decider=FakeDecider(keep=True),
    )
    assert result.matched is False
    assert result.reason == 'ambiguous_no_context:co-op'


def test_soft_prior_candidate_is_filtered() -> None:
    did = 'did:plc:softpriortest0000000000001'
    kept = match_post(
        'Direct action works when neighbors show up.',
        author_did=did,
        soft_prior_dids={did},
        decider=FakeDecider(keep=True, score=0.8),
    )
    assert kept.matched is True
    assert kept.reason == 'decision_keep:soft_prior_ambiguous:direct action:0.800'

    rejected = match_post(
        'Direct action works when neighbors show up.',
        author_did=did,
        soft_prior_dids={did},
        decider=FakeDecider(keep=False, score=0.1),
    )
    assert rejected.matched is False
    assert rejected.reason.startswith('decision_reject:soft_prior_ambiguous:direct action:')


def test_unavailable_decider_drops_candidate() -> None:
    from server.decider import UnavailableDecider

    result = match_post(
        'Mutual credit clearing beats landlord title.',
        decider=UnavailableDecider(),
    )
    assert result.matched is False
    assert result.reason == 'decision_unscored:strong_positive'
