"""Unit tests for the OpenRouter decision-model client (no network)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from server.decider import (
    _KEEP_QUESTION,
    PROMPT_VERSION,
    Decision,
    FakeDecider,
    OpenRouterDecider,
    decision_from_payload,
)
from server.matcher import match_post


def test_prompt_rejects_socialize_all_housing_foil() -> None:
    keep = _KEEP_QUESTION['keep']
    note = 'socialize all housing'
    assert note in keep['instructions'].lower()
    assert 'out of scope' in keep['instructions'].lower()
    assert note in keep['criteria']['false'].lower()
    assert 'tenant ownership' in keep['criteria']['false'].lower()


def test_prompt_excludes_biological_mutualism() -> None:
    keep = _KEEP_QUESTION['keep']
    note = 'biological mutualism (symbiosis between species)'
    assert note in keep['instructions'].lower()
    assert 'out of scope' in keep['instructions'].lower()
    assert note in keep['criteria']['false'].lower()


def test_prompt_rejects_social_club_invites() -> None:
    keep = _KEEP_QUESTION['keep']
    assert PROMPT_VERSION == 'institution-keep-v5'
    assert 'fandom, roleplay, and adult social-club invites' in keep['instructions'].lower()
    assert 'vrchat' in keep['criteria']['false'].lower()
    assert 'roleplay packs' in keep['criteria']['false'].lower()


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
    seen: list[tuple[str, list[str]]] = []

    def judge(text: str, images: Sequence[str]) -> dict[str, object]:
        seen.append((text, list(images)))
        return {'answers': {'keep': {'type': 'noul', 'noul': 0.2}}}

    decider = OpenRouterDecider(api_key='test', keep_min=0.5, _judge=judge)
    decision = decider.decide('Mutual credit for neighbors.')
    assert decision == Decision(False, 0.2)
    assert seen == [('Mutual credit for neighbors.', [])]
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


def test_openrouter_decider_sends_image_parts_and_keeps_text_state(monkeypatch) -> None:
    import json

    captured: list[dict[str, object]] = []

    class _Response:
        def read(self) -> bytes:
            return b'{"answers":{"keep":{"type":"noul","noul":0.2}}}'

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

    def fake_urlopen(request, timeout: float) -> _Response:
        del timeout
        captured.append(json.loads(request.data.decode()))
        return _Response()

    monkeypatch.setattr('server.decider.urllib.request.urlopen', fake_urlopen)
    url = 'data:image/jpeg;base64,' + ('A' * 16)
    decider = OpenRouterDecider(api_key='test', timeout_s=8)

    text_only = decider.decide('Mutual credit for neighbors.')
    assert text_only == Decision(False, 0.2)
    assert captured[0]['state'] == {'post': 'Mutual credit for neighbors.'}

    with_image = decider.decide(
        'Mutual credit for neighbors.',
        images=[url, 'https://evil.example/a.jpg', url],
    )
    assert with_image == Decision(False, 0.2)
    assert captured[1]['state'] == {
        'post': 'Mutual credit for neighbors.',
        'images': [{'type': 'image_url', 'image_url': {'url': url}}],
    }
    assert decider.decide('Mutual credit for neighbors.', images=[url]) == Decision(False, 0.2)
    assert len(captured) == 2


def test_match_post_passes_embed_images_to_decider(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_urls(embed: object, *, author_did: str | None) -> list[str]:
        seen['embed'] = embed
        seen['author_did'] = author_did
        return ['data:image/jpeg;base64,YQ==']

    monkeypatch.setattr('server.matcher.image_data_urls', fake_urls)

    class _Spy:
        def decide(self, haystack: str, *, images: Sequence[str] | None = None) -> Decision:
            seen['haystack'] = haystack
            seen['images'] = images
            return Decision(True, 0.8)

    result = match_post(
        'Mutual credit clearing beats landlord title.',
        author_did='did:plc:abc123',
        embed={'$type': 'app.bsky.embed.images'},
        decider=_Spy(),
    )
    assert result.matched is True
    assert seen['author_did'] == 'did:plc:abc123'
    assert seen['images'] == ['data:image/jpeg;base64,YQ==']
    assert 'Mutual credit' in str(seen['haystack'])


def test_match_post_without_decider_does_not_fetch_images(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(embed: object, *, author_did: str | None) -> list[str]:
        del embed, author_did
        raise AssertionError('image fetch on the regex floor')

    monkeypatch.setattr('server.matcher.image_data_urls', boom)
    result = match_post(
        'Mutual credit clearing beats landlord title.',
        embed={'$type': 'app.bsky.embed.images'},
    )
    assert result.matched is True
    assert result.reason == 'strong_positive'


def test_unavailable_decider_drops_candidate() -> None:
    from server.decider import UnavailableDecider

    result = match_post(
        'Mutual credit clearing beats landlord title.',
        decider=UnavailableDecider(),
    )
    assert result.matched is False
    assert result.reason == 'decision_unscored:strong_positive'
