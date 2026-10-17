"""Unit tests for the DeepSeek quality-rubric classifier (no network)."""

from __future__ import annotations

from server.classifier import (
    SYSTEM_PROMPT,
    ClassifierModel,
    DeepSeekClassifier,
    FakeClassifier,
    RubricThresholds,
    admit_grades,
    classify_candidate,
    composite_score,
    extract_features,
    grades_from_payload,
    load_model,
)
from server.matcher import (
    is_opaque_record_embed,
    looks_like_solicit,
    match_post,
)


def test_prompt_excludes_biological_mutualism() -> None:
    note = 'biological mutualism (symbiosis between species)'
    assert note in SYSTEM_PROMPT.lower()


def test_load_default_model_stub() -> None:
    model = load_model()
    assert model.version == 'deepseek_v4_flash'
    assert model.threshold > 0


def test_extract_features_ambiguous_term() -> None:
    feats = extract_features(
        'Our co-op refuses landlords',
        term='co-op',
        has_event_cue=False,
        has_local_venue=False,
    )
    assert feats['has_ambiguous'] == 1.0
    assert feats['has_event_cue'] == 0.0


def test_fake_classifier_keeps_substring() -> None:
    fake = FakeClassifier(keep_substrings=('refuses landlords',))
    decision = classify_candidate(
        'Our co-op shares tools and refuses landlords.',
        term='co-op',
        has_event_cue=False,
        has_local_venue=False,
        classifier=fake,
    )
    assert decision is not None
    assert decision.matched is True
    assert decision.reason.startswith('classifier:')
    assert decision.grades is not None
    assert decision.grades.solicit < 0.35


def test_deepseek_classifier_uses_injected_judge() -> None:
    clf = DeepSeekClassifier(
        api_key='test',
        thresholds=RubricThresholds(composite_min=0.7),
        _judge=lambda _text: {
            'keep': True,
            'thematic_fit': 0.91,
            'positive_valence': 0.88,
            'wow': 0.80,
            'solicit': 0.05,
            'rationale': 'aligned',
        },
    )
    decision = clf.classify(
        'Horizontal assemblies without bosses',
        term='autonomous',
        has_event_cue=False,
        has_local_venue=False,
    )
    assert decision is not None
    assert decision.matched is True
    assert decision.score == composite_score(0.91, 0.88, 0.80)
    assert decision.reason == 'classifier:ambiguous:autonomous'


def test_deepseek_classifier_accepts_legacy_score_payload() -> None:
    clf = DeepSeekClassifier(
        api_key='test',
        threshold=0.7,
        _judge=lambda _text: {'keep': True, 'score': 0.91, 'rationale': 'aligned'},
    )
    decision = clf.classify(
        'Horizontal assemblies without bosses',
        term='autonomous',
        has_event_cue=False,
        has_local_venue=False,
    )
    assert decision is not None
    assert decision.matched is True
    assert decision.score == 0.91


def test_deepseek_classifier_respects_thematic_gate() -> None:
    clf = DeepSeekClassifier(
        api_key='test',
        thresholds=RubricThresholds(thematic_min=0.8, composite_min=0.5),
        _judge=lambda _text: {
            'keep': True,
            'thematic_fit': 0.5,
            'positive_valence': 0.9,
            'wow': 0.9,
            'solicit': 0.0,
            'rationale': 'weak theme',
        },
    )
    decision = clf.classify(
        'maybe related',
        term='punk',
        has_event_cue=False,
        has_local_venue=False,
    )
    assert decision is not None
    assert decision.matched is False
    assert decision.reason == 'quality_reject:thematic_fit'


def test_deepseek_classifier_rejects_high_solicit() -> None:
    clf = DeepSeekClassifier(
        api_key='test',
        _judge=lambda _text: {
            'keep': True,
            'thematic_fit': 0.95,
            'positive_valence': 0.7,
            'wow': 0.6,
            'solicit': 0.9,
            'rationale': 'venmo ask',
        },
    )
    decision = clf.classify(
        'Mutual aid please Venmo me for rent',
        term=None,
        has_event_cue=False,
        has_local_venue=False,
    )
    assert decision is not None
    assert decision.matched is False
    assert decision.reason == 'quality_reject:solicit'


def test_grades_from_payload_accepts_thematic_typos() -> None:
    grades = grades_from_payload(
        {
            'keep': True,
            'thematic_fic': 0.82,
            'positive_valence': 0.7,
            'wow': 0.6,
            'solicit': 0.05,
        }
    )
    assert grades.thematic_fit == 0.82
    assert grades.composite == composite_score(0.82, 0.7, 0.6)


def test_admit_grades_composite_floor() -> None:
    grades = grades_from_payload(
        {
            'thematic_fit': 0.72,
            'positive_valence': 0.56,
            'wow': 0.46,
            'solicit': 0.1,
        }
    )
    # 0.45*0.72 + 0.25*0.56 + 0.30*0.46 = 0.324 + 0.14 + 0.138 = 0.602
    assert admit_grades(grades, RubricThresholds(composite_min=0.68)) == 'quality_reject:composite'
    assert admit_grades(grades, RubricThresholds(composite_min=0.60)) is None


def test_match_post_regex_floor_without_decider() -> None:
    strong = match_post('Mutual credit clearing beats landlord title.')
    assert strong.matched is True
    assert strong.reason == 'strong_positive'

    leftover = match_post('Our co-op shares tools citywide and refuses landlords.')
    assert leftover.matched is False
    assert leftover.reason == 'ambiguous_no_context:co-op'


def test_solicit_cue_vetoes_strong_positive_without_classifier() -> None:
    text = 'Mutualism and mutual credit are great — please Venmo @me for rent #mutualaid'
    assert looks_like_solicit(text)
    result = match_post(text)
    assert result.matched is False
    assert result.reason.startswith('solicit_cue_unscored:')


def test_solicit_cue_passes_when_decider_keeps() -> None:
    from server.decider import FakeDecider

    result = match_post(
        'Mutual credit association goal: donate via our collective PayPal.',
        decider=FakeDecider(keep=True, score=0.8),
    )
    assert result.matched is True
    assert result.reason == 'decision_keep:strong_positive:0.800'


def test_solicit_cue_drops_when_decider_rejects() -> None:
    from server.decider import FakeDecider

    result = match_post(
        'Mutualism mutual credit — Venmo me if you can spare anything for rent.',
        decider=FakeDecider(keep=False, score=0.04),
    )
    assert result.matched is False
    assert result.reason == 'decision_reject:strong_positive:0.040'


def test_allowlist_solicit_cue_still_gated() -> None:
    from server.decider import FakeDecider

    result = match_post(
        'CashApp me for mutual aid please',
        author_did='did:plc:allowlisted000000000000001',
        allowlist_dids={'did:plc:allowlisted000000000000001'},
        decider=FakeDecider(keep=False, score=0.02),
    )
    assert result.matched is False
    assert result.reason == 'decision_reject:allowlist_did:0.020'


def test_helpsky_hashtag_soup_is_solicit_shaped() -> None:
    text = '💕 💸 #HelpSky #MutualAid #HelpFolksLive #MABoost #MADBoost'
    assert looks_like_solicit(text)
    result = match_post(text)
    assert result.matched is False


def test_opaque_quote_with_mutual_aid_is_solicit_shaped() -> None:
    embed = {
        '$type': 'app.bsky.embed.record',
        'record': {
            'uri': 'at://did:plc:quoted000000000000000001/app.bsky.feed.post/3example',
            'cid': 'bafyreiopaquequote000000000000000000000000000000001',
        },
    }
    assert is_opaque_record_embed(embed)
    assert looks_like_solicit('#MutualAid #HelpSky', embed=embed)
    # Bare mutual aid + opaque quote is solicit-shaped, but without a
    # provisional keep (mutual aid alone is not strong) it drops earlier.
    bare = match_post('Mutual aid Friday', embed=embed)
    assert bare.matched is False
    strong_solicit = match_post('Mutualism mutual credit #MutualAid Friday', embed=embed)
    assert strong_solicit.matched is False
    assert strong_solicit.reason.startswith('solicit_cue_unscored:')


def test_hydrated_quote_uses_nested_text_not_opaque() -> None:
    embed = {
        '$type': 'app.bsky.embed.record#view',
        'record': {
            'uri': 'at://did:plc:quoted000000000000000001/app.bsky.feed.post/3example',
            'cid': 'bafyreiopaquequote000000000000000000000000000000001',
            'value': {
                'text': 'Neighborhood mutual credit circle restocked tonight.',
                '$type': 'app.bsky.feed.post',
            },
        },
    }
    assert not is_opaque_record_embed(embed)
    # Outer boost tags still solicit-gate via HelpSky / money cues.
    assert looks_like_solicit('#HelpSky #MutualAid', embed=embed)


def test_mutual_aid_praxis_without_mutualist_economics_drops() -> None:
    result = match_post(
        'Neighborhood mutual aid fridge restocked — take what you need, share what you can.'
    )
    assert result.matched is False


def test_classifier_model_stub_never_keeps() -> None:
    stub = ClassifierModel(version='x', threshold=0.5, weights={'bias': 10.0})
    assert (
        stub.classify(
            'anything',
            term='anarchy',
            has_event_cue=False,
            has_local_venue=False,
        )
        is None
    )
