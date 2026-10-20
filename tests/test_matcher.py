import json
from pathlib import Path
from typing import Any

import pytest
from server.allowlists import (
    load_allowlist_dids,
    load_allowlist_handles,
    load_blocklist_dids,
    load_blocklist_handles,
)
from server.matcher import extract_alt_text, is_opaque_record_embed, match_post

ALL_CASES = json.loads(
    (Path(__file__).resolve().parents[1] / 'data' / 'eval_cases.json').read_text(encoding='utf-8')
)
CASES = [c for c in ALL_CASES if c.get('regression', True)]

ALLOWLIST_HANDLES = load_allowlist_handles()
ALLOWLIST_DIDS = load_allowlist_dids()
BLOCKLIST_HANDLES = load_blocklist_handles()
BLOCKLIST_DIDS = load_blocklist_dids()


@pytest.mark.parametrize('case', CASES, ids=[c['id'] for c in CASES])
def test_eval_case(case: dict[str, Any]) -> None:
    soft_prior_dids: set[str] = set()
    if case.get('soft_prior') and case.get('author_did'):
        soft_prior_dids.add(str(case['author_did']))
    langs = case.get('langs')
    result = match_post(
        case.get('text', ''),
        alt_text=case.get('alt_text', ''),
        langs=langs if isinstance(langs, list) else None,
        author_did=case.get('author_did'),
        author_handle=case.get('author_handle'),
        allowlist_dids=ALLOWLIST_DIDS,
        allowlist_handles=ALLOWLIST_HANDLES,
        blocklist_dids=BLOCKLIST_DIDS,
        blocklist_handles=BLOCKLIST_HANDLES,
        soft_prior_dids=soft_prior_dids,
        embed=case.get('embed'),
    )
    assert result.matched is bool(case['expected']), (
        f'{case["id"]}: expected={case["expected"]} got={result.matched} '
        f'reason={result.reason} note={case.get("note")}'
    )


def test_soft_prior_unlocks_bare_ambiguous_not_hard_negative() -> None:
    did = 'did:plc:softpriortest0000000000001'
    priors = {did}
    keep = match_post(
        'Direct action works when neighbors show up.',
        author_did=did,
        soft_prior_dids=priors,
    )
    assert keep.matched is True
    assert keep.reason == 'soft_prior_ambiguous:direct action'

    drop = match_post(
        'Direct action works when neighbors show up.',
        author_did=did,
        soft_prior_dids=set(),
    )
    assert drop.matched is False
    assert drop.reason == 'ambiguous_no_context:direct action'

    blocked = match_post(
        'Ancap tip: privatization fixes everything.',
        author_did=did,
        soft_prior_dids=priors,
    )
    assert blocked.matched is False
    assert blocked.reason in {'hard_negative', 'entity_out_of_scope:anarcho_capitalism'}


def test_allowlist_did_matches_without_handle_or_keywords() -> None:
    did = 'did:plc:allowlisttest00000000000001'
    result = match_post(
        'Thanks for reading — more updates tomorrow.',
        author_did=did,
        allowlist_dids={did},
        allowlist_handles=set(),
    )
    assert result.matched is True
    assert result.reason == 'allowlist_did'


def test_blocklist_drops_even_with_strong_text() -> None:
    result = match_post(
        'Proudhonian mutualism reading group on mutual credit tonight.',
        author_did='did:plc:blocklisted00000000000001',
        author_handle='blocked.bsky.social',
        blocklist_dids={'did:plc:blocklisted00000000000001'},
        blocklist_handles={'blocked.bsky.social'},
    )
    assert result.matched is False
    assert result.reason == 'blocklist_did'

    by_handle = match_post(
        'Proudhonian mutualism reading group on mutual credit tonight.',
        author_handle='blocked.bsky.social',
        blocklist_handles={'blocked.bsky.social'},
    )
    assert by_handle.matched is False
    assert by_handle.reason == 'blocklist_handle'


def test_blocklist_beats_allowlist() -> None:
    did = 'did:plc:allowandblock00000000000001'
    result = match_post(
        'Our newsletter is out for subscribers.',
        author_did=did,
        allowlist_dids={did},
        blocklist_dids={did},
    )
    assert result.matched is False
    assert result.reason == 'blocklist_did'


def test_event_in_scope_venue_requires_cue_and_venue() -> None:
    keep = match_post('Really Really Free Market this Saturday in the park — bring what you can.')
    assert keep.matched is True
    assert keep.reason.startswith('event_in_scope_venue:')

    no_cue = match_post('The really really free market archive needs a new index.')
    assert no_cue.matched is False

    off_topic = match_post('Tickets on sale for Saturday comedy night at The Fillmore. Doors at 7.')
    assert off_topic.matched is False


def test_critique_of_ancap_still_keeps() -> None:
    result = match_post(
        'Mutualism is not anarcho-capitalism. Mutual credit abolishes capitalist title.'
    )
    assert result.matched is True
    assert result.reason in {'strong_positive_over_negative', 'strong_positive'}


def test_bare_mutual_aid_does_not_keep() -> None:
    result = match_post(
        'Neighborhood mutual aid fridge restocked — take what you need, share what you can.'
    )
    assert result.matched is False


def test_mutual_aid_with_mutualist_economics_keeps() -> None:
    result = match_post(
        'Neighborhood mutual aid pairs with mutual credit so reciprocity is not charity.'
    )
    assert result.matched is True


def test_ambiguous_co_op_stays_on_the_regex_floor() -> None:
    from server.decider import FakeDecider

    result = match_post(
        'Our co-op shares tools citywide and refuses landlords.',
        decider=FakeDecider(keep=True),
    )
    assert result.matched is False
    assert result.reason == 'ambiguous_no_context:co-op'


def test_extract_alt_text_clips_external_description() -> None:
    embed = {
        'external': {
            'title': 'Mutual credit zine',
            'description': 'x' * 500,
        }
    }
    alt = extract_alt_text(embed)
    assert 'Mutual credit zine' in alt
    assert len(alt) < 500


def test_extract_alt_text_from_hydrated_quote_value() -> None:
    embed = {
        '$type': 'app.bsky.embed.record#view',
        'record': {
            'uri': 'at://did:plc:x/app.bsky.feed.post/3y',
            'value': {'text': 'Quoted mutual credit note', 'embed': None},
        },
    }
    assert 'Quoted mutual credit note' in extract_alt_text(embed)
    assert not is_opaque_record_embed(embed)


def test_is_opaque_record_embed_jetstream_shape() -> None:
    embed = {
        '$type': 'app.bsky.embed.record',
        'record': {
            'uri': 'at://did:plc:x/app.bsky.feed.post/3y',
            'cid': 'bafyreiexample',
        },
    }
    assert is_opaque_record_embed(embed)
    assert extract_alt_text(embed) == ''
