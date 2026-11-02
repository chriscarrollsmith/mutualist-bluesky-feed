import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from server.allowlists import (
    load_allowlist_dids,
    load_allowlist_handles,
    load_blocklist_dids,
    load_blocklist_handles,
)
from server.decider import Decision, SpiritDecision, UnavailableDecider
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


def test_bookchin_and_institutions_keep() -> None:
    municipal = match_post(
        "Bookchin's libertarian municipalism treats the city assembly as government."
    )
    assert municipal.matched is True
    assert municipal.reason == 'strong_positive'

    rojava = match_post("The Rojava commune runs a bakery cooperative and a women's council.")
    assert rojava.matched is True
    assert rojava.reason.startswith('ambiguous_with_context:')

    critique = match_post('Bookchin rejected armed struggle in favor of the municipal assembly.')
    assert critique.matched is True
    assert critique.reason == 'strong_positive_over_negative'


def test_social_club_invite_does_not_keep_ideology_namedrop() -> None:
    drop = match_post(
        'Our political foundation is communalism, libertarian municipalism, '
        'and democratic confederalism.',
        alt_text=(
            'Come join your wolf goddess’s pack. This 18+ community hangs out '
            'in VRChat. Join the Discord server — a little communal shrine.'
        ),
    )
    assert drop.matched is False
    assert drop.reason == 'hard_negative'

    keep = match_post(
        "Bookchin's libertarian municipalism treats the city assembly as a way "
        'to govern without the nation-state.'
    )
    assert keep.matched is True
    assert keep.reason == 'strong_positive'


def test_commercial_peoples_bank_earnings_drop() -> None:
    earnings = match_post(
        'Peoples Bank turns $1.4 million profit in third quarter https://ift.tt/x1GHMqB'
    )
    assert earnings.matched is False
    assert earnings.reason == 'hard_negative'

    dividends = match_post("People's Bank declared a dividend for shareholders this morning.")
    assert dividends.matched is False
    assert dividends.reason == 'hard_negative'

    institution = match_post(
        "Proudhon's People's Bank issued mutual credit, not a profit every quarter."
    )
    assert institution.matched is True

    bare = match_post("The People's Bank is a mutual credit desk for the neighborhood.")
    assert bare.matched is True
    assert bare.reason == 'strong_positive'


def test_socialize_all_housing_drops_proudhon_foil() -> None:
    jacobin = match_post(
        'Socialize All Housing. Arthur Mülberger, drawing on Pierre-Joseph Proudhon, '
        'offered tenants a path to become owners.'
    )
    assert jacobin.matched is False
    assert jacobin.reason == 'hard_negative'

    rebuttal = match_post(
        'Proudhon rejected the call to socialize all housing in favor of occupancy-and-use.'
    )
    assert rebuttal.matched is True

    commune = match_post(
        'Kropotkin argued the federated commune should socialize all housing as common heritage.'
    )
    assert commune.matched is True


def test_institution_collisions_and_offensive_violence_drop() -> None:
    academic = match_post('Social ecology is an interesting academic field.')
    assert academic.matched is False

    systems = match_post(
        'Social-ecological systems research shows how fisheries respond to climate stress.'
    )
    assert systems.matched is False

    sectarian = match_post('Communalism is tearing the election apart along religious lines.')
    assert sectarian.matched is False

    brussels = match_post('EU confederalism is the debate over sovereignty in Brussels.')
    assert brussels.matched is False

    war = match_post('Rojava forces launched an offensive across the river last night.')
    assert war.matched is False

    martyrs = match_post('Democratic confederalism honors the martyrs of the offensive.')
    assert martyrs.matched is False
    assert martyrs.reason == 'hard_negative'


def test_ambiguous_co_op_stays_on_the_regex_floor() -> None:
    from server.decider import FakeDecider

    result = match_post(
        'Our co-op shares tools citywide and refuses landlords.',
        decider=FakeDecider(keep=True),
    )
    assert result.matched is False
    assert result.reason == 'ambiguous_no_context:co-op'


def test_food_and_nonprofit_coops_are_strong_and_game_co_op_stays_dropped() -> None:
    food = match_post(
        'The Canberra Food Co-op has been a food cooperative for fifty years, '
        'dividing bulk orders among members.'
    )
    assert food.matched is True
    assert food.reason == 'strong_positive'

    nonprofit = match_post(
        'Link title only.',
        alt_text=(
            'What if social media was a nonprofit co-op? CoSocial is a '
            'self-funding, self-governing Mastodon cooperative.'
        ),
    )
    assert nonprofit.matched is True
    assert nonprofit.reason == 'strong_positive'

    game = match_post('Loot-shooter chaos with online co-op for 2-4 players.')
    assert game.matched is False
    assert game.reason == 'ambiguous_no_context:co-op'

    supermarket = match_post(
        "Hilltop honey beat Rowse, Aldi, Co-op and Sainsbury's in a taste test."
    )
    assert supermarket.matched is False

    snark = match_post(
        'The worker run co-op game dev laid off workers and called it a labor victory.'
    )
    assert snark.matched is False
    assert snark.reason == 'ambiguous_no_context:co-op'

    grocery = match_post(
        'vtdigger.org/2026/10/31/u...',
        alt_text=(
            'Upper Valley’s member-owned co-op battles rising food prices - VTDigger '
            'The grocery cooperative is struggling to stick to its roots and '
            'keep prices affordable.'
        ),
    )
    assert grocery.matched is True
    assert grocery.reason == 'strong_positive'

    grocery_coop = match_post('The neighborhood grocery co-op kept member prices below the chain.')
    assert grocery_coop.matched is True
    assert grocery_coop.reason == 'strong_positive'

    cooperation = match_post(
        'Nonprofit cooperation between the two agencies improved the grant process.'
    )
    assert cooperation.matched is False

    grocery_cooperation = match_post(
        'Grocery cooperation between the two chains improved the supply contract.'
    )
    assert grocery_cooperation.matched is False


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


class _SpiritScript:
    """Spirit-lane double. Core ``decide`` is unused and would be a test bug."""

    def __init__(self, decision: SpiritDecision | None) -> None:
        self.decision = decision
        self.calls = 0

    def decide(self, haystack: str, *, images: Sequence[str] | None = None) -> Decision | None:
        del haystack, images
        raise AssertionError('core decide called on a spirit-only double')

    def decide_spirit(
        self, haystack: str, *, images: Sequence[str] | None = None
    ) -> SpiritDecision | None:
        del haystack, images
        self.calls += 1
        return self.decision


def _spirit(
    *,
    spirit: float = 0.9,
    voice: float = 0.9,
    account: float = 0.9,
    lift: float = 0.9,
) -> SpiritDecision:
    return SpiritDecision(spirit=spirit, voice=voice, account=account, lift=lift)


def test_spirit_surface_stays_dropped_without_a_decider() -> None:
    repair = match_post('The sewing team at the repair cafe fixed a coat and a shoe.')
    assert repair.matched is False
    assert repair.reason == 'no_match'
    housing = match_post('One challenge of living in co-op housing is individual preference.')
    assert housing.matched is False
    assert housing.reason == 'ambiguous_no_context:co-op'


def test_spirit_voice_and_account_paths_keep() -> None:
    scene = match_post(
        'I woke up out of sorts, walked the cohousing field, and felt human again.',
        decider=_SpiritScript(_spirit(voice=0.99, account=0.40)),
    )
    assert scene.matched is True
    assert scene.reason == 'decision_keep:spirit_voice:0.990'

    explainer = match_post(
        'In cohousing it is easy to connect at common meals and work days.',
        decider=_SpiritScript(_spirit(voice=0.05, account=0.96)),
    )
    assert explainer.matched is True
    assert explainer.reason == 'decision_keep:spirit_account:0.960'


def test_spirit_gate_rejects_theme_lift_and_thin_paths() -> None:
    metaphor = match_post(
        'A shared compute cluster is akin to a tool library.',
        decider=_SpiritScript(_spirit(spirit=0.08, voice=0.19, account=0.20, lift=0.18)),
    )
    assert metaphor.matched is False
    assert metaphor.reason == 'decision_reject:spirit_theme:0.080'

    complaint = match_post(
        'My housing co-op meetings are often a circus.',
        decider=_SpiritScript(_spirit(spirit=0.70, voice=0.95, account=0.57, lift=0.47)),
    )
    assert complaint.matched is False
    assert complaint.reason == 'decision_reject:spirit_lift:0.470'

    fragment = match_post(
        'The community fridge even had this amazing microwave.',
        decider=_SpiritScript(_spirit(spirit=0.90, voice=0.14, account=0.66, lift=0.66)),
    )
    assert fragment.matched is False
    assert fragment.reason == 'decision_reject:spirit_path:0.660'


def test_spirit_solicit_does_not_call_the_decider() -> None:
    script = _SpiritScript(_spirit())
    result = match_post(
        'Please donate to my GoFundMe so I can stock the community fridge.',
        decider=script,
    )
    assert result.matched is False
    assert result.reason == 'solicit_cue_unscored:spirit'
    assert script.calls == 0


def test_spirit_lane_does_not_override_core_or_out_of_scope() -> None:
    from server.decider import FakeDecider

    core = match_post(
        'Mutual credit clearing at the repair cafe beats landlord title.',
        decider=FakeDecider(keep=True, score=0.88),
    )
    assert core.matched is True
    assert core.reason == 'decision_keep:strong_positive:0.880'

    blocked = match_post(
        'Anarcho-capitalism is the real repair cafe.',
        decider=_SpiritScript(_spirit()),
    )
    assert blocked.matched is False
    assert blocked.reason.startswith('entity_out_of_scope:')


def test_unavailable_decider_drops_spirit_candidate() -> None:
    result = match_post(
        'The sewing team at the repair cafe fixed a coat and a shoe.',
        decider=UnavailableDecider(),
    )
    assert result.matched is False
    assert result.reason == 'decision_unscored:spirit'


def test_spirit_surface_word_order() -> None:
    script = _SpiritScript(_spirit(voice=0.2, account=0.91))
    spaced = match_post(
        'A housing co op where neighbors share food and tools.',
        decider=script,
    )
    assert spaced.matched is True
    assert spaced.reason == 'decision_keep:spirit_account:0.910'
