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

    charter = match_post(
        'Democracy is not only for governments. We govern our AI use together '
        'according to a cooperative charter: '
        'https://git.inference.coop/co-op/docs/src/branch/main/charter.md '
        'What policies would you want around this technology?'
    )
    assert charter.matched is True
    assert charter.reason == 'entity_in_scope:inference_cooperative'

    charter_phrase = match_post('We govern our AI use together according to a cooperative charter.')
    assert charter_phrase.matched is True
    assert charter_phrase.reason == 'strong_positive'

    charter_school = match_post('The Cooperative Charter School announced a bake sale for Friday.')
    assert charter_school.matched is False

    charter_academy = match_post(
        'Families toured the Cooperative Charter Academy before the open house.'
    )
    assert charter_academy.matched is False


def test_worker_owned_farming_and_housing_coops_are_strong() -> None:
    worker_owned = match_post(
        'workers in the charity rating industry should transform their workplace '
        'into a worker-owned cooperative'
    )
    assert worker_owned.matched is True
    assert worker_owned.reason == 'strong_positive'

    workplace = match_post(
        'sometimes i forget i work with a "worker-owned cooperative of abolitionists" '
        'and the mission statement is longer than the shift'
    )
    assert workplace.matched is True
    assert workplace.reason == 'strong_positive'

    owned_company = match_post(
        'The worker-owned company laid off staff and called the remaining shop a team.'
    )
    assert owned_company.matched is False

    farming = match_post(
        'High point was visiting the Burin Land and Farming Cooperative in Palestine, '
        'which works on local food sovereignty.'
    )
    assert farming.matched is True
    assert farming.reason == 'strong_positive'

    farming_game = match_post('Stardew farming co-op with four friends until the asteroid hits.')
    assert farming_game.matched is False

    agricole = match_post(
        'Face à la crise du maïs à Kananga, une coopérative agricole lance sa récolte '
        'sur le marché au prix social.'
    )
    assert agricole.matched is True
    assert agricole.reason == 'strong_positive'

    loomio = match_post(
        'A retrospective of my time supporting the founding of the Loomio Cooperative.'
    )
    assert loomio.matched is True
    assert loomio.reason == 'strong_positive'

    loomio_tool = match_post('We ran the book club poll on Loomio and closed it at nine.')
    assert loomio_tool.matched is False

    homeowner = match_post(
        'Each put down $10,000 and became a homeowner at C Street Co-op, '
        'about $800 a month against $1,400 rent.'
    )
    assert homeowner.matched is True
    assert homeowner.reason == 'strong_positive'

    game = match_post('Loot-shooter chaos with online co-op for 2-4 players.')
    assert game.matched is False

    shifts = match_post(
        'Here are the next set of volunteer labour shifts available at Karma Co-op '
        'for store opening, closing and cashier duty.'
    )
    assert shifts.matched is True
    assert shifts.reason == 'strong_positive'

    producers = match_post(
        'Les producteurs d’oignon de Pitoa se sont regroupés autour de Riskou Djarné, '
        'une coopérative créée en 2015 avec conseil d’administration.'
    )
    assert producers.matched is True
    assert producers.reason == 'strong_positive'

    cooperation = match_post(
        'Worker-owned cooperation between the two agencies improved the grant process.'
    )
    assert cooperation.matched is False


def test_plural_workers_software_and_french_coops_are_strong() -> None:
    workers = match_post('To me Socialism is just another word for Workers cooperative.')
    assert workers.matched is True
    assert workers.reason == 'strong_positive'

    workers_apostrophe = match_post("The workers' co-op bought the lathe with member shares.")
    assert workers_apostrophe.matched is True
    assert workers_apostrophe.reason == 'strong_positive'

    software = match_post(
        'Hey folks! I was wondering if any oomfs know about running a sottware cooperative? '
        'Legal / social / financial structure, day to day, how to get started, whatever.'
    )
    assert software.matched is True
    assert software.reason == 'strong_positive'

    software_hyphen = match_post('We are starting a software co-op for the studio tools.')
    assert software_hyphen.matched is True
    assert software_hyphen.reason == 'strong_positive'

    software_cooperation = match_post(
        'The agencies signed a software cooperation agreement and a grant.'
    )
    assert software_cooperation.matched is False

    shop = match_post(
        'Vous voulez faire vos courses dans une coopérative engagée '
        'au centre ville de Strasbourg ? Venez vous renseigner au magasin Coopalim'
    )
    assert shop.matched is True
    assert shop.reason == 'strong_positive'

    platform = match_post(
        'One of our members is Platformcoops in Germany. '
        'We design and support cooperative systems that redistribute ownership.'
    )
    assert platform.matched is True
    assert platform.reason == 'strong_positive'

    platform_agreement = match_post(
        'The ministries signed a platform cooperation agreement on data sharing.'
    )
    assert platform_agreement.matched is False

    rehab = match_post(
        'Portrait de ce Stéphanois, qui préside une coopérative qui réhabilite des appartements '
        'pour les mettre à disposition des plus fragiles.'
    )
    assert rehab.matched is True
    assert rehab.reason == 'strong_positive'

    farming = match_post(
        'Ghassan Najjar, co-founder of Burin’s Land and Farming Agricultural Cooperative, '
        'was arrested for peaceful activism to protect the olive harvest.'
    )
    assert farming.matched is True
    assert farming.reason == 'strong_positive'

    inference = match_post(
        'A bit of a milestone!',
        alt_text='101 members of the Inference Cooperative!',
    )
    assert inference.matched is True
    assert inference.reason == 'entity_in_scope:inference_cooperative'

    milestone_alone = match_post('A bit of a milestone!')
    assert milestone_alone.matched is False

    supermarket = match_post(
        'The Co-op was originally membership owned and members used to get a divi. '
        'If it is still member owned it might be an alternative to the big guys.'
    )
    assert supermarket.matched is False

    invest = match_post(
        'La coopérative TyGAMES c’est pour très bientôt ! '
        'Il faudrait encore 23 parts sociales souscrites. Investir. Rejoindre la coopérative.'
    )
    assert invest.matched is False

    research = match_post(
        'The hemp industry applied for a government grant to set up a Cooperative Research Centre.'
    )
    assert research.matched is False

    game = match_post('EXIT: The Game — Family Game Cooperative Game Puzzle a Day Escape Room')
    assert game.matched is False


def test_named_cooperative_forms_from_the_floor_are_strong() -> None:
    mondragon = match_post(
        "Spain's Mondragon cooperative group employs over 71,000 people. "
        'Built around worker cooperation, not billionaire ownership.'
    )
    assert mondragon.matched is True
    assert mondragon.reason == 'strong_positive'

    farmers = match_post(
        'Farmer cooperatives must be prepared for crises, not only for bigger markets.'
    )
    assert farmers.matched is True
    assert farmers.reason == 'strong_positive'

    processor = match_post(
        'Glacier Processing Cooperative is upgrading its plant to better serve producers.'
    )
    assert processor.matched is True
    assert processor.reason == 'strong_positive'

    bookshop = match_post(
        'our bookshop cooperative had its AGM and voted to become a not-for-profit '
        'so the surplus keeps a bookshop in the area'
    )
    assert bookshop.matched is True
    assert bookshop.reason == 'strong_positive'

    media = match_post(
        '1 hour webinar about the project',
        alt_text=('Media co-ops: How journalists and news organizations are building together'),
    )
    assert media.matched is True
    assert media.reason == 'strong_positive'

    social_media = match_post('Our social media co-op campaign ships the app on Friday.')
    assert social_media.matched is False

    gallery = match_post(
        'Sylvia Sleigh painted the members of the longest running all-female '
        'artists cooperative gallery in the US.'
    )
    assert gallery.matched is True
    assert gallery.reason == 'strong_positive'

    joined = match_post("I've officially become a member of a co-op art gallery.")
    assert joined.matched is True
    assert joined.reason == 'strong_positive'

    umbrella = match_post(
        'Are there any cooperative umbrella orgs that offer benefits and payroll '
        'to enrolled freelancer members?'
    )
    assert umbrella.matched is True
    assert umbrella.reason == 'strong_positive'

    lab = match_post('Wish we can start our own non-profit/cooperative lab testing service in BC.')
    assert lab.matched is True
    assert lab.reason == 'strong_positive'

    farms = match_post(
        'Knouse Foods is a co-op of just over 100 family farms — 15,000 acres of apples.'
    )
    assert farms.matched is True
    assert farms.reason == 'strong_positive'

    trades = match_post(
        'Maçonnerie, couverture, plomberie, électricité…',
        alt_text='Huit entreprises s’unissent en coopérative',
    )
    assert trades.matched is True
    assert trades.reason == 'strong_positive'

    inference = match_post(
        'Could universities join the co-op on their own terms? '
        'https://git.inference.coop/co-op/support/issues/32'
    )
    assert inference.matched is True
    assert inference.reason == 'entity_in_scope:inference_cooperative'

    bare = match_post('Could universities join the co-op on their own terms?')
    assert bare.matched is False

    purchase = match_post('Council authorized a cooperative purchase agreement with Motorola.')
    assert purchase.matched is False

    heist = match_post('French cooperative bank Crédit Mutuel Arkéa was targeted in a cyberattack.')
    assert heist.matched is False

    cooperative_adj = match_post("You've been very cooperative.")
    assert cooperative_adj.matched is False


def test_cooperative_forms_missed_on_2026_11_09() -> None:
    movement = match_post(
        'I had no idea that the White Poppy movement came about in 1933... '
        'www.co-operativeheritage.coop/Blog/the-co-operative-association'
    )
    assert movement.matched is True
    assert movement.reason == 'strong_positive'

    movement_alt = match_post(
        'A note on remembrance.',
        alt_text=(
            'The Co-op Movement and the White Poppy. '
            'The Co-operative Association with the White Poppy.'
        ),
    )
    assert movement_alt.matched is True
    assert movement_alt.reason == 'strong_positive'

    newsroom = match_post(
        'Start.coop is running an accelerator for worker-led and cooperative newsrooms. '
        'The cooperative model can serve readers better than media owned by billionaires.'
    )
    assert newsroom.matched is True
    assert newsroom.reason == 'strong_positive'

    bookshop = match_post(
        'NUMAX a reçu un prix.',
        alt_text=(
            'Cette coopérative galicienne, qui partage ses locaux avec une salle de cinéma, '
            'ne dispose que de 45 m² consacrés à la vente de livres.'
        ),
    )
    assert bookshop.matched is True
    assert bookshop.reason == 'strong_positive'

    dairies = match_post(
        'California Dairies is a co-op of about 300 dairies. The owners milk cows.'
    )
    assert dairies.matched is True
    assert dairies.reason == 'strong_positive'

    seniors = match_post(
        'Elle a opté pour l’indépendance.',
        alt_text=(
            "Aurora, 73 ans, résidente d'une coopérative pour seniors : "
            'ici, ce n’est pas une maison de retraite.'
        ),
    )
    assert seniors.matched is True
    assert seniors.reason == 'strong_positive'

    communities = match_post(
        'La Regione investe sui borghi.',
        alt_text='360mila euro per sostenere le cooperative di comunità nei borghi montani.',
    )
    assert communities.matched is True
    assert communities.reason == 'strong_positive'

    cave = match_post(
        'Le Chai des Vignerons, la cave coopérative de Lézignan-Corbières, '
        'enregistre enfin une récolte normale.'
    )
    assert cave.matched is True
    assert cave.reason == 'strong_positive'

    member = match_post(
        'OSHA threatened me at the co-op I pay to be a member of. '
        "I'm not an employee and there is no boss."
    )
    assert member.matched is True
    assert member.reason == 'strong_positive'

    pledge = match_post(
        'The owner of Bloomin’ Farm Cooperative is asking 7,000 people to pledge $500 '
        'each so the community will form a cooperative and keep the land.'
    )
    assert pledge.matched is False

    supermarket = match_post('The local Co-op put the chicken breast in a security box.')
    assert supermarket.matched is False

    heist = match_post('French cooperative bank Crédit Mutuel Arkéa was targeted in a cyberattack.')
    assert heist.matched is False


def test_cooperative_forms_missed_on_2026_11_10() -> None:
    dairy = match_post(
        'The store-brand butter comes from Foremost Farms, a Wisconsin co-op '
        'owned by hundreds of dairy farms across seven states.'
    )
    assert dairy.matched is True
    assert dairy.reason == 'strong_positive'

    apex = match_post(
        'Join Co-operatives UK online to find out more about co-ops, the benefits '
        'they offer and how to get a co-op business off the ground.'
    )
    assert apex.matched is True
    assert apex.reason == 'strong_positive'

    wholesale = match_post(
        "1953 advert for the Co-op's Progress washing machine manufactured at the "
        'CWS Keighley Ironworks.'
    )
    assert wholesale.matched is True
    assert wholesale.reason == 'strong_positive'

    bookshop = match_post(
        'Lecture musicale au sein de notre librairie coopérative, pour les 50 ans de la librairie.'
    )
    assert bookshop.matched is True
    assert bookshop.reason == 'strong_positive'

    members = match_post(
        "They're now full voting members of the cooperative and will help shape Northsky."
    )
    assert members.matched is True
    assert members.reason == 'strong_positive'

    farms = match_post(
        'Black farmers open a grocery store.',
        alt_text=(
            'Fountain Heights Farms Cooperative opens a grocery store in Birmingham '
            'after nearly 70 years of food scarcity.'
        ),
    )
    assert farms.matched is True
    assert farms.reason == 'strong_positive'

    scop = match_post(
        'Les salarié·es de Scop-Ti, près de Marseille, autogèrent leur usine de thé depuis 2014.'
    )
    assert scop.matched is True
    assert scop.reason == 'strong_positive'

    jobs = match_post(
        'The founders of ChiFresh Kitchen built a network of co-ops to provide jobs and homes.'
    )
    assert jobs.matched is True
    assert jobs.reason == 'strong_positive'

    school = match_post(
        'Jean Pham, a worker-owner of Underpainting Collective, a cooperative school of fine art.'
    )
    assert school.matched is True
    assert school.reason == 'strong_positive'

    platforms = match_post(
        'Why Don’t Artists Join Together in Co-Op Platforms? '
        'Most of the money in the creative economy goes to corporations and intermediaries.'
    )
    assert platforms.matched is True
    assert platforms.reason == 'strong_positive'

    hoax = match_post(
        'Apple Corps is going to do Imagine Multimedia. There was a vote after '
        "Paul and Ringo's apparent deaths.",
        alt_text=(
            'Why Don’t Artists Join Together in Co-Op Platforms? '
            'Most of the money in the creative economy goes to corporations.'
        ),
    )
    assert hoax.matched is False
    assert hoax.reason == 'hard_negative'

    house = match_post(
        'Building cooperative: This will be a house that suits everyone. '
        'https://www.zeit.de/geld/2026-11/baugemeinschaft-hausbau'
    )
    assert house.matched is True
    assert house.reason == 'strong_positive'

    cleaning = match_post(
        'Green, toxin-free cleaning from a Pioneer Valley co-op that pays fair wages.'
    )
    assert cleaning.matched is True
    assert cleaning.reason == 'strong_positive'

    solar = match_post(
        'A free Sanibel solar co-op is offering education, evaluations, and '
        'bulk-purchase savings for residents who want backup power.'
    )
    assert solar.matched is True
    assert solar.reason == 'strong_positive'

    community = match_post('Homeless by GSA: The housing association bankrupting a community co-op')
    assert community.matched is True
    assert community.reason == 'strong_positive'

    shares = match_post('Buy shares in our solar co-op. The offer closes Friday.')
    assert shares.matched is False

    kickstarter = match_post(
        "We're a worker-cooperative founded by veteran gamedevs on socialist principles, "
        "and we're launching on Kickstarter next Tuesday."
    )
    assert kickstarter.matched is False

    benefit = match_post(
        'We’ll be playing some new songs at this benefit for our children’s place, '
        'a co-op preschool in Burbank.'
    )
    assert benefit.matched is False

    pledge = match_post(
        'Sun Prairie farm owner seeks 7,000 pledges to turn farm into full cooperative.'
    )
    assert pledge.matched is False

    merger = match_post(
        'The CMA has approved plans for Southern Co-op and The Co-operative Group to join forces.'
    )
    assert merger.matched is False


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
