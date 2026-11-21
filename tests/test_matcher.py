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
    assert keep.reason == 'strong_positive'

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


def test_cooperative_forms_missed_on_2026_11_11() -> None:
    cases = [
        'Discover how a member-owned co-op built the Seattle REI flagship.',
        'The co-op has no ad inventory to fill: member dues fund the service.',
        'Bison dip is made by a co-op. Upstate Niagara is owned by about 230 family dairy farms.',
        'Brighton and Hove Energy Services Co-op is opening a solar communities programme.',
        'Berkshire Co-op operating in the red for third year, finances in dire straits.',
        'Learn how we can help your co-op build a stronger financial foundation.',
        'lmao wait wft ace hardware is a co-op???',
        'We are now accepting bookings for the West Village Rehearsal Co-Op (WVRC).',
        'A Sabby co-op member with a monthly subscription should update payment.',
        'Wisconsin Food Hub Cooperative is bringing locally grown produce into school cafeterias.',
        'The county distributed tools to 50 Savings and Credit Cooperative Societies.',
        'Organic Valley operates as a farmer-owned cooperative rather than a corporation.',
        'Voxeurop is the first publication managed by a European press cooperative.',
        'VGS is a publishing cooperative managed by a voluntary board of directors.',
        'Adieux au patronat: gestion ouvrière dans une usine reprise en coopérative.',
        'Frédéric Mauduit, nouveau président de la coopérative Sherpa.',
        'If we have private equity should we not have cooperative equity?',
        'Jersey Homesteads was a cooperative community planned for garment workers.',
        'ICA helps bring cooperative practice closer together.',
        "C'est une coopérative qui réfléchit, investit, et innove.",
        'La coopérative a racheté une station service dont le propriétaire partait.',
        'Can a cooperative model help university spin-outs scale?',
        'Début de chantier pour la coopérative d’habitation La Solidarité.',
        'My new local pub is a gem. It is a cooperative, and the beer is local.',
    ]
    for text in cases:
        result = match_post(text)
        assert result.matched is True, text
        assert result.reason == 'strong_positive', text

    emilia = match_post(
        '',
        alt_text=(
            "Cooperative, un valore aggiunto per l'Emilia-Romagna. "
            "I risultati dell'indagine dell'Osservatorio regionale sulla Cooperazione."
        ),
    )
    assert emilia.matched is True
    assert emilia.reason == 'strong_positive'

    party = match_post('Friendsgiving dinner at the West Village Rehearsal Co-Op from 5 to 8.')
    assert party.matched is False

    renters = match_post(
        'Workers get first dibs to turn it into a cooperative, and generous grants '
        'from the state. Euthanise the renters already.'
    )
    assert renters.matched is False

    extension = match_post(
        'As a Master Gardener with Cooperative Extension, I am appalled that the '
        'county removed Extension funding.'
    )
    assert extension.matched is False

    tools = match_post(
        'What would it take for newcomers to actually join and use cooperative tools online?'
    )
    assert tools.matched is False

    electric = match_post('The rural electric co-op is raising the kilowatt rate this winter.')
    assert electric.matched is False

    hockey = match_post('The school board will seek legal advice on a boys hockey co-op.')
    assert hockey.matched is False

    dues_only = match_post('Reminder: member dues for the chess club are due Friday.')
    assert dues_only.matched is False


def test_cooperative_forms_missed_on_2026_11_12() -> None:
    cases = [
        'Four reporters started a co-op after their papers closed.',
        "That's how the Co-op started. Pooling saved money, then they opened a storefront.",
        'I am glad to see a co-op emerging around the ATmosphere.',
        'They are going to build Assembly as a co-op.',
        'Software to start and run a co-op or other democratic workplace.',
        'Governance software forged in co-op principles.',
        'Introducing the Atmosphere Community Cooperative, governed by its members.',
        'Edinburgh Community Solar Co-operative owns and operates the panels.',
        'Lock in your co-op shifts early or you will be sorting potatoes.',
        'Who we are: a member-governed home. Atmosphere Co-op.',
        "L'Alliance coopérative internationale ouvre PortailCoop.",
        "La coopérative foncière eclOOre mobilise l'épargne citoyenne.",
        'French cooperative law solved this with the titre participatif.',
        'A cooperative software initiative that helps shops collectively own their booking tools.',
        'Jersey Homesteads was planned as a cooperative community for garment workers.',
        'Come to the workshop, and join our co-op.',
        'Honoured to support as the cooperative gets set up.',
        'Really Really Free Market: Clothing Swap 11/15',
    ]
    for text in cases:
        result = match_post(text)
        assert result.matched is True, text

    drops = [
        'Luma Island adds four-player cross-platform co-op on December 2.',
        'I started a co-op game server for the weekend.',
        'The co-op shifts to a new map after the boss.',
        'Join our co-op game tonight.',
        (
            'The zoning code cannot be changed to support the Community Land Trust, '
            'which I support, unless staff support the changes, which I do not support #a2council'
        ),
        'Atmosphere Community Cooperative has been verified by @bsky.app.',
        'The really really free market archive needs a new index.',
        'The rural electric co-op is raising the kilowatt rate this winter.',
    ]
    for text in drops:
        result = match_post(text)
        assert result.matched is False, text

    platform = match_post('Our platform co-op shares surplus with the workers who build it.')
    assert platform.matched is True
    assert platform.reason == 'strong_positive'


def test_cooperative_forms_missed_on_2026_11_13() -> None:
    cases = [
        'The Really Really Free Market is one of my favorite things in LA.',
        'Some scenes from the Really Really Free Market.',
        "The Finland Co-op, Minnesota's oldest continuing co-op since 1913.",
        'Surrey ACORN holds a Save Our Co-op, Save Our Homes Day of Action.',
        'As a multi-stakeholder co-operative, the writers are paid by subscribers.',
        'The app is owned by the taxi drivers.',
        'Its fishermen still run their own co-op, as they have since 1948.',
        'I want to make the social platform a customer-owned co-op.',
        'Our co-op windfarm has been spinning idle for most of the month.',
        'The Cooperative Wholesale Society office and warehouse in Manchester.',
        'I organized eight families into a loose co-op at the farmers market.',
        'If people want a non-profit grocery, they can form a co-op.',
        'A new platform to build knowledge for a stronger cooperative movement.',
        'The cooperative village of Hjortshøj has been a living laboratory since 1992.',
        'Wakefern celebrated its cooperative roots at the annual meeting.',
        'The World Cooperative Monitor tracks the largest cooperatives.',
        'The Cambridge Artists Cooperative crafts a shop of member works.',
        'Subscribe to our journalist-owned cooperative.',
        'The credit cooperative society pools savings for cheap loans.',
        'De-Google Together is hosted by Bunk Computer Cooperative.',
        'Il festival è promosso da Confcooperative.',
        'Le quatrième lieu de vie de la coopérative ouvre ses portes.',
    ]
    for text in cases:
        result = match_post(text)
        assert result.matched is True, text

    drops = [
        'Edinburgh Community Solar Co-operative launches new share offer.',
        (
            'A clear bipartisan win is massive building reform. '
            'And they even brought up community land trusts!'
        ),
        'Pico Park is a platform co-op puzzle game for two to eight players.',
        (
            'Every Walmart will be a trans-run co-op and the Walton family '
            'will be flensed for winter heating.'
        ),
        'The community land trust holds the land so occupancy is not a landlord title.',
    ]
    results = [match_post(text) for text in drops]
    assert results[0].matched is False
    assert results[1].matched is False
    assert results[2].matched is False
    assert results[3].matched is False
    assert results[4].matched is True
    assert results[4].reason == 'strong_positive'


def test_cooperative_forms_missed_on_2026_11_14() -> None:
    cases = [
        'Bring them to be repaired at Karma Co-op. RepairCafe Toronto is co-hosting.',
        'A poster for the Alexandria Clay Co-op pottery sale.',
        'Co-op City Times: November 18 Open Board Meeting Agenda.',
        'A Barcelona cooperative offers a non-profit, right-to-use model for long-term homes.',
        'This film is about their felt shoe and textile cooperative in Kyrgyzstan.',
        'City council is weighing a statue against our community college childcare cooperative.',
        'La centrale a été développée avec la coopérative citoyenne BretiSun.',
        'Equal Exchange is a cooperative specialising in coffee grown by women.',
        'Civil Law indivisible reserves lock cooperative equity in the common fund.',
        (
            'The colony aimed at a self-sustaining life through '
            'cooperative farming and light industry.'
        ),
        "La chaîne d'épiceries passe du statut de coopérative à celui de SRL.",
        'Cooperative member fees cover the cold storage so stores can buy food near wholesale.',
    ]
    for text in cases:
        result = match_post(text)
        assert result.matched is True, text
        assert result.reason == 'strong_positive'

    drops = [
        'The sewing team at the repair cafe fixed a coat.',
        'His pitch was a cooperative farming community. The tract became a summer resort.',
        'The right to use this software is included with the license.',
        'Central Virginia Electric Cooperative cut the ribbon on a solar facility.',
        'Cooperative Bank of Kenya raised its interim dividend.',
        'Our co-op escape room game arrives on Steam in two days.',
        'We’ll be playing songs at a benefit for a co-op preschool in Burbank.',
        'Thanks to our co-op student, the toy section is restocked.',
        'Has cooperative federalism become permissive federalism?',
        'Pico Park is a platform co-op puzzle game for two to eight players.',
        'Edinburgh Community Solar Co-operative launches new share offer.',
        'A shooting in Co-op City left two people injured.',
        'Spain’s cooperative answer to the housing crisis.',
    ]
    for text in drops:
        result = match_post(text)
        assert result.matched is False, text


def test_cooperative_forms_missed_on_2026_11_15() -> None:
    cases = [
        'The directory internet, rebuilt as a co-op. 0% platform fees. No ads. No VC.',
        (
            'We created a "co-op within a co-op" and let young people build it. '
            'One called it a community business, not one where it is every person '
            'for themselves.'
        ),
        'Tillamook is suing the co-op that owns Darigold to keep a milk pooling agreement alive.',
        'Maryland & Virginia Milk Producers, a co-op since 1920: 750 member farms.',
        "Helena's club burned, then rose again. Now a public co-op, its Rathskeller is open.",
        "Land O'Lakes is an American agricultural cooperative known for its butter.",
        "Monet fundó la Société Anonyme Coopérative d'Artistes-Peintres en 1873.",
        '282 doctors founded a cooperative in Adana and press their own olive oil.',
        'Se domattina le cooperative sociali andassero in black out, il paese si ferma.',
        'Welfare, Gardini: "Paese si ferma senza cooperative".',
        'The Cooperative Cure for Capitalism',
    ]
    for text in cases:
        result = match_post(text)
        assert result.matched is True, text
        assert result.reason == 'strong_positive'

    amesbury = match_post(
        "Remember when Amesbury's new Co-op opened?",
        alt_text=(
            'Crowds gathered to celebrate the opening of a new Co-op superstore '
            'in Amesbury, continuing a relationship stretching back to the early 20th century.'
        ),
    )
    assert amesbury.matched is True
    assert amesbury.reason == 'strong_positive'

    mobile = match_post(
        'Good Morning from the land of the Finns',
        alt_text="Customers about to enter cooperative Mäki-Matti's mobile shop 'Auto-Matti'.",
    )
    assert mobile.matched is True
    assert mobile.reason == 'strong_positive'

    preschool = match_post(
        '',
        alt_text=(
            'Parents and educators warn state funding changes threaten '
            'cooperative preschool programs.'
        ),
    )
    assert preschool.matched is True
    assert preschool.reason == 'strong_positive'

    darigold = match_post(
        'Tillamook is suing over a terminated agreement.',
        alt_text='The lawsuit is against the dairy cooperative that owns Darigold.',
    )
    assert darigold.matched is True
    assert darigold.reason == 'strong_positive'

    drops = [
        "I haven't played a co-op since last year.",
        'Muse at Co-op Live in Manchester was a laser-soaked masterclass.',
        'Please join me in supporting co-op! coopradio.org/donate/',
        'A woman in my co-op brought pillows to the Board Meeting.',
        'The soul veg popup at the co-op was excellent.',
        'We’ll be playing songs at a benefit for a co-op preschool in Burbank.',
        (
            'pour remettre en cause le système, passer à une propriété coopérative, '
            'collective ou publique, sortir le rail de la sphère privée.'
        ),
        'The rural electric co-op is raising the kilowatt rate this winter.',
        'Cooperative Bank of Kenya raised its interim dividend.',
    ]
    for text in drops:
        result = match_post(text)
        assert result.matched is False, text


def test_cooperative_forms_missed_on_2026_11_16() -> None:
    cases = [
        (
            "The Calavo sticker on the avocado began as a growers' co-op in 1924. "
            'California avocado farmers copied the Sunkist model.'
        ),
        (
            'In 2005 I set up a wind turbine co-op to give a local community cheap energy. '
            'I had the land, the council and the funding.'
        ),
        (
            'Discovered her on the co-op music service. '
            'It supports a cooperative digital music platform.'
        ),
        'I am daydreaming about running a cooperative grocery store.',
        (
            'The Dr. U.S. Awasthi Initiative in Cooperative Economics '
            'funds research on cooperatives.'
        ),
        'Maybe time for them to just go ahead and start a new coffee cooperative.',
        (
            'I joined this cooperative thing to make it think the way I do, and if I fail, '
            'I hate it and leave.'
        ),
        'The district is exploring a health insurance cooperative that pools shared risk.',
    ]
    for text in cases:
        result = match_post(text)
        assert result.matched is True, text
        assert result.reason == 'strong_positive'

    olive = match_post(
        'Spain’s olive oil intervention hits trouble just days after new rules take effect.',
        alt_text=(
            'A Spanish olive oil cooperative has stopped buying oil from local producers '
            'following the government’s latest intervention in the market.'
        ),
    )
    assert olive.matched is True
    assert olive.reason == 'strong_positive'

    news = match_post(
        'thesouthlander.com give us your email so you see our story tomorrow morning',
        alt_text="The Southlander The Southland's Only Investigative News Cooperative",
    )
    assert news.matched is True
    assert news.reason == 'strong_positive'

    drops = [
        'We’ll be playing songs at a benefit for a co-op preschool in Burbank.',
        'The rural electric co-op is raising the kilowatt rate this winter.',
        'Kit Carson Electric Co-op has maintained its base electric rates since 2016.',
        'A Dakota co-op apartment once owned by Rex Reed is listed for $5.99 million.',
        (
            'Let the very tractors weep with pride: '
            'the cabbage cooperative have coaxed a second harvest.'
        ),
        'Red billed Blue Magpies practice cooperative breeding.',
        'It is unclear whether cooperative overtures from Trump officials will survive.',
        'Tommy Douglas led the Cooperative Commonwealth Federation.',
        'Gift ideas: Palestinian Soap Cooperative olive oil soap blocks.',
        'The theft happened outside the Co-op store in Doune.',
        'A co-op restaurant management sim for 1-4 players.',
    ]
    for text in drops:
        result = match_post(text)
        assert result.matched is False, text


def test_cooperative_forms_missed_on_2026_11_17() -> None:
    cases = [
        "Third Act volunteers can now join Clean Energy Credit Union's financial cooperative.",
        'Our little cooperative made sure to have D&O insurance before users joined our PDS.',
        'Our next Co-op Conversation is on cooperative democracy and the boardroom.',
        'Consum, the Valencian cooperative, opens a 1,135 square metre supermarket.',
        'The Irish Organic Milk Suppliers Cooperative established a dairy producer organisation.',
        'Sandrine organise un job dating pour notre coopérative MEDICOOP.',
        'In a just world, every tech company would be a leftist co-op.',
        'Chico Natural Foods Co-op’s board found its footing through a GM transition.',
    ]
    for text in cases:
        result = match_post(text)
        assert result.matched is True, text
        assert result.reason == 'strong_positive'

    doula = match_post(
        'Doula care is growing in North Carolina’s rural counties.',
        alt_text='A Chatham County group supports a doula cooperative.',
    )
    assert doula.matched is True
    assert doula.reason == 'strong_positive'

    ambulance = match_post(
        'www.ctaq.com',
        alt_text=(
            'CTAQ : Coopérative des techniciens ambulanciers du Québec. '
            'La CTAQ offre un service paramédical à ses membres.'
        ),
    )
    assert ambulance.matched is True
    assert ambulance.reason == 'strong_positive'

    grain = match_post(
        'WA’s harvest is piling into the grain network.',
        alt_text='Cooperative Bulk Handling Group logged 665,000 tonnes at Kwinana.',
    )
    assert grain.matched is True
    assert grain.reason == 'strong_positive'

    forestry = match_post(
        'New episode of Forestry Now.',
        alt_text='Victor Barber from Western Forestry Co Op explains how harvesting works.',
    )
    assert forestry.matched is True
    assert forestry.reason == 'strong_positive'

    solar = match_post(
        'Coconut Creek was asked to publicize a homeowner buying group.',
        alt_text=(
            'Solar United Neighbors urged the city to publicize a free solar co-op. '
            'The nonprofit uses group purchasing and is not seeking city funding.'
        ),
    )
    assert solar.matched is True
    assert solar.reason == 'strong_positive'

    wind = match_post(
        'Minstens de helft van de windmolens blijft in handen van de buurt.',
        alt_text=(
            'De Windvogel is samen met drie Utrechtse energiecoöperaties '
            'het lokaal eigendom aan het organiseren.'
        ),
    )
    assert wind.matched is True
    assert wind.reason == 'strong_positive'

    films = match_post(
        'GRIEF screens November 18 at the Filmmakers Co-op in NYC.',
        alt_text='A filmmaker-member of the co-op curated the program.',
    )
    assert films.matched is True
    assert films.reason == 'strong_positive'

    drops = [
        'Give monthly to Drivers Cooperative Colorado and the match is 100%.',
        'The rural electric co-op is raising the kilowatt rate this winter.',
        'Co-operative Bank is cutting mortgage rates by 20 basis points.',
        'Delivery app got a mad offer on co-op veg.',
        'A $6M co-op in the Dakota was Rex Reed’s longtime home.',
        'Helldivers 2 co-op with the squad this weekend.',
        'Red billed Blue Magpies practice cooperative breeding.',
        '2027 Summer Co op Manufacturing Technology, Swiftwater PA.',
        'The board renewed a unified bowling co-op with the Catholic school.',
        'We’ll be playing songs at a benefit for a co-op preschool in Burbank.',
        'The solar co-op published a share offer for new members.',
        'Mohave Electric Cooperative congratulates its Citizen of the Year.',
        'Two officers were shot at Co-op City in the Bronx.',
        'Our little cooperative game night is on Thursday.',
    ]
    for text in drops:
        result = match_post(text)
        assert result.matched is False, text


def test_cooperative_forms_missed_on_2026_11_18() -> None:
    cases = [
        (
            'En France, le solaire citoyen et solidaire avec la coopérative SolarCoop. '
            'Kits et wébinaires.'
        ),
        ('Black farmers in a cooperative in Alabama opened a grocery store in Fountain Heights.'),
        'Fram Farmers, a cooperative owned by its farmers, won a new cooperative award.',
        'Forming a Publishing Co-op: The Ups & Downs.',
        "New Zealand's fertiliser co-ops manage risk for farmers who own them.",
        'Our developer co-op has been building MVPs and mobile apps for startups.',
        'In favour of credit unions & other cooperative models that follow the law.',
        'Robert Owen founded the cooperative community at New Harmony, Indiana.',
        'The Bridge of Blessing Farmers Agriculture Cooperative received a livelihood grant.',
        'At Multipharma, our cooperative model gives a clear answer: the people who count on us.',
    ]
    for text in cases:
        result = match_post(text)
        assert result.matched is True, text
        assert result.reason == 'strong_positive', text

    podcast = match_post(
        'What do Ungagged do? Have a look.',
        alt_text=(
            'Descriptive text about the not-for-profit cooperative podcast and article platform.'
        ),
    )
    assert podcast.matched is True
    assert podcast.reason == 'strong_positive'

    solar_card = match_post(
        'Un petit coup de projecteur.',
        alt_text='Société coopérative photovoltaïque. Commandez vos panneaux solaires.',
    )
    assert solar_card.matched is True
    assert solar_card.reason == 'strong_positive'

    kickstarter = match_post(
        'Full trailer for Delta x Dungeon Breaker by Timeless Way Interactive: '
        'an equal ownership worker cooperative #kickstarter #gametrailer'
    )
    assert kickstarter.matched is False
    assert kickstarter.reason == 'hard_negative'

    housing = match_post(
        'Laurie Fitz invites residents to share what it is really like '
        'transitioning into cooperative housing.'
    )
    assert housing.matched is False

    drops = [
        '2027 Summer Co-op Manufacturing Science, Framingham, MA.',
        'Engineering co-op prep for next semester.',
        'Il est toujours possible de prendre des parts de la coopérative.',
        'Kick off the year-end fundraising campaign for our members.',
        "Ranger's Path leaves Early Access with two-player co-op.",
        'Two officers were shot at Co-op City in the Bronx.',
    ]
    for text in drops:
        result = match_post(text)
        assert result.matched is False, text


def test_cooperative_forms_missed_on_2026_11_19() -> None:
    cases = [
        (
            'Does selling mean I am an owner? The article is about selling on '
            'Artisans Cooperative and becoming a member-owner.'
        ),
        ('The Sacramento-based almond cooperative will reopen membership after a 10-year closure.'),
        (
            'Ocean Spray is a cooperative of farm business owners, which has '
            'very little to do with treatment of farm workers.'
        ),
        ('A conversation on the work of Nundah Community Enterprises Cooperative.'),
        (
            'La coopérative L’Arban accompagne particuliers et collectivités '
            'dans l’éco-rénovation de logements.'
        ),
        'The coffee shop is a co-op now, and everyone is growing their own food.',
        "It's a co op site, not like bandcamp, which venture capital keeps passing around.",
        (
            'Timeless Way Interactive is organized as a worker owned gamedev '
            'cooperative developing Delta x Dungeon Breaker.'
        ),
        (
            'Group Health Cooperative member Joe Gardiner received a new heart, '
            'and Group Health pays the cost.'
        ),
        (
            'The Technology Freedom Cooperative is a project to break up with Big Tech '
            'and share an archive server.'
        ),
        ('Join the webinar on cooperative banking experiences from India, Malaysia and Australia.'),
        'Parkdale’s Danu Social House became a co-op, a neighbourhood third space.',
        'From a £9m grocery to a 1,400-farmer co-op.',
        'Seven residents signed up after joining a solar co-op workshop.',
    ]
    for text in cases:
        result = match_post(text)
        assert result.matched is True, text
        assert result.reason == 'strong_positive', text

    writer = match_post(
        'Olivia Charles on music and finally living as herself.',
        alt_text='Olivia Charles: The Flaming Hydra Interview a fiery writer-owned cooperative',
    )
    assert writer.matched is True
    assert writer.reason == 'strong_positive'

    nundah_alt = match_post(
        'Really enjoyed this conversation with Dr Peter Westoby.',
        alt_text='Richard Warner, Co-ordinator of Nundah Community Enterprise Co-op',
    )
    assert nundah_alt.matched is True
    assert nundah_alt.reason == 'strong_positive'

    artist = match_post(
        'Wednesday advert of the week.',
        alt_text='Aberdeen City Arts Board artist co-op. Original designs.',
    )
    assert artist.matched is True
    assert artist.reason == 'strong_positive'

    drops = [
        'French cooperative bank Crédit Mutuel Arkéa was targeted in a cyberattack.',
        (
            'Help keep Drivers Cooperative Colorado rolling: new monthly gifts '
            'are matched 100%, up to $100.'
        ),
        'Former Co-op building to become a fitness studio and ironing business.',
        'Midwest Electric is returning $2.5 million to members who had electric service.',
        'Central Electric Cooperative is accepting applications for its youth trip.',
        'White-fronted nun-birds are cooperative-breeding pairs plus helpers.',
        (
            'It is a Mitchell-Lama co-op, where the city controls the price and '
            'owners cannot resell at a profit.'
        ),
        'An introduction with EmpowerLocal and the Center for Cooperative Media.',
        "Ranger's Path leaves Early Access with two-player co-op.",
        "You've been very cooperative.",
        'Hegseth arrived to sign a cooperative defense agreement.',
        "Iowa's largest farm cooperative is suing its former CEO.",
        'Thank you Tillamook cooperative.',
        'We work in a cooperative manner wherein 51% of profit goes to upgrades.',
        'Shein attiva in Italia il programma Cooperative Logistics.',
        'Full trailer for an equal ownership worker cooperative #kickstarter',
        'The worker owned electric cooperative posted a new rate notice.',
        'I manage a co-op owned & operated by leftists. We want to donate to campaigns.',
        'Support us through the Co-op Local Community Fund on the Co-op app.',
        'Golding Homes completes a stock transfer from a housing co-op.',
        'Night of the Living Toys lets you stuff your co-op partners into boxes.',
    ]
    for text in drops:
        result = match_post(text)
        assert result.matched is False, text


def test_cooperative_forms_missed_on_2026_11_20() -> None:
    cases = [
        'Hensall seems small on the map, its farmer owned co-op exports dry beans.',
        "Artisans who are anti AI should consider trying Artisan's Cooperative!",
        'Premi per Incontro, cooperativa di comunità, e Almaflores.',
        'A weeklong conference for worker-owned newsrooms just wrapped up.',
        "Bonnington Square's squatters' cooperative preserved terraced homes.",
        'Nürnberg explores a fan cooperative to fund stadium redevelopment.',
        'FC Nürnberg is considering a supporter cooperative for the stadium.',
        (
            'In a cooperative business, there is no distance between the CEO '
            'and the farmers who own the company.'
        ),
        'Free art at City Art Cooperative Gallery runs through November 27.',
        'We collectively buy a few GPUs and rotate them through the co-op.',
        'Hippie fans put together a co-op to open a juice bar club.',
        'Beyond The Music is the co-operative global music and media conference.',
        'Intrigued by this cooperative filmfestival of collaboratively made works.',
        'Wonder what a co-op platform would look like for workers locked out of processors.',
    ]
    for text in cases:
        result = match_post(text)
        assert result.matched is True, text
        assert result.reason == 'strong_positive', text

    fontstand = match_post(
        'More cooperatives, today for type foundries.',
        alt_text='Pssst, the Fontstand Cooperative now exists',
    )
    assert fontstand.matched is True
    assert fontstand.reason == 'strong_positive'

    leaders = match_post(
        'I wrote an essay about why collective action is how we survive.',
        alt_text=(
            'Grounded Together: Why Cooperative Organizing Works When Systems Fail. '
            'The groundstop stranded worker-owned media leaders.'
        ),
    )
    assert leaders.matched is True
    assert leaders.reason == 'strong_positive'

    principles = match_post(
        '',
        alt_text=(
            'Co-Operative Principles Are The Foundation Of Development at the Sahakari awards.'
        ),
    )
    assert principles.matched is True
    assert principles.reason == 'strong_positive'

    started = match_post(
        'One idea sparked by the layoff wave.',
        alt_text='Someone who wants to start a #cooperative instead of another corporation.',
    )
    assert started.matched is True
    assert started.reason == 'strong_positive'

    media = match_post(
        'A report on media run on #cooperative principles is coming out.',
        alt_text=(
            'Learning from cooperatives: media organizations inspired by cooperative principles.'
        ),
    )
    assert media.matched is True
    assert media.reason == 'strong_positive'

    drops = [
        'It is a Mitchell-Lama co-op, where the city controls the price.',
        'Starting a co-op discord server for streamers to promote themselves.',
        "Let's start a cooperative game tonight.",
        'The board game teaches cooperative principles of play.',
        'Support us through the Co-op Local Community Fund on the Co-op app.',
        'Central Electric Cooperative is accepting applications for its youth trip.',
        'Thank you Tillamook cooperative.',
        'A fan cooperative board game night at the library.',
        'Fans put together a co-op game for the poster.',
        'Wonder what a co-op platform game would look like.',
        'Couch co-op pixel platformer for two cats.',
        'White-fronted nun-birds are cooperative-breeding pairs plus helpers.',
        'Full trailer for an equal ownership worker cooperative #kickstarter',
    ]
    for text in drops:
        result = match_post(text)
        assert result.matched is False, text


def test_cooperative_forms_missed_on_2026_11_21() -> None:
    cases = [
        (
            'Magnum Photos is a cooperative of acclaimed, independent photographers '
            'who document daily life.'
        ),
        ('The Brick House is a cooperative of writers, artists, and editors founded in 2020.'),
        'VatN plays at The Monarch, a cooperative musician workerspace in Louisville.',
        ('Rabobank is a cooperative bank established by farmers in the late nineteenth century.'),
        'Windcoop est une coopérative maritime basée à Lorient.',
        'Le moyen, rejoindre la coopérative d’activité et d’emploi coop alpha.',
        (
            'A renewable energy cooperative is bringing pathways for trading work '
            'to Chicago’s south side.'
        ),
        'He has been living at the farm the Longo maï collective took over.',
        'Leading Austrian wine producer and cooperative Domäne Wachau unveiled the vintage.',
        'She has been leading the cranberry growers’ cooperative for a few months.',
        'Not a member? Join the co-op at subvert.fm',
    ]
    for text in cases:
        result = match_post(text)
        assert result.matched is True, text
        assert result.reason == 'strong_positive', text

    sighting = match_post('I saw Gaby Hoffmann checking out at the Park Slope Food Coop')
    assert sighting.matched is False
    assert sighting.reason == 'hard_negative'

    stocking = match_post('i stocked nut butters at the park slope food co-op with peter sarsgaard')
    assert stocking.matched is True
    assert stocking.reason == 'strong_positive'

    meeting = match_post('I saw the food co-op board post the meeting notes.')
    assert meeting.matched is True
    assert meeting.reason == 'strong_positive'

    drops = [
        'Cooperative Bank of Kenya raised its interim dividend.',
        'Pedernales Electric Cooperative approved a 5% rate reduction.',
        'Spirit Island is a cooperative strategy game where island spirits drive off invaders.',
        'Double your gift to Drivers Cooperative Colorado through December 8.',
        'La coopérative lance une levée de fonds citoyenne pour son modèle agroécologique.',
        'One challenge of living in co-op housing is individual preference.',
        'Join the co-op at discord.gg for the raid night.',
    ]
    for text in drops:
        result = match_post(text)
        assert result.matched is False, text


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

    cooperative_housing = match_post(
        'Residents described transitioning into cooperative housing at the center.',
        decider=script,
    )
    assert cooperative_housing.matched is True
    assert cooperative_housing.reason == 'decision_keep:spirit_account:0.910'
