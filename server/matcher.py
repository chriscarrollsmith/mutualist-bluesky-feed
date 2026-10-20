"""Mutualist post matcher.

Keeps posts about durable institutions people can use while firms and states
still dominate: mutual credit, possession, cooperatives, commons, assemblies,
and confederations. Proudhonian and left-market traditions are the core.
Bookchin, libertarian municipalism, and democratic confederalism are in scope
when the post is about those institutions. Contract and assembly both belong.
Kropotkin stays for mutual aid and the federated commune.

Rejects common false positives:
- Anarcho-capitalism / right-"libertarian" market fundamentalism
- Chaos / entertainment uses of "anarchy" without political content
- Crypto/Web3 "decentralized" jargon without mutualist values
- Bare ambiguous terms without institutional context
- Bare mutual aid and charity
- Personal fundraising / extractive money asks
- Offensive violence, martyrdom, and armed-struggle celebration
  (defiance and non-violent resistance may still keep)

Recall without keywords comes from author allowlists and from soft author
priors. A prior is three decision-model keeps and no rejections in the window;
regex matches do not earn one. Author blocklists drop an account even when the
text would otherwise keep. When a decision model is enabled, it keep/rejects every candidate that
passes this floor (see ``server/decider.py``). Bare ambiguous terms stay
dropped unless a soft prior lets them through to that filter.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

from server.decider import (
    KEEP_PREFIX,
    REJECT_PREFIX,
    UNSCORED_PREFIX,
    DecisionBackend,
    get_default_decider,
)
from server.post_images import image_data_urls
from server.topic_entities import TopicEntities, default_topic_entities


@dataclass(frozen=True)
class MatchResult:
    matched: bool
    reason: str


# Phrases that almost always mean an in-scope institution.
_STRONG_POSITIVE = re.compile(
    r"""
    (?:
        \#mutualism\b
      | \#mutualist\b
      | \#agorism\b
      | \bmutualism\b
      | \bmutualists?\b
      | \bproudhon\b
      | proudhonian
      | josiah\s+warren
      | benjamin\s+tucker
      | william\s+b\.?\s+greene
      | dyer\s+lum
      | kevin\s+carson
      | \bkropotkin\b
      | center\s+for\s+a\s+stateless\s+society
      | \bc4ss\b
      | occupancy[\s-]and[\s-]use
      | possession\s+(?:not|vs\.?|versus)\s+property
      | property\s+is\s+theft
      | mutual\s+credit
      | free\s+bank(?:ing|s)?
      | people'?s\s+bank
      | cost\s+the\s+limit\s+of\s+price
      | equitable\s+commerce
      | \bagorism\b
      | \bagorists?\b
      | left[\s-]?wing\s+market\s+anarch
      | market\s+anarchis(?:m|t)s?
      | free[\s-]market\s+anti[\s-]?capital
      | anti[\s-]?capitalist\s+free[\s-]market
      | mutualist\s+economics?
      | \bbookchin\b
      | libertarian\s+municipalism
      | democratic\s+confederalism
      | municipal\s+assembl(?:y|ies)
      | neighborhood\s+assembl(?:y|ies)
      | community\s+land\s+trusts?
      | \btime[\s-]?banks?\b
      | worker\s+cooperatives?
      | worker\s+co[\s-]?ops?
      | platform\s+cooperatives?
      | platform\s+co[\s-]?ops?
      | community\s+currenc(?:y|ies)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Ambiguous terms that need institutional context (or the decider).
# Includes bare anarchism, mutual aid, and broad left-market labels.
_AMBIGUOUS_TERM = re.compile(
    r"""
    (?:
        \#anarchism\b
      | \#anarchist\b
      | \#anarchy\b
      | \#mutualaid\b
      | \#directaction\b
      | \banarchism\b
      | \banarchists?\b
      | \banarchy\b
      | \blibertarian\b
      | libertarian\s+socialis(?:m|t)s?
      | individualist\s+anarch
      | \bautonom(?:y|ous|ist)s?\b
      | direct\s+action
      | mutual\s+aid
      | \bcommunes?\b
      | decentraliz(?:e|ed|ing|ation)
      | \bcooperatives?\b
      | \bco[\s-]?ops?\b
      | free\s+association
      | \breciprocity\b
      | \bsocialis(?:m|t)s?\b
      | free\s+the\s+people
      | no\s+rulers
      | power\s+to\s+the\s+people
      | without\s+(?:bosses|rulers|masters|landlords)
      | anti[\s-]?authoritarian
      | anti[\s-]?capitalis(?:m|t)
      | social[\s-]ecology\b
      | \bcommunalism\b
      | \bconfederalism\b
      | dual\s+power
      | \brojava\b
      | \baanes\b
      | \bdaanes\b
      | shop\s+committee
      | solidarity\s+union
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Context that unlocks ambiguous terms toward keep.
# Intentionally excludes bare "anarchy" / "anarchist" / "anarchism" and bare
# "mutual aid" — those are the ambiguous terms being gated.
_MUTUALIST_CONTEXT = re.compile(
    r"""
    (?:
        \bmutualism\b
      | \bmutualists?\b
      | \bproudhon\b
      | proudhonian
      | josiah\s+warren
      | benjamin\s+tucker
      | william\s+b\.?\s+greene
      | dyer\s+lum
      | kevin\s+carson
      | \bkropotkin\b
      | center\s+for\s+a\s+stateless\s+society
      | \bc4ss\b
      | occupancy[\s-]and[\s-]use
      | possession\s+(?:not|vs\.?|versus)\s+property
      | property\s+is\s+theft
      | mutual\s+credit
      | free\s+bank(?:ing|s)?
      | people'?s\s+bank
      | cost\s+the\s+limit\s+of\s+price
      | equitable\s+commerce
      | \bagorism\b
      | \bagorists?\b
      | left[\s-]?wing\s+market
      | market\s+anarch
      | free[\s-]market\s+anti[\s-]?capital
      | anti[\s-]?capitalist\s+free[\s-]market
      | mutualist\s+economics?
      | left[\s-]?libertarian
      | \bbookchin\b
      | libertarian\s+municipalism
      | democratic\s+confederalism
      | municipal\s+assembl
      | neighborhood\s+assembl
      | community\s+land\s+trusts?
      | \btime[\s-]?banks?\b
      | worker\s+cooperatives?
      | worker\s+co[\s-]?ops?
      | platform\s+cooperatives?
      | platform\s+co[\s-]?ops?
      | community\s+currenc(?:y|ies)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Subjects that stay dropped until an institutional cue is also present.
# "Social ecology" alone is academic or Bookchin; "communalism" is often
# sectarian politics; "confederalism" is often the EU; "Rojava" is often war.
_INSTITUTION_SUBJECT = re.compile(
    r"""
    (?:
        social[\s-]ecology\b
      | \bcommunalism\b
      | \bconfederalism\b
      | dual\s+power
      | \brojava\b
      | \baanes\b
      | \bdaanes\b
      | shop\s+committee
      | solidarity\s+union
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Cues that unlock those subjects. Bare "council" and bare "assembly" are
# intentionally absent.
_INSTITUTION_CUE = re.compile(
    r"""
    (?:
        \bcommunes?\b
      | \bcooperatives?\b
      | \bco[\s-]?ops?\b
      | municipal\s+assembl
      | neighborhood\s+assembl
      | people'?s\s+assembl
      | women'?s\s+(?:council|assembl|cooperative|co[\s-]?op)
      | community\s+land\s+trusts?
      | \btime[\s-]?banks?\b
      | worker\s+co(?:operatives?|[\s-]?ops?)
      | platform\s+co(?:operatives?|[\s-]?ops?)
      | community\s+currenc
      | \bbookchin\b
      | libertarian\s+municipalism
      | democratic\s+confederalism
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Hard negatives that always win over an otherwise-strong phrase when they
# clearly indicate right-"libertarian" / ancap / entertainment collisions.
# Collisions that should not be rescued by a nearby strong mutualist phrase.
# Plain "anarcho-capitalism" is intentionally absent from this set so critiques
# like "mutualism is not anarcho-capitalism" can still keep via
# strong_positive_over_negative.
#
# "Free-market anarchism" is NOT a hard negative: left-wing market anarchists
# use that phrase. Ancap is caught via Rothbard/Hoppe/Mises/voluntaryism/
# property-rights anarchism / explicit ancap labels instead.
_HARD_NEGATIVE_BLOCKS_STRONG = re.compile(
    r"""
    (?:
        sons\s+of\s+anarchy
      | anarchy\s+online
      | decentralized\s+finance
      | \bdefi\b
      | \bweb3\b
      | sovereign\s+citizens?
      | mises\s+institute
      | \brothbard\b
      | \bhoppean\b
      | hans[\s-]hermann\s+hoppe
      | \bvoluntaryism\b
      | property[\s-]rights\s+anarch
      | \bmartyrs?\b
      | \bmartyrdom\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_HARD_NEGATIVE = re.compile(
    r"""
    (?:
        anarcho[\s-]?capitalis(?:m|t)s?
      | \bancaps?\b
      | \banarcho[\s-]?cap\b
      | \brothbard\b
      | \bhoppean\b
      | hans[\s-]hermann\s+hoppe
      | mises\s+institute
      | \bvoluntaryism\b
      | property[\s-]rights\s+anarch
      | sons\s+of\s+anarchy
      | anarchy\s+online
      | decentralized\s+(?:finance|exchange|autonomous\s+organization)
      | \bdefi\b
      | \bnfts?\b.{0,40}decentral
      | decentral.{0,40}\bnfts?\b
      | \bweb3\b
      | sovereign\s+citizens?
      | this\s+is\s+anarchy!
      | pure\s+chaos\s+anarchy
      | state\s+of\s+anarchy
      | \bmartyrs?\b
      | \bmartyrdom\b
      | armed\s+struggle
      | people'?s\s+war
      | \binsurrectionary\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Upcoming-event phrasing unlocks high-confidence mutualist venues/projects.
_EVENT_CUE = re.compile(
    r"""
    \b(?:
        tonight|tomorrow|this\s+weekend|this\s+saturday|this\s+sunday|
        next\s+(?:friday|saturday|sunday|week)|
        doors(?:\s+at|\s+open)|tickets?|presale|save\s+the\s+date|join\s+us|
        open\s+mic|festival|bookfair|distro|teach[\s-]?in|assembly|
        \d{1,2}:\d{2}\s*(?:am|pm)|(?<!\d)\d{1,2}\s*(?:am|pm)\b|
        january|february|march|april|june|july|august|september|
        october|november|december
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_IN_SCOPE_EVENT_VENUE = re.compile(
    r"""
    (?:
        mutualis[mt]\s+(?:meetup|reading|circle|conference|forum|bookfair)
      | mutual\s+credit\s+(?:workshop|teach[\s-]?in|meetup)
      | free\s+bank(?:ing)?\s+(?:workshop|teach[\s-]?in|meetup)
      | occupancy[\s-]and[\s-]use
      | proudhon\s+(?:reading|circle|meetup)
      | really\s+really\s+free\s+market
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Mutual-aid token used to pair boost/ask vernacular with the phrase.
_MUTUAL_AID_TOKEN = re.compile(
    r'(?:\#mutualaid\b|mutual\s+aid)',
    re.IGNORECASE,
)

# Cheap cue that a provisional keep may be a personal money-ask. Forces a
# quality-rubric pass (solicit gate) when the classifier is available; when it
# is not, the post is dropped for precision.
#
# Includes Bluesky "HelpSky" fundraising hashtag culture and short boost/ask
# shapes that otherwise sail through on bare ``#mutualaid`` / "mutual aid".
_SOLICIT_CUE = re.compile(
    r"""
    (?:
        \bvenmo\b
      | \bcash\s*app\b
      | \bcashapp\b
      | \bpaypal(?:\.me)?\b
      | gofund\.?\s*me
      | \bgofundme\b
      | \bko-?fi\b
      | \bpatreon\b
      | please\s+(?:donate|venmo|cash\s*app|send\s+(?:money|cash|\$))
      | \bdonat(?:e|ions?)\s+to\s+me\b
      | mutual\s+aid\s+request
      | \bneed(?:s|ing)?\s+(?:mutual\s+aid|donations?|money|rent|funds?)\b
      | \bcan\s+you\s+(?:spare|send|donate)\b
      | \bsend\s+(?:me\s+)?(?:\$|money|venmo|cash)
      | \bdm\s+me\s+for\s+(?:my\s+)?(?:venmo|cash\s*app|cashapp|paypal)
      | \#helpsky\b
      | \#helpfolkslive\b
      | \#mutualaidrequest\b
      | \#mutualaidboost\b
      | \#mutualaidsaveslives\b
      | \#urgentmutualaid\b
      | \#emergencyma\b
      | \#maboost\b
      | \#madboost\b
      | \#fundsky\b
      | \#lgbtqma\b
      | \#bipocma\b
      | 💸
      | \banything\s+helps\b
      | \brent\s+due\b
      | \b(?:urgent|immediate)\b.{0,60}(?:\#mutualaid\b|mutual\s+aid|\#emergencyma\b)
      | \bplease\s+(?:help|assist|donate|boost)\b.{0,60}
        (?:\#mutualaid\b|mutual\s+aid|venmo|cash|\$|donat|rent)
      | (?:\#mutualaid\b|mutual\s+aid).{0,60}\bplease\s+(?:help|assist|donate|boost)\b
      | \b(?:bumping|boosting|amplifying|re-?sharing|passing\s+along)\b
        .{0,80}(?:\#mutualaid\b|mutual\s+aid)
      | (?:\#mutualaid\b|mutual\s+aid).{0,40}\b(?:plz|pls|please)\b
      | \b(?:plz|pls)\b.{0,40}(?:\#mutualaid\b|mutual\s+aid)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_HANDLE_MENTION = re.compile(r'(?<![\w.])@[\w.-]+', flags=re.UNICODE)

_EMBED_TEXT_MAX = 320


def _normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text or '').strip()


def _clip_embed_text(value: object, *, max_len: int = _EMBED_TEXT_MAX) -> str:
    text = str(value or '').strip()
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip()


def _record_embed_payload(embed: object | None) -> dict[str, Any] | None:
    """Return the inner record/ref dict for quote embeds, if present."""
    if isinstance(embed, dict):
        record_obj = embed.get('record')
        if not isinstance(record_obj, dict):
            return None
        record = cast(dict[str, Any], record_obj)
        # recordWithMedia nests the quote under record.record.
        nested_obj = record.get('record')
        if isinstance(nested_obj, dict):
            nested = cast(dict[str, Any], nested_obj)
            if 'uri' in nested or 'cid' in nested or 'value' in nested or 'text' in nested:
                return nested
        return record
    record_attr = getattr(embed, 'record', None) if embed is not None else None
    if record_attr is None:
        return None
    nested_attr = getattr(record_attr, 'record', None)
    if nested_attr is not None and (
        getattr(nested_attr, 'uri', None)
        or getattr(nested_attr, 'cid', None)
        or getattr(nested_attr, 'value', None)
        or getattr(nested_attr, 'text', None)
    ):
        if isinstance(nested_attr, dict):
            return cast(dict[str, Any], nested_attr)
        uri = getattr(nested_attr, 'uri', None)
        return {'uri': uri} if uri else None
    return cast(dict[str, Any], record_attr) if isinstance(record_attr, dict) else None


def _quoted_record_text(record_payload: dict[str, Any]) -> str:
    """Best-effort text from a hydrated quote payload (AppView) or empty."""
    for key in ('value', 'record'):
        nested = record_payload.get(key)
        if isinstance(nested, dict):
            text = nested.get('text')
            if isinstance(text, str) and text.strip():
                return text
    text = record_payload.get('text')
    return text if isinstance(text, str) else ''


def is_opaque_record_embed(embed: object | None) -> bool:
    """True when a quote embed is only a strong ref (Jetstream create shape).

    Firehose records carry ``uri``/``cid`` without the quoted post body. Boost
    accounts wrap personal fundraisers this way while the outer text is just
    ``#MutualAid #HelpSky`` — without hydration we cannot see PayPal/Ko-fi in
    the quote, so mutual-aid + opaque quote is treated as solicit-shaped.
    """
    payload = _record_embed_payload(embed)
    if not payload:
        return False
    if _quoted_record_text(payload).strip():
        return False
    uri = payload.get('uri')
    cid = payload.get('cid')
    return bool(uri or cid)


def extract_alt_text(embed: object | None) -> str:
    """Pull alt text from common Bluesky embed shapes (dict or SDK-like)."""
    if not embed:
        return ''

    chunks: list[str] = []

    if isinstance(embed, dict):
        images = embed.get('images')
        if isinstance(images, list):
            for image in images:
                if not isinstance(image, dict):
                    continue
                alt = image.get('alt')
                if alt:
                    chunks.append(_clip_embed_text(alt))
        external = embed.get('external')
        if isinstance(external, dict):
            title = external.get('title')
            if title:
                chunks.append(_clip_embed_text(title))
            description = external.get('description')
            if description:
                chunks.append(_clip_embed_text(description))
        media = embed.get('media')
        if isinstance(media, dict):
            chunks.append(extract_alt_text(media))
        record_payload = _record_embed_payload(embed)
        if record_payload is not None:
            quoted = _quoted_record_text(record_payload)
            if quoted:
                chunks.append(_clip_embed_text(quoted))
            for key in ('value', 'record'):
                nested = record_payload.get(key)
                if isinstance(nested, dict):
                    chunks.append(extract_alt_text(nested.get('embed')))
                    break
            else:
                chunks.append(extract_alt_text(record_payload.get('embed')))
        return ' '.join(chunk for chunk in chunks if chunk)

    images = getattr(embed, 'images', None)
    if isinstance(images, list):
        for image in images:
            alt = getattr(image, 'alt', None)
            if alt:
                chunks.append(_clip_embed_text(alt))
    external = getattr(embed, 'external', None)
    if external is not None:
        title = getattr(external, 'title', None)
        if title:
            chunks.append(_clip_embed_text(title))
        description = getattr(external, 'description', None)
        if description:
            chunks.append(_clip_embed_text(description))
    return ' '.join(chunk for chunk in chunks if chunk)


def combine_text(text: str = '', *, alt_text: str = '', langs: Iterable[str] | None = None) -> str:
    del langs
    return _normalize(f'{text} {alt_text}')


def _normalize_langs(langs: Iterable[str] | None) -> list[str]:
    if not langs:
        return []
    out: list[str] = []
    for lang in langs:
        token = str(lang or '').strip().lower().replace('_', '-')
        if token:
            out.append(token)
    return out


def _haystack_without_handles(haystack: str) -> str:
    return _HANDLE_MENTION.sub(' ', haystack)


def _soft_prior_ambiguous(
    author_did: str | None,
    soft_prior_dids: set[str],
    term: str,
) -> MatchResult | None:
    if author_did and author_did in soft_prior_dids:
        return MatchResult(True, f'soft_prior_ambiguous:{term}')
    return None


def _match_in_scope_event(haystack: str) -> MatchResult | None:
    if not _EVENT_CUE.search(haystack):
        return None
    match = _IN_SCOPE_EVENT_VENUE.search(haystack)
    if not match:
        return None
    venue = re.sub(r'\s+', ' ', match.group(0).lower())
    return MatchResult(True, f'event_in_scope_venue:{venue}')


def looks_like_solicit(
    haystack: str,
    *,
    embed: object | None = None,
    opaque_record_embed: bool | None = None,
) -> bool:
    """True when text/alt looks like a personal money-ask or fundraising boost.

    Also true for mutual-aid posts that quote another record without hydrated
    quote text (common Jetstream shape for HelpSky-style boost accounts).
    """
    if _SOLICIT_CUE.search(haystack or ''):
        return True
    opaque = is_opaque_record_embed(embed) if opaque_record_embed is None else opaque_record_embed
    return bool(opaque and _MUTUAL_AID_TOKEN.search(haystack or ''))


def _content_filter(
    haystack: str,
    provisional: MatchResult,
    *,
    decider: DecisionBackend | None,
    embed: object | None = None,
    author_did: str | None = None,
) -> MatchResult:
    """Keep or reject a regex candidate.

    With no decider, the regex result stands, except solicit-shaped keeps,
    which drop. With a decider, its yes/no is authoritative. A decline
    (timeout, missing key, bad payload) drops the post. Images on the embed
    are downloaded and sent with the post; a failed download omits that image.
    """
    if not provisional.matched:
        return provisional
    if decider is None:
        if looks_like_solicit(haystack, embed=embed):
            return MatchResult(False, f'solicit_cue_unscored:{provisional.reason}')
        return provisional
    images = image_data_urls(embed, author_did=author_did) or None
    decision = decider.decide(haystack, images=images)
    if decision is None:
        return MatchResult(False, f'{UNSCORED_PREFIX}{provisional.reason}')
    if decision.keep:
        return MatchResult(True, f'{KEEP_PREFIX}{provisional.reason}:{decision.score:.3f}')
    return MatchResult(False, f'{REJECT_PREFIX}{provisional.reason}:{decision.score:.3f}')


def match_post(
    text: str,
    *,
    alt_text: str = '',
    langs: Iterable[str] | None = None,
    author_did: str | None = None,
    author_handle: str | None = None,
    allowlist_dids: set[str] | None = None,
    allowlist_handles: set[str] | None = None,
    blocklist_dids: set[str] | None = None,
    blocklist_handles: set[str] | None = None,
    soft_prior_dids: set[str] | None = None,
    decider: DecisionBackend | None = None,
    topic_entities: TopicEntities | None = None,
    embed: object | None = None,
    use_default_decider: bool = False,
) -> MatchResult:
    """Return whether a post belongs in the mutualism feed.

    Decision order: blocklist → allowlist → out-of-scope entity → hard
    negative / in-scope entity / strong regex → event+venue →
    ambiguous+context → soft prior → ambiguous leftover → drop.

    When a decision model is active, it keep/rejects every candidate that
    passes the regex floor, including soft-prior ambiguous terms. Bare
    ambiguous terms with no prior stay dropped. Pass ``decider`` to inject
    one. ``use_default_decider`` is the indexer path: a disabled filter
    leaves the regex floor in place and still drops solicit-shaped keeps.

    ``embed`` is the raw Bluesky embed object (Jetstream or AppView). Opaque
    quote refs paired with mutual-aid language are solicit-shaped even when
    the quoted body is not hydrated yet.
    """
    allowlist_dids = allowlist_dids or set()
    allowlist_handles = {h.lower() for h in (allowlist_handles or set())}
    blocklist_dids = blocklist_dids or set()
    blocklist_handles = {h.lower() for h in (blocklist_handles or set())}
    soft_prior_dids = soft_prior_dids or set()
    entities = topic_entities if topic_entities is not None else default_topic_entities()
    lang_tags = _normalize_langs(langs)
    if decider is not None:
        backend: DecisionBackend | None = decider
    elif use_default_decider:
        backend = get_default_decider()
    else:
        backend = None

    if author_did and author_did in blocklist_dids:
        return MatchResult(False, 'blocklist_did')
    if author_handle and author_handle.lower() in blocklist_handles:
        return MatchResult(False, 'blocklist_handle')

    haystack = combine_text(text, alt_text=alt_text, langs=lang_tags)

    if author_did and author_did in allowlist_dids:
        provisional = MatchResult(True, 'allowlist_did')
        return _content_filter(
            haystack, provisional, decider=backend, embed=embed, author_did=author_did
        )
    if author_handle and author_handle.lower() in allowlist_handles:
        provisional = MatchResult(True, 'allowlist_handle')
        return _content_filter(
            haystack, provisional, decider=backend, embed=embed, author_did=author_did
        )

    if not haystack:
        return MatchResult(False, 'empty')

    entity = entities.lookup(haystack)
    strong_hit = _STRONG_POSITIVE.search(haystack)

    if entity is not None and entity.scope == 'out_of_scope' and not strong_hit:
        return MatchResult(False, f'entity_out_of_scope:{entity.entity_id}')

    if _HARD_NEGATIVE.search(haystack):
        if strong_hit and not _HARD_NEGATIVE_BLOCKS_STRONG.search(haystack):
            return _content_filter(
                haystack,
                MatchResult(True, 'strong_positive_over_negative'),
                decider=backend,
                embed=embed,
                author_did=author_did,
            )
        return MatchResult(False, 'hard_negative')

    if entity is not None and entity.scope == 'in_scope':
        return _content_filter(
            haystack,
            MatchResult(True, f'entity_in_scope:{entity.entity_id}'),
            decider=backend,
            embed=embed,
            author_did=author_did,
        )

    if strong_hit:
        return _content_filter(
            haystack,
            MatchResult(True, 'strong_positive'),
            decider=backend,
            embed=embed,
            author_did=author_did,
        )

    event_match = _match_in_scope_event(haystack)
    if event_match is not None:
        return _content_filter(
            haystack, event_match, decider=backend, embed=embed, author_did=author_did
        )

    term_haystack = _haystack_without_handles(haystack)
    ambiguous_hits = _AMBIGUOUS_TERM.findall(term_haystack)
    if ambiguous_hits:
        distinct = {re.sub(r'\s+', ' ', h.lower()) for h in ambiguous_hits}
        term = sorted(distinct)[0]

        paired = _INSTITUTION_SUBJECT.search(term_haystack) and _INSTITUTION_CUE.search(
            term_haystack
        )
        if _MUTUALIST_CONTEXT.search(term_haystack) or _STRONG_POSITIVE.search(haystack) or paired:
            return _content_filter(
                haystack,
                MatchResult(True, f'ambiguous_with_context:{term}'),
                decider=backend,
                embed=embed,
                author_did=author_did,
            )

        prior = _soft_prior_ambiguous(author_did, soft_prior_dids, term)
        if prior:
            return _content_filter(
                haystack, prior, decider=backend, embed=embed, author_did=author_did
            )

        return MatchResult(False, f'ambiguous_no_context:{term}')

    return MatchResult(False, 'no_match')
