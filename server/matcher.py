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
- Fandom, roleplay, and adult social-club invites that only name an ideology
- Offensive violence, martyrdom, and armed-struggle celebration
  (defiance and non-violent resistance may still keep)
- State-socialist calls to socialize all housing, even when Proudhon is
  named only as the foil (occupancy-and-use, possession, mutual credit,
  and Kropotkin on the federated commune still keep)

Recall without keywords comes from author allowlists and from soft author
priors. A prior is three decision-model keeps and no rejections in the window;
regex matches do not earn one. Author blocklists drop an account even when the
text would otherwise keep. When a decision model is enabled, it keep/rejects every candidate that
passes this floor (see ``server/decider.py``). Bare ambiguous terms stay
dropped unless a soft prior lets them through to that filter. A second lane
rescues posts the core floor drops when they name cohousing or a housing
co-op, a repair cafe, a tool library, a community fridge, or commoning.
That lane runs only while a decider is active, and it fails closed otherwise.
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
    SpiritDecision,
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
      | worker(?:s['’]?|['’]s)?\s+cooperatives?\b
      | worker(?:s['’]?|['’]s)?\s+co[\s-]?ops?\b
      | platform\s+cooperatives?\b
      | (?<![\w-])platform\s+co[\s-]?ops?\b(?!\s+(?:puzzle|game|games|mode|gameplay)\b)
      | \bplatformco-?ops?\b
      | (?:soft|sott)ware\s+co[\s-]?(?:operatives?|ops?)\b
      | faire\s+(?:vos|ses|mes)\s+courses.{0,80}coop[eé]rative\b
      | coop[eé]rative\s+qui\s+r[eé]habilite\s+des\s+appartements\b
      | food\s+co[\s-]?(?:operatives?|ops?)\b
      | grocery\s+co[\s-]?(?:operatives?|ops?)\b
      | non[\s-]?profit\s+co[\s-]?(?:operatives?|ops?)\b
      | \bco-?operative\s+charters?\b(?!\s+(?:schools?|academ(?:y|ies)|campus)\b)
      | worker[\s-]owned\s+co[\s-]?(?:operatives?|ops?)\b
      | farming(?:\s+\w+){0,1}\s+cooperatives?\b
      | coop[eé]ratives?\s+agricoles?\b
      | loomio\s+co[\s-]?(?:operatives?|ops?)\b
      | (?:became|become)\s+(?:a\s+)?homeowners?\s+at\s+
        (?:[\w'’-]+\s+){0,5}co[\s-]?ops?\b
      | volunteer\s+labou?r\s+shifts?.{0,80}co[\s-]?ops?\b
      | coop[eé]ratives?.{0,60}conseil\s+d['’]administration
      | community\s+currenc(?:y|ies)
      | \bmondragon\b
      | farmer\s+cooperatives?\b
      | processing\s+cooperatives?\b
      | bookshop\s+co-?operatives?\b
      | (?<!social\s)media\s+co-?ops?\b
      | artists?\s+cooperative\s+galler(?:y|ies)\b
      | co-?op\s+art\s+galler(?:y|ies)\b
      | cooperative\s+umbrella\s+orgs?\b
      | non[\s-]?profit/co-?operatives?\b
      | co-?op\s+of\s+(?:[\w'’-]+\s+){0,6}family\s+farms\b
      | co-?op\s+movement\b
      | co-?operativeheritage\.coop\b
      | cooperative\s+newsrooms?\b
      | coop[eé]rative.{0,160}vente\s+de\s+livres
      | co-?op\s+of\s+(?:[\w'’-]+\s+){0,6}dairies\b
      | coop[eé]rative\s+pour\s+seniors\b
      | cooperativ[ae]\s+di\s+comunit[aà]\b
      | cave\s+coop[eé]rative\b
      | co-?ops?\b.{0,40}\bpay\s+to\s+be\s+a\s+members?\b
      | entreprises\s+s['’]unissent\s+en\s+coop[eé]rative\b
      | co-?op\s+owned\s+by\s+(?:[\w'’-]+\s+){0,8}dairy\s+farms\b
      | co-?operatives\s+uk\b
      | \bcws\b
      | co-?operative\s+home\s+magazine\b
      | librairie\s+coop[eé]rative\b
      | voting\s+members?\s+of\s+the\s+cooperative\b
      | \bfarms?\s+cooperatives?\s+opens?\b
      | coop[eé]rative\s+ouvri[eè]re\b
      | autog[eè]rent\s+leur\s+usine\b
      | network\s+of\s+co-?ops\b
      | worker-owners?\b
      | co-?op\s+platforms?\b(?!\s+(?:game|games|shooter|fighter)\b)
      | \bbaugemeinschaft\b
      | co-?ops?\s+that\s+pays?\s+fair\s+wages\b
      | socially\s+responsible\s+cooperative\s+business\b
      | solar\s+co-?ops?\b.{0,160}\bbulk[\s-]purchase
      | community\s+co-?ops?\b
      | member-owned\s+co-?ops?\b
      | co-?ops?\b.{0,80}\bmember\s+dues\b
      | co-?op\b.{0,180}\bfamily\s+dairy\s+farms\b
      | energy\s+services\s+co-?op\b
      | co-?op\s+operating\s+in\s+the\s+red\b
      | co-?op\s+build\s+a\s+stronger\s+financial\s+foundation\b
      | ace\s+hardware\s+is\s+a\s+co-?op\b
      | rehearsal\s+co-?op\s+\(wvrc\)
      | co-?op\s+member\s+with\s+a\s+monthly\s+subscription\b
      | food\s+hub\s+cooperatives?\b
      | savings\s+and\s+credit\s+cooperative\b
      | farmer[\s-]owned\s+co[\s-]?(?:operative|op)s?\b
      | press\s+cooperatives?\b
      | publishing\s+cooperatives?\b
      | reprise\s+en\s+coop[eé]rative\b
      | pr[eé]sident\s+de\s+la\s+coop[eé]rative\b
      | cooperative\s+equity\b
      | cooperative\s+community\s+planned\b
      | bring\s+cooperative\s+practice\b
      | coop[eé]rative\s+qui\s+r[eé]fl[eé]chit\b
      | coop[eé]rative\s+a\s+rachet[eé]\b
      | osservatorio\s+regionale\s+sulla\s+cooperazione\b
      | cooperative\s+model\s+help\s+university\b
      | coop[eé]rative\s+d['’]habitation\b
      | \blocal\s+pub\b.{0,80}\bcooperative\b
      | started\s+a\s+co-?op\b(?!\s+(?:game|server|mode|campaign|run|session|night|playthrough)\b)
      | co-?ops?\s+(?:or|and)\s+other\s+democratic\b
      | forged\s+in\s+co-?op\s+principles
      | build\s+\w+\s+as\s+a\s+co-?op\b
      | community\s+co-?operatives?\b
      | community\s+solar\s+co-?ops?\b
      | community\s+solar\s+co-?operatives?\b
      | co-?op\s+shifts?\b(?!\s+(?:to|into|toward|towards)\b)
      | co-?op\b.{0,200}member-governed
      | member-governed.{0,200}co-?op\b
      | co-?op\b.{0,120}pooling\s+saved
      | pooling\s+saved.{0,120}co-?op\b
      | co-?op\s+emerging\b
      | alliance\s+coop[eé]rative\s+internationale
      | coop[eé]rative\s+fonci[eè]re
      | titre\s+participatif
      | collectively\s+own\b.{0,80}\b(?:tools|booking)
      | planned\s+as\s+a\s+cooperative\s+community
      | cooperative\s+software\s+initiative
      | join\s+our\s+co-?op\b(?!\s+game\b)
      | cooperative\s+gets\s+set\s+up
      | really\s*(?:\(\s*)?really[!)]*\s+free\s+markets?\b(?!\s+archive\b)
      | fishing\s+boat\s+cooperatives?\b
      | weaving\s+cooperatives?\b
      | franconian\s+cooperatives?\b
      | wine\s+co-?operatives?\b(?!\s+(?:game|games|board|mode|expansion)\b)
      | oldest\s+continuing\s+co-?ops?\b
      | save\s+our\s+co-?ops?\b
      | multi-stakeholder\s+co-?operatives?\b
      | owned\s+by\s+the\s+taxi\s+drivers\b
      | run\s+their\s+own\s+co-?op\b
      | customer-owned\s+co-?ops?\b
      | co-?op\s+wind\s?farms?\b
      | cooperative\s+wholesale\s+society\b
      | loose\s+co-?op\b
      | form\s+a\s+co-?op\b(?!\s+(?:team|party|lobby|group|mode|game)\b)
      | cooperative\s+movement\b
      | cooperative\s+villages?\b
      | artists?\s+cooperatives?\b
      | computer\s+cooperatives?\b
      | credit\s+cooperative\s+societ(?:y|ies)\b
      | journalist-owned\s+cooperatives?\b
      | \bconfcooperative\b
      | cooperative\s+roots\b
      | world\s+cooperative\s+monitor\b
      | lieu\s+de\s+vie\s+de\s+la\s+coop
      | repaired\s+at\s+(?:[\w'’-]+\s+){0,6}co-?ops?\b
      | \brepaircafe\b.{0,200}\bco-?ops?\b
      | \bco-?ops?\b.{0,200}\brepaircafe\b
      | clay\s+co-?ops?\b
      | co-?op\s+city\b.{0,180}\bboard\s+meeting\b
      | right-to-use\s+model\b
      | textile\s+cooperatives?\b
      | childcare\s+cooperatives?\b
      | child[\s-]care\s+cooperatives?\b
      | coop[eé]rative\s+citoyenne\b
      | \bis\s+a\s+cooperative\s+speciali[sz]ing\b
      | indivisible\s+reserves\b
      | cooperative\s+farming\s+and\s+light\s+industry\b
      | statut\s+de\s+coop[eé]rative\b
      | cooperative\s+member\s+fees\b
      | rebuilt\s+as\s+a\s+co-?op\b
      | co-?op\s+within\s+a\s+co-?op\b
      | milk\s+pooling\b
      | dairy\s+cooperatives?\b
      | a\s+co-?op\s+since\s+\d{4}\b
      | co-?op\s+superstores?\b
      | now\s+a\s+public\s+co-?op\b
      | cooperative\s+preschools?\b
      | cooperative.{0,80}mobile\s+shops?\b
      | agricultural\s+cooperatives?\b
      | coop[eé]rative\s+d['’]artistes\b
      | founded\s+a\s+cooperative\b(?!\s+(?:game|board)\b)
      | cooperative\s+sociali\b
      | senza\s+cooperative\b
      | cooperative\s+cure\s+for\s+capitalism\b
      | growers['’]?\s+co-?ops?\b
      | wind\s+turbine\s+co-?ops?\b
      | co-?op\s+music\s+services?\b
      | cooperative\s+digital\s+music\s+platforms?\b
      | olive\s+oil\s+cooperatives?\b
      | news\s+cooperatives?\b
      | cooperative\s+grocery\s+stores?\b
      | cooperative\s+economics\b
      | coffee\s+cooperatives?\b
      | joined\s+(?:a|the|this|our|their)\s+cooperatives?\b
      | health\s+insurance\s+cooperatives?\b
      | doula\s+cooperatives?\b
      | milk\s+suppliers\s+cooperatives?\b
      | forestry\s+co[\s-]?(?:operatives?|ops?)\b
        (?!\s+(?:(?:board\s+)?games?|mode|gameplay)\b)
      | natural\s+foods\s+co[\s-]?ops?\b
      | solar\s+co-?ops?\b.{0,200}\bgroup[\s-]purchasing
      | energieco[oö]peraties?\b
      | filmmakers?\s+co-?ops?\b
      | the\s+[A-Za-z]+\s+cooperative,\s+opens\b
      | coop[eé]rative\s+des\s+techniciens\b
      | cooperative\s+bulk\s+handling\b
      | financial\s+cooperatives?\b
      | our\s+little\s+cooperative\b(?!\s+(?:game|games|mode|gameplay|campaign)\b)
      | co-?op\s+conversations?\b
      | notre\s+coop[eé]rative\b
      | leftist\s+co-?ops?\b
      | \bsolarcoop\b
      | soci[eé]t[eé]\s+coop[eé]rative\s+photovolta
      | solaire\s+citoyen.{0,80}coop[eé]rative
      | farmers\s+in\s+a\s+cooperative.{0,160}grocery\s+store
      | cooperative\s+owned\s+by\s+its\s+farmers\b
      | publishing\s+co-?ops?\b(?!\s+(?:program|internship|placement|job|position)\b)
      | fertili[sz]er\s+co-?ops?\b
      | \bour\s+developer\s+co-?ops?\b
      | credit\s+unions?.{0,80}cooperative\s+models?\b
      | founded\s+the\s+cooperative\s+community\b
      | agriculture\s+cooperatives?\b
      | not-for-profit\s+cooperatives?\b
      | our\s+cooperative\s+model\b(?!\s+(?:of\s+)?(?:play|gameplay|game|gaming)\b)
      | \bartisans\s+cooperatives?\b
      | \bartisans?['’]s?\s+cooperatives?\b
      | \balmond\s+cooperatives?\b
      | cooperative\s+of\s+farm\s+business\s+owners\b
      | community\s+enterprises?\s+co(?:-?ops?|operatives?)\b
      | coop[eé]rative.{0,220}[eé]co-r[eé]novation
      | writer[\s-]owned\s+cooperatives?\b
      | coffee\s+shop\s+is\s+a\s+co-?op\b
      | \bco\s+op\s+site\b
      | worker[\s-]owned\s+(?!electric\b|utility\b|rural\b)\w+\s+cooperatives?\b
      | \bgroup\s+health\s+cooperative\b
      | \btechnology\s+freedom\s+cooperative\b
      | \bcooperative\s+banking\b
      | became\s+a\s+co-?op\b(?!\s+(?:game|mode|server|session)\b)
      | artists?\s+co-?ops?\b
      | \bfarmer\s+co-?ops?\b
      | joining\s+a\s+solar\s+co-?op\b
      | worker[\s-]owned\s+newsrooms?\b
      | worker[\s-]owned\s+media\s+leaders\b
      | squatters?['’]?\s+cooperatives?\b
      | \bfontstand\s+cooperative\b
      | \bfan\s+cooperatives?\b(?!\s+(?:game|games|board|mode)\b)
      | \bsupporter\s+cooperatives?\b
      | cooperative\s+business\b.{0,200}\bfarmers\s+who\s+own\b
      | art\s+cooperative\s+galler(?:y|ies)\b
      | start\s+a\s+\#?cooperatives?\b(?!\s+(?:game|games|board|mode|server|campaign)\b)
      | co-?operative\s+principles\b(?!\s+of\s+(?:play|gameplay|games?|the\s+game)\b)
      | collectively\s+buy\b.{0,160}\bco-?ops?\b
      | put\s+together\s+a\s+co-?op\b(?!\s+(?:game|games|team|server|mode)\b)
      | co-?operative\s+global\s+music\b
      | cooperative\s+\#?filmfestivals?\b
      | \bcooperative\s+of\b(?:\s+[\w,]+){0,8}\s+
        (?:photographers?|writers?|artists?|editors?|journalists?)\b
      | coop[eé]rative\s+maritime\b
      | coop[eé]rative\s+d['’]activit[eé]\s+et\s+d['’]emploi\b
      | renewable\s+energy\s+cooperatives?\b
      | \blongo\s+ma[iï]\b
      | wine\s+producers?\s+and\s+cooperatives?\b
      | grower(?:s['’]?|['’]s)\s+cooperatives?\b
        (?!\s+(?:(?:board\s+)?games?|mode|gameplay)\b)
      | cooperative\s+musician\s+workerspaces?\b
      | cooperative\s+banks?\s+established\s+by\s+farmers\b
      | \#wineco-?operatives?\b
      | cooperative\s+grain\s+warehouses?\b
      | our\s+cooperative\s+community\b
        (?!\s+(?:of\s+)?(?:play|gameplay|games?|gaming|players?|server)\b)
      | territorio\s+colectivo\b
      | coop[eé]rative\s+[aà]\s+but\s+non[\s-]?lucratif\b
      | cooperative\s+publishing\b(?!\s+(?:game|games|board|mode)\b)
      | join\s+the\s+co-?op\s+at\s+subvert\.fm\b
      | ace\s+hardware\s+cooperatives?\b
      | workshop\s+cooperatives?\b
        (?!\s+(?:game|games|board|mode|gameplay|agreement)\b)
      | la\s+coop[eé]rative\s+de\s+
        (?!(?:jeu|jeux|games?|cr[eé]dits?|logement)\b)
      | \bcooperative\s+inc\b
      | marketing\s+cooperatives?\b(?!\s+(?:game|games|board|mode)\b)
      | industrial\s+cooperatives?\b
      | processing\s+co-?ops?\b(?!\s+(?:game|games|mode|gameplay)\b)
      | neighborhood\s+farm\s+co-?ops?\b
      | cooperative\s+news\s+outlets?\b
      | coop[eé]rative\s+radiophonique\b
      | coop[eé]rative\s+laiti[eè]res?\b
      | mod[eè]le\s+de\s+la\s+coop[eé]rative\b
        (?!\s+(?:de\s+)?(?:jeu|jeux|games?|logement|cr[eé]dits?)\b)
      | transform[eé]e\s+en\s+coop[eé]rative\b
        (?!\s+(?:de\s+)?(?:jeu|jeux|games?|logement|cr[eé]dits?)\b)
      | subvert\.fm\s+cooperatives?\b
      | international\s+cooperative\s+alliance\b
      | cooperative\s+power\b.{0,200}\b(?:non[\s-]?profit|energy\s+model)\b
      | cranberry\s+cooperatives?\b
        (?!\s+(?:(?:board\s+)?games?|mode|gameplay)\b)
      | grower[\s-]owned\s+co[\s-]?(?:operative|op)s?\b
        (?!\s+(?:(?:board\s+)?games?|mode|gameplay)\b)
      | cooperative\s+agriculture\b
        (?!\s+(?:(?:board\s+)?games?|mode|gameplay)\b)
      | \bof\s+cooperative\s+business\b(?!\s+agreements?\b)
      | mushroom\s+cooperatives?\b
        (?!\s+(?:(?:board\s+)?games?|mode|gameplay)\b)
      | co-?op\s+studio\s+development\b.{0,240}(?:cooperatives?\b|worker-?led\b)
      | (?:cooperatives?\b|worker-?led\b).{0,240}co-?op\s+studio\s+development\b
      | (?<!martial\s)arts\s+cooperatives?\b
        (?!\s+(?:(?:board\s+)?games?|mode|gameplay)\b)
      | filmmaking\s+cooperatives?\b
        (?!\s+(?:(?:board\s+)?games?|mode|gameplay)\b)
      | (?<!social\s)media\s+cooperatives?\b
        (?!\s+(?:(?:board\s+)?games?|mode|gameplay)\b)
      | inference[\s-]cooperatives?\b
      | artists?(?:['’]s?|s['’])?\s+cooperatives?\b
        (?!\s+(?:(?:board\s+)?games?|mode|gameplay)\b)
      | (?<!parts\s)(?<!part\s)(?<!investir\s)(?<!sociales\s)
        pour\s+la\s+coop[eé]rative\b
        (?!\s+de\s+(?:jeu|jeux|games?|cr[eé]dits?|logement)\b)
      | coop[eé]ratives?\s+de\s+sant[eé]\b
      | through\s+a\s+cooperative\s+lens\b
      | delaying\s+grape\s+payments\b.{0,200}cooperatives?\b
      | cooperatives?\b.{0,200}delaying\s+grape\s+payments\b
      | ocean\s+spray\s+cooperatives?\b
      | pari\s+de\s+la\s+coop[eé]rative\b
        (?!\s+de\s+(?:jeu|jeux|games?|logement|cr[eé]dits?)\b)
      | cooperative\s+non[\s-]?profits?\b
        (?!\s+(?:(?:board\s+)?games?|mode|gameplay|housing|agreements?)\b)
      | cooperative\s+art\s+galler(?:y|ies)\b
        (?!\s+(?:(?:board\s+)?games?|mode|gameplay)\b)
      | farmers?(?:['’]s|s['’])?\s+co-?ops?\b
        (?!\s+(?:(?:board\s+)?games?|mode|gameplay|electric|utility)\b)
      | kubsitu\s+cooperatives?\b
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
      | worker(?:s['’]?|['’]s)?\s+cooperatives?\b
      | worker(?:s['’]?|['’]s)?\s+co[\s-]?ops?\b
      | platform\s+cooperatives?\b
      | (?<![\w-])platform\s+co[\s-]?ops?\b(?!\s+(?:puzzle|game|games|mode|gameplay)\b)
      | \bplatformco-?ops?\b
      | (?:soft|sott)ware\s+co[\s-]?(?:operatives?|ops?)\b
      | faire\s+(?:vos|ses|mes)\s+courses.{0,80}coop[eé]rative\b
      | coop[eé]rative\s+qui\s+r[eé]habilite\s+des\s+appartements\b
      | food\s+co[\s-]?(?:operatives?|ops?)\b
      | grocery\s+co[\s-]?(?:operatives?|ops?)\b
      | non[\s-]?profit\s+co[\s-]?(?:operatives?|ops?)\b
      | \bco-?operative\s+charters?\b(?!\s+(?:schools?|academ(?:y|ies)|campus)\b)
      | worker[\s-]owned\s+co[\s-]?(?:operatives?|ops?)\b
      | farming(?:\s+\w+){0,1}\s+cooperatives?\b
      | coop[eé]ratives?\s+agricoles?\b
      | loomio\s+co[\s-]?(?:operatives?|ops?)\b
      | (?:became|become)\s+(?:a\s+)?homeowners?\s+at\s+
        (?:[\w'’-]+\s+){0,5}co[\s-]?ops?\b
      | volunteer\s+labou?r\s+shifts?.{0,80}co[\s-]?ops?\b
      | coop[eé]ratives?.{0,60}conseil\s+d['’]administration
      | community\s+currenc(?:y|ies)
      | \bmondragon\b
      | farmer\s+cooperatives?\b
      | processing\s+cooperatives?\b
      | bookshop\s+co-?operatives?\b
      | (?<!social\s)media\s+co-?ops?\b
      | artists?\s+cooperative\s+galler(?:y|ies)\b
      | co-?op\s+art\s+galler(?:y|ies)\b
      | cooperative\s+umbrella\s+orgs?\b
      | non[\s-]?profit/co-?operatives?\b
      | co-?op\s+of\s+(?:[\w'’-]+\s+){0,6}family\s+farms\b
      | co-?op\s+movement\b
      | co-?operativeheritage\.coop\b
      | cooperative\s+newsrooms?\b
      | coop[eé]rative.{0,160}vente\s+de\s+livres
      | co-?op\s+of\s+(?:[\w'’-]+\s+){0,6}dairies\b
      | coop[eé]rative\s+pour\s+seniors\b
      | cooperativ[ae]\s+di\s+comunit[aà]\b
      | cave\s+coop[eé]rative\b
      | co-?ops?\b.{0,40}\bpay\s+to\s+be\s+a\s+members?\b
      | entreprises\s+s['’]unissent\s+en\s+coop[eé]rative\b
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
      | worker(?:s['’]?|['’]s)?\s+co(?:operatives?|[\s-]?ops?)\b
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
#
# VRChat, roleplay packs, and shrine invites are the same kind of collision:
# an in-scope ideology name does not make a fandom or adult social club
# into a commune, assembly, or cooperative. A death hoax in the post's own
# words is not rescued by an in-scope link card. Seeing someone check out
# at a food co-op is a sighting, not the cooperative. A chamber
# anniversary feature that calls a store a local business and community
# cooperative is an ad for the shop. A matching-gift drive is a pledge.
# Mondragon the federation is not a given name followed by Mondragon.
# A fuel-price board that names a community cooperative is a rate notice.
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
      | \bvrchat\b
      | tiny\s+sweethearts
      | communal\s+shrine
      | \bgoddess(?:['’]s)?\s+pack\b
      | apparent\s+deaths
      | \#a2council\b
      | has\s+been\s+verified\s+by\s+@
      | \bshare\s+offers?\b
      | \bkickstarters?\b
      | monthly\s+gifts?\b.{0,40}\bmatch(?:ed|ing)?\b
      | bipartisan.{0,200}community\s+land\s+trusts?
      | brought\s+up\s+community\s+land\s+trusts?
      | \bsaw\b.{0,60}\bchecking\s+out\b.{0,80}\bfood\s+co[\s-]?(?:operatives?|ops?)\b
      | local\s+business\s+and\s+community\s+co-?operatives?\b
      | cheapest\s+(?:diesel|petrol|unleaded)\b.{0,240}co-?(?:operatives?|ops?)\b
      | co-?(?:operatives?|ops?)\b.{0,240}cheapest\s+(?:diesel|petrol|unleaded)\b
      | \b\d+(?:\.\d+)?p/l\b.{0,200}co-?(?:operatives?|ops?)\b
      | co-?(?:operatives?|ops?)\b.{0,200}\b\d+(?:\.\d+)?p/l\b
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
      | \bvrchat\b
      | tiny\s+sweethearts
      | communal\s+shrine
      | \bgoddess(?:['’]s)?\s+pack\b
      | apparent\s+deaths
      | \#a2council\b
      | has\s+been\s+verified\s+by\s+@
      | \bshare\s+offers?\b
      | \bkickstarters?\b
      | monthly\s+gifts?\b.{0,40}\bmatch(?:ed|ing)?\b
      | bipartisan.{0,200}community\s+land\s+trusts?
      | brought\s+up\s+community\s+land\s+trusts?
      | \bsaw\b.{0,60}\bchecking\s+out\b.{0,80}\bfood\s+co[\s-]?(?:operatives?|ops?)\b
      | local\s+business\s+and\s+community\s+co-?operatives?\b
      | cheapest\s+(?:diesel|petrol|unleaded)\b.{0,240}co-?(?:operatives?|ops?)\b
      | co-?(?:operatives?|ops?)\b.{0,240}cheapest\s+(?:diesel|petrol|unleaded)\b
      | \b\d+(?:\.\d+)?p/l\b.{0,200}co-?(?:operatives?|ops?)\b
      | co-?(?:operatives?|ops?)\b.{0,200}\b\d+(?:\.\d+)?p/l\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Engels-style housing slogan. Link cards often name Proudhon in the blurb
# while the title argues the opposite thesis. A mutualist or Kropotkin frame
# may quote the slogan and still keep.
_SOCIALIZE_ALL_HOUSING = re.compile(
    r'\bsociali[sz]e\s+all\s+housing\b',
    re.IGNORECASE,
)
_SOCIALIZE_ALL_HOUSING_RESCUE = re.compile(
    r"""
    (?:
        occupancy[\s-]and[\s-]use
      | possession\s+(?:not|vs\.?|versus)\s+property
      | property\s+is\s+theft
      | \bkropotkin\b
      | federated\s+commune
      | mutual\s+credit
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Commercial and central banks often use the name "People's Bank" / "Peoples Bank".
# A quarterly profit report is not Proudhon's Banque du Peuple. Mutual credit,
# free banking, or Proudhon in the same post may still discuss profit and keep.
_COMMERCIAL_PEOPLES_BANK = re.compile(
    r"""
    (?:
        people'?s\s+bank\b.{0,120}\bprofit\b.{0,40}\bquarter\b
      | \bprofit\b.{0,40}\bquarter\b.{0,120}people'?s\s+bank\b
      | people'?s\s+bank\b.{0,120}\b(?:earnings|net\s+income|shareholders?|dividends?)\b
      | \b(?:earnings|net\s+income|shareholders?|dividends?)\b.{0,120}people'?s\s+bank\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
_COMMERCIAL_PEOPLES_BANK_RESCUE = re.compile(
    r"""
    (?:
        \bproudhon\b
      | mutual\s+credit
      | mutual\s+bank
      | banque\s+du\s+peuple
      | william\s+b\.?\s+greene
      | occupancy[\s-]and[\s-]use
      | cost\s+the\s+limit\s+of\s+price
      | equitable\s+commerce
      | \bfree\s+banking\b
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
        \b\d{1,2}/\d{1,2}\b|
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
      | really\s*(?:\(\s*)?really[!)]*\s+free\s+markets?
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

# Practices the core floor drops. With a decider they get the spirit questions;
# without one they stay dropped, so offline eval does not keep the whole surface.
_SPIRIT_SURFACE = re.compile(
    r"""
    (?:
        housing\s+co[\s-]?ops?\b
      | housing\s+cooperatives?\b
      | cooperative\s+housing\b
      | co[\s-]?ops?\s+housing\b
      | co[\s-]?housing\b
      | repair\s+caf[eé]s?\b
      | tool\s+librar(?:y|ies)\b
      | community\s+fridges?\b
      | \bcommoning\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_HANDLE_MENTION = re.compile(r'(?<![\w.])@[\w.-]+', flags=re.UNICODE)

_EMBED_TEXT_MAX = 320


def _normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text or '').strip()


# A capitalized given name plus Mondragon is a person. The federation is the
# bare name, including after a lowercase particle such as "de Mondragon".
_GIVEN_NAME_MONDRAGON = re.compile(r'\b([A-Z][a-z]{1,20}) Mondragon\b')


def _without_given_name_mondragon(haystack: str) -> str:
    blank = ' ' * len('Mondragon')
    return _GIVEN_NAME_MONDRAGON.sub(lambda match: f'{match.group(1)} {blank}', haystack)


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


def _spirit_filter(
    haystack: str,
    *,
    decider: DecisionBackend,
    embed: object | None,
    author_did: str | None,
) -> MatchResult:
    """Keep a non-core practice only when the spirit lane clears. Fail closed."""
    if looks_like_solicit(haystack, embed=embed):
        return MatchResult(False, 'solicit_cue_unscored:spirit')
    method = getattr(decider, 'decide_spirit', None)
    if not callable(method):
        return MatchResult(False, f'{UNSCORED_PREFIX}spirit')
    images = image_data_urls(embed, author_did=author_did) or None
    decision = method(haystack, images=images)
    if not isinstance(decision, SpiritDecision):
        return MatchResult(False, f'{UNSCORED_PREFIX}spirit')
    keep, label = decision.admission()
    prefix = KEEP_PREFIX if keep else REJECT_PREFIX
    return MatchResult(keep, f'{prefix}{label}')


def _maybe_spirit(
    haystack: str,
    *,
    decider: DecisionBackend | None,
    embed: object | None,
    author_did: str | None,
) -> MatchResult | None:
    """Return a spirit-lane result, or None to keep the core drop reason."""
    if decider is None or _SPIRIT_SURFACE.search(haystack) is None:
        return None
    return _spirit_filter(haystack, decider=decider, embed=embed, author_did=author_did)


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
    ambiguous+context → soft prior → spirit surface → drop.

    When a decision model is active, it keep/rejects every candidate that
    passes the regex floor, including soft-prior ambiguous terms. Bare
    ambiguous terms with no prior stay dropped, unless the text names a
    spirit-lane practice (cohousing, housing co-op, repair cafe, tool
    library, community fridge, commoning). That lane fails closed without
    a decider. Pass ``decider`` to inject one. ``use_default_decider`` is
    the indexer path: a disabled filter leaves the regex floor in place and
    still drops solicit-shaped keeps.

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

    haystack = _without_given_name_mondragon(combine_text(text, alt_text=alt_text, langs=lang_tags))

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

    if _SOCIALIZE_ALL_HOUSING.search(haystack) and not _SOCIALIZE_ALL_HOUSING_RESCUE.search(
        haystack
    ):
        return MatchResult(False, 'hard_negative')

    if _COMMERCIAL_PEOPLES_BANK.search(haystack) and not _COMMERCIAL_PEOPLES_BANK_RESCUE.search(
        haystack
    ):
        return MatchResult(False, 'hard_negative')

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

        spirit = _maybe_spirit(haystack, decider=backend, embed=embed, author_did=author_did)
        if spirit is not None:
            return spirit
        return MatchResult(False, f'ambiguous_no_context:{term}')

    spirit = _maybe_spirit(haystack, decider=backend, embed=embed, author_did=author_did)
    if spirit is not None:
        return spirit
    return MatchResult(False, 'no_match')
