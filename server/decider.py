"""Perplexity decision-model content filter (OpenRouter Decisions API).

The regex floor in ``server.matcher`` proposes candidates. This module
keep/rejects those candidates with ``perplexity/pplx-decider-v1.2``.
Application code thresholds the returned yes-probability (``noul``). When a
candidate has images, those JPEG data URLs are sent in ``state`` next to the
post text. Network errors and unusable payloads decline (``None``) so the
matcher can fail closed.

``get_default_decider`` is off unless ``DECISION_ENABLED`` is set, so offline
eval and unit tests stay on the regex floor. When the flag is on and no API
key is configured, the client still declines and the matcher drops.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from server.scope_text import BIOLOGICAL_EXCLUSION

DEFAULT_MODEL = 'perplexity/pplx-decider-v1.2'
DEFAULT_API_URL = 'https://openrouter.ai/api/alpha/decisions'
DEFAULT_KEEP_MIN = 0.5
DEFAULT_TIMEOUT_S = 8.0
DEFAULT_MAX_TEXT_CHARS = 2000
PROMPT_VERSION = 'institution-keep-v8'
SPIRIT_PROMPT_VERSION = 'spirit-account-v2'
SPIRIT_MIN = 0.58
VOICE_MIN = 0.70
ACCOUNT_MIN = 0.70
LIFT_MIN = 0.64

KEEP_PREFIX = 'decision_keep:'
REJECT_PREFIX = 'decision_reject:'
UNSCORED_PREFIX = 'decision_unscored:'
_MAX_IMAGE_PARTS = 4
_MAX_DATA_URL_CHARS = 2_000_000
_DATA_URL = re.compile(r'^data:image/(?:jpeg|png|webp);base64,[A-Za-z0-9+/=]+$')

_KEEP_QUESTION: dict[str, Any] = {
    'keep': {
        'type': 'noul',
        'instructions': (
            'Should this Bluesky post be kept in a curated feed of durable '
            'non-state institutions — mutual credit, possession, cooperatives, '
            'commons, assemblies, and confederations — that people can use while '
            'firms and states still dominate, and whose aim is to reduce dependence '
            'on both? In scope: Proudhonian and left-market traditions (reciprocity, '
            'occupancy-and-use, mutual credit, free banking, equitable commerce, '
            'agorism, free-market anti-capitalism); Kropotkin on mutual aid and the '
            'federated commune; Bookchin, libertarian municipalism, and social ecology '
            'when the post is about communes or assemblies; democratic confederalism '
            "when the post is about councils, cooperatives, or women's institutions. "
            'Contract and assembly both belong. ' + BIOLOGICAL_EXCLUSION + ' is out of scope. '
            'Fandom, roleplay, and adult social-club invites are out of scope even '
            'when they name an in-scope ideology. '
            'Engels-style calls to socialize all housing are out of scope, including '
            'when Proudhon is named only to dismiss tenant ownership. '
            "A commercial or central bank that only shares the name People's Bank "
            'or Peoples Bank is out of scope, including quarterly profit and earnings '
            "reports. Proudhon's Bank of the People and mutual credit stay in scope. "
            'A co-operative share offer is out of scope. A community land trust '
            'mentioned only inside bipartisan building reform, or a governor or '
            'mayor housing partnership, is out of scope. A land trust that '
            'holds land for occupancy stays in scope, and a co-op that runs '
            'its service stays in scope when the post is not a share offer.'
        ),
        'criteria': {
            'true': (
                'The post substantially celebrates or practices an in-scope '
                'institution, in a constructive tone. Defiance and non-violent '
                'resistance are fine. Argument between contract and assembly is '
                'fine. Collective org news without a personal hardship pitch is fine. '
                'A co-op explaining that member dues fund the service is in scope, '
                'including when the contrast is an ad-funded platform.'
            ),
            'false': (
                'Off-topic, hostile, unclear, or out of scope. Includes right-wing '
                'or reactionary framing, scare coverage of socialism or anarchism, '
                'anarcho-capitalism, chaos or entertainment uses of anarchy, crypto '
                'or Web3 jargon, bare mutual aid, personal fundraising, charity, '
                'state-capacity or nation-state reform, Engels-style calls to '
                'socialize all housing (including when Proudhon is named only to '
                'dismiss tenant ownership), commercial or central-bank earnings '
                "under the name People's Bank or Peoples Bank (profit, dividends, "
                "shareholders) when the post is not about mutual credit or Proudhon's "
                'Bank of the People, a co-operative share offer, a community land '
                'trust named only inside bipartisan building reform or a governor '
                'or mayor housing partnership, academic social-ecological '
                'systems, sectarian communalism, EU confederalism, fandom, roleplay, '
                'or adult social-club invites (VRChat, roleplay packs, shrine servers), '
                'syndicalist or '
                'union cheerleading with no durable institution, insurrection, '
                'armed-struggle footage, martyrdom, party war communiqués, '
                + BIOLOGICAL_EXCLUSION
                + ', hate, or offensive use of violence.'
            ),
        },
    }
}

# Second lane for practices the core floor drops. A post must clear spirit and
# lift, then either voice (a person inside the practice) or account (a plain
# explanation of how it works and why someone would join or start it).
_SPIRIT_QUESTIONS: dict[str, Any] = {
    'spirit': {
        'type': 'noul',
        'instructions': (
            'Is this Bluesky post spiritually mutualist, even if it never names '
            'Proudhon or anarchism? A mutualist spirit here means '
            'people meeting needs through a lived practice that reduces dependence '
            'on firms, landlords, or the state: cooperative or cohousing life, a '
            'tool library, a repair cafe, a community fridge or community pantry '
            'used as a commons, or commoning. The practice has to be the subject '
            'of the post, not one stop in a personal diary. A metaphor does not '
            'count (a software tool library, a personal book library, '
            "'community' as an audience). Out of scope: personal fundraising, "
            'charity appeals, handing food you just bought to a stranger, an '
            'errand list that mentions dropping leftovers at a community fridge '
            'among other stops, crime or disaster news about a fridge or building, '
            'state housing policy, party campaigns, award blurbs, academic theory '
            'with no lived practice, slogans, and pure event logistics.'
        ),
        'criteria': {
            'true': (
                'A real mutualist-spirited practice is what the post is about: '
                'someone living in it, doing the shared work, or showing how the '
                'arrangement works.'
            ),
            'false': (
                'False friend, news brief, policy, fundraising, charity appeal, '
                'street charity, a personal errand log, a leftover drop-off '
                'mentioned in passing, academic aside, slogan, or the practice '
                'is only a metaphor.'
            ),
        },
    },
    'voice': {
        'type': 'noul',
        'instructions': (
            'Is a particular person present in this post? A human voice means a '
            "scene, a decision, a body at work, a named day's labor, or someone "
            'thinking on the page. First person helps but is not required if a '
            "specific person's action is shown. Press releases, award blurbs, "
            'hour changes, ticket links, slogan cards, headline-plus-URL, and '
            'organization marketing are not a human voice.'
        ),
        'criteria': {
            'true': 'A particular person and a concrete moment are in the post.',
            'false': (
                'Institutional copy, a flyer, a headline, a statistic with no person, or a slogan.'
            ),
        },
    },
    'account': {
        'type': 'noul',
        'instructions': (
            'Does this Bluesky post carry a mutualist-spirited practice to a reader '
            'who does not already live it? The practices are cooperative housing, '
            'cohousing, a tool library, a repair cafe, a community fridge, or commoning. '
            'Yes when a person is shown inside the practice, or when the post explains '
            'in plain language what the practice is like, how it works, or why someone '
            'would take part, start one, or trust it. A link plus a real sentence of '
            'explanation counts. No when the post is only a time, place, or hour change; '
            'a headline that something opened, closed, burned, or broke ground; a grant, '
            'award, or funding notice; a slogan shorter than a thought; a complaint; '
            'or a promotional shout-out that never says how the practice works.'
        ),
        'criteria': {
            'true': (
                'A scene of the practice, or a plain explanation of what it is like '
                'and why it is worth joining or starting.'
            ),
            'false': (
                'Logistics, news headline, grant or award notice, slogan, complaint, '
                'or promo that does not explain the practice.'
            ),
        },
    },
    'lift': {
        'type': 'noul',
        'instructions': (
            'Is this post positively inflected and value-additive for a reader who '
            'already follows a feed of mutualist institutions? Positively inflected '
            'means constructive, warm, or defiant without cruelty. Grief beside '
            'ongoing practice can pass. Value-additive means the reader comes away '
            'with a picture of the practice, a method, or a felt reason to trust it. '
            'Doom, pile-ons, cynicism, irony with nothing left over, duplicate '
            'talking points, and mundane mentions that teach nothing are out. '
            'Warmth alone is not lift: a thank-you for snacks, cheesies, or a pie, '
            'with no picture of how the shared arrangement works, teaches nothing.'
        ),
        'criteria': {
            'true': (
                "Constructive and worth a stranger's time. The reader learns or feels the practice."
            ),
            'false': (
                'Thin, cruel, cynical, doom-soaked, a restatement with no residue, '
                'or a warm errand anecdote that never shows how the practice works.'
            ),
        },
    },
}


@dataclass(frozen=True)
class Decision:
    keep: bool
    score: float


@dataclass(frozen=True)
class SpiritDecision:
    """Four spirit-lane scores. ``label`` is the matcher reason suffix."""

    spirit: float
    voice: float
    account: float
    lift: float

    def admission(self) -> tuple[bool, str]:
        """Return whether the post clears the spirit lane, and a reason suffix."""
        if self.spirit < SPIRIT_MIN:
            return False, f'spirit_theme:{self.spirit:.3f}'
        if self.lift < LIFT_MIN:
            return False, f'spirit_lift:{self.lift:.3f}'
        if self.voice >= VOICE_MIN:
            return True, f'spirit_voice:{self.voice:.3f}'
        if self.account >= ACCOUNT_MIN:
            return True, f'spirit_account:{self.account:.3f}'
        best = max(self.voice, self.account)
        return False, f'spirit_path:{best:.3f}'


class DecisionBackend(Protocol):
    def decide(self, haystack: str, *, images: Sequence[str] | None = None) -> Decision | None: ...


JudgeFn = Callable[[str, Sequence[str]], dict[str, Any]]


def image_parts(images: Sequence[str] | None) -> list[dict[str, Any]]:
    """Keep up to four valid image data URLs, in order, without duplicates."""
    parts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for url in images or ():
        if not isinstance(url, str) or url in seen or len(url) > _MAX_DATA_URL_CHARS:
            continue
        if _DATA_URL.match(url) is None:
            continue
        seen.add(url)
        parts.append({'type': 'image_url', 'image_url': {'url': url}})
        if len(parts) >= _MAX_IMAGE_PARTS:
            break
    return parts


def decision_state(text: str, images: Sequence[str] | None = None) -> dict[str, Any]:
    """State object for one classification. Text-only posts stay ``{'post': text}``."""
    state: dict[str, Any] = {}
    if text:
        state['post'] = text
    parts = image_parts(images)
    if parts:
        state['images'] = parts
    if not state:
        raise ValueError('empty decision state')
    return state


def decision_enabled() -> bool:
    raw = os.environ.get('DECISION_ENABLED', '').strip().lower()
    return raw in {'1', 'true', 't', 'yes', 'y'}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def decision_from_payload(payload: dict[str, Any], *, keep_min: float) -> Decision:
    """Parse a Decisions API body. Raises ValueError on an unusable payload."""
    answers = payload.get('answers')
    if not isinstance(answers, dict):
        raise ValueError('decision response missing answers')
    keep = answers.get('keep')
    if not isinstance(keep, dict):
        raise ValueError('decision response missing keep')
    if keep.get('type') != 'noul':
        raise ValueError('decision keep answer is not noul')
    score = float(keep['noul'])
    if score < 0.0 or score > 1.0:
        raise ValueError('decision noul out of range')
    return Decision(score >= keep_min, score)


def spirit_from_payload(payload: dict[str, Any]) -> SpiritDecision:
    """Parse spirit, voice, account, and lift. Raises ValueError if any is unusable."""
    answers = payload.get('answers')
    if not isinstance(answers, dict):
        raise ValueError('decision response missing answers')
    scores: dict[str, float] = {}
    for key in ('spirit', 'voice', 'account', 'lift'):
        node = answers.get(key)
        if not isinstance(node, dict):
            raise ValueError(f'decision response missing {key}')
        if node.get('type') != 'noul':
            raise ValueError(f'decision {key} answer is not noul')
        score = float(node['noul'])
        if score < 0.0 or score > 1.0:
            raise ValueError(f'decision {key} noul out of range')
        scores[key] = score
    return SpiritDecision(
        spirit=scores['spirit'],
        voice=scores['voice'],
        account=scores['account'],
        lift=scores['lift'],
    )


@dataclass
class OpenRouterDecider:
    """Live OpenRouter decider with a small in-process cache."""

    api_key: str
    model: str = DEFAULT_MODEL
    api_url: str = DEFAULT_API_URL
    keep_min: float = DEFAULT_KEEP_MIN
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS
    _judge: JudgeFn | None = None
    _cache: dict[str, Decision | None] | None = None
    _spirit_cache: dict[str, SpiritDecision | None] | None = None

    def __post_init__(self) -> None:
        if self._cache is None:
            self._cache = {}
        if self._spirit_cache is None:
            self._spirit_cache = {}

    def decide(self, haystack: str, *, images: Sequence[str] | None = None) -> Decision | None:
        text = (haystack or '').strip()
        urls = [part['image_url']['url'] for part in image_parts(images)]
        if not text and not urls:
            return None
        clipped = text[: self.max_text_chars]
        image_tag = hashlib.sha256('\n'.join(urls).encode()).hexdigest() if urls else ''
        cache_key = hashlib.sha256(
            f'{self.model}|{PROMPT_VERSION}|{self.keep_min:.3f}|{clipped}|{image_tag}'.encode()
        ).hexdigest()
        assert self._cache is not None
        if cache_key in self._cache:
            return self._cache[cache_key]
        try:
            payload = (
                self._judge(clipped, urls)
                if self._judge
                else self._call_api(clipped, urls, _KEEP_QUESTION)
            )
            decision = decision_from_payload(payload, keep_min=self.keep_min)
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ):
            self._cache[cache_key] = None
            return None
        self._cache[cache_key] = decision
        return decision

    def decide_spirit(
        self, haystack: str, *, images: Sequence[str] | None = None
    ) -> SpiritDecision | None:
        """Score a non-core practice. A network or payload failure declines."""
        text = (haystack or '').strip()
        urls = [part['image_url']['url'] for part in image_parts(images)]
        if not text and not urls:
            return None
        clipped = text[: self.max_text_chars]
        image_tag = hashlib.sha256('\n'.join(urls).encode()).hexdigest() if urls else ''
        cache_key = hashlib.sha256(
            f'{self.model}|{SPIRIT_PROMPT_VERSION}|{clipped}|{image_tag}'.encode()
        ).hexdigest()
        assert self._spirit_cache is not None
        if cache_key in self._spirit_cache:
            return self._spirit_cache[cache_key]
        try:
            payload = (
                self._judge(clipped, urls)
                if self._judge
                else self._call_api(clipped, urls, _SPIRIT_QUESTIONS)
            )
            decision = spirit_from_payload(payload)
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ):
            self._spirit_cache[cache_key] = None
            return None
        self._spirit_cache[cache_key] = decision
        return decision

    def _call_api(
        self,
        text: str,
        images: Sequence[str],
        questions: dict[str, Any],
    ) -> dict[str, Any]:
        body = {
            'model': self.model,
            'state': decision_state(text, images),
            'questions': questions,
        }
        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(body).encode('utf-8'),
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'User-Agent': 'mutualist-bluesky-feed-decider/1.0',
            },
            method='POST',
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            payload = json.loads(response.read().decode('utf-8'))
        if not isinstance(payload, dict):
            raise ValueError('decision response is not an object')
        return payload


@dataclass(frozen=True)
class UnavailableDecider:
    """Enabled filter with no client. ``decide`` declines so callers fail closed."""

    def decide(self, haystack: str, *, images: Sequence[str] | None = None) -> Decision | None:
        del haystack, images
        return None

    def decide_spirit(
        self, haystack: str, *, images: Sequence[str] | None = None
    ) -> SpiritDecision | None:
        del haystack, images
        return None


@dataclass(frozen=True)
class FakeDecider:
    """Deterministic backend for unit tests (no network)."""

    keep: bool = True
    keep_substrings: tuple[str, ...] = ()
    reject_substrings: tuple[str, ...] = ()
    score: float = 0.91

    def decide(self, haystack: str, *, images: Sequence[str] | None = None) -> Decision | None:
        del images
        lowered = haystack.lower()
        for needle in self.reject_substrings:
            if needle.lower() in lowered:
                return Decision(False, 0.05)
        if self.keep_substrings:
            for needle in self.keep_substrings:
                if needle.lower() in lowered:
                    return Decision(True, self.score)
            return Decision(False, 0.05)
        return Decision(self.keep, self.score)


_client: DecisionBackend | None = None
_client_ready = False


def build_default_decider() -> DecisionBackend | None:
    if not decision_enabled():
        return None
    api_key = os.environ.get('OPENROUTER_API_KEY', '').strip()
    if not api_key:
        return UnavailableDecider()
    return OpenRouterDecider(
        api_key=api_key,
        model=os.environ.get('DECISION_MODEL', DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        api_url=os.environ.get('DECISION_API_URL', DEFAULT_API_URL).strip() or DEFAULT_API_URL,
        keep_min=_env_float('DECISION_KEEP_MIN', DEFAULT_KEEP_MIN),
        timeout_s=_env_float('DECISION_TIMEOUT_S', DEFAULT_TIMEOUT_S),
        max_text_chars=int(_env_float('DECISION_MAX_TEXT_CHARS', float(DEFAULT_MAX_TEXT_CHARS))),
    )


def get_default_decider() -> DecisionBackend | None:
    """Return the process decider, or None when the filter is disabled.

    Disabled (the test and eval default) leaves the regex floor in place.
    Enabled with a missing key returns a client that declines every post.
    """
    global _client, _client_ready
    if not decision_enabled():
        return None
    if not _client_ready:
        _client = build_default_decider()
        _client_ready = True
    return _client


def clear_decider_cache() -> None:
    """Test helper: drop the cached default client."""
    global _client, _client_ready
    _client = None
    _client_ready = False
