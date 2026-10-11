"""Quality rubric classifier (DeepSeek).

Not consulted during ingest. The feed's content filter is ``server.decider``.
This module scores four dimensions via ``deepseek-v4-flash`` for tests and for
offline rubric experiments:

- thematic_fit: Proudhonian mutualism / left-market mutualist alignment
- positive_valence: constructive / generative tone (hate/violence low)
- wow: humor, insight, craft, memorability
- solicit: personal fundraising / extractive money-ask (inverted gate)

Network calls are optional: when the classifier is disabled or no API key is
configured, this module declines (``None``) so the matcher keeps its regex-floor
drop reason (or drops solicit-shaped provisional keeps). Tests inject a
``ClassifierBackend``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

DEFAULT_MODEL = 'deepseek-v4-flash'
DEFAULT_API_URL = 'https://api.deepseek.com/v1/chat/completions'
# Legacy single-score default; still used as fallback composite min.
DEFAULT_THRESHOLD = 0.72
DEFAULT_THEMATIC_MIN = 0.70
DEFAULT_VALENCE_MIN = 0.55
DEFAULT_WOW_MIN = 0.45
DEFAULT_SOLICIT_MAX = 0.35
DEFAULT_COMPOSITE_MIN = 0.68
DEFAULT_TIMEOUT_S = 8.0
DEFAULT_MAX_TEXT_CHARS = 1200
RUBRIC_VERSION = 'quality-v1'

THEMATIC_WEIGHT = 0.45
VALENCE_WEIGHT = 0.25
WOW_WEIGHT = 0.30

SYSTEM_PROMPT = """\
You score Bluesky posts for a curated mutualism feed that prefers
high-quality, constructive, memorable posts — not personal fundraising,
hate, or violence.

Mutualism here means Proudhonian and closely related left-market currents:
reciprocity, possession / occupancy-and-use, mutual credit, free banking,
equitable commerce, agorism, free-market anti-capitalism, libertarian
socialism when framed that way, and Kropotkin. Sibling anarchist tendencies
(anarcho-communism, syndicalism, Bakuninist collectivism, Bookchin/social
ecology, Food Not Bombs, IWW, CrimethInc, etc.) are out of scope unless the
post is substantially about mutualist economics. Bare "mutual aid" without
mutualist-economics framing is out of scope.

Score each dimension in [0,1] using this shared scale:
- 0.0–0.29 clear miss / opposite of the dimension
- 0.30–0.49 weak / incidental
- 0.50–0.69 present but thin
- 0.70–0.84 solid, clear signal
- 0.85–1.0 strong / exemplary

Dimensions:
- thematic_fit: how substantially the post is about or celebrating mutualism
  / left-market mutualist economics (Proudhon, Tucker, Warren, mutual credit,
  occupancy-and-use, agorism, C4SS, Kevin Carson, Kropotkin) even if it never
  says "mutualism".
- positive_valence: constructive, generative, solidarity-forward tone
  (practical how-tos, joyful wins, witty demolition of hierarchy). Defiance
  and non-violent resistance can score mid/high. Low for doomspirals,
  factional pile-ons, cruelty-as-politics, hate, or celebration of violence.
  Grief paired with constructive praxis can still score mid/high.
- wow: humor, insight, craft, surprise, or memorable framing. Mid for
  competent on-theme reportage; low for boilerplate slogans and link dumps.
- solicit: personal / extractive money-ask intensity. HIGH for Venmo/CashApp/
  GoFundMe/PayPal-me asks, "I'm short on rent", "mutual aid request" for an
  individual. LOW for non-monetary mutualist praxis, project updates, or
  announcing a collective org fundraiser as news without a personal hardship
  pitch.

Also set keep=false for category errors regardless of scores:
- anarcho-capitalism / right-"libertarian" market fundamentalism
- chaos / entertainment "anarchy" with no political content
- crypto/Web3/DeFi "decentralized" jargon without mutualist values
- generic left or general-anarchist slogans with no mutualist signal
- sibling anarchist tendencies without mutualist economics
- authoritarian state-socialist celebration of centralized power
- hate speech, dehumanization, or celebration / advocacy of violence

Return JSON only with these exact keys:
{"keep": true|false,
 "thematic_fit": 0.0-1.0,
 "positive_valence": 0.0-1.0,
 "wow": 0.0-1.0,
 "solicit": 0.0-1.0,
 "rationale": "<short reason>"}

Do not rename keys (especially thematic_fit). Prefer precision when unsure.
"""


@dataclass(frozen=True)
class RubricThresholds:
    thematic_min: float = DEFAULT_THEMATIC_MIN
    valence_min: float = DEFAULT_VALENCE_MIN
    wow_min: float = DEFAULT_WOW_MIN
    solicit_max: float = DEFAULT_SOLICIT_MAX
    composite_min: float = DEFAULT_COMPOSITE_MIN

    def cache_token(self) -> str:
        return (
            f'{self.thematic_min:.3f}|{self.valence_min:.3f}|{self.wow_min:.3f}|'
            f'{self.solicit_max:.3f}|{self.composite_min:.3f}'
        )


@dataclass(frozen=True)
class GradeBreakdown:
    thematic_fit: float
    positive_valence: float
    wow: float
    solicit: float
    composite: float


@dataclass(frozen=True)
class ClassifierDecision:
    matched: bool
    reason: str
    score: float
    grades: GradeBreakdown | None = None


class ClassifierBackend(Protocol):
    def classify(
        self,
        haystack: str,
        *,
        term: str | None,
        has_event_cue: bool,
        has_local_venue: bool,
    ) -> ClassifierDecision | None: ...


JudgeFn = Callable[[str], dict[str, Any]]


def composite_score(thematic_fit: float, positive_valence: float, wow: float) -> float:
    return THEMATIC_WEIGHT * thematic_fit + VALENCE_WEIGHT * positive_valence + WOW_WEIGHT * wow


def admit_grades(grades: GradeBreakdown, thresholds: RubricThresholds) -> str | None:
    """Return a reject reason, or None when grades clear all gates."""
    if grades.thematic_fit < thresholds.thematic_min:
        return 'quality_reject:thematic_fit'
    if grades.solicit > thresholds.solicit_max:
        return 'quality_reject:solicit'
    if grades.positive_valence < thresholds.valence_min:
        return 'quality_reject:positive_valence'
    if grades.wow < thresholds.wow_min:
        return 'quality_reject:wow'
    if grades.composite < thresholds.composite_min:
        return 'quality_reject:composite'
    return None


def thresholds_from_env() -> RubricThresholds:
    # CLASSIFIER_THRESHOLD remains a supported alias for the composite floor.
    composite_default = _env_float('CLASSIFIER_THRESHOLD', DEFAULT_COMPOSITE_MIN)
    return RubricThresholds(
        thematic_min=_env_float('CLASSIFIER_THEMATIC_MIN', DEFAULT_THEMATIC_MIN),
        valence_min=_env_float('CLASSIFIER_VALENCE_MIN', DEFAULT_VALENCE_MIN),
        wow_min=_env_float('CLASSIFIER_WOW_MIN', DEFAULT_WOW_MIN),
        solicit_max=_env_float('CLASSIFIER_SOLICIT_MAX', DEFAULT_SOLICIT_MAX),
        composite_min=_env_float('CLASSIFIER_COMPOSITE_MIN', composite_default),
    )


@dataclass(frozen=True)
class ClassifierModel:
    """Compatibility shim / offline stub for tests that pass a local scorer."""

    version: str
    threshold: float
    weights: dict[str, float]

    def score(self, features: dict[str, float]) -> float:
        total = self.weights.get('bias', 0.0)
        for name, value in features.items():
            if name == 'bias':
                continue
            total += self.weights.get(name, 0.0) * value
        return total

    def classify(
        self,
        haystack: str,
        *,
        term: str | None,
        has_event_cue: bool,
        has_local_venue: bool,
    ) -> ClassifierDecision | None:
        del haystack, has_event_cue, has_local_venue
        # Offline stub: never keep unless tests override via FakeBackend.
        del term
        return None


@dataclass
class DeepSeekClassifier:
    """Live DeepSeek multi-dimensional quality scorer with a small cache."""

    api_key: str
    model: str = DEFAULT_MODEL
    api_url: str = DEFAULT_API_URL
    thresholds: RubricThresholds | None = None
    # Legacy alias mirrored into thresholds.composite_min when thresholds omitted.
    threshold: float = DEFAULT_COMPOSITE_MIN
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS
    _cache: dict[str, ClassifierDecision | None] | None = None
    _judge: JudgeFn | None = None

    def __post_init__(self) -> None:
        if self._cache is None:
            self._cache = {}
        if self.thresholds is None:
            self.thresholds = RubricThresholds(composite_min=self.threshold)
        else:
            self.threshold = self.thresholds.composite_min

    def classify(
        self,
        haystack: str,
        *,
        term: str | None,
        has_event_cue: bool,
        has_local_venue: bool,
    ) -> ClassifierDecision | None:
        text = (haystack or '').strip()
        if not text:
            return None
        assert self.thresholds is not None
        clipped = text[: self.max_text_chars]
        cache_key = hashlib.sha256(
            f'{self.model}|{RUBRIC_VERSION}|{self.thresholds.cache_token()}|{term}|'
            f'{int(has_event_cue)}|{int(has_local_venue)}|{clipped}'.encode()
        ).hexdigest()
        assert self._cache is not None
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            payload = self._judge(clipped) if self._judge else self._call_api(clipped, term=term)
        except urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError:
            self._cache[cache_key] = None
            return None

        decision = _decision_from_payload(payload, term=term, thresholds=self.thresholds)
        self._cache[cache_key] = decision
        return decision

    def _call_api(self, text: str, *, term: str | None) -> dict[str, Any]:
        user = {
            'text': text,
            'ambiguous_term': term,
            'instruction': (
                'Score thematic_fit, positive_valence, wow, and solicit for the '
                'mutualist feed quality rubric.'
            ),
        }
        body = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': json.dumps(user, ensure_ascii=False)},
            ],
            'response_format': {'type': 'json_object'},
            'temperature': 0.0,
            'max_tokens': 320,
            'thinking': {'type': 'disabled'},
        }
        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(body).encode('utf-8'),
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'User-Agent': 'mutualist-bluesky-feed-classifier/0.2',
            },
            method='POST',
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            raw = json.loads(response.read().decode('utf-8'))
        content = raw['choices'][0]['message']['content']
        return _parse_json_object(content)


@dataclass(frozen=True)
class FakeClassifier:
    """Deterministic backend for unit tests (no network)."""

    keep_terms: frozenset[str] = frozenset()
    keep_substrings: tuple[str, ...] = ()
    reject_substrings: tuple[str, ...] = ()
    score: float = 0.9
    solicit: float = 0.1
    thresholds: RubricThresholds | None = None

    def classify(
        self,
        haystack: str,
        *,
        term: str | None,
        has_event_cue: bool,
        has_local_venue: bool,
    ) -> ClassifierDecision | None:
        del has_event_cue, has_local_venue
        lowered = haystack.lower()
        for needle in self.reject_substrings:
            if needle.lower() in lowered:
                grades = _fake_grades(self.score, solicit=0.95)
                return ClassifierDecision(
                    False,
                    'quality_reject:solicit',
                    grades.composite,
                    grades,
                )
        thresholds = self.thresholds or RubricThresholds()
        if term and term.lower() in self.keep_terms:
            grades = _fake_grades(self.score, solicit=self.solicit)
            reject = admit_grades(grades, thresholds)
            if reject:
                return ClassifierDecision(False, reject, grades.composite, grades)
            return ClassifierDecision(
                True, f'classifier:ambiguous:{term.lower()}', grades.composite, grades
            )
        for needle in self.keep_substrings:
            if needle.lower() in lowered:
                grades = _fake_grades(self.score, solicit=self.solicit)
                reject = admit_grades(grades, thresholds)
                if reject:
                    return ClassifierDecision(False, reject, grades.composite, grades)
                label = f'ambiguous:{term.lower()}' if term else 'quality'
                return ClassifierDecision(True, f'classifier:{label}', grades.composite, grades)
        return None


def _fake_grades(score: float, *, solicit: float) -> GradeBreakdown:
    clamped = _clamp01(score)
    return GradeBreakdown(
        thematic_fit=clamped,
        positive_valence=clamped,
        wow=clamped,
        solicit=_clamp01(solicit),
        composite=composite_score(clamped, clamped, clamped),
    )


def _parse_json_object(content: str) -> dict[str, Any]:
    text = (content or '').strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError('classifier response is not a JSON object')
    return data


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _payload_float(payload: dict[str, Any], key: str, default: float) -> float:
    try:
        return _clamp01(float(payload.get(key, default)))
    except TypeError, ValueError:
        return _clamp01(default)


def _lookup_dim(
    payload: dict[str, Any],
    names: tuple[str, ...],
    *,
    default: float,
    prefix: str | None = None,
) -> float:
    """Read a dimension, accepting aliases and minor key typos."""
    for name in names:
        if name in payload:
            return _payload_float(payload, name, default)
    if prefix:
        for key in payload:
            if isinstance(key, str) and key.startswith(prefix) and key not in names:
                return _payload_float(payload, key, default)
    return _clamp01(default)


def grades_from_payload(payload: dict[str, Any]) -> GradeBreakdown:
    """Parse multi-dim grades; legacy ``score`` maps onto thematic/valence/wow."""
    legacy = _payload_float(payload, 'score', 0.0)
    multi_keys = ('thematic_fit', 'positive_valence', 'wow', 'solicit')
    alias_keys = ('thematic', 'theme', 'values_alignment', 'valence', 'positive')
    has_multi = any(key in payload for key in multi_keys) or any(
        isinstance(key, str) and (key.startswith('thematic') or key in alias_keys)
        for key in payload
    )
    if has_multi:
        thematic = _lookup_dim(
            payload,
            ('thematic_fit', 'thematic', 'theme', 'values_alignment', 'score'),
            default=legacy,
            prefix='thematic',
        )
        valence = _lookup_dim(
            payload,
            ('positive_valence', 'valence', 'positive', 'score'),
            default=legacy,
            prefix='positive',
        )
        wow = _lookup_dim(payload, ('wow', 'wow_factor', 'score'), default=legacy)
        solicit = _lookup_dim(
            payload,
            ('solicit', 'solicitation', 'money_ask', 'fundraising'),
            default=0.0,
            prefix='solicit',
        )
    else:
        thematic = legacy
        valence = legacy
        wow = legacy
        solicit = 0.0
    return GradeBreakdown(
        thematic_fit=thematic,
        positive_valence=valence,
        wow=wow,
        solicit=solicit,
        composite=composite_score(thematic, valence, wow),
    )


def _decision_from_payload(
    payload: dict[str, Any],
    *,
    term: str | None,
    thresholds: RubricThresholds,
) -> ClassifierDecision | None:
    grades = grades_from_payload(payload)
    keep = bool(payload.get('keep'))
    if term:
        label = f'ambiguous:{term}'
    else:
        label = 'quality'

    if not keep:
        reason = f'quality_reject:keep_false:{label}'
        return ClassifierDecision(False, reason, grades.composite, grades)

    reject = admit_grades(grades, thresholds)
    if reject:
        return ClassifierDecision(False, reject, grades.composite, grades)
    return ClassifierDecision(True, f'classifier:{label}', grades.composite, grades)


def classifier_enabled() -> bool:
    raw = os.environ.get('CLASSIFIER_ENABLED', 'true').strip().lower()
    return raw in {'1', 'true', 't', 'yes', 'y'}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def build_default_classifier() -> ClassifierBackend | None:
    if not classifier_enabled():
        return None
    api_key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
    if not api_key:
        return None
    thresholds = thresholds_from_env()
    return DeepSeekClassifier(
        api_key=api_key,
        model=os.environ.get('CLASSIFIER_MODEL', DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        api_url=os.environ.get('DEEPSEEK_API_URL', DEFAULT_API_URL).strip() or DEFAULT_API_URL,
        thresholds=thresholds,
        threshold=thresholds.composite_min,
        timeout_s=_env_float('CLASSIFIER_TIMEOUT_S', DEFAULT_TIMEOUT_S),
        max_text_chars=int(_env_float('CLASSIFIER_MAX_TEXT_CHARS', float(DEFAULT_MAX_TEXT_CHARS))),
    )


@lru_cache(maxsize=1)
def _cached_default_classifier() -> ClassifierBackend | None:
    return build_default_classifier()


def clear_model_cache() -> None:
    """Test helper: drop the cached default classifier."""
    _cached_default_classifier.cache_clear()


def load_model(path: Any = None) -> ClassifierModel:
    """Back-compat helper used by older tests; returns a no-op stub."""
    del path
    return ClassifierModel(version='deepseek_v4_flash', threshold=DEFAULT_COMPOSITE_MIN, weights={})


def extract_features(
    haystack: str,
    *,
    term: str | None,
    has_event_cue: bool,
    has_local_venue: bool,
) -> dict[str, float]:
    """Lightweight feature view for debugging / offline scripts."""
    return {
        'has_event_cue': 1.0 if has_event_cue else 0.0,
        'has_local_venue': 1.0 if has_local_venue else 0.0,
        'has_ambiguous': 1.0 if term else 0.0,
        'text_len': float(min(len(haystack or ''), 2000)),
    }


def classify_candidate(
    haystack: str,
    *,
    term: str | None,
    has_event_cue: bool,
    has_local_venue: bool,
    classifier: ClassifierBackend | None = None,
    model: ClassifierBackend | None = None,
    allow_without_term: bool = False,
) -> ClassifierDecision | None:
    """Score a candidate; return a decision (keep or graded reject) or None.

    ``None`` means unscored (disabled / error / no backend). Graded rejects
    return ``ClassifierDecision(matched=False, ...)``.

    Production ambiguous leftovers always pass a latchable ``term``. Solicit
    rechecks on provisional keeps set ``allow_without_term=True``.
    """
    backend = classifier if classifier is not None else model
    if backend is None:
        if not term and not allow_without_term:
            return None
        backend = _cached_default_classifier()
    if backend is None:
        return None

    return backend.classify(
        haystack,
        term=term,
        has_event_cue=has_event_cue,
        has_local_venue=has_local_venue,
    )
