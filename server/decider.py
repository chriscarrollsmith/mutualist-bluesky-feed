"""Perplexity decision-model content filter (OpenRouter Decisions API).

The regex floor in ``server.matcher`` proposes candidates. This module
keep/rejects those candidates with ``perplexity/pplx-decider-v1.1-27b``.
Application code thresholds the returned yes-probability (``noul``). Network
errors and unusable payloads decline (``None``) so the matcher can fail closed.

``get_default_decider`` is off unless ``DECISION_ENABLED`` is set, so offline
eval and unit tests stay on the regex floor. When the flag is on and no API
key is configured, the client still declines and the matcher drops.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

DEFAULT_MODEL = 'perplexity/pplx-decider-v1.1-27b'
DEFAULT_API_URL = 'https://openrouter.ai/api/alpha/decisions'
DEFAULT_KEEP_MIN = 0.5
DEFAULT_TIMEOUT_S = 8.0
DEFAULT_MAX_TEXT_CHARS = 2000
PROMPT_VERSION = 'mutualism-keep-v1'

KEEP_PREFIX = 'decision_keep:'
REJECT_PREFIX = 'decision_reject:'
UNSCORED_PREFIX = 'decision_unscored:'

_KEEP_QUESTION: dict[str, Any] = {
    'keep': {
        'type': 'noul',
        'instructions': (
            'Should this Bluesky post be kept in a curated feed of Proudhonian '
            'mutualism and closely related left-market currents (reciprocity, '
            'occupancy-and-use, mutual credit, free banking, equitable commerce, '
            'agorism, free-market anti-capitalism, libertarian socialism when '
            'framed that way, and Kropotkin)?'
        ),
        'criteria': {
            'true': (
                'The post substantially celebrates or practices in-scope mutualist '
                'economics, in a constructive tone. Defiance and non-violent '
                'resistance are fine. Collective org news without a personal '
                'hardship pitch is fine.'
            ),
            'false': (
                'Off-topic, hostile, unclear, or out of scope. Includes right-wing '
                'or reactionary framing, scare coverage of socialism or anarchism, '
                'anarcho-capitalism, chaos or entertainment uses of anarchy, crypto '
                'or Web3 jargon, sibling anarchist tendencies without mutualist '
                'economics, bare mutual aid, personal fundraising, hate, or '
                'celebration of violence.'
            ),
        },
    }
}


@dataclass(frozen=True)
class Decision:
    keep: bool
    score: float


class DecisionBackend(Protocol):
    def decide(self, haystack: str) -> Decision | None: ...


JudgeFn = Callable[[str], dict[str, Any]]


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

    def __post_init__(self) -> None:
        if self._cache is None:
            self._cache = {}

    def decide(self, haystack: str) -> Decision | None:
        text = (haystack or '').strip()
        if not text:
            return None
        clipped = text[: self.max_text_chars]
        cache_key = hashlib.sha256(
            f'{self.model}|{PROMPT_VERSION}|{self.keep_min:.3f}|{clipped}'.encode()
        ).hexdigest()
        assert self._cache is not None
        if cache_key in self._cache:
            return self._cache[cache_key]
        try:
            payload = self._judge(clipped) if self._judge else self._call_api(clipped)
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

    def _call_api(self, text: str) -> dict[str, Any]:
        body = {
            'model': self.model,
            'state': {'post': text},
            'questions': _KEEP_QUESTION,
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

    def decide(self, haystack: str) -> Decision | None:
        del haystack
        return None


@dataclass(frozen=True)
class FakeDecider:
    """Deterministic backend for unit tests (no network)."""

    keep: bool = True
    keep_substrings: tuple[str, ...] = ()
    reject_substrings: tuple[str, ...] = ()
    score: float = 0.91

    def decide(self, haystack: str) -> Decision | None:
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
