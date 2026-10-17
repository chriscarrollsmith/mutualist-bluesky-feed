"""Opt-in live checks for the Perplexity decider.

Run with ``uv run pytest -m live``. The default suite skips this module.
"""

from __future__ import annotations

import base64
import json
import os
import struct
import urllib.request
import zlib
from pathlib import Path

import pytest
from dotenv import load_dotenv
from server.decider import _KEEP_QUESTION, OpenRouterDecider, decision_state
from server.matcher import match_post

pytestmark = pytest.mark.live

_MUTUALIST = (
    'Neighborhood mutual credit clearing this month: we price shared tools at cost, '
    'occupancy-and-use rather than landlord title. Proudhonian mutualism in practice.'
)
_HOSTILE = (
    'Proudhonian mutualism is a socialist plot. Breitbart is right about the '
    'dangers of a massive socialist welfare state — vote them out.'
)


def _live_decider() -> OpenRouterDecider:
    load_dotenv(Path(__file__).resolve().parents[1] / '.env')
    api_key = os.environ.get('OPENROUTER_API_KEY', '').strip()
    if not api_key:
        pytest.skip('OPENROUTER_API_KEY is not set')
    return OpenRouterDecider(api_key=api_key, timeout_s=30.0)


def test_live_decider_keeps_mutualist_post_and_rejects_hostile_one() -> None:
    decider = _live_decider()

    kept = decider.decide(_MUTUALIST)
    assert kept is not None, 'mutualist post was unscored'
    assert kept.keep, f'expected keep, noul={kept.score:.3f}'

    rejected = decider.decide(_HOSTILE)
    assert rejected is not None, 'hostile post was unscored'
    assert not rejected.keep, f'expected reject, noul={rejected.score:.3f}'


def test_live_match_post_filters_regex_candidates() -> None:
    decider = _live_decider()

    keep = match_post(_MUTUALIST, decider=decider)
    assert keep.matched, f'expected decision_keep, reason={keep.reason}'
    assert keep.reason.startswith('decision_keep:')

    drop = match_post(_HOSTILE, decider=decider)
    assert not drop.matched, f'expected decision_reject, reason={drop.reason}'
    assert drop.reason.startswith('decision_reject:')


def _png_data_url(rgb: tuple[int, int, int]) -> str:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack('>I', len(data))
            + tag
            + data
            + struct.pack('>I', zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b''.join(b'\x00' + bytes(rgb) * 32 for _ in range(32))
    ihdr = struct.pack('>IIBBBBB', 32, 32, 8, 2, 0, 0, 0)
    blob = (
        b'\x89PNG\r\n\x1a\n'
        + chunk(b'IHDR', ihdr)
        + chunk(b'IDAT', zlib.compress(raw, 9))
        + chunk(b'IEND', b'')
    )
    return 'data:image/png;base64,' + base64.b64encode(blob).decode('ascii')


def test_live_decider_reads_attached_image() -> None:
    decider = _live_decider()
    red = _png_data_url((220, 20, 20))
    blue = _png_data_url((20, 40, 220))
    text = 'Bluesky post with no written caption.'

    decision = decider.decide(text, images=[red])
    assert decision is not None, 'image classification was unscored'

    def color_of(image: str) -> str:
        body = {
            'model': decider.model,
            'state': decision_state(text, [image]),
            'questions': {
                **_KEEP_QUESTION,
                'dominant_color': {
                    'type': 'choice',
                    'instructions': 'What is the dominant color of the attached image?',
                    'criteria': {
                        'red': 'mostly red',
                        'blue': 'mostly blue',
                        'other': 'neither',
                    },
                },
            },
        }
        request = urllib.request.Request(
            decider.api_url,
            data=json.dumps(body).encode(),
            headers={
                'Authorization': f'Bearer {decider.api_key}',
                'Content-Type': 'application/json',
            },
            method='POST',
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode())
        choice = payload['answers']['dominant_color']['choice']
        assert isinstance(choice, str)
        return choice

    assert color_of(red) == 'red'
    assert color_of(blue) == 'blue'
