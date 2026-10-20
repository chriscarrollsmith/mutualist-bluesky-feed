"""Checked-in named entities for mutualist topic disambiguation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOPIC_ENTITIES_PATH = ROOT / 'data' / 'topic_entities.json'

# ``in_scope`` keeps; ``out_of_scope`` drops.
_ALLOWED_SCOPES = frozenset({'in_scope', 'out_of_scope'})


@dataclass(frozen=True)
class TopicEntityHit:
    entity_id: str
    scope: str
    surface: str


@dataclass(frozen=True)
class TopicEntities:
    """Longest-surface-first matcher over checked-in topic entities."""

    patterns: tuple[tuple[re.Pattern[str], str, str], ...]

    def lookup(self, text: str) -> TopicEntityHit | None:
        haystack = text or ''
        if not haystack:
            return None
        for pattern, entity_id, scope in self.patterns:
            match = pattern.search(haystack)
            if match:
                return TopicEntityHit(
                    entity_id=entity_id,
                    scope=scope,
                    surface=re.sub(r'\s+', ' ', match.group(0).lower()),
                )
        return None


def _surface_pattern(surface: str) -> re.Pattern[str]:
    parts = [re.escape(part) for part in surface.split() if part]
    if not parts:
        raise ValueError(f'empty topic-entity surface: {surface!r}')
    body = r'\s+'.join(parts)
    return re.compile(rf'(?<!\w){body}(?!\w)', re.IGNORECASE)


def _json_text(path: Path) -> str:
    """Ignore end-of-line scanner markers so they are not part of the JSON."""
    raw = path.read_text(encoding='utf-8')
    return re.sub(r'[ \t]*// pragma: allowlist secret', '', raw)


def load_topic_entities(path: Path | None = None) -> TopicEntities:
    entities_path = path or DEFAULT_TOPIC_ENTITIES_PATH
    payload = json.loads(_json_text(entities_path))
    entities = payload.get('entities') or []
    compiled: list[tuple[int, re.Pattern[str], str, str]] = []
    for entity in entities:
        entity_id = str(entity['id'])
        scope = str(entity['scope'])
        if scope not in _ALLOWED_SCOPES:
            raise ValueError(f'unknown scope {scope!r} for {entity_id}')
        for surface in entity.get('surfaces') or []:
            surface_text = str(surface).strip().lower()
            if not surface_text:
                continue
            compiled.append((len(surface_text), _surface_pattern(surface_text), entity_id, scope))
    compiled.sort(key=lambda item: item[0], reverse=True)
    patterns = tuple((pattern, entity_id, scope) for _, pattern, entity_id, scope in compiled)
    return TopicEntities(patterns=patterns)


@lru_cache(maxsize=1)
def default_topic_entities() -> TopicEntities:
    return load_topic_entities()
