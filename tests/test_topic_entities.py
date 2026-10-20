"""Topic-entity disambiguation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from server.matcher import match_post
from server.topic_entities import default_topic_entities, load_topic_entities


def test_default_topic_entities_longer_surface_wins() -> None:
    entities = default_topic_entities()
    hit = entities.lookup('Rewatching Sons of Anarchy season 3 tonight.')
    assert hit is not None
    assert hit.scope == 'out_of_scope'
    assert hit.entity_id == 'sons_of_anarchy'


def test_match_post_out_of_scope_and_in_scope_entities() -> None:
    other = match_post('Who still plays Anarchy Online in 2026?')
    assert other.matched is False
    assert other.reason in {'entity_out_of_scope:anarchy_online', 'hard_negative'}

    sibling = match_post('Black Rose Anarchist Federation published a new primer.')
    assert sibling.matched is False
    assert sibling.reason == 'entity_out_of_scope:black_rose_anarchist_federation'

    in_scope = match_post('New C4SS essay on left-wing market anarchism.')
    assert in_scope.matched is True
    assert in_scope.reason in {'entity_in_scope:c4ss', 'strong_positive'}


def test_load_topic_entities_rejects_unknown_scope(tmp_path: Path) -> None:
    path = tmp_path / 'bad.json'
    path.write_text(
        '{"entities":[{"id":"x","scope":"mars","surfaces":["x"]}]}',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='unknown scope'):
        load_topic_entities(path)
