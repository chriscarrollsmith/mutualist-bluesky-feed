"""Unit tests for allowlist file loading and DID file generation helpers."""

from __future__ import annotations

from pathlib import Path

from scripts.resolve_allowlist_dids import build_did_file, list_kind_from_handles_path
from server.allowlists import (
    load_allowlist_dids,
    load_allowlist_handles,
    load_blocklist_dids,
    load_blocklist_handles,
    load_list_file,
)


def test_load_list_file_skips_comments_and_blanks(tmp_path: Path) -> None:
    path = tmp_path / 'list.txt'
    path.write_text('# comment\n\nalpha\nBeta\n', encoding='utf-8')
    assert load_list_file(path) == ['alpha', 'Beta']


def test_production_allowlists_are_aligned() -> None:
    handles = load_allowlist_handles()
    dids = load_allowlist_dids()
    # Empty is valid until mutualist voices are curated; keep DID/handle counts aligned.
    assert all(h == h.lower() for h in handles)
    assert all(d.startswith('did:') for d in dids)
    assert len(dids) == len(handles)


def test_production_blocklists_are_aligned() -> None:
    handles = load_blocklist_handles()
    dids = load_blocklist_dids()
    assert all(h == h.lower() for h in handles)
    assert all(d.startswith('did:') for d in dids)
    assert len(dids) == len(handles)
    assert 'bypophoenix.bsky.social' in handles
    assert 'anarchist-ai.bsky.social' in handles


def test_build_did_file_includes_handle_provenance() -> None:
    text = build_did_file(
        [
            ('example.bsky.social', 'did:plc:exampleauthor'),
            ('other.example', 'did:plc:exampleother'),
        ]
    )
    assert '# example.bsky.social' in text
    assert 'did:plc:exampleauthor' in text
    assert 'resolve_allowlist_dids.py' in text
    blocked = build_did_file(
        [('bypophoenix.bsky.social', 'did:plc:vdykwvsuhnim6beywhcqje7r')],
        kind='blocklist',
    )
    assert '# bypophoenix.bsky.social' in blocked
    assert 'blocklist_handles.txt' in blocked
    assert list_kind_from_handles_path(Path('data/blocklist_handles.txt')) == 'blocklist'
