"""Indexer path tests: allowlisted DIDs and soft author priors."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from server.database import AuthorDecision, Post, SubscriptionState, db


@pytest.fixture()
def isolated_db() -> Any:
    """Rebuild post tables so each test starts empty."""
    db.connect(reuse_if_open=True)
    db.drop_tables([Post, SubscriptionState, AuthorDecision], safe=True)
    db.create_tables([Post, SubscriptionState, AuthorDecision])
    yield
    AuthorDecision.delete().execute()
    Post.delete().execute()


def _create_event(*, author: str, text: str, uri_key: str = 'abc') -> dict[str, Any]:
    now = datetime.now(UTC).isoformat().replace('+00:00', 'Z')
    return {
        'operation': 'create',
        'uri': f'at://{author}/app.bsky.feed.post/{uri_key}',
        'cid': 'bafytestcid',
        'author': author,
        'record': {
            '$type': 'app.bsky.feed.post',
            'text': text,
            'createdAt': now,
        },
    }


def test_handle_event_indexes_allowlisted_did_without_topic_phrase(
    isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from server import config
    from server.indexer import handle_event

    author = 'did:plc:allowlisttest00000000000001'
    monkeypatch.setattr(config, 'ALLOWLIST_DIDS', {author})
    event = _create_event(author=author, text='Our newsletter is out for subscribers.')
    handle_event(event)

    row = Post.get(Post.uri == event['uri'])
    assert row.author_did == author
    assert row.match_reason == 'allowlist_did'


def test_handle_event_skips_unknown_author_without_topic_phrase(isolated_db: None) -> None:
    from server.indexer import handle_event

    event = _create_event(
        author='did:plc:notallowlisted000000000000',
        text='Our newsletter is out for subscribers.',
        uri_key='xyz',
    )
    handle_event(event)
    assert Post.select().where(Post.uri == event['uri']).count() == 0


def test_regex_keep_does_not_earn_soft_prior(isolated_db: None) -> None:
    from server.author_priors import author_has_soft_prior
    from server.indexer import handle_event

    author = 'did:plc:indexersoftprior0000000001'
    event = _create_event(author=author, text='Mutual credit clearing update.')
    handle_event(event)
    row = Post.get(Post.uri == event['uri'])
    assert row.match_reason == 'strong_positive'
    assert not author_has_soft_prior(author)
    assert AuthorDecision.select().count() == 0


def test_decider_keeps_earn_soft_prior(isolated_db: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from server import config
    from server.author_priors import author_has_soft_prior
    from server.decider import FakeDecider
    from server.indexer import handle_event

    author = 'did:plc:indexersoftprior0000000001'
    assert author not in config.ALLOWLIST_DIDS
    monkeypatch.setattr(
        'server.matcher.get_default_decider',
        lambda: FakeDecider(keep=True, score=0.9),
    )

    for i in range(config.SOFT_PRIOR_MIN_KEEPS):
        handle_event(
            _create_event(
                author=author,
                text=f'Mutual credit clearing update {i}.',
                uri_key=f'strong{i}',
            )
        )
    assert author_has_soft_prior(author)
    assert AuthorDecision.select().where(AuthorDecision.kept == 1).count() == (
        config.SOFT_PRIOR_MIN_KEEPS
    )

    bare = _create_event(
        author=author,
        text='Direct action works when neighbors show up.',
        uri_key='bare',
    )
    handle_event(bare)
    row = Post.get(Post.uri == bare['uri'])
    assert (row.match_reason or '').startswith('decision_keep:soft_prior_ambiguous:direct action:')


def test_decider_rejection_blocks_soft_prior(
    isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from server import config
    from server.author_priors import author_has_soft_prior
    from server.decider import FakeDecider
    from server.indexer import handle_event

    author = 'did:plc:indexerrejectprior000000001'
    monkeypatch.setattr(
        'server.matcher.get_default_decider',
        lambda: FakeDecider(reject_substrings=('socialist',), keep=True, score=0.9),
    )
    for i in range(config.SOFT_PRIOR_MIN_KEEPS):
        handle_event(
            _create_event(
                author=author,
                text=f'Mutual credit clearing update {i}.',
                uri_key=f'strong{i}',
            )
        )
    assert author_has_soft_prior(author)

    hostile = _create_event(
        author=author,
        text='Proudhonian mutualism is just socialist chaos.',
        uri_key='hostile',
    )
    handle_event(hostile)
    assert Post.select().where(Post.uri == hostile['uri']).count() == 0
    assert not author_has_soft_prior(author)


def test_handle_event_skips_blocklisted_did_with_strong_text(isolated_db: None) -> None:
    from server import config
    from server.indexer import handle_event

    assert config.BLOCKLIST_DIDS, 'production DID blocklist must be populated'
    author = sorted(config.BLOCKLIST_DIDS)[0]
    event = _create_event(author=author, text='Neighborhood mutual credit clearing tonight.')
    handle_event(event)
    assert Post.select().where(Post.uri == event['uri']).count() == 0


def test_purge_blocklisted_posts_deletes_indexed_rows(isolated_db: None) -> None:
    from server import config
    from server.indexer import purge_blocklisted_posts

    assert config.BLOCKLIST_DIDS
    author = sorted(config.BLOCKLIST_DIDS)[0]
    Post.create(
        uri=f'at://{author}/app.bsky.feed.post/old',
        cid='bafytestcid',
        author_did=author,
        match_reason='strong_positive',
    )
    other = 'did:plc:notblocklisted0000000000001'
    Post.create(
        uri=f'at://{other}/app.bsky.feed.post/keep',
        cid='bafytestcid2',
        author_did=other,
        match_reason='strong_positive',
    )
    assert purge_blocklisted_posts() == 1
    assert Post.select().where(Post.author_did == author).count() == 0
    assert Post.select().where(Post.author_did == other).count() == 1


def test_handle_event_archived_uses_event_time_not_wall_clock(
    isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch-up must not drop posts that were fresh at firehose time_us."""
    from datetime import timedelta

    from server import config
    from server.indexer import handle_event

    author = 'did:plc:allowlisttest00000000000001'
    monkeypatch.setattr(config, 'ALLOWLIST_DIDS', {author})
    # Created ~30h before wall clock, but event time_us is only 2h after createdAt.
    created = datetime.now(UTC) - timedelta(hours=30)
    event_time = created + timedelta(hours=2)
    event = {
        'operation': 'create',
        'uri': f'at://{author}/app.bsky.feed.post/catchup1',
        'cid': 'bafytestcid',
        'author': author,
        'time_us': int(event_time.timestamp() * 1_000_000),
        'record': {
            '$type': 'app.bsky.feed.post',
            'text': 'Catch-up bulletin for subscribers.',
            'createdAt': created.isoformat().replace('+00:00', 'Z'),
        },
    }
    handle_event(event)
    assert Post.select().where(Post.uri == event['uri']).count() == 1

    # Same createdAt with a live-tail time_us (wall-aligned) is archived.
    live = dict(event)
    live['uri'] = f'at://{author}/app.bsky.feed.post/catchup2'
    live['time_us'] = int(datetime.now(UTC).timestamp() * 1_000_000)
    handle_event(live)
    assert Post.select().where(Post.uri == live['uri']).count() == 0
