"""Unit tests for decision-earned soft author priors."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from server.author_priors import author_has_soft_prior, record_decision
from server.database import AuthorDecision, Post, SubscriptionState, db, utc_now


@pytest.fixture()
def isolated_db() -> Any:
    db.connect(reuse_if_open=True)
    db.drop_tables([Post, SubscriptionState, AuthorDecision], safe=True)
    db.create_tables([Post, SubscriptionState, AuthorDecision])
    yield
    AuthorDecision.delete().execute()
    Post.delete().execute()


def test_three_keeps_earn_soft_prior(isolated_db: None) -> None:
    from server import config

    did = 'did:plc:priorearnertest00000000001'
    assert not author_has_soft_prior(did)
    for _ in range(config.SOFT_PRIOR_MIN_KEEPS - 1):
        record_decision(did, kept=True)
        assert not author_has_soft_prior(did)
    record_decision(did, kept=True)
    assert author_has_soft_prior(did)


def test_rejection_blocks_soft_prior(isolated_db: None) -> None:
    from server import config

    did = 'did:plc:priorrejecttest000000000001'
    for _ in range(config.SOFT_PRIOR_MIN_KEEPS):
        record_decision(did, kept=True)
    assert author_has_soft_prior(did)
    record_decision(did, kept=False)
    assert not author_has_soft_prior(did)


def test_soft_prior_ignores_decisions_outside_window(isolated_db: None) -> None:
    from server import config

    did = 'did:plc:priorexpiredtest0000000001'
    for _ in range(config.SOFT_PRIOR_MIN_KEEPS):
        record_decision(did, kept=True)
    assert author_has_soft_prior(did)

    stale = utc_now() - timedelta(days=config.SOFT_PRIOR_WINDOW_DAYS + 1)
    AuthorDecision.update(decided_at=stale).where(AuthorDecision.author_did == did).execute()
    assert not author_has_soft_prior(did)

    record_decision(did, kept=False)
    for row in AuthorDecision.select().where(AuthorDecision.author_did == did):
        if not row.kept:
            row.decided_at = stale
            row.save()
    for _ in range(config.SOFT_PRIOR_MIN_KEEPS):
        record_decision(did, kept=True)
    assert author_has_soft_prior(did)
