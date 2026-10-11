"""Soft author priors earned from decision-model keep/reject outcomes.

A prior unlocks bare ambiguous terms so they reach the content filter. It does
not skip that filter, and regex matches do not earn one. An author has a prior
after at least ``SOFT_PRIOR_MIN_KEEPS`` keeps and zero rejections in the last
``SOFT_PRIOR_WINDOW_DAYS`` days.
"""

from __future__ import annotations

from datetime import timedelta

from server import config
from server.database import AuthorDecision, db, utc_now


def author_has_soft_prior(author_did: str | None) -> bool:
    """True when recent decider keeps are enough and unrejected."""
    if not author_did:
        return False
    keeps, rejects = _window_counts(author_did)
    return keeps >= config.SOFT_PRIOR_MIN_KEEPS and rejects == 0


def record_decision(author_did: str | None, *, kept: bool) -> None:
    """Append one decider outcome for ``author_did``."""
    if not author_did:
        return
    with db.atomic():
        AuthorDecision.create(author_did=author_did, kept=kept, decided_at=utc_now())


def _window_counts(author_did: str) -> tuple[int, int]:
    cutoff = utc_now() - timedelta(days=config.SOFT_PRIOR_WINDOW_DAYS)
    keeps = 0
    rejects = 0
    query = AuthorDecision.select(AuthorDecision.kept).where(
        (AuthorDecision.author_did == author_did) & (AuthorDecision.decided_at >= cutoff)
    )
    for row in query:
        if row.kept:
            keeps += 1
        else:
            rejects += 1
    return keeps, rejects
