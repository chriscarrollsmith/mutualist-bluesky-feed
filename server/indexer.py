"""Apply matcher rules to Jetstream post events and update the DB."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import peewee

from server import config
from server.author_priors import author_has_soft_prior, record_decision
from server.database import Post, db, prune_old_posts, utc_now
from server.decider import KEEP_PREFIX, REJECT_PREFIX
from server.logger import logger
from server.matcher import extract_alt_text, match_post

_last_prune = utc_now()


def _parse_created_at(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        # Bluesky timestamps are ISO-8601; normalize Z.
        normalized = value.replace('Z', '+00:00')
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is not None:
            dt = dt.astimezone(UTC).replace(tzinfo=None)
        return dt
    except ValueError:
        return None


def _is_archived(created_at: datetime | None, *, reference: datetime | None = None) -> bool:
    """True when created_at is more than one day before ``reference``.

    ``reference`` defaults to wall clock. During Jetstream catch-up, pass the
    event's ``time_us`` so a lagged consumer does not drop the backlog as
    soon as wall-clock age crosses 24h.
    """
    if not created_at:
        return False
    ref = reference if reference is not None else utc_now()
    return ref - created_at > timedelta(days=1)


def _event_reference_time(event: dict[str, Any]) -> datetime | None:
    time_us = event.get('time_us')
    if not isinstance(time_us, int) or time_us <= 0:
        return None
    return datetime.fromtimestamp(time_us / 1_000_000, tz=UTC).replace(tzinfo=None)


def _maybe_prune() -> None:
    global _last_prune
    now = utc_now()
    if now - _last_prune < timedelta(minutes=30):
        return
    deleted = prune_old_posts()
    _last_prune = now
    if deleted:
        logger.info('pruned %s old posts', deleted)


def _is_muted(text: str, alt_text: str) -> bool:
    if not config.MUTED_KEYWORDS:
        return False
    haystack = f'{text} {alt_text}'.lower()
    return any(keyword in haystack for keyword in config.MUTED_KEYWORDS)


def handle_engagement_event(event: dict[str, Any]) -> None:
    """Increment like/repost counts when engagement targets an indexed post."""
    operation = event.get('operation')
    subject_uri = event.get('subject_uri')
    kind = event.get('engagement')
    if operation != 'create' or not subject_uri or kind not in {'like', 'repost'}:
        return

    field = Post.like_count if kind == 'like' else Post.repost_count
    updated = Post.update({field: field + 1}).where(Post.uri == subject_uri).execute()
    if updated:
        logger.debug('engagement %s +1 for %s', kind, subject_uri)


def handle_event(event: dict[str, Any]) -> None:
    if event.get('engagement'):
        handle_engagement_event(event)
        return

    operation = event.get('operation')
    uri = event.get('uri')
    if not uri or not operation:
        return

    if operation == 'delete':
        deleted = Post.delete().where(Post.uri == uri).execute()
        if deleted:
            logger.debug('deleted %s', uri)
        return

    if operation != 'create':
        return

    record = event.get('record') or {}
    if config.IGNORE_REPLY_POSTS and record.get('reply'):
        return

    created_at = _parse_created_at(record.get('createdAt'))
    if config.IGNORE_ARCHIVED_POSTS and _is_archived(
        created_at, reference=_event_reference_time(event)
    ):
        return

    text = record.get('text') or ''
    embed = record.get('embed')
    alt_text = extract_alt_text(embed)
    if _is_muted(text, alt_text):
        logger.debug('muted %s', uri)
        return

    langs = record.get('langs') or []
    if not isinstance(langs, list):
        langs = []
    author_did = event.get('author')
    soft_prior_dids = {author_did} if author_has_soft_prior(author_did) else set()
    result = match_post(
        text,
        alt_text=alt_text,
        langs=langs,
        author_did=author_did,
        allowlist_dids=config.ALLOWLIST_DIDS,
        allowlist_handles=config.ALLOWLIST_HANDLES,
        blocklist_dids=config.BLOCKLIST_DIDS,
        blocklist_handles=config.BLOCKLIST_HANDLES,
        soft_prior_dids=soft_prior_dids,
        embed=embed,
        use_default_decider=True,
    )
    if result.reason.startswith(REJECT_PREFIX):
        record_decision(author_did, kept=False)
    if not result.matched:
        return

    reply = record.get('reply') or {}
    parent = (reply.get('parent') or {}).get('uri')
    root = (reply.get('root') or {}).get('uri')

    try:
        with db.atomic():
            Post.create(
                uri=uri,
                cid=event.get('cid') or '',
                author_did=author_did,
                reply_parent=parent,
                reply_root=root,
                match_reason=result.reason,
                created_at=created_at,
                like_count=0,
                repost_count=0,
            )
    except peewee.IntegrityError:
        return
    if result.reason.startswith(KEEP_PREFIX):
        record_decision(author_did, kept=True)

    logger.info('indexed %s reason=%s', uri, result.reason)
    _maybe_prune()


def purge_blocklisted_posts() -> int:
    """Delete indexed posts whose author is on the production blocklist."""
    if not config.BLOCKLIST_DIDS:
        return 0
    deleted = Post.delete().where(Post.author_did << list(config.BLOCKLIST_DIDS)).execute()
    if deleted:
        logger.info('purged %s posts from blocklisted authors', deleted)
    return deleted
