from datetime import UTC, datetime, timedelta
from pathlib import Path

import peewee

from server import config


def utc_now() -> datetime:
    """Naive UTC timestamp compatible with existing SQLite rows."""
    return datetime.now(UTC).replace(tzinfo=None)


_db_path = Path(config.DATABASE_PATH)
_is_memory = str(_db_path) == ':memory:'
if not _is_memory:
    _db_path.parent.mkdir(parents=True, exist_ok=True)

if _is_memory:
    # Plain ":memory:" is per-connection. FastAPI runs sync routes (e.g.
    # /healthz) on worker threads, so share one in-memory DB across them.
    # Keep at least one connection open (see connect() below) or SQLite
    # discards the shared database when the last connection closes.
    db = peewee.SqliteDatabase(
        'file:capital_region_feed?mode=memory&cache=shared',
        uri=True,
    )
else:
    db = peewee.SqliteDatabase(
        str(_db_path),
        pragmas={
            'journal_mode': 'wal',
            'synchronous': 'normal',
        },
    )


class BaseModel(peewee.Model):
    class Meta:
        database = db


class Post(BaseModel):
    uri = peewee.CharField(unique=True, index=True)
    cid = peewee.CharField()
    author_did = peewee.CharField(null=True, index=True)
    reply_parent = peewee.CharField(null=True, default=None)
    reply_root = peewee.CharField(null=True, default=None)
    match_reason = peewee.CharField(null=True, default=None)
    created_at = peewee.DateTimeField(null=True, index=True)
    indexed_at = peewee.DateTimeField(default=utc_now, index=True)
    like_count = peewee.IntegerField(default=0)
    repost_count = peewee.IntegerField(default=0)


class SubscriptionState(BaseModel):
    service = peewee.CharField(unique=True)
    cursor = peewee.BigIntegerField()


class AuthorLocalStats(BaseModel):
    """Per-author counter rows stored next to the feed database."""

    author_did = peewee.CharField(primary_key=True)
    strong_match_count = peewee.IntegerField(default=0)
    last_strong_at = peewee.DateTimeField(null=True, index=True)


class AuthorDecision(BaseModel):
    """Decider keep/reject events used for soft author priors."""

    author_did = peewee.CharField(index=True)
    kept = peewee.BooleanField()
    decided_at = peewee.DateTimeField(index=True)


def _ensure_columns() -> None:
    """Add columns introduced after the initial schema (SQLite has no auto-migrate)."""
    rows = db.execute_sql('PRAGMA table_info("post")').fetchall()
    columns = {row[1] for row in rows}
    if 'like_count' not in columns:
        db.execute_sql('ALTER TABLE "post" ADD COLUMN like_count INTEGER DEFAULT 0 NOT NULL')
    if 'repost_count' not in columns:
        db.execute_sql('ALTER TABLE "post" ADD COLUMN repost_count INTEGER DEFAULT 0 NOT NULL')


if db.is_closed():
    db.connect()
    db.create_tables([Post, SubscriptionState, AuthorLocalStats, AuthorDecision])
    _ensure_columns()


def prune_old_posts(retention_days: int | None = None) -> int:
    days = config.POST_RETENTION_DAYS if retention_days is None else retention_days
    cutoff = utc_now() - timedelta(days=days)
    deleted = Post.delete().where(Post.indexed_at < cutoff).execute()
    decision_cutoff = utc_now() - timedelta(days=config.SOFT_PRIOR_WINDOW_DAYS)
    AuthorDecision.delete().where(AuthorDecision.decided_at < decision_cutoff).execute()
    return deleted
