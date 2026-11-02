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
        'file:mutualist_feed?mode=memory&cache=shared',
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
    # Hash of normalized post text. Null when the text is empty or dedupe is off.
    content_hash = peewee.CharField(max_length=64, null=True, unique=True)


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


def _post_columns() -> set[str]:
    rows = db.execute_sql('PRAGMA table_info("post")').fetchall()
    return {row[1] for row in rows}


def _ensure_content_hash_column() -> None:
    """Add content_hash before Peewee builds indexes on an existing table.

    ``create_tables`` skips ``CREATE TABLE`` when ``post`` already exists, then
    still issues ``CREATE UNIQUE INDEX`` for new fields. A missing column on
    that statement fails the startup.
    """
    existing = db.execute_sql(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'post'"
    ).fetchall()
    if not existing:
        return
    if 'content_hash' not in _post_columns():
        db.execute_sql('ALTER TABLE "post" ADD COLUMN content_hash VARCHAR(64)')


def _ensure_columns() -> None:
    """Add columns introduced after the initial schema (SQLite has no auto-migrate)."""
    columns = _post_columns()
    if 'like_count' not in columns:
        db.execute_sql('ALTER TABLE "post" ADD COLUMN like_count INTEGER DEFAULT 0 NOT NULL')
    if 'repost_count' not in columns:
        db.execute_sql('ALTER TABLE "post" ADD COLUMN repost_count INTEGER DEFAULT 0 NOT NULL')
    if 'content_hash' not in columns:
        db.execute_sql('ALTER TABLE "post" ADD COLUMN content_hash VARCHAR(64)')
    # Peewee names the unique index "{table}_{column}" when creating a fresh
    # database. Keep that name so an existing volume picks up the same index.
    db.execute_sql(
        'CREATE UNIQUE INDEX IF NOT EXISTS "post_content_hash" ON "post" ("content_hash")'
    )


if db.is_closed():
    db.connect()
    _ensure_content_hash_column()
    db.create_tables([Post, SubscriptionState, AuthorLocalStats, AuthorDecision])
    _ensure_columns()


def prune_old_posts(retention_days: int | None = None) -> int:
    days = config.POST_RETENTION_DAYS if retention_days is None else retention_days
    cutoff = utc_now() - timedelta(days=days)
    expired = Post.select(Post.content_hash).where(
        (Post.indexed_at < cutoff) & Post.content_hash.is_null(False)
    )
    expired_hashes = [row.content_hash for row in expired if row.content_hash]
    deleted = Post.delete().where(Post.indexed_at < cutoff).execute()
    if expired_hashes:
        # Local import: dedupe loads this module for the Post model.
        from server.dedupe import forget_fingerprints

        forget_fingerprints(expired_hashes)
    decision_cutoff = utc_now() - timedelta(days=config.SOFT_PRIOR_WINDOW_DAYS)
    AuthorDecision.delete().where(AuthorDecision.decided_at < decision_cutoff).execute()
    return deleted
