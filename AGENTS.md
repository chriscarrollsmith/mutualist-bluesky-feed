# AGENTS.md

## Cursor Cloud specific instructions

Single Python 3.14 service: a self-hosted Bluesky custom feed generator for
Proudhonian and libertarian-municipalist institutions. See `README.md` for the full architecture and
command reference. Dependencies are managed with [uv](https://docs.astral.sh/uv/)
(`pyproject.toml` + `uv.lock`); the project virtualenv lives at `.venv`.

### Running / testing / lint

- Prefer `uv run …` (or activate `.venv` after `uv sync`).
- Install/sync: `uv sync`
- Tests (no network needed): `uv run pytest -q`
- Live decider check (opt-in, needs `OPENROUTER_API_KEY`): `uv run pytest -m live`
- Matcher precision/recall eval (no network needed):
  `uv run python scripts/eval_filter.py`
- Daily false-negative planner (no network): `uv run python scripts/fn_search.py summary`
  then `plan`. Record judged results with `record`. Do not pass
  `DEFAULT_SEARCH_QUERIES` / `HIGH_PRECISION_QUERIES`.
- Lint/format: `uv run ruff check .` and `uv run ruff format .`
- Type check: `uv run ty check`
- Run the server (dev): `uv run python -m server` (uvicorn on `PORT`, default
  8080).

### Non-obvious caveats

- `.env` is gitignored. For local dev, copy `.env.example` to `.env` and override
  values that are production-only:
  - `FEEDGEN_HOSTNAME=localhost` and `SERVICE_DID=did:web:localhost` — otherwise
    `/.well-known/did.json` returns 404 locally, and the app refuses to start if
    `FEEDGEN_HOSTNAME` is unset or equal to the shell's `HOSTNAME=cursor`.
  - `DATABASE_PATH=./feed_database.db` — the default `/data/...` path is the Fly.io
    volume mount and is not writable locally.
  - `DECISION_ENABLED=false` in unit tests (set via `tests/conftest.py`) so
    pytest stays offline. Production sets `DECISION_ENABLED=true` and
    `OPENROUTER_API_KEY`; the Perplexity decider keep/rejects every regex
    candidate. A missing key or a failed request drops the post. Image
    candidates also download JPEG thumbnails from `cdn.bsky.app`. A failed
    download omits that image; `DECISION_IMAGES=false` skips the download.
- `server/config.py` calls `load_dotenv(override=True)`, so values in `.env`
  win over exported shell variables.
- The Jetstream consumer runs in a background daemon thread started from the
  FastAPI lifespan in `server/app.py`. The feed is empty until a matching post
  arrives.
- `publish_feed.py` requires real Bluesky app-password credentials and mutates a
  live feed record — do NOT run it in cloud agents.
- After editing `data/allowlist_handles.txt`, refresh DIDs with
  `uv run python scripts/resolve_allowlist_dids.py` before relying on Jetstream
  recall.
- After editing `data/blocklist_handles.txt`, refresh DIDs with
  `uv run python scripts/resolve_allowlist_dids.py --handles data/blocklist_handles.txt --output data/blocklist_dids.txt`.
- The false-negative planner never calls Bluesky. `searchPosts` often returns
  403 from a datacenter IP; record the planned query with `--status blocked`
  and do not invent a zero. A miss is a post that belongs in the feed and that
  the indexer would drop (`match_post` after the reply, archive, blocklist, and
  mute checks). Replies are not misses while `IGNORE_REPLY_POSTS` is set.
  Commit `data/fn_search_strategies.json`, `data/fn_search_log.jsonl`, the eval
  case, and the matcher change together.
