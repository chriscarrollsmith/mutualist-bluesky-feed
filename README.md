# Mutualist Bluesky Feed

Self-hosted Bluesky custom feed that aggregates and celebrates **Proudhonian
mutualism** — reciprocity, possession, mutual credit — plus closely related
left-market currents (agorism, free-market anti-capitalism, libertarian
socialism when framed that way). Kropotkin is in scope; sibling anarchist
tendencies (ancom, syndicalism, Bakuninism, FNB/IWW/CrimethInc, etc.) are not.
Bare mutual aid keeps only when paired with mutualist economics. The feed aims
to stay hate- and violence-free; defiance and non-violent resistance are fine.

Architecture forked from [capital-region-feed](https://github.com/chriscarrollsmith/capital-region-feed).

## Architecture

```
Jetstream (posts/likes/reposts) → matcher → decider → SQLite → getFeedSkeleton → Bluesky AppView
```

Bluesky only asks this service for post URIs; AppView hydrates them.

| Path | Role |
|------|------|
| `server/matcher.py` | Regex floor (strong / ambiguous / hard negatives) |
| `server/decider.py` | Perplexity decider keep/reject on regex candidates |
| `server/jetstream.py` | Websocket consumer + cursor persistence |
| `server/indexer.py` | Match → DB writes / deletes |
| `server/app.py` | XRPC + `did:web` endpoints |
| `data/eval_cases.json` | Stratified keep/drop fixtures |
| `publish_feed.py` | Create/update the generator record |

## Matching policy

Decision order:

```
blocklist → allowlist → gazetteer other/local → hard negatives → strong regex →
event+venue → (ambiguous + context) → soft prior → Perplexity decider → drop
```

Anything that passes the regex floor is a candidate. With `DECISION_ENABLED=true`,
`perplexity/pplx-decider-v1.2` (OpenRouter Decisions API) keep/rejects that
candidate. Keep when the yes-probability (`noul`) is at least `DECISION_KEEP_MIN`
(default 0.5). A timeout, a bad payload, or an enabled filter with no
`OPENROUTER_API_KEY` drops the post.

Offline eval and unit tests leave the filter disabled. The regex floor then
stands, except personal money-asks (Venmo, CashApp, GoFundMe, `#HelpSky`,
`#MutualAidRequest`, `💸`, “please help”, opaque quote + `#mutualaid`, …), which
still drop.

**Regex candidates** (then the decider, when enabled):

- Strong phrases (`mutualism`, `Proudhon`, `mutual credit`, `occupancy-and-use`,
  `agorism`, `free-market anti-capitalism`, `Kropotkin`, `C4SS`, …)
- Ambiguous terms (`anarchism`, `mutual aid`, `libertarian socialism`, `co-op`,
  …) with **mutualist** context (not general-anarchist context)
- Allowlisted authors (`data/allowlist_handles.txt` / `allowlist_dids.txt`)
- Soft-prior authors using a bare ambiguous term
- Event phrasing + mutualist venue/project (`mutualist reading circle`,
  `Really Really Free Market`, …)

**Soft prior:** three decider keeps and no decider rejections in the last
`SOFT_PRIOR_WINDOW_DAYS` (30). A prior lets bare ambiguous terms through to the
decider. It does not skip the decider, and a regex match does not earn one.

The decider's reject criteria include right-wing or reactionary framing,
anarcho-capitalism, chaos or entertainment "anarchy", crypto jargon, sibling
anarchist tendencies without mutualist economics, biological mutualism
(symbiosis between species), bare mutual aid, personal fundraising, hate, and
celebration of violence. Unclear posts are rejects.

**Drop** when the author is blocklisted (`data/blocklist_handles.txt` /
`blocklist_dids.txt`), even if the text would otherwise keep.

**Drop** hard negatives / false friends:

- Anarcho-capitalism / ancap / Rothbard–Hoppe / Mises Institute framing
- Sons of Anarchy, Anarchy Online, “state of anarchy” chaos news
- DeFi / Web3 / NFT “decentralized” jargon without mutualist values
- Sibling anarchist tendencies without mutualist economics
- Bare mutual aid without mutualist-economics framing
- Personal fundraising wrapped as “mutual aid”

## Local setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14+.

```bash
uv sync
cp .env.example .env
# For local dev:
# FEEDGEN_HOSTNAME=localhost
# SERVICE_DID=did:web:localhost
# DATABASE_PATH=./feed_database.db
# FEED_URI=at://did:plc:test/app.bsky.feed.generator/mutualism
# DECISION_ENABLED=false     # optional offline
# OPENROUTER_API_KEY=…       # required when DECISION_ENABLED=true
```

### Lint, type-check, and test

```bash
uv run ruff check .
uv run ruff format .
uv run ty check
uv run pytest -q
uv run python scripts/eval_filter.py --verbose
```

### Run the feedgen locally

```bash
uv run python -m server
```

Endpoints:

- `GET /healthz`
- `GET /.well-known/did.json`
- `GET /xrpc/app.bsky.feed.describeFeedGenerator`
- `GET /xrpc/app.bsky.feed.getFeedSkeleton?feed=<FEED_URI>`

## Publish the feed record

On your machine (not in a cloud agent), with a Bluesky app password:

```bash
# HANDLE, PASSWORD, FEEDGEN_HOSTNAME, RECORD_NAME=mutualism, DISPLAY_NAME, DESCRIPTION
uv run python publish_feed.py
```

Put the printed URI into `FEED_URI` / Fly env. That creates
`…/app.bsky.feed.generator/mutualism`. Delete the old `…/anarchism` generator
record from the same account when you no longer want it listed:

```bash
uv run python unpublish_feed.py anarchism
```

## Deploy on Fly.io

The live Fly app hostname may still be `anarchist-bluesky-feed.fly.dev` until
you create a renamed app; `FEED_URI` should point at the `mutualism` rkey.

```bash
fly apps create mutualist-bluesky-feed   # optional rename; or keep existing app
fly volumes create feed_data --region ewr --size 1
fly secrets set OPENROUTER_API_KEY=…
fly deploy
```

`fly.toml` keeps **one always-on machine** (`auto_stop_machines = "off"`) because Jetstream must stay connected. Set `OPENROUTER_API_KEY` as a Fly secret. `DECISION_ENABLED=true` with no key drops every regex candidate.

Confirm:

```bash
curl -s https://anarchist-bluesky-feed.fly.dev/healthz
curl -s https://anarchist-bluesky-feed.fly.dev/.well-known/did.json
curl -s "https://anarchist-bluesky-feed.fly.dev/xrpc/app.bsky.feed.getFeedSkeleton?feed=$FEED_URI"
```

## Suggested iteration loop

```bash
uv run python scripts/collect_eval_sample.py authors --from-allowlist > /tmp/sample-authors.jsonl
uv run python scripts/collect_eval_sample.py near-miss > /tmp/sample-near-miss.jsonl
uv run python scripts/collect_eval_sample.py events > /tmp/sample-events.jsonl

# Optional LLM proposals (human confirm still required)
uv run python scripts/llm_label_judge.py --input /tmp/sample-near-miss.jsonl --output /tmp/proposed.jsonl

uv run python scripts/append_eval_cases.py --input /tmp/labeled.jsonl
uv run python scripts/eval_filter.py && uv run pytest -q
```

After matcher changes, audit/purge stale rows with `scripts/audit_indexed_feed.py`.
If Jetstream skips a lag gap, backfill with `scripts/backfill_gap.py`.
