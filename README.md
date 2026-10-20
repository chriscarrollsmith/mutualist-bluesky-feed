# Mutualist Bluesky Feed

Self-hosted Bluesky custom feed for durable institutions — mutual credit,
possession, cooperatives, commons, assemblies, confederations — that people
can use while firms and states still dominate, and whose aim is to reduce
dependence on both. Proudhonian and left-market traditions are the core.
Libertarian municipalism, Bookchin's social ecology when it is about communes
or assemblies, and democratic confederalism when it is about councils,
cooperatives, or women's institutions are in scope too. Contract and assembly
both belong. Kropotkin stays for mutual aid and the federated commune.
Offensive violence, insurrection, martyrdom, and party war communiqués stay
out. Bare mutual aid keeps only when paired with one of those institutions.
The feed aims to stay hate-free; defiance and non-violent resistance are fine.

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
| `server/fn_search.py` | Daily false-negative query planner and log |
| `publish_feed.py` | Create/update the generator record |

## Matching policy

Decision order:

```
blocklist → allowlist → out-of-scope entity → hard negatives → in-scope entity / strong regex →
event+venue → (ambiguous + context) → soft prior → Perplexity decider → drop
```

Anything that passes the regex floor is a candidate. With `DECISION_ENABLED=true`,
`perplexity/pplx-decider-v1.2` (OpenRouter Decisions API) keep/rejects that
candidate. Keep when the yes-probability (`noul`) is at least `DECISION_KEEP_MIN`
(default 0.5). A timeout, a bad payload, or an enabled filter with no
`OPENROUTER_API_KEY` drops the post. Candidates with images also send JPEG
thumbnails from the Bluesky CDN. A failed download omits that image and the
text decision still runs. `DECISION_IMAGES=false` classifies on text only.

Offline eval and unit tests leave the filter disabled. The regex floor then
stands, except personal money-asks (Venmo, CashApp, GoFundMe, `#HelpSky`,
`#MutualAidRequest`, `💸`, “please help”, opaque quote + `#mutualaid`, …), which
still drop.

**Regex candidates** (then the decider, when enabled):

- Strong phrases (`Proudhon`, `mutual credit`, `occupancy-and-use`, `agorism`,
  `Bookchin`, `libertarian municipalism`, `democratic confederalism`,
  `worker cooperative`, `community land trust`, `time bank`, `Kropotkin`,
  `C4SS`, …)
- Ambiguous terms (`anarchism`, `mutual aid`, `co-op`, `social ecology`,
  `Rojava`, `communalism`, `confederalism`, …) with institutional context
- Allowlisted authors (`data/allowlist_handles.txt` / `allowlist_dids.txt`)
- Soft-prior authors using a bare ambiguous term
- Event phrasing + an in-scope venue or project (`mutualist reading circle`,
  `Really Really Free Market`, …)

**Soft prior:** three decider keeps and no decider rejections in the last
`SOFT_PRIOR_WINDOW_DAYS` (30). A prior lets bare ambiguous terms through to the
decider. It does not skip the decider, and a regex match does not earn one.

The decider keeps posts that practice an in-scope institution. Contract and
assembly both count. It rejects right-wing or reactionary framing,
anarcho-capitalism, chaos or entertainment "anarchy", crypto jargon, bare
mutual aid, charity, state-capacity politics, academic social-ecological
systems, sectarian communalism, EU confederalism, syndicalist cheerleading
with no durable institution, insurrection, martyrdom, party war communiqués,
symbiosis between species, personal fundraising, hate, and offensive use of
violence. Unclear posts are rejects.

**Drop** when the author is blocklisted (`data/blocklist_handles.txt` /
`blocklist_dids.txt`), even if the text would otherwise keep.

**Drop** hard negatives / false friends:

- Anarcho-capitalism / ancap / Rothbard–Hoppe / Mises Institute framing
- Sons of Anarchy, Anarchy Online, "state of anarchy" chaos news
- DeFi / Web3 / NFT "decentralized" jargon without institutional values
- Martyrdom, including when an in-scope name is also present
- Bare mutual aid, charity, and personal fundraising wrapped as "mutual aid"
- Sectarian communalism, EU confederalism, and social-ecological systems
  research with no institutional cue

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

### Daily false-positive and false-negative review

The two errors live in different samples. The feed database only stores posts the
matcher kept, and retention deletes them. Sampling the served feed shows false
positives and nothing else:

```bash
uv run python scripts/audit_indexed_feed.py --source feed --feed "$FEED_URI" --limit 100
```

Recall is measured by asking the public AppView `searchPosts` for posts the
matcher would drop. `scripts/fn_search.py` is the planner and the log. It does
not call Bluesky. `searchPosts` is unauthenticated
(`GET https://api.bsky.app/xrpc/app.bsky.feed.searchPosts`, `sort=latest`,
`since` 24 hours ago). Page with `cursor` when a query needs more than one page.
The client already lives in `scripts/backfill_gap.py` as `iter_search_posts`.
A datacenter IP often gets 403; record that attempt as blocked and do not invent
a zero.

Do not spend the day on the high-precision backfill presets (`mutualism`,
`Proudhon`, and the rest of `DEFAULT_SEARCH_QUERIES` in `scripts/backfill_gap.py`).
Those mostly return posts the matcher already keeps, so the log looks like “no
misses” and the planner learns the wrong lesson. `HIGH_PRECISION_QUERIES` in
`server/fn_search.py` is that denylist.

```bash
uv run python scripts/fn_search.py summary
uv run python scripts/fn_search.py plan
```

`plan` prints a slate of 8. The seed is the UTC date, so a rerun the same day is
stable. `--seed` and `--now` override it. The slate mixes:

- **exploit** — queries that have already produced false negatives, including
  queries that were not in the original catalog, weighted by historical
  false-negative count. A query that ran in the last 36 hours is weighted at
  `0.25`.
- **explore** — the least-used query from the least-used family in
  `data/fn_search_strategies.json`. Families are coarse recall hypotheses
  (`bare_term`, `practice`, `figure`, `current`, `event_phrase`, `hashtag`),
  because entropy is computed over families.

With no completed miss yet, the whole slate is explore. After that,
`explore_fraction = 0.35 + (1 - diversity) * 0.45`, where diversity is the
Shannon entropy of family counts divided by the entropy of a uniform
distribution over the families in use. A slate larger than one query keeps at
least one exploit slot and one explore slot. Unfilled exploit slots go to
explore.

Judge each in-window post with the indexer decision: drop replies when
`IGNORE_REPLY_POSTS` is set, drop archived and muted posts, drop blocklisted
authors, then `match_post`. A hit is a false negative only when the post belongs
in the feed and that decision would drop it. Off-topic noise the matcher
correctly drops, posts it already keeps, and replies are not false negatives.
`hits` is how many in-window posts were judged.

```bash
uv run python scripts/fn_search.py record \
  --query "tool library" --family practice \
  --hits 4 --false-negatives 1 \
  --false-negative-id "at://did:plc:example/app.bsky.feed.post/abc" \
  --notes "shared tools, no strong institutional phrase"
uv run python scripts/fn_search.py record \
  --query "mutual aid" --family bare_term --status blocked
```

Recording appends one JSON object to `data/fn_search_log.jsonl`. On `ok`, when
the false-negative count is at least one and the query is not already listed,
it is appended to the named family (created if needed) so a later exploit day
can select it. `--no-remember` skips that. Re-recording a known query keeps the
catalog’s original casing and refuses a different family. The planned eight are
the minimum: an extra query the day’s reading suggests is recorded the same way.

For each real miss, add an `expected: true` eval case and change the matcher
the same day. Commit the catalog, the log, the eval case, and the matcher
together. Exploit only has something to reuse because every miss was written
down with its query, and explore stays spread out because every attempt,
including the empty ones, was written down too.
