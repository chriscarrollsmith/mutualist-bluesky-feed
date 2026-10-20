# Backlog

Living backlog for the mutualist Bluesky feed.

## Product goals

- **Precision:** Keep out ancap / chaos / speculative-crypto false friends,
  offensive violence, empty scenetalk, and personal fundraising wrapped as
  “mutual aid.”
- **Recall:** Surface inspiring **proto-anarchist** / decentralized-leftist
  content — including authors who never say “anarchist” — plus allowlisted
  collectives and mutual-aid *praxis*.
- **Content filter:** The Perplexity decider keep/rejects every regex candidate.
  A soft prior is three decider keeps and no rejections in 30 days.
- **Tone:** Prefer constructive, respectable, high-wow posts. Governance is to
  be spread out and humanized. Contract and assembly both count. State-capacity
  politics stays out.

## Active roadmap — proto-anarchist reclaim

Tracking epic: [#5](https://github.com/chriscarrollsmith/mutualist-bluesky-feed/issues/5)

Implementation order:

| # | Issue | Focus |
|---|-------|-------|
| 1 | [#6](https://github.com/chriscarrollsmith/mutualist-bluesky-feed/issues/6) | Eval fixtures for latent keep/drop strata |
| 2 | [#7](https://github.com/chriscarrollsmith/mutualist-bluesky-feed/issues/7) | Docs / product thesis sync |
| 3 | [#8](https://github.com/chriscarrollsmith/mutualist-bluesky-feed/issues/8) | Classifier thematic_fit rewrite |
| 4 | [#9](https://github.com/chriscarrollsmith/mutualist-bluesky-feed/issues/9) | Compound latent latch |
| 5 | [#10](https://github.com/chriscarrollsmith/mutualist-bluesky-feed/issues/10) | Quality-gate empty scenetalk |
| 6 | [#11](https://github.com/chriscarrollsmith/mutualist-bluesky-feed/issues/11) | Domain recall: OSS/OSH, CLTs, time banks, Sarafu, Rojava |
| 7 | [#12](https://github.com/chriscarrollsmith/mutualist-bluesky-feed/issues/12) | Crypto nuance: ZK identity in, speculation out |
| 8 | [#13](https://github.com/chriscarrollsmith/mutualist-bluesky-feed/issues/13) | Allowlist: PoH, BrightID, Hypercerts, Circles UBI, Regen, Sarafu, Rojava |
| 9 | [#14](https://github.com/chriscarrollsmith/mutualist-bluesky-feed/issues/14) | Soft priors from latent quality keeps |

## P0 — Ship / harden v1

### B-001 — Grow allowlist of aligned orgs
- Curate publishers and organizers of mutual credit, cooperatives, land trusts,
  assemblies, and similar institutions. Screen with `scripts/screen_allowlist_candidates.py`.
- Keep DIDs in sync via `scripts/resolve_allowlist_dids.py`.
- Roadmap expansion: [#13](https://github.com/chriscarrollsmith/mutualist-bluesky-feed/issues/13).

### B-002 — Expand eval set from live near-misses
- Sample with `collect_eval_sample.py` (authors / near-miss / events).
- Optional LLM proposals via `llm_label_judge.py` — human confirm before append.
- Latent/proto stratum: [#6](https://github.com/chriscarrollsmith/mutualist-bluesky-feed/issues/6).

### B-003 — Tune quality-rubric gates on holdout
- Log classifier keep/drop reasons (incl. `quality_reject:*` / solicit cues).
- Adjust thematic / valence / wow / solicit / composite gates only with
  stratified eval evidence.
- Prompt thesis rewrite: [#8](https://github.com/chriscarrollsmith/mutualist-bluesky-feed/issues/8).

### B-004 — Publish + Fly cutover
- `publish_feed.py` with real credentials on an operator machine.
- Set `FEED_URI` + `DEEPSEEK_API_KEY` on Fly; verify `did.json` + skeleton.

## P1 — Quality

### B-010 — Multilingual anarchist terms
- Spanish/French/German strong positives and hard negatives.

### B-011 — Soft-prior analytics
- Dashboard or script for authors approaching the soft-prior threshold.
- Latent-keep flywheel: [#14](https://github.com/chriscarrollsmith/mutualist-bluesky-feed/issues/14).

### B-013 — Quality rubric ranking
- Optionally store grade breakdowns on indexed posts and soft-boost wow /
  valence in `RANKING_MODE` experiments once the index is non-empty.
