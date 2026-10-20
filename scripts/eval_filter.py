#!/usr/bin/env python3
"""Evaluate matcher precision/recall against labeled fixtures.

Product policy (see README): optimize for **both** false positives and false
negatives. Ancap / chaos / crypto false friends, sibling anarchist tendencies,
bare mutual aid, and personal fundraising must stay out; mutualist economics
and allowlisted mutualist voices should match.

Reports aggregate metrics plus stratification by ``bucket`` / ``signal`` and a
``dev`` vs ``holdout`` split so matcher changes are not judged only on the set
used while iterating. Prefer reading author- and event-signal recall alongside
``skyfeed_fp`` / ``precision_gate`` precision — not aggregate F1 alone.

Case schema (``data/eval_cases.json``):

- ``id``, ``text``, ``expected`` (bool) — required
- ``signal``: ``text`` | ``author`` | ``event`` — how the topic cue arrives
- ``bucket``: reason-oriented stratum (e.g. ``skyfeed_fp``,
  ``allowlisted_author``, ``in_scope_event``, ``solicit_reject``)
- ``split``: ``dev`` (iterate freely) or ``holdout`` (report separately)
- ``regression``: if false, scored in reports but ignored for exit code / pytest
  (known recall gaps until author/event/AI backlog items land)
- ``soft_prior``: if true with ``author_did``, treat that DID as soft-prior eligible
  for this case (earned priors in production come from decider keep/reject rows)
- ``langs``: optional Bluesky language tags
- Bucket ``ambiguous_classifier`` covers bare ambiguous terms. They stay dropped
  on the regex floor; a soft prior is what lets them reach the decider
- Bucket ``solicit_reject`` covers personal money-asks that must stay dropped
  when the decider is offline
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.allowlists import (  # noqa: E402
    load_allowlist_dids,
    load_allowlist_handles,
    load_blocklist_dids,
    load_blocklist_handles,
)
from server.matcher import match_post  # noqa: E402


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def add(self, *, predicted: bool, expected: bool) -> None:
        if predicted and expected:
            self.tp += 1
        elif predicted and not expected:
            self.fp += 1
        elif not predicted and not expected:
            self.tn += 1
        else:
            self.fn += 1

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total else 0.0


@dataclass
class EvalReport:
    overall: Confusion = field(default_factory=Confusion)
    by_split: dict[str, Confusion] = field(default_factory=lambda: defaultdict(Confusion))
    by_signal: dict[str, Confusion] = field(default_factory=lambda: defaultdict(Confusion))
    by_bucket: dict[str, Confusion] = field(default_factory=lambda: defaultdict(Confusion))
    failures: list[str] = field(default_factory=list)
    regression_failures: list[str] = field(default_factory=list)


def evaluate_cases(
    cases: list[dict[str, Any]],
    *,
    allowlist_handles: set[str] | None = None,
    allowlist_dids: set[str] | None = None,
    blocklist_handles: set[str] | None = None,
    blocklist_dids: set[str] | None = None,
    soft_prior_dids: set[str] | None = None,
    verbose: bool = False,
) -> EvalReport:
    handles = allowlist_handles if allowlist_handles is not None else load_allowlist_handles()
    dids = allowlist_dids if allowlist_dids is not None else load_allowlist_dids()
    blocked_handles = (
        blocklist_handles if blocklist_handles is not None else load_blocklist_handles()
    )
    blocked_dids = blocklist_dids if blocklist_dids is not None else load_blocklist_dids()
    report = EvalReport()

    for case in cases:
        case_soft: set[str] = set(soft_prior_dids or ())
        if case.get('soft_prior') and case.get('author_did'):
            case_soft.add(str(case['author_did']))
        langs = case.get('langs')
        result = match_post(
            case.get('text', ''),
            alt_text=case.get('alt_text', ''),
            langs=langs if isinstance(langs, list) else None,
            author_did=case.get('author_did'),
            author_handle=case.get('author_handle'),
            allowlist_dids=dids,
            allowlist_handles=handles,
            blocklist_dids=blocked_dids,
            blocklist_handles=blocked_handles,
            soft_prior_dids=case_soft,
            embed=case.get('embed'),
        )
        expected = bool(case['expected'])
        predicted = result.matched
        signal = str(case.get('signal') or 'text')
        bucket = str(case.get('bucket') or 'unspecified')
        split = str(case.get('split') or 'dev')
        regression = case.get('regression', True)
        if not isinstance(regression, bool):
            regression = bool(regression)

        report.overall.add(predicted=predicted, expected=expected)
        report.by_split[split].add(predicted=predicted, expected=expected)
        report.by_signal[signal].add(predicted=predicted, expected=expected)
        report.by_bucket[bucket].add(predicted=predicted, expected=expected)

        ok = predicted == expected
        if not ok:
            kind = 'FP' if predicted and not expected else 'FN'
            line = (
                f'{kind} {case["id"]} [{split}/{signal}/{bucket}] '
                f'reason={result.reason} :: {case.get("text", "")[:90]}'
            )
            report.failures.append(line)
            if regression:
                report.regression_failures.append(line)

        if verbose:
            mark = 'OK' if ok else 'MISS'
            gap = '' if regression else ' gap'
            print(
                f'[{mark}{gap}] {case["id"]} split={split} signal={signal} '
                f'bucket={bucket} expected={expected} got={predicted} ({result.reason})'
            )

    return report


def _fmt_rate(value: float, *, defined: bool) -> str:
    return f'{value:.3f}' if defined else 'n/a'


def _fmt_confusion(name: str, c: Confusion) -> str:
    return (
        f'{name}: n={c.total} tp={c.tp} fp={c.fp} tn={c.tn} fn={c.fn} '
        f'precision={_fmt_rate(c.precision, defined=(c.tp + c.fp) > 0)} '
        f'recall={_fmt_rate(c.recall, defined=(c.tp + c.fn) > 0)} '
        f'f1={_fmt_rate(c.f1, defined=(c.tp + c.fp) > 0 and (c.tp + c.fn) > 0)} '
        f'accuracy={c.accuracy:.3f}'
    )


def print_report(report: EvalReport) -> None:
    print(_fmt_confusion('all', report.overall))
    print()
    print('By split:')
    for key in sorted(report.by_split):
        print(f'  {_fmt_confusion(key, report.by_split[key])}')
    print()
    print('By signal:')
    for key in sorted(report.by_signal):
        print(f'  {_fmt_confusion(key, report.by_signal[key])}')
    print()
    print('By bucket:')
    for key in sorted(report.by_bucket):
        print(f'  {_fmt_confusion(key, report.by_bucket[key])}')

    if report.failures:
        print('\nFailures:')
        for line in report.failures:
            print(f'  - {line}')
    if report.regression_failures != report.failures:
        print(
            f'\nRegression failures: {len(report.regression_failures)} '
            f'(of {len(report.failures)} total misses; non-regression gaps ignored for exit)'
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--cases',
        type=Path,
        default=ROOT / 'data' / 'eval_cases.json',
        help='Path to labeled eval JSON',
    )
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Fail on any miss, including regression=false gap cases',
    )
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding='utf-8'))
    report = evaluate_cases(cases, verbose=args.verbose)
    print_report(report)

    if args.strict:
        return 1 if report.failures else 0
    return 1 if report.regression_failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
