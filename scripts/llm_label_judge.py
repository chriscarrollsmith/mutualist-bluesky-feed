#!/usr/bin/env python3
"""Offline LLM judge: propose keep/drop labels for unlabeled eval candidates.

Reads JSONL from ``scripts/collect_eval_sample.py`` (``expected: null``), calls an
OpenAI-compatible chat API to propose ``expected`` + rationale, and writes JSONL
for **human review**. Never writes ``data/eval_cases.json`` — humans confirm,
then:

    uv run python scripts/append_eval_cases.py --input /tmp/confirmed.jsonl

Defaults to DeepSeek (``DEEPSEEK_API_KEY``, ``deepseek-v4-flash``,
``https://api.deepseek.com/v1/chat/completions``). Falls back to
``OPENAI_API_KEY`` + OpenAI URL if DeepSeek is unset. Offline label bootstrap
only — the live feed path uses ``server/classifier.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from server.scope_text import BIOLOGICAL_EXCLUSION

ROOT = Path(__file__).resolve().parents[1]
# Prefer repo .env values (same pattern as server/config.py).
load_dotenv(ROOT / '.env', override=True)

JsonObject = dict[str, Any]
JudgeFn = Callable[[JsonObject], JsonObject]

DEFAULT_DEEPSEEK_MODEL = 'deepseek-v4-flash'
DEFAULT_DEEPSEEK_API_URL = 'https://api.deepseek.com/v1/chat/completions'
DEFAULT_OPENAI_MODEL = 'gpt-4o-mini'
DEFAULT_OPENAI_API_URL = 'https://api.openai.com/v1/chat/completions'

SYSTEM_PROMPT = (
    """\
You label Bluesky posts for a curated feed of durable non-state institutions.
Prefer high-quality, constructive, memorable posts. Personal fundraising,
hate, and offensive violence stay out.

In scope: Proudhonian and left-market traditions (reciprocity, possession /
occupancy-and-use, mutual credit, free banking, agorism, free-market
anti-capitalism); Kropotkin on mutual aid and the federated commune; Bookchin,
libertarian municipalism, and social ecology when the post is about communes
or assemblies; democratic confederalism when the post is about councils,
cooperatives, or women's institutions. Contract and assembly both belong.
Bare "mutual aid" without one of those institutions is out of scope. """
    + BIOLOGICAL_EXCLUSION
    + """ is out of scope.

Return JSON only: {"expected": true|false, "rationale": "<short reason>",
"confidence": "high"|"medium"|"low"}.

Keep (expected=true) when the post is about or celebrating an in-scope
institution, from a clearly aligned org/voice, or announces such an event —
including argument between contract and assembly. Prefer constructive tone
and substance over boilerplate slogans. Defiance and non-violent resistance
can keep.

Drop (expected=false) for:
- Anarcho-capitalism / right-"libertarian" market fundamentalism
- Chaos / entertainment "anarchy" (Sons of Anarchy, Anarchy Online, sports riots)
- Crypto/Web3/DeFi "decentralized" jargon without institutional values
- Bare ambiguous left or general-anarchist terms with no institutional signal
- Syndicalist or union cheerleading with no durable institution
- State-capacity politics and reforms whose agent is the nation-state
- Academic social-ecological systems, sectarian communalism, and EU confederalism
- Insurrection, armed-struggle footage, martyrdom, and party war communiqués
- """
    + BIOLOGICAL_EXCLUSION
    + """
- Personal money-asks (Venmo/CashApp/GoFundMe/PayPal-me, "I'm short on rent",
  individual "mutual aid request") even when wrapped in institutional language
- Hate speech or offensive use of violence

Prefer precision on false friends and solicits. If unsure, set confidence to
low and lean drop for bare slogans or personal fundraising; lean keep only
when the institution is clear and the post has no extractive ask.
"""
)


def resolve_api_defaults(
    *,
    api_key: str | None = None,
    model: str | None = None,
    api_url: str | None = None,
) -> tuple[str, str, str]:
    """Pick DeepSeek by default; fall back to OpenAI when only that key is set."""
    deepseek_key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
    openai_key = os.environ.get('OPENAI_API_KEY', '').strip()

    if api_key:
        key = api_key
        # Explicit key: keep caller model/url, else DeepSeek defaults.
        resolved_model = model or DEFAULT_DEEPSEEK_MODEL
        resolved_url = api_url or DEFAULT_DEEPSEEK_API_URL
        return key, resolved_model, resolved_url

    if deepseek_key:
        return (
            deepseek_key,
            model or DEFAULT_DEEPSEEK_MODEL,
            api_url or DEFAULT_DEEPSEEK_API_URL,
        )
    if openai_key:
        return (
            openai_key,
            model or DEFAULT_OPENAI_MODEL,
            api_url or DEFAULT_OPENAI_API_URL,
        )
    return '', model or DEFAULT_DEEPSEEK_MODEL, api_url or DEFAULT_DEEPSEEK_API_URL


def read_jsonl(path: Path | None) -> list[JsonObject]:
    if path is None:
        raw = sys.stdin.read()
    else:
        raw = path.read_text(encoding='utf-8')
    rows: list[JsonObject] = []
    for line_no, line in enumerate(raw.splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith('#'):
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f'invalid JSON on line {line_no}: {exc}') from exc
        if not isinstance(payload, dict):
            raise ValueError(f'line {line_no} must be a JSON object')
        rows.append(payload)
    return rows


def _candidate_user_message(row: JsonObject) -> str:
    parts = [
        f'id: {row.get("id")}',
        f'text: {row.get("text")}',
        f'signal_hint: {row.get("signal")}',
        f'bucket_hint: {row.get("bucket")}',
    ]
    if row.get('author_handle'):
        parts.append(f'author_handle: {row["author_handle"]}')
    if row.get('author_did'):
        parts.append(f'author_did: {row["author_did"]}')
    if row.get('langs'):
        parts.append(f'langs: {row["langs"]}')
    if row.get('note'):
        parts.append(f'collector_note: {row["note"]}')
    return '\n'.join(parts)


def parse_judge_response(content: str) -> JsonObject:
    text = content.strip()
    if text.startswith('```'):
        lines = text.splitlines()
        # Drop opening/closing fences.
        if lines and lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        text = '\n'.join(lines).strip()
    # Some models wrap JSON in prose; take the outermost object if needed.
    if not text.startswith('{'):
        start = text.find('{')
        end = text.rfind('}')
        if start >= 0 and end > start:
            text = text[start : end + 1]
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError('judge response must be a JSON object')
    expected = payload.get('expected')
    if not isinstance(expected, bool):
        raise ValueError(f'expected must be a boolean, got {expected!r}')
    rationale = str(payload.get('rationale') or '').strip()
    confidence = str(payload.get('confidence') or 'medium').strip().lower()
    if confidence not in {'high', 'medium', 'low'}:
        confidence = 'medium'
    return {
        'expected': expected,
        'rationale': rationale,
        'confidence': confidence,
    }


def chat_completions_judge(
    row: JsonObject,
    *,
    api_key: str,
    model: str,
    api_url: str,
    timeout: float = 120.0,
) -> JsonObject:
    body: JsonObject = {
        'model': model,
        'temperature': 0,
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': _candidate_user_message(row)},
        ],
    }
    # DeepSeek V4 supports OpenAI-style JSON mode; keep it when available.
    body['response_format'] = {'type': 'json_object'}
    # Non-thinking mode keeps label bootstrap fast/cheap on deepseek-v4-flash.
    body['thinking'] = {'type': 'disabled'}
    request = urllib.request.Request(
        api_url,
        data=json.dumps(body).encode('utf-8'),
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'mutualist-bluesky-feed-llm-label/0.1',
        },
        method='POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'LLM API HTTP {exc.code}: {detail[:400]}') from exc

    try:
        message = payload['choices'][0]['message']
        content = message.get('content')
        # Reasoning models may leave content empty and put text elsewhere.
        if content is None or content == '':
            content = message.get('reasoning_content') or message.get('reasoning')
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f'unexpected LLM response shape: {payload!r}') from exc
    if content is None or content == '':
        raise RuntimeError(f'empty LLM message content: {payload!r}')
    return parse_judge_response(str(content))


# Backward-compatible alias used by older call sites / docs.
openai_chat_judge = chat_completions_judge


def apply_proposal(row: JsonObject, proposal: JsonObject) -> JsonObject:
    """Attach LLM proposal fields; leave human confirmation as the gate."""
    out = dict(row)
    out['expected'] = proposal['expected']
    out['proposed_by'] = 'llm'
    out['proposed_expected'] = proposal['expected']
    out['proposed_confidence'] = proposal['confidence']
    rationale = proposal.get('rationale') or ''
    prior_note = str(row.get('note') or '').strip()
    llm_note = (
        f'llm propose expected={proposal["expected"]} ({proposal["confidence"]}): {rationale}'
    )
    out['note'] = f'{prior_note} | {llm_note}'.strip(' |') if prior_note else llm_note
    out['needs_human_confirm'] = True
    return out


def judge_rows(
    rows: list[JsonObject],
    *,
    judge: JudgeFn,
    only_unlabeled: bool = True,
    skip_errors: bool = False,
) -> tuple[list[JsonObject], dict[str, int]]:
    stats = {
        'proposed': 0,
        'skipped_labeled': 0,
        'skipped_empty': 0,
        'errors': 0,
    }
    out: list[JsonObject] = []
    for row in rows:
        if only_unlabeled and row.get('expected') is not None:
            stats['skipped_labeled'] += 1
            continue
        text = row.get('text')
        if not isinstance(text, str) or not str(row.get('id') or '').strip():
            stats['skipped_empty'] += 1
            continue
        try:
            proposal = judge(row)
            out.append(apply_proposal(row, proposal))
            stats['proposed'] += 1
        except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
            stats['errors'] += 1
            if not skip_errors:
                raise RuntimeError(f'judge failed for {row.get("id")!r}: {exc}') from exc
            err_row = dict(row)
            err_row['proposed_by'] = 'llm'
            err_row['needs_human_confirm'] = True
            err_row['note'] = f'{row.get("note") or ""} | llm error: {exc}'.strip(' |')
            out.append(err_row)
    return out, stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--input',
        type=Path,
        help='Unlabeled JSONL (default: stdin)',
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Write proposed JSONL here (default: stdout)',
    )
    parser.add_argument(
        '--api-key',
        default='',
        help='API key (default: DEEPSEEK_API_KEY, else OPENAI_API_KEY)',
    )
    parser.add_argument(
        '--model',
        default='',
        help=f'Model id (default: {DEFAULT_DEEPSEEK_MODEL} with DeepSeek)',
    )
    parser.add_argument(
        '--api-url',
        default='',
        help=f'Chat completions URL (default: {DEFAULT_DEEPSEEK_API_URL})',
    )
    parser.add_argument(
        '--include-labeled',
        action='store_true',
        help='Also re-judge rows that already have expected set',
    )
    parser.add_argument(
        '--skip-errors',
        action='store_true',
        help='Keep going when a single row fails; mark note with error',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=0,
        help='Max unlabeled rows to judge (0 = all)',
    )
    return parser


def main(argv: list[str] | None = None, *, judge: JudgeFn | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        rows = read_jsonl(args.input)
    except ValueError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1

    if args.limit and args.limit > 0:
        selected: list[JsonObject] = []
        for row in rows:
            if row.get('expected') is not None and not args.include_labeled:
                continue
            selected.append(row)
            if len(selected) >= args.limit:
                break
        rows = selected

    if judge is None:
        api_key, model, api_url = resolve_api_defaults(
            api_key=args.api_key or None,
            model=args.model or None,
            api_url=args.api_url or None,
        )
        if not api_key:
            print(
                'error: set DEEPSEEK_API_KEY (preferred) or OPENAI_API_KEY, or pass --api-key',
                file=sys.stderr,
            )
            return 1
        print(f'using model={model} api_url={api_url}', file=sys.stderr)

        def judge(row: JsonObject) -> JsonObject:
            return chat_completions_judge(
                row,
                api_key=api_key,
                model=model,
                api_url=api_url,
            )

    try:
        proposed, stats = judge_rows(
            rows,
            judge=judge,
            only_unlabeled=not args.include_labeled,
            skip_errors=args.skip_errors,
        )
    except RuntimeError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1

    lines = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in proposed)
    if args.output is None:
        sys.stdout.write(lines)
    else:
        args.output.write_text(lines, encoding='utf-8')

    print(
        f'proposed={stats["proposed"]} '
        f'skipped_labeled={stats["skipped_labeled"]} '
        f'skipped_empty={stats["skipped_empty"]} '
        f'errors={stats["errors"]}',
        file=sys.stderr,
    )
    print(
        'Review proposals, then append only human-confirmed rows with '
        'scripts/append_eval_cases.py (do not trust LLM labels blindly).',
        file=sys.stderr,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
