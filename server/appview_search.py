"""AppView ``searchPosts`` request rules.

The public AppView serves one unauthenticated page. A ``cursor`` on that
request is refused with HTTP 403 and the body "Request forbidden by
administrative rules." Further pages need a session from
``com.atproto.server.createSession`` (``Authorization: Bearer``).

A query that starts with ``#`` is not sent. The same endpoint returns HTTP 400
``InvalidRequest`` for it. Callers search the remaining text.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

DEFAULT_API_HOST = 'https://api.bsky.app'
DEFAULT_USER_AGENT = (
    'mutualist-bluesky-feed-backfill/0.1 '
    '(+https://github.com/chriscarrollsmith/mutualist-bluesky-feed)'
)
FN_SEARCH_USER_AGENT = (
    'mutualist-bluesky-feed-fn-search/0.1 '
    '(+https://github.com/chriscarrollsmith/mutualist-bluesky-feed)'
)
SESSION_HOST = 'https://bsky.social'

JsonObject = dict[str, Any]
Opener = Callable[..., Any]


class UnauthenticatedCursorError(Exception):
    """``searchPosts`` will not follow a cursor unless the request has a session."""


def search_query_text(query: str) -> str:
    """Return the ``q`` value to send. One leading ``#`` is removed."""
    text = ' '.join(query.split())
    if text.startswith('#'):
        text = text[1:].strip()
    if not text:
        raise ValueError(f'{query!r} is empty after stripping #')
    return text


def format_api_time(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')


def search_posts_params(
    query: str,
    *,
    since: datetime,
    until: datetime,
    limit: int = 100,
    cursor: str | None = None,
    access_token: str | None = None,
) -> dict[str, Any]:
    """Build ``searchPosts`` query params. A cursor without a token is an error."""
    if cursor and not access_token:
        raise UnauthenticatedCursorError(
            'searchPosts does not serve a cursor without a session; kept the first page'
        )
    if limit < 1 or limit > 100:
        raise ValueError('searchPosts limit must be from 1 to 100')
    params: dict[str, Any] = {
        'q': search_query_text(query),
        'limit': limit,
        'sort': 'latest',
        'since': format_api_time(since),
        'until': format_api_time(until),
    }
    if cursor:
        params['cursor'] = cursor
    return params


def search_posts_url(api_host: str, params: dict[str, Any]) -> str:
    base = api_host.rstrip('/') + '/xrpc/app.bsky.feed.searchPosts'
    return base + '?' + urllib.parse.urlencode(params)


def appview_request(
    url: str,
    *,
    access_token: str | None = None,
    user_agent: str = DEFAULT_USER_AGENT,
) -> urllib.request.Request:
    headers = {'Accept': 'application/json', 'User-Agent': user_agent}
    if access_token:
        headers['Authorization'] = f'Bearer {access_token}'
    return urllib.request.Request(url, headers=headers)


def fetch_json(
    url: str,
    *,
    access_token: str | None = None,
    timeout: float = 30.0,
    user_agent: str = DEFAULT_USER_AGENT,
    opener: Opener = urllib.request.urlopen,
) -> JsonObject:
    request = appview_request(url, access_token=access_token, user_agent=user_agent)
    with opener(request, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode('utf-8'))
    if not isinstance(payload, dict):
        raise ValueError('AppView response was not a JSON object')
    return payload


def create_session(
    handle: str,
    password: str,
    *,
    host: str = SESSION_HOST,
    timeout: float = 20.0,
    opener: Opener = urllib.request.urlopen,
) -> str:
    """Return an access token. The password is not logged."""
    body = json.dumps({'identifier': handle, 'password': password}).encode('utf-8')
    request = urllib.request.Request(
        host.rstrip('/') + '/xrpc/com.atproto.server.createSession',
        data=body,
        headers={
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'User-Agent': DEFAULT_USER_AGENT,
        },
        method='POST',
    )
    with opener(request, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode('utf-8'))
    token = payload.get('accessJwt') if isinstance(payload, dict) else None
    if not isinstance(token, str) or not token:
        raise RuntimeError('createSession returned no accessJwt')
    return token
