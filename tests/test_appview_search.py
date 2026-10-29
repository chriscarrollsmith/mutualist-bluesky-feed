"""Request rules for AppView searchPosts. No network."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from server.appview_search import (
    UnauthenticatedCursorError,
    appview_request,
    create_session,
    search_posts_params,
    search_query_text,
)


def test_search_query_text_strips_one_leading_hash() -> None:
    assert search_query_text('#mutual aid') == 'mutual aid'
    assert search_query_text('  #free   school ') == 'free school'
    assert search_query_text('c4ss #note') == 'c4ss #note'
    with pytest.raises(ValueError, match='empty'):
        search_query_text('#')


def test_cursor_requires_a_session() -> None:
    since = datetime(2026, 10, 28, tzinfo=UTC)
    until = datetime(2026, 10, 29, tzinfo=UTC)
    with pytest.raises(UnauthenticatedCursorError):
        search_posts_params('free school', since=since, until=until, cursor='page-2')
    params = search_posts_params(
        '#free school',
        since=since,
        until=until,
        cursor='page-2',
        access_token='token',
        limit=5,
    )
    assert params['q'] == 'free school'
    assert params['cursor'] == 'page-2'
    assert params['limit'] == 5
    assert 'cursor' not in search_posts_params('free school', since=since, until=until)


def test_appview_request_sets_bearer_only_with_a_token() -> None:
    bare = appview_request('https://api.bsky.app/xrpc/app.bsky.feed.searchPosts')
    assert bare.get_header('Authorization') is None
    authed = appview_request(
        'https://api.bsky.app/xrpc/app.bsky.feed.searchPosts',
        access_token='abc',
    )
    assert authed.get_header('Authorization') == 'Bearer abc'


def test_create_session_returns_access_token_without_echoing_the_password() -> None:
    seen: dict[str, object] = {}

    class _Response:
        def read(self) -> bytes:
            return json.dumps({'accessJwt': 'access-token', 'refreshJwt': 'refresh'}).encode()

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

    def opener(request, timeout: float = 20):  # type: ignore[no-untyped-def]
        del timeout
        seen['method'] = request.get_method()
        seen['url'] = request.full_url
        body = json.loads(request.data.decode('utf-8'))
        seen['identifier'] = body['identifier']
        seen['password_sent'] = body['password'] == 'secret-value'
        assert request.get_header('Authorization') is None
        return _Response()

    token = create_session('user.bsky.social', 'secret-value', opener=opener)
    assert token == 'access-token'
    assert seen['method'] == 'POST'
    assert str(seen['url']).endswith('/xrpc/com.atproto.server.createSession')
    assert seen['identifier'] == 'user.bsky.social'
    assert seen['password_sent'] is True
