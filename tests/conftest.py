import os

import pytest

# Keep config imports safe if a test pulls in server packages beyond matcher.
os.environ.setdefault('FEEDGEN_HOSTNAME', 'test.example')
os.environ.setdefault('HOSTNAME', 'test.example')
os.environ.setdefault(
    'FEED_URI',
    'at://did:plc:test/app.bsky.feed.generator/mutualism',
)
os.environ.setdefault('DATABASE_PATH', ':memory:')
# Unit tests stay offline. Live decider calls are covered with FakeDecider.
os.environ.setdefault('CLASSIFIER_ENABLED', 'false')
os.environ.setdefault('DECISION_ENABLED', 'false')


@pytest.fixture(autouse=True)
def _decision_filter_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep pytest on the regex floor even if .env enables the decider."""
    monkeypatch.setenv('DECISION_ENABLED', 'false')
    from server.decider import clear_decider_cache

    clear_decider_cache()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        'markers',
        'live: call the live OpenRouter decider (opt in with pytest -m live)',
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip live API tests unless the mark expression selects them."""
    expr = config.option.markexpr or ''
    if 'live' in expr.replace('not live', ''):
        return
    skip_live = pytest.mark.skip(reason='opt-in live API test; run pytest -m live')
    for item in items:
        if item.get_closest_marker('live') is not None:
            item.add_marker(skip_live)
