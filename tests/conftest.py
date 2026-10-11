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
