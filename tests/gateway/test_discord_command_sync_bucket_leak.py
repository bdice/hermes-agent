"""Command-sync must not leak its 30s rate-limit cap into shared buckets.

Hermes bounds slash-command-sync 429 sleeps by swapping HTTPClient.max_ratelimit_timeout
to 30s for the sync window. discord.py bakes that value into each Ratelimit bucket when the
bucket is first created, so a bucket born during the window keeps the cap forever and its
acquire() then raises RateLimited locally — without contacting Discord — whenever the bucket
expires >30s out. That silently killed Discord auto-thread creation for minutes after any
prior thread start (the start-thread bucket's window is ~300s). The sync must therefore drop
buckets created during the swap so they are recreated with the restored timeout.
"""
from __future__ import annotations

import asyncio
import sys

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import (
    _DISCORD_COMMAND_SYNC_MAX_RATE_LIMIT_SLEEP_SECONDS,
    DiscordAdapter,
)


def _real_ratelimit_classes():
    """Load discord.py's real Ratelimit/RateLimited past the suite's discord stub (conftest)."""
    saved = {n: sys.modules.pop(n) for n in list(sys.modules)
             if n == "discord" or n.startswith("discord.")}
    try:
        from discord.errors import RateLimited
        from discord.http import Ratelimit
        return RateLimited, Ratelimit
    finally:
        for n in [n for n in sys.modules if n == "discord" or n.startswith("discord.")]:
            if n not in saved:
                del sys.modules[n]
        sys.modules.update(saved)


class _Bucket:
    """Captures the timeout at creation, like discord.py's Ratelimit does."""

    def __init__(self, cap: object) -> None:
        self._max_ratelimit_timeout = cap
        self.outgoing = 0
        self._pending_requests: list[object] = []


class _FakeHttp:
    """Mirrors discord.py HTTPClient's lazy bucket creation (the bake-at-creation behavior)."""

    def __init__(self) -> None:
        self.max_ratelimit_timeout = None
        self._buckets: dict[str, _Bucket] = {}

    def get_ratelimit(self, key: str) -> _Bucket:
        return self._buckets.setdefault(key, _Bucket(self.max_ratelimit_timeout))


def _adapter_for_sync(http: _FakeHttp) -> DiscordAdapter:
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake"))
    adapter._client = type("C", (), {"http": http, "application_id": 1})()
    adapter._get_discord_command_sync_policy = lambda: "safe"
    adapter._desired_command_sync_fingerprint = lambda: "fp"
    adapter._command_sync_skip_reason = lambda app_id, fingerprint: None
    adapter._record_command_sync_attempt = lambda app_id, fingerprint: None
    adapter._record_command_sync_rate_limit = lambda app_id, fingerprint, retry_after: None
    adapter._record_command_sync_success = lambda app_id, fingerprint, summary: None
    return adapter


@pytest.mark.asyncio
async def test_sync_window_buckets_are_dropped_after_restore() -> None:
    http = _FakeHttp()
    adapter = _adapter_for_sync(http)
    http.get_ratelimit("pre-existing-route")  # created before the swap: must survive

    observed: dict[str, object] = {}

    async def _safe_sync_slash_commands():
        observed["timeout_during"] = http.max_ratelimit_timeout
        http.get_ratelimit("start-thread-bucket")  # born during the window: must be dropped
        busy = http.get_ratelimit("busy-bucket")  # in flight during teardown: must be kept
        busy.outgoing = 1
        return {"total": 0, "unchanged": 0, "updated": 0, "recreated": 0, "created": 0, "deleted": 0}

    adapter._safe_sync_slash_commands = _safe_sync_slash_commands

    await adapter._run_post_connect_initialization()

    assert observed["timeout_during"] == _DISCORD_COMMAND_SYNC_MAX_RATE_LIMIT_SLEEP_SECONDS
    assert http.max_ratelimit_timeout is None  # restored
    assert "pre-existing-route" in http._buckets
    assert "busy-bucket" in http._buckets  # never orphan requests mid-flight
    assert "start-thread-bucket" not in http._buckets
    # The next request recreates it with the restored (None) cap, i.e. discord.py waits
    # out the window instead of raising RateLimited locally.
    assert http.get_ratelimit("start-thread-bucket")._max_ratelimit_timeout is None


@pytest.mark.asyncio
async def test_poisoned_bucket_raises_locally_but_clean_bucket_waits() -> None:
    """Grounding: this is the exact local-raise the leaked 30s cap causes."""
    RateLimited, Ratelimit = _real_ratelimit_classes()
    loop = asyncio.get_running_loop()

    poisoned = Ratelimit(_DISCORD_COMMAND_SYNC_MAX_RATE_LIMIT_SLEEP_SECONDS)
    poisoned.limit = poisoned.remaining = 10  # quota available; only the expiry matters
    poisoned.expires = loop.time() + 100
    with pytest.raises(RateLimited):
        await poisoned.acquire()

    clean = Ratelimit(None)
    clean.limit = clean.remaining = 10
    clean.expires = loop.time() + 100
    await clean.acquire()  # no pre-emptive raise — discord.py proceeds and self-paces
