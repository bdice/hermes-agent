"""Auto-follow joins must arm voice mode on the reply-bound text channel.

The runner's auto-follow join handler binds replies to the user's last text channel, but voice
mode lived on whatever channel the last manual /voice join had armed. When those differ, spoken
turns in the bound channel answered text-only instead of playing TTS in the VC. The handler must
arm "all" on the channel it actually bound.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.platforms.base import SessionSource
from gateway.run_voice import GatewayVoiceMixin

_GUILD, _USER, _BOUND_TEXT = 1, 42, 700


class _Runner(GatewayVoiceMixin):
    def __init__(self) -> None:
        self._voice_mode: dict[str, str] = {}
        self.saved = 0

    def _save_voice_modes(self) -> None:
        self.saved += 1

    def _adapter_profile_for_source(self, source) -> None:
        return None  # default profile: key shape is "<platform>:<chat_id>"


@pytest.mark.asyncio
async def test_autofollow_join_arms_voice_mode_on_bound_channel() -> None:
    runner = _Runner()
    adapter = MagicMock()
    adapter._voice_auto_follow_bound = {_GUILD: _BOUND_TEXT}
    adapter._voice_auto_follow_user_id = {_GUILD: str(_USER)}
    adapter._owner_profile = None
    adapter._auto_tts_enabled_chats = set()
    adapter._auto_tts_disabled_chats = set()
    from gateway.config import Platform
    adapter.build_source.side_effect = lambda **kw: SessionSource(
        platform=Platform.DISCORD, chat_id=kw["chat_id"], chat_type=kw["chat_type"],
        user_id=kw["user_id"], guild_id=kw.get("guild_id"))
    channel = SimpleNamespace(guild=SimpleNamespace(id=_GUILD))

    async def _join(source, guild_id):
        return "joined"

    runner._join_voice_channel_for_source = AsyncMock(side_effect=_join)

    result = await runner._handle_voice_autofollow_join(channel, _BOUND_TEXT, adapter=adapter)

    assert result == "joined"
    runner._join_voice_channel_for_source.assert_awaited_once()
    # The binding channel — not a stale manual /voice join's channel — carries "all".
    assert runner._voice_mode == {"discord:700": "all"}
    # And the adapter's auto-TTS set mirrors it so _should_send_voice_reply fires.
    assert "700" in adapter._auto_tts_enabled_chats


@pytest.mark.asyncio
async def test_autofollow_join_rejects_stale_binding() -> None:
    runner = _Runner()
    adapter = MagicMock()
    adapter._voice_auto_follow_bound = {_GUILD: 800}  # bound elsewhere
    channel = SimpleNamespace(guild=SimpleNamespace(id=_GUILD))
    runner._join_voice_channel_for_source = AsyncMock()

    result = await runner._handle_voice_autofollow_join(channel, _BOUND_TEXT, adapter=adapter)

    assert result == ""
    runner._join_voice_channel_for_source.assert_not_awaited()
    assert runner._voice_mode == {}
