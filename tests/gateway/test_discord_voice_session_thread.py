"""One thread per voice conversation session.

The join binds voice replies to a text channel, so a whole spoken conversation — transcript
echoes and replies — lands in the channel stream. The first utterance of a session must open a
single thread under the bound channel, move the binding there (dropping the stale join-time
source so turns route to the thread's session), and reuse it for every later utterance. Leaving
the voice channel ends the session: the next conversation opens a fresh thread. Thread creation
failure keeps replies inline rather than dropping the turn.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter, VoiceReceiver

_GUILD, _USER, _PARENT, _THREAD = 1, 42, 700, 900


class _FakeThread:
    def __init__(self, channel_id: int) -> None:
        self.id = channel_id


def _adapter_with_client():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake"))
    parent = SimpleNamespace(
        id=_PARENT, name="hermes",
        create_thread=AsyncMock(return_value=_FakeThread(_THREAD)))
    parent.parent = None  # a text channel's parent is itself; set post-construction
    thread = SimpleNamespace(id=_THREAD, name="voice", parent=parent)
    client = MagicMock()
    client.get_channel.side_effect = lambda cid: {
        _PARENT: parent, _THREAD: thread}.get(cid)
    adapter._client = client
    return adapter, parent, thread


def _patch_stt(transcript: str):
    from contextlib import ExitStack
    import unittest.mock as m

    stack = ExitStack()
    stack.enter_context(m.patch.object(VoiceReceiver, "pcm_to_wav"))
    stack.enter_context(m.patch("tools.transcription_tools.transcribe_audio",
                                return_value={"success": True, "transcript": transcript}))
    stack.enter_context(m.patch("tools.voice_mode_transcript.is_whisper_hallucination",
                                return_value=False))
    return stack


class _FakeDMChannel:
    pass


def _patch_discord_types(monkeypatch) -> None:
    import discord
    monkeypatch.setattr(discord, "Thread", _FakeThread, raising=False)
    monkeypatch.setattr(discord, "DMChannel", _FakeDMChannel, raising=False)


@pytest.mark.asyncio
async def test_first_utterance_opens_session_thread_and_rebinds(monkeypatch) -> None:
    _patch_discord_types(monkeypatch)
    adapter, parent, _thread = _adapter_with_client()
    adapter._voice_text_channels[_GUILD] = _PARENT
    adapter._voice_sources[_GUILD] = {"chat_id": str(_PARENT)}  # join-time parent source
    adapter._voice_input_callback = callback = AsyncMock()

    with _patch_stt("what's the score"):
        await adapter._process_voice_input(_GUILD, _USER, b"\x00" * 9600, _PARENT)

    parent.create_thread.assert_awaited_once()
    assert adapter._voice_session_threads[_GUILD] == _THREAD
    assert adapter._voice_text_channels[_GUILD] == _THREAD
    assert _GUILD not in adapter._voice_sources  # stale parent source dropped
    callback.assert_awaited_once()


@pytest.mark.asyncio
async def test_later_utterances_reuse_the_session_thread(monkeypatch) -> None:
    _patch_discord_types(monkeypatch)
    adapter, parent, _thread = _adapter_with_client()
    adapter._voice_text_channels[_GUILD] = _THREAD
    adapter._voice_session_threads[_GUILD] = _THREAD
    adapter._voice_input_callback = AsyncMock()

    with _patch_stt("second turn"):
        await adapter._process_voice_input(_GUILD, _USER, b"\x00" * 9600, _THREAD)

    parent.create_thread.assert_not_awaited()
    assert adapter._voice_text_channels[_GUILD] == _THREAD


@pytest.mark.asyncio
async def test_leave_ends_the_session_so_next_conversation_threads_fresh() -> None:
    adapter, _parent, _thread = _adapter_with_client()
    adapter._voice_text_channels[_GUILD] = _THREAD
    adapter._voice_sources[_GUILD] = {"chat_id": str(_THREAD)}
    adapter._voice_session_threads[_GUILD] = _THREAD

    await adapter.leave_voice_channel(_GUILD)

    assert _GUILD not in adapter._voice_session_threads
    assert _GUILD not in adapter._voice_text_channels


@pytest.mark.asyncio
async def test_thread_creation_failure_keeps_turns_inline(monkeypatch) -> None:
    import discord
    monkeypatch.setattr(discord, "Thread", _FakeThread, raising=False)
    monkeypatch.setattr(discord, "DMChannel", object, raising=False)
    adapter, parent, _thread = _adapter_with_client()
    parent.create_thread = AsyncMock(side_effect=RuntimeError("forum full"))
    adapter._voice_text_channels[_GUILD] = _PARENT
    adapter._voice_sources[_GUILD] = {"chat_id": str(_PARENT)}
    adapter._voice_input_callback = callback = AsyncMock()

    with _patch_stt("still works"):
        await adapter._process_voice_input(_GUILD, _USER, b"\x00" * 9600, _PARENT)

    assert adapter._voice_text_channels[_GUILD] == _PARENT  # binding untouched
    assert _GUILD not in adapter._voice_session_threads
    callback.assert_awaited_once()  # the turn is not lost
