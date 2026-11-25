"""A custom endpoint's declared per-model output cap rides on requests that set no max_tokens."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from agent import model_metadata as mm
from agent.auxiliary_client import _build_call_kwargs

_PROXY = "https://inference-proxy.example/v1"
_CATALOG = {
    "big": {"name": "big", "context_length": 1_000_000, "max_completion_tokens": 128_000},
    "small": {"name": "small", "context_length": 200_000, "max_completion_tokens": 16_384},
    "undeclared": {"name": "undeclared", "context_length": 200_000},
    "whole-window": {"name": "whole-window", "context_length": 32_768, "max_completion_tokens": 32_768},
}


@pytest.fixture
def catalog():
    with patch.object(mm, "fetch_endpoint_model_metadata", return_value=_CATALOG) as fetch:
        yield fetch


@pytest.mark.parametrize(
    ("model", "expected"),
    [("big", 128_000), ("small", 16_384), ("undeclared", None), ("whole-window", None), ("missing", None)],
)
def test_declared_cap_is_per_model(catalog, model, expected) -> None:
    assert mm.declared_max_output_tokens(model, _PROXY, "key") == expected


@pytest.mark.parametrize(
    "base_url",
    ["https://openrouter.ai/api/v1", "https://integrate.api.nvidia.com/v1", "http://127.0.0.1:8080/v1", ""],
)
def test_known_and_local_endpoints_are_not_probed(catalog, base_url) -> None:
    assert mm.declared_max_output_tokens("big", base_url) is None
    catalog.assert_not_called()


def _aux_cap(model: str, max_tokens: int | None = None) -> int | None:
    kwargs = _build_call_kwargs(
        "custom", model, [{"role": "user", "content": "hi"}], max_tokens=max_tokens,
        base_url=_PROXY, task="compression", api_key="key",
    )
    return kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")


def test_aux_request_without_cap_sends_the_declared_cap(catalog) -> None:
    assert _aux_cap("big") == 128_000
    assert _aux_cap("small") == 16_384
    assert _aux_cap("undeclared") is None


def test_aux_caller_cap_is_not_replaced(catalog) -> None:
    # Custom routes do not forward caller caps; the declared limit must not sneak in instead.
    assert _aux_cap("big", max_tokens=1500) is None


def _main_cap(monkeypatch, model: str, base_url: str = _PROXY, max_tokens: int | None = None) -> int | None:
    from tests.agent.test_provider_parity import _make_agent

    agent = _make_agent(monkeypatch, "custom", base_url=base_url, model=model)
    agent.max_tokens = max_tokens
    kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
    return kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")


def test_main_request_without_cap_sends_the_declared_cap(monkeypatch, catalog) -> None:
    assert _main_cap(monkeypatch, "big") == 128_000
    assert _main_cap(monkeypatch, "small") == 16_384
    assert _main_cap(monkeypatch, "undeclared") is None


def test_main_explicit_cap_wins(monkeypatch, catalog) -> None:
    assert _main_cap(monkeypatch, "big", max_tokens=2048) == 2048
