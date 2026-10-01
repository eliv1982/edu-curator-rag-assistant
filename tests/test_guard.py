"""Sanity check for the offline guard in conftest.py."""

import os
import socket

import pytest


def test_openai_key_is_removed():
    assert "OPENAI_API_KEY" not in os.environ


def test_external_network_is_blocked():
    with pytest.raises(RuntimeError, match="blocked in tests"):
        socket.create_connection(("api.openai.com", 443), timeout=1)
