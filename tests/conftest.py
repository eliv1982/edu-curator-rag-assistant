"""
Shared test setup. Every test runs offline and without an OpenAI key.
"""

import socket
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _is_local(host) -> bool:
    if isinstance(host, bytes):
        host = host.decode(errors="replace")
    return host is None or host in _LOCAL_HOSTS or str(host).startswith("127.")


@pytest.fixture(autouse=True)
def offline_no_key(monkeypatch):
    """
    Guarantee that no test can reach OpenAI (or any non-local host).

    - OPENAI_API_KEY is removed, so nothing can authenticate by accident.
    - SHOW_RECENT_QUESTIONS is removed, so tests start from the privacy default.
    - Non-loopback connections and DNS lookups raise immediately. Loopback is
      allowed because asyncio uses it internally (socketpair on Windows).
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SHOW_RECENT_QUESTIONS", raising=False)

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def guarded_connect(self, address):
        if isinstance(address, tuple) and not _is_local(address[0]):
            raise RuntimeError(f"Network access is blocked in tests: {address!r}")
        return real_connect(self, address)

    def guarded_connect_ex(self, address):
        if isinstance(address, tuple) and not _is_local(address[0]):
            raise RuntimeError(f"Network access is blocked in tests: {address!r}")
        return real_connect_ex(self, address)

    def guarded_getaddrinfo(host, *args, **kwargs):
        if not _is_local(host):
            raise RuntimeError(f"DNS lookup is blocked in tests: {host!r}")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def assignments_csv(tmp_path, monkeypatch):
    """
    Factory: write an assignments CSV into tmp_path and point rag_pipeline at it.

    Usage: assignments_csv(rows) or assignments_csv(rows, header=..., encoding=...).
    Each row is a (module, assignment, deadline, status, description) tuple.
    """
    from src import rag_pipeline

    def write(
        rows,
        header="module,assignment,deadline,status,description",
        encoding="utf-8",
    ):
        path = tmp_path / "assignments.csv"
        lines = [header] + [",".join(f'"{cell}"' for cell in row) for row in rows]
        path.write_text("\n".join(lines) + "\n", encoding=encoding, newline="")
        monkeypatch.setattr(rag_pipeline, "ASSIGNMENTS_CSV", path)
        return path

    return write
