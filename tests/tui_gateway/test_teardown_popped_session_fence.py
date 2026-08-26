"""Regression tests for post-flush forced-reap lease fencing."""

import contextlib
import os
import sqlite3
import threading
from types import SimpleNamespace

import pytest


class _TurnThread:
    def __init__(self, *, settle_on_grace_join=False):
        self._alive = True
        self._settle = settle_on_grace_join
        self.join_calls = []
        self._lock = threading.Lock()

    def is_alive(self):
        with self._lock:
            return self._alive

    def join(self, timeout=None):
        with self._lock:
            self.join_calls.append(timeout)
            if self._settle:
                self._alive = False


@contextlib.contextmanager
def _open_db(db_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path)
    try:
        yield db
    finally:
        db.close()


def _make_agent(session_id, holder, db_path, *, tail="tail"):
    from hermes_state import SessionDB

    def persist(messages):
        with SessionDB(db_path) as db:
            db.append_messages_batch(
                session_id=session_id,
                messages=messages,
                turn_lease_holder=holder,
                turn_lease_ttl_seconds=300.0,
            )

    return SimpleNamespace(
        session_id=session_id,
        model="test-model",
        platform="tui",
        _active_session_turn_lease_holder=holder,
        _session_messages=[{"role": "assistant", "content": tail}],
        _persist_session=persist,
        commit_memory_session=lambda history: None,
        close=lambda: None,
    )


def _make_session(agent, turn, *, session_key="sess-fence"):
    return {
        "agent": agent,
        "history": [],
        "history_lock": threading.Lock(),
        "session_key": session_key,
        "_finalized": False,
        "_run_thread": turn,
        "source": "tui",
    }


def _contents(db_path, session_id):
    with sqlite3.connect(db_path) as conn:
        return [row[0] for row in conn.execute(
            "SELECT content FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        )]


def test_force_reap_flushes_then_fences_stale_holder(tmp_path, monkeypatch):
    """A wedged turn's buffered tail lands before its holder is fenced."""
    from hermes_state import SessionDB, SessionTurnLeaseLostError
    from tui_gateway import server as srv

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    db.create_session("sess-fence", source="tui")
    holder = f"pid={os.getpid()}:turn=wedged"
    assert db.try_acquire_session_turn_lease(
        "sess-fence", holder, ttl_seconds=300.0
    )
    session = _make_session(
        _make_agent("sess-fence", holder, db_path),
        _TurnThread(settle_on_grace_join=False),
    )
    monkeypatch.setattr(srv, "_session_db", lambda s: _open_db(db_path))

    assert srv._teardown_popped_session(
        session, end_reason="ws_orphan_reap"
    ) is True
    assert "tail" in _contents(db_path, "sess-fence")
    assert db.get_session("sess-fence")["ended_at"] is not None

    with pytest.raises(SessionTurnLeaseLostError):
        db.append_messages_batch(
            session_id="sess-fence",
            messages=[{"role": "assistant", "content": "late"}],
            turn_lease_holder=holder,
            turn_lease_ttl_seconds=300.0,
        )
    assert db.try_acquire_session_turn_lease(
        "sess-fence", "pid=other:turn=next", ttl_seconds=300.0
    )
    db.release_session_turn_lease("sess-fence", "pid=other:turn=next")


def test_thread_settles_within_grace_does_not_fence_owner(tmp_path, monkeypatch):
    """A turn that settles inside grace keeps its live owner lease."""
    from hermes_state import SessionDB
    from tui_gateway import server as srv

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    db.create_session("sess-fence", source="tui")
    holder = f"pid={os.getpid()}:turn=healthy"
    assert db.try_acquire_session_turn_lease(
        "sess-fence", holder, ttl_seconds=300.0
    )
    session = _make_session(
        _make_agent("sess-fence", holder, db_path),
        _TurnThread(settle_on_grace_join=True),
    )
    monkeypatch.setattr(srv, "_session_db", lambda s: _open_db(db_path))

    assert srv._teardown_popped_session(
        session, end_reason="ws_orphan_reap"
    ) is True
    assert db.try_acquire_session_turn_lease(
        "sess-fence", "pid=other:turn=next", ttl_seconds=5.0
    ) is False
    assert "tail" in _contents(db_path, "sess-fence")
    db.release_session_turn_lease("sess-fence", holder)


def test_end_session_holder_delete_does_not_revoke_successor(tmp_path):
    """The atomic end operation is holder-qualified and idempotent."""
    from hermes_state import SessionDB, SessionTurnLeaseLostError

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    db.create_session("sess-fence", source="tui")
    old = "pid=old:turn=wedged"
    new = "pid=new:turn=successor"
    assert db.try_acquire_session_turn_lease(
        "sess-fence", old, ttl_seconds=300.0
    )
    db.end_session("sess-fence", "ws_orphan_reap", turn_lease_holder=old)
    assert db.try_acquire_session_turn_lease(
        "sess-fence", new, ttl_seconds=300.0
    )
    db.end_session("sess-fence", "ws_orphan_reap", turn_lease_holder=old)
    assert db.append_messages_batch(
        session_id="sess-fence",
        messages=[{"role": "assistant", "content": "successor"}],
        turn_lease_holder=new,
        turn_lease_ttl_seconds=300.0,
    ) == 1
    with pytest.raises(SessionTurnLeaseLostError):
        db.append_messages_batch(
            session_id="sess-fence",
            messages=[{"role": "assistant", "content": "old"}],
            turn_lease_holder=old,
            turn_lease_ttl_seconds=300.0,
        )
    db.release_session_turn_lease("sess-fence", new)
