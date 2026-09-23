"""Regression tests for the migration advisory-lock leak (#4611).

A failure between ``pg_try_advisory_lock`` and the end of the migration used to
leave the session-level advisory lock behind: the ``finally`` block ran
``pg_advisory_unlock`` on a connection whose transaction had already aborted, so
the unlock raised ``InFailedSqlTransaction``, replaced the original error in the
log, and the lock stayed on the backend. Behind a transaction-mode pooler
(PgBouncer) that leaked lock blocks every later migrator until the pooler
recycles the backend.

These tests drive the real ``run_migrations`` lock/unlock path against a fake
SQLAlchemy connection and engine, mirroring the read-only ``CREATE EXTENSION``
failure from the issue.
"""

import logging
import re

import pytest
from psycopg2 import errors as pg_errors

from hindsight_api import migrations as migrations_module
from hindsight_api._pg_extensions import extension_schema
from hindsight_api._vector_index import bootstrap_extension


class _Scalar:
    def __init__(self, value):
        self._value = value

    def scalar(self):
        return self._value


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _Row:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row

    def fetchall(self):
        return [self._row] if self._row is not None else []


class FakeMigrationConnection:
    """Fake SQLAlchemy connection for the advisory-lock body of run_migrations.

    Answers the statements the lock body issues (catalog lookups, advisory
    locks, CREATE EXTENSION). Pass ``fail_on`` to make one statement raise,
    which aborts the (fake) transaction: every further execute then raises
    ``InFailedSqlTransaction``, like psycopg2 on a real aborted transaction.
    """

    def __init__(self, fail_on=None, installed=()):
        self.fail_on = fail_on
        if isinstance(installed, dict):
            self.installed = dict(installed)
        else:
            self.installed = {name: ("public", True) for name in installed}
        self.lock_acquired = False
        self.lock_released = False
        self.lock_released_after_abort = False
        self.commits = 0
        self.rollbacks = 0
        self.statements = []
        self.open_txn = False
        self._aborted = False

    def execute(self, statement, params=None, *args, **kwargs):
        sql = str(statement)
        self.statements.append(sql)
        self.open_txn = True
        if self._aborted:
            raise pg_errors.InFailedSqlTransaction(
                "current transaction is aborted, commands ignored until end of transaction block"
            )
        if self.fail_on and self.fail_on in sql:
            self._aborted = True
            raise pg_errors.ReadOnlySqlTransaction("cannot execute CREATE EXTENSION in a read-only transaction")
        if "pg_try_advisory_lock" in sql:
            self.lock_acquired = True
            return _Scalar(True)
        if "pg_advisory_unlock" in sql:
            self.lock_released = True
            self.lock_released_after_abort = self._aborted
            return _Scalar(True)
        if "FROM pg_extension" in sql:
            name = (params or {}).get("extension_name") or (params or {}).get("name")
            if name in self.installed:
                return _Row(self.installed[name])
            return _Row(None)
        if "= ANY(:names)" in sql:
            names = (params or {}).get("names") or []
            public = (params or {}).get("public") or "public"
            return _Rows(
                [
                    (name,)
                    for name, (schema, relocatable) in self.installed.items()
                    if name in names and schema != public and relocatable
                ]
            )
        match = re.search(r"CREATE EXTENSION IF NOT EXISTS (\w+)", sql)
        if match:
            self.installed[match.group(1)] = ("public", True)
        return _Scalar(None)

    def commit(self):
        self.open_txn = False
        # A COMMIT on an aborted transaction is a ROLLBACK in PostgreSQL.
        if self._aborted:
            self._aborted = False
            self.rollbacks += 1
        else:
            self.commits += 1

    def rollback(self):
        self.open_txn = False
        self._aborted = False
        self.rollbacks += 1


class _FakeEngine:
    """Replaces create_engine(...) so run_migrations uses the fake connection."""

    def __init__(self, conn):
        self._conn = conn
        self.disposed = False

    def connect(self):
        return _FakeCtx(self._conn)

    def dispose(self):
        self.disposed = True


class _FakeCtx:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, exc_type, exc, tb):
        return False


def _patch_engine(monkeypatch, conn):
    engine = _FakeEngine(conn)
    monkeypatch.setattr(migrations_module, "create_engine", lambda url, poolclass=None: engine)
    return engine


def _isolate_run_migrations(monkeypatch):
    """Keep run_migrations off the subprocess-isolation and real-migration paths."""
    monkeypatch.setattr(migrations_module, "_should_isolate_migrations", lambda: False)
    monkeypatch.setattr(migrations_module, "is_oracle_url", lambda url: False)
    monkeypatch.setattr(migrations_module, "to_libpq_url", lambda url: url)
    monkeypatch.setattr(migrations_module, "configured_vector_extension", lambda: "pgvector")
    monkeypatch.setattr(migrations_module, "_run_migrations_internal", lambda *a, **k: None)


def _error_chain(excinfo) -> list[str]:
    names = []
    err = excinfo.value
    while err is not None:
        names.append(type(err).__name__)
        err = err.__cause__
    return names


def test_lock_released_when_a_migration_fails(monkeypatch):
    """The #4611 core: any failure inside the lock body must still release the lock.

    A failed statement aborts the transaction; the pre-fix finally ran
    pg_advisory_unlock on the aborted transaction, which raised
    InFailedSqlTransaction and left the lock on the backend. The fix rolls back
    first, so the unlock succeeds.
    """
    conn = FakeMigrationConnection(installed={"vector", "pg_trgm"})
    _patch_engine(monkeypatch, conn)
    _isolate_run_migrations(monkeypatch)

    def failing_internal(*a, **k):
        # A failed migration statement leaves the transaction aborted.
        conn.execute("SELECT count(*) FROM nonexistent_table")
        raise pg_errors.InternalError('relation "nonexistent" does not exist')

    monkeypatch.setattr(migrations_module, "_run_migrations_internal", failing_internal)

    with pytest.raises(RuntimeError):
        migrations_module.run_migrations("postgresql://user:***@host/db")

    assert conn.lock_acquired, "the lock must be acquired before the failure"
    assert conn.rollbacks >= 1, "a rollback must run before the unlock"
    assert conn.lock_released, "the advisory lock must be released despite the failure"
    assert not conn.lock_released_after_abort, "the unlock must not run on an aborted txn"


def test_original_error_is_not_masked_by_the_unlock(monkeypatch):
    """The migration's own error must surface, not InFailedSqlTransaction."""
    conn = FakeMigrationConnection(installed={"vector", "pg_trgm"})
    _patch_engine(monkeypatch, conn)
    _isolate_run_migrations(monkeypatch)

    def failing_internal(*a, **k):
        conn.execute("SELECT count(*) FROM nonexistent_table")
        raise pg_errors.InternalError('relation "nonexistent_table" does not exist')

    monkeypatch.setattr(migrations_module, "_run_migrations_internal", failing_internal)

    with pytest.raises(RuntimeError) as excinfo:
        migrations_module.run_migrations("postgresql://user:***@host/db")

    causes = _error_chain(excinfo)
    assert any("InternalError" in c for c in causes), causes
    assert not any("InFailedSqlTransaction" in c for c in causes), causes


def test_read_only_backend_skips_noop_ddl_and_succeeds(monkeypatch):
    """The #4611 trigger: installed extension on a read-only session.

    CREATE EXTENSION IF NOT EXISTS on an installed extension is a no-op on a
    writable session but an error on a read-only one (PgBouncer transaction
    mode carries default_transaction_read_only across clients). With the fix
    the pointless DDL is skipped and the migration run succeeds.
    """
    conn = FakeMigrationConnection(installed={"vector"}, fail_on="CREATE EXTENSION IF NOT EXISTS vector")
    _patch_engine(monkeypatch, conn)
    _isolate_run_migrations(monkeypatch)

    migrations_module.run_migrations("postgresql://user:***@host/db")

    assert conn.lock_acquired
    assert conn.lock_released
    assert not any("CREATE EXTENSION" in s for s in conn.statements), (
        "no CREATE EXTENSION may be issued when the extension is already installed"
    )


def test_unlock_failure_does_not_mask_the_original_error(monkeypatch, caplog):
    """If the rollback path itself fails, the unlock failure is logged, not raised.

    The surfaced error is the one the migration body raised (here the first
    failing rollback, OperationalError) — never an InFailedSqlTransaction from
    the finally block.
    """
    conn = FakeMigrationConnection(fail_on="CREATE EXTENSION IF NOT EXISTS vector")
    _patch_engine(monkeypatch, conn)
    _isolate_run_migrations(monkeypatch)

    def broken_rollback():
        conn.rollbacks += 1
        raise pg_errors.OperationalError("connection already closed")

    conn.rollback = broken_rollback

    with pytest.raises(RuntimeError) as excinfo:
        migrations_module.run_migrations("postgresql://user:***@host/db")

    causes = _error_chain(excinfo)
    assert not any("InFailedSqlTransaction" in c for c in causes), causes
    release_failures = [r for r in caplog.records if "Failed to release migration advisory lock" in r.getMessage()]
    assert release_failures, "an unlock failure must be logged so the lost lock is discoverable"


def test_run_migrations_succeeds_on_a_healthy_backend(monkeypatch):
    """Happy path still works: lock acquired, migrations run, lock released."""
    conn = FakeMigrationConnection(installed={"vector", "pg_trgm"})
    _patch_engine(monkeypatch, conn)
    _isolate_run_migrations(monkeypatch)

    migrations_module.run_migrations("postgresql://user:pass@host/db")

    assert conn.lock_acquired
    assert conn.lock_released
    assert not conn.lock_released_after_abort


def test_bootstrap_extension_skips_ddl_when_installed():
    """Fix 2: bootstrap must not run CREATE EXTENSION on an installed extension.

    On a read-only session that statement fails even though it would do
    nothing, aborting the transaction and triggering the lock leak.
    """
    conn = _BootstrapRecordingConnection(installed={"vector", "pg_trgm"})

    bootstrap_extension(conn, "pgvector")

    assert not any("CREATE EXTENSION" in s for s in conn.statements), (
        "CREATE EXTENSION must be skipped when the extension is already installed"
    )


def test_bootstrap_extension_still_installs_when_missing():
    conn = _BootstrapRecordingConnection(installed=set())

    bootstrap_extension(conn, "pgvector")

    assert any("CREATE EXTENSION IF NOT EXISTS vector" in s for s in conn.statements)


class _BootstrapRecordingConnection:
    """Minimal connection recording statements, for the bootstrap unit tests."""

    def __init__(self, installed):
        if isinstance(installed, dict):
            self.installed = dict(installed)
        else:
            self.installed = {name: ("public", True) for name in installed}
        self.statements = []

    def execute(self, statement, params=None, *args, **kwargs):
        sql = str(statement)
        self.statements.append(sql)
        if "FROM pg_extension" in sql:
            name = (params or {}).get("name")
            if name in self.installed:
                return _Row(self.installed[name])
            return _Row(None)
        if "current_setting('search_path')" in sql:
            return _Scalar('"$user", public')
        if "set_config('search_path'" in sql:
            return _Scalar(None)
        match = re.search(r"CREATE EXTENSION IF NOT EXISTS (\w+)", sql)
        if match:
            self.installed[match.group(1)] = ("public", True)
        return _Scalar(None)

    def commit(self):
        pass

    def rollback(self):
        pass


class _WaitingConnection(FakeMigrationConnection):
    """A migrator that finds the lock taken by another worker at first."""

    def __init__(self, locks_error=None):
        super().__init__(installed={"vector", "pg_trgm"})
        self.other_holder = True
        self.polls = 0
        self.holder_lookups = 0
        self.locks_error = locks_error

    def execute(self, statement, params=None, *args, **kwargs):
        sql = str(statement)
        if "pg_try_advisory_lock" in sql and not self._aborted:
            self.polls += 1
            if self.other_holder:
                return _Scalar(False)
        if "FROM pg_locks" in sql:
            self.holder_lookups += 1
            self.statements.append(sql)
            self.open_txn = True
            if self.locks_error is not None:
                self._aborted = True
                raise self.locks_error
            return _Row((4242, "hindsight-api", "10.0.0.7", "CREATE INDEX CONCURRENTLY ..."))
        return super().execute(statement, params, *args, **kwargs)


def test_waiting_worker_polls_and_commits_between_polls(monkeypatch):
    """A worker that can't get the lock keeps polling, without an open txn."""
    conn = _WaitingConnection()
    _patch_engine(monkeypatch, conn)
    _isolate_run_migrations(monkeypatch)

    def fake_sleep(_seconds):
        conn.other_holder = False

    monkeypatch.setattr(migrations_module.time, "sleep", fake_sleep)

    migrations_module.run_migrations("postgresql://user:pass@host/db")

    assert conn.polls >= 2, "the worker must poll until the lock is free"
    assert conn.commits >= 1, "the worker must commit between polls"


def test_waiting_worker_logs_while_polling(monkeypatch, caplog):
    """Fix 3: a worker waiting on a held lock must say so, not look hung."""
    conn = _WaitingConnection()
    _patch_engine(monkeypatch, conn)
    _isolate_run_migrations(monkeypatch)

    def fake_sleep(_seconds):
        conn.other_holder = False

    monkeypatch.setattr(migrations_module.time, "sleep", fake_sleep)

    with caplog.at_level(logging.INFO, logger="hindsight_api.migrations"):
        migrations_module.run_migrations("postgresql://user:pass@host/db")

    waiting_logs = [r for r in caplog.records if "waiting" in r.getMessage().lower()]
    assert waiting_logs, "a worker polling for the lock must log while waiting"
    message = waiting_logs[0].getMessage()
    assert "pid=4242" in message, "the log must name the backend holding the lock"
    # The remedy it points at has to be a knob that exists.
    assert "HINDSIGHT_API_MIGRATION_DATABASE_URL" in message


def test_waiting_worker_holds_no_snapshot_across_the_sleep(monkeypatch):
    """The holder lookup must not reopen a txn that spans the poll sleep.

    The commit in the wait loop exists so this connection holds no snapshot
    while waiting, which would otherwise block a CREATE INDEX CONCURRENTLY in
    the migration worker.  A diagnostic SELECT issued after that commit
    quietly undoes it.
    """
    conn = _WaitingConnection()
    _patch_engine(monkeypatch, conn)
    _isolate_run_migrations(monkeypatch)

    seen = []

    def fake_sleep(_seconds):
        seen.append(conn.open_txn)
        conn.other_holder = False

    monkeypatch.setattr(migrations_module.time, "sleep", fake_sleep)

    migrations_module.run_migrations("postgresql://user:pass@host/db")

    assert conn.holder_lookups >= 1, "the holder lookup must actually run"
    assert seen and not any(seen), "no transaction may be open while the worker sleeps"


def test_failed_holder_lookup_does_not_break_the_wait_loop(monkeypatch, caplog):
    """A best-effort diagnostic must never turn a wait into a failed migration.

    A failing pg_locks lookup leaves the transaction aborted; without a
    rollback the next pg_try_advisory_lock raises InFailedSqlTransaction and
    the migration fails although the lock was merely busy.
    """
    conn = _WaitingConnection(locks_error=pg_errors.InsufficientPrivilege("permission denied for view pg_locks"))
    _patch_engine(monkeypatch, conn)
    _isolate_run_migrations(monkeypatch)

    def fake_sleep(_seconds):
        conn.other_holder = False

    monkeypatch.setattr(migrations_module.time, "sleep", fake_sleep)

    with caplog.at_level(logging.DEBUG, logger="hindsight_api.migrations"):
        migrations_module.run_migrations("postgresql://user:pass@host/db")

    assert conn.lock_acquired and conn.lock_released
    waiting_logs = [r for r in caplog.records if "waiting" in r.getMessage().lower()]
    assert waiting_logs, "the wait is still reported when the holder cannot be identified"


def test_extension_schema_helper_contract():
    """Guard the helper fix 2 relies on: installed -> schema, absent -> None."""
    assert extension_schema(_BootstrapRecordingConnection(installed={"vector"}), "vector") == "public"
    assert extension_schema(_BootstrapRecordingConnection(installed=set()), "vector") is None
