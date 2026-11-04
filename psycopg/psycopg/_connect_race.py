"""
Race several connection attempts concurrently.

When a connection is requested to several endpoints (for instance because
several hosts are specified in the connection string), normally the endpoints
are tried one after the other. The functions in this module allow instead to
start all the attempts concurrently and to return the first one that succeeds,
while all the losing connections are disposed of as soon as the race ends
(including on timeout and on interruption).

Both a blocking and an asyncio implementation are provided; they share the
same semantics:

- the attempts are started in the order produced by
  `~psycopg.conninfo.conninfo_attempts()` (which is random if
  `load_balance_hosts=random` is requested), and run concurrently within a
  single shared timeout;
- if `target_session_attrs=prefer-standby` is requested, the standby attempts
  race first and the attempts accepting any server type only race if no
  standby could be obtained, so that a primary cannot win while a standby is
  available;
- on total failure, the errors of every attempt are reported in the stable
  order in which the attempts were planned, regardless of completion order.
"""

# Copyright (C) 2025 The Psycopg Team

from __future__ import annotations

import asyncio
import logging
import selectors
from time import monotonic
from typing import TYPE_CHECKING, Any

from . import errors as e
from . import generators, pq, waiting
from ._enums import Ready as Ready
from ._enums import Wait as Wait
from .conninfo import make_conninfo
from ._conninfo_utils import get_param

if TYPE_CHECKING:
    from .abc import ConnDict, ConnMapping
    from ._connection_base import BaseConnection

WAIT_R = Wait.R
WAIT_W = Wait.W
READY_NONE = Ready.NONE
READY_R = Ready.R
READY_W = Ready.W

logger = logging.getLogger("psycopg")


def race_connect(
    cls: type[BaseConnection[Any]],
    attempts: list[ConnDict],
    params: ConnMapping,
    *,
    timeout: float,
    interval: float,
) -> tuple[BaseConnection[Any] | None, list[tuple[e.Error, str]]]:
    """
    Run the connection *attempts* concurrently until the first success.

    :return: the winning connection, or `!None` if no attempt succeeded,
        together with the errors of the failed attempts, in attempts order.

    All the attempts share a single deadline, *timeout* seconds from the start
    of the race: the timeout is not multiplied by the number of attempts.
    """
    deadline = monotonic() + timeout
    conn_errors: list[tuple[e.Error, str]] = []
    rv: BaseConnection[Any] | None = None
    for wave in _race_waves(attempts, params):
        rv, wave_errors = _race_wave_sync(cls, wave, deadline, interval)
        conn_errors.extend(wave_errors)
        if rv is not None:
            return rv, conn_errors
    return None, conn_errors


async def race_connect_async(
    cls: type[BaseConnection[Any]],
    attempts: list[ConnDict],
    params: ConnMapping,
    *,
    timeout: float,
    interval: float,
) -> tuple[BaseConnection[Any] | None, list[tuple[e.Error, str]]]:
    """
    Async version of `race_connect()`.
    """
    deadline = monotonic() + timeout
    conn_errors: list[tuple[e.Error, str]] = []
    rv: BaseConnection[Any] | None = None
    for wave in _race_waves(attempts, params):
        rv, wave_errors = await _race_wave_async(cls, wave, deadline, interval)
        conn_errors.extend(wave_errors)
        if rv is not None:
            return rv, conn_errors
    return None, conn_errors


def _race_waves(attempts: list[ConnDict], params: ConnMapping) -> list[list[ConnDict]]:
    """
    Split the attempts in preference waves to race.

    With `target_session_attrs=prefer-standby`, the attempts list is composed
    of a first set only accepting standbys, followed by a set accepting any
    server; race the former first to preserve the preference.
    """
    if get_param(params, "target_session_attrs") == "prefer-standby":
        n = len(attempts) // 2
        if n and len(attempts) == 2 * n:
            return [attempts[:n], attempts[n:]]
    return [attempts]


def _attempt_descr(attempt: ConnDict) -> str:
    tdescr = (attempt.get("host"), attempt.get("port"), attempt.get("hostaddr"))
    return "host: %r, port: %r, hostaddr: %r" % tdescr


def _timeout_error() -> e.ConnectionTimeout:
    return e.ConnectionTimeout("connection timeout expired")


# Sync implementation


class _SyncAttempt:
    """
    State of one connection attempt driven by a selector.

    The attempt owns its `!PGconn` until it wins the race, in which case the
    ownership is transferred to the returned connection. In every other case
    the connection is finished in `release()`.
    """

    __slots__ = (
        "conninfo",
        "descr",
        "pgconn",
        "gen",
        "fileno",
        "events",
        "registered",
        "done",
        "succeeded",
        "error",
    )

    def __init__(self, conninfo: str, descr: str):
        self.conninfo = conninfo
        self.descr = descr
        self.pgconn = pq.PGconn.connect_start(conninfo.encode())
        self.gen = generators._connect_poll(self.pgconn, conninfo)
        self.fileno = -1
        self.events = 0
        self.registered = False
        self.done = False
        self.succeeded = False
        self.error: e.Error | None = None
        logger.debug("connection attempt: %s", descr)

    def _wait_events(self, wait: int) -> int:
        events = 0
        if wait & WAIT_R:
            events |= selectors.EVENT_READ
        if wait & WAIT_W:
            events |= selectors.EVENT_WRITE
        return events

    def _register(self, sel: selectors.BaseSelector, fileno: int, wait: int) -> None:
        self.fileno = fileno
        self.events = self._wait_events(wait)
        sel.register(fileno, self.events, self)
        self.registered = True

    def _unregister(self, sel: selectors.BaseSelector) -> None:
        if self.registered:
            sel.unregister(self.fileno)
            self.registered = False

    def _set_wait(self, sel: selectors.BaseSelector, fileno: int, wait: int) -> None:
        events = self._wait_events(wait)
        if self.registered and fileno == self.fileno and events == self.events:
            return
        self._unregister(sel)
        self._register(sel, fileno, wait)

    def start(self, sel: selectors.BaseSelector) -> None:
        """Perform the first poll and register the connection socket."""
        try:
            fileno, wait = next(self.gen)
        except StopIteration:
            self.done = True
            self.succeeded = True
            logger.debug("connection succeeded: %s", self.descr)
        except e.Error as ex:
            self.done = True
            self.error = ex
            logger.debug("connection failed: %s: %s", self.descr, ex)
        else:
            self._register(sel, fileno, wait)

    def advance(self, sel: selectors.BaseSelector, ready: int) -> None:
        """Resume the generator once its socket is reported ready."""
        try:
            fileno, wait = self.gen.send(ready)
        except StopIteration:
            self._unregister(sel)
            self.done = True
            self.succeeded = True
            logger.debug("connection succeeded: %s", self.descr)
        except e.Error as ex:
            self._unregister(sel)
            self.done = True
            self.error = ex
            logger.debug("connection failed: %s: %s", self.descr, ex)
        else:
            self._set_wait(sel, fileno, wait)

    def release(self, sel: selectors.BaseSelector) -> None:
        """Close a connection that didn't win the race."""
        self._unregister(sel)
        # finish() is idempotent: the generator might have called it already
        # on failure.
        self.pgconn.finish()


def _race_wave_sync(
    cls: type[BaseConnection[Any]],
    wave: list[ConnDict],
    deadline: float,
    interval: float,
) -> tuple[BaseConnection[Any] | None, list[tuple[e.Error, str]]]:
    states = [
        _SyncAttempt(make_conninfo("", **attempt), _attempt_descr(attempt))
        for attempt in wave
    ]

    winner: _SyncAttempt | None = None
    with selectors.DefaultSelector() as sel:
        try:
            for st in states:
                st.start(sel)

            while winner is None:
                # An attempt might complete without blocking, at start time.
                winner = next((st for st in states if st.succeeded), None)
                if winner is not None:
                    break

                pending = [st for st in states if not st.done]
                if not pending:
                    break

                remaining = deadline - monotonic()
                if remaining <= 0:
                    break

                events = sel.select(timeout=min(interval, remaining))

                if not events:
                    # No socket reported ready: nudge every pending attempt
                    # with READY_NONE, as wait_conn() does, then re-check the
                    # deadline.
                    for st in pending:
                        st.advance(sel, READY_NONE)
                    continue

                ready_by_state: dict[int, int] = {}
                for key, mask in events:
                    st = key.data
                    ready = 0
                    if mask & selectors.EVENT_READ:
                        ready |= READY_R
                    if mask & selectors.EVENT_WRITE:
                        ready |= READY_W
                    ready_by_state[id(st)] = ready_by_state.get(id(st), 0) | ready

                # Resolve the ties between attempts completing in the same
                # poll in attempts order, which is randomised if requested.
                for st in pending:
                    ready = ready_by_state.get(id(st))
                    if ready is not None:
                        st.advance(sel, ready)
                        if st.succeeded:
                            winner = st
                            break

        except BaseException:
            # On interruption (Ctrl-C, SystemExit, ...) release every
            # in-flight connection immediately before propagating.
            for st in states:
                st.release(sel)
            raise

        rv: BaseConnection[Any] | None
        if winner is not None:
            rv = cls(winner.pgconn)
        else:
            rv = None

        wave_errors: list[tuple[e.Error, str]] = []
        for st in states:
            if st is winner:
                continue
            if st.error is not None:
                wave_errors.append((st.error, st.descr))
            elif not st.done:
                # The attempt was still in progress when the deadline expired.
                tex = _timeout_error()
                logger.debug("connection failed: %s: %s", st.descr, tex)
                wave_errors.append((tex, st.descr))
            st.release(sel)

    return rv, wave_errors


# Async implementation


class _AsyncAttempt:
    """
    State of one connection attempt racing in an asyncio loop.
    """

    __slots__ = ("conninfo", "descr", "pgconn")

    def __init__(self, conninfo: str, descr: str):
        self.conninfo = conninfo
        self.descr = descr
        self.pgconn = pq.PGconn.connect_start(conninfo.encode())
        logger.debug("connection attempt: %s", descr)


async def _race_wave_async(
    cls: type[BaseConnection[Any]],
    wave: list[ConnDict],
    deadline: float,
    interval: float,
) -> tuple[BaseConnection[Any] | None, list[tuple[e.Error, str]]]:
    states = [
        _AsyncAttempt(make_conninfo("", **attempt), _attempt_descr(attempt))
        for attempt in wave
    ]

    async def attempt(st: _AsyncAttempt) -> pq.PGconn:
        gen = generators._connect_poll(st.pgconn, st.conninfo)
        return await waiting.wait_conn_async(
            gen, interval=interval, timeout=max(0.0, deadline - monotonic())
        )

    tasks = [asyncio.create_task(attempt(st)) for st in states]
    index = {t: i for i, t in enumerate(tasks)}
    winner_idx = -1

    try:
        pending: set[asyncio.Task[Any]] = set(tasks)
        while pending and winner_idx < 0:
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            done, pending = await asyncio.wait(
                pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
            )
            for t in done:
                i = index[t]
                try:
                    t.result()
                except e.Error as ex:
                    logger.debug("connection failed: %s: %s", states[i].descr, ex)
                else:
                    winner_idx = i
                    logger.debug("connection succeeded: %s", states[i].descr)
                    break
    finally:
        # Cancel the attempts still racing and let them actually stop before
        # closing their sockets. If the enclosing operation is interrupted
        # while waiting, proceed nonetheless: the tasks are already
        # cancelling and the sockets must be closed promptly.
        for t in tasks:
            if not t.done():
                t.cancel()
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except BaseException:
            results = [asyncio.CancelledError()] * len(tasks)
        for i, st in enumerate(states):
            if i != winner_idx:
                st.pgconn.finish()

    if winner_idx >= 0:
        return cls(states[winner_idx].pgconn), []

    wave_errors: list[tuple[e.Error, str]] = []
    for st, result in zip(states, results):
        if isinstance(result, e._WaitTimeout):
            # The attempt hit the shared deadline inside wait_conn_async().
            ex = _timeout_error()
            logger.debug("connection failed: %s: %s", st.descr, ex)
        elif isinstance(result, e.Error):
            ex = result
        else:
            # CancelledError: the attempt was stopped while still in progress,
            # either because the race ended or because it was interrupted.
            ex = _timeout_error()
            logger.debug("connection failed: %s: %s", st.descr, ex)
        wave_errors.append((ex, st.descr))

    return None, wave_errors
