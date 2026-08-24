"""Collectors must not outlive the run.

A collector that has put items on the queue will not exit until they are
flushed, and SIGTERM is swallowed by the MolmoSpaces rollout runner. Both
together left eight collectors holding 24 GB of GPU memory an hour after a run
finished.
"""

from __future__ import annotations

import queue as queue_module

from rlt.train_parallel import shutdown


class FakeQueue:
    def __init__(self, items: int = 3) -> None:
        self.items = items
        self.closed = False
        self.cancelled = False

    def get(self, timeout=None):
        if self.items <= 0:
            raise queue_module.Empty
        self.items -= 1
        return ("rows", [])

    def close(self):
        self.closed = True

    def cancel_join_thread(self):
        self.cancelled = True


class FakeWorker:
    """Stays alive until killed, like a collector whose SIGTERM was ignored."""

    def __init__(self, pid: int, dies_on_join: bool) -> None:
        self.pid = pid
        self.alive = True
        self.dies_on_join = dies_on_join
        self.killed = False

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        if self.dies_on_join:
            self.alive = False

    def kill(self):
        self.killed = True
        self.alive = False


def test_stubborn_collectors_are_killed():
    workers = [FakeWorker(1, dies_on_join=True), FakeWorker(2, dies_on_join=False)]
    rows_queue = FakeQueue()

    shutdown(workers, rows_queue, grace=0.5)

    assert not any(w.is_alive() for w in workers), "a collector survived the shutdown"
    assert workers[0].killed is False, "a cooperative collector should not need SIGKILL"
    assert workers[1].killed is True, "a stubborn collector must be killed"
    assert rows_queue.closed and rows_queue.cancelled


def test_the_queue_is_drained_so_collectors_can_flush():
    workers = [FakeWorker(1, dies_on_join=True)]
    rows_queue = FakeQueue(items=5)
    shutdown(workers, rows_queue, grace=0.5)
    assert rows_queue.items == 0, "leftover items keep a collector from exiting"
