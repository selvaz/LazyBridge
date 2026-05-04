"""Event hub + LazyBridge ``EventExporter`` adapter.

The hub is the single in-memory pub/sub between the producing
``Session`` and any number of HTTP/SSE subscribers. It keeps a small
ring buffer so a browser tab opened mid-run can still see recent
context, hands each subscriber its own bounded queue so a slow client
can't back-pressure the producer, and stamps a monotonic sequence
number on every event so the UI can detect drops.

The exporter is a thin shim that satisfies the
``lazybridge.exporters.EventExporter`` protocol — its ``export()``
just normalises the payload and forwards into the hub.
"""

from __future__ import annotations

import queue
import threading
from collections import deque
from typing import Any

from lazybridge.ext.viz._normalizer import normalise_event

_RING_BUFFER_SIZE = 500  # events kept for late-joining subscribers
_SUBSCRIBER_QUEUE_SIZE = 1000  # per-tab cap; drop oldest when full


class EventHub:
    """Thread-safe pub/sub over enriched event dicts.

    Each subscriber gets its own bounded :class:`queue.Queue`. The
    producer (``publish``) never blocks: if a subscriber's queue is
    full we drop the *oldest* event for that subscriber and bump a
    counter. The UI can show a "missed N events" badge from the gap
    in sequence numbers.
    """

    def __init__(self, *, ring_size: int = _RING_BUFFER_SIZE) -> None:
        self._lock = threading.Lock()
        self._subs: list[queue.Queue[dict[str, Any]]] = []
        self._ring: deque[dict[str, Any]] = deque(maxlen=ring_size)
        self._seq = 0
        self._closed = False

    @property
    def seq(self) -> int:
        """Monotonic counter of total events ever published."""
        return self._seq

    def publish(self, event: dict[str, Any]) -> None:
        """Stamp + fan out an already-normalised event."""
        if self._closed:
            return
        with self._lock:
            self._seq += 1
            stamped = {**event, "_seq": self._seq}
            self._ring.append(stamped)
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(stamped)
            except queue.Full:
                # Drop oldest, then enqueue. Done outside the hub lock
                # so a single slow tab can't stall every other one.
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(stamped)
                except queue.Full:
                    pass

    def subscribe(self, *, replay_recent: bool = True) -> queue.Queue[dict[str, Any]]:
        """Register a new subscriber and return its queue.

        When ``replay_recent`` is set (the default) the subscriber
        receives the current ring-buffer contents up front, so a
        late-joining browser tab does not start with a blank canvas.
        """
        q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=_SUBSCRIBER_QUEUE_SIZE)
        with self._lock:
            if replay_recent:
                for e in self._ring:
                    try:
                        q.put_nowait(e)
                    except queue.Full:
                        break
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            try:
                self._subs.remove(q)
            except ValueError:
                pass

    def snapshot(self) -> list[dict[str, Any]]:
        """Copy of the ring buffer, ordered oldest-first."""
        with self._lock:
            return list(self._ring)

    def close(self) -> None:
        """Stop accepting new events; wake every subscriber with ``None``."""
        with self._lock:
            self._closed = True
            subs = list(self._subs)
            self._subs.clear()
        for q in subs:
            try:
                q.put_nowait({"_seq": -1, "event_type": "_closed"})
            except queue.Full:
                pass


class HubExporter:
    """``EventExporter`` adapter that forwards into an :class:`EventHub`."""

    def __init__(self, hub: EventHub) -> None:
        self._hub = hub

    def export(self, event: dict[str, Any]) -> None:
        self._hub.publish(normalise_event(event))
