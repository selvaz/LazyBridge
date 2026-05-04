"""EventHub fan-out: subscribe, publish, late-join replay, ring buffer."""

from __future__ import annotations

import queue

from lazybridge.ext.viz.exporter import EventHub, HubExporter


def test_publish_fans_out_to_subscriber():
    hub = EventHub()
    q = hub.subscribe()
    hub.publish({"event_type": "agent_start", "agent_name": "a"})
    ev = q.get(timeout=1.0)
    assert ev["event_type"] == "agent_start"
    assert ev["_seq"] == 1


def test_seq_is_monotonic():
    hub = EventHub()
    q = hub.subscribe()
    for i in range(5):
        hub.publish({"event_type": "loop_step", "i": i})
    seqs = [q.get_nowait()["_seq"] for _ in range(5)]
    assert seqs == [1, 2, 3, 4, 5]


def test_late_subscriber_replays_ring_buffer():
    hub = EventHub(ring_size=3)
    for i in range(5):
        hub.publish({"event_type": "x", "i": i})
    q = hub.subscribe()  # replay_recent=True by default
    seen = []
    while not q.empty():
        seen.append(q.get_nowait()["i"])
    # Only the last 3 survive in the ring
    assert seen == [2, 3, 4]


def test_late_subscriber_no_replay_when_opted_out():
    hub = EventHub()
    hub.publish({"event_type": "x"})
    q = hub.subscribe(replay_recent=False)
    assert q.empty()


def test_unsubscribe_stops_delivery():
    hub = EventHub()
    q = hub.subscribe()
    hub.unsubscribe(q)
    hub.publish({"event_type": "x"})
    try:
        q.get_nowait()
        raise AssertionError("should be empty")
    except queue.Empty:
        pass


def test_close_wakes_subscribers():
    hub = EventHub()
    q = hub.subscribe()
    hub.close()
    msg = q.get(timeout=1.0)
    assert msg["event_type"] == "_closed"


def test_hub_exporter_satisfies_event_protocol():
    from lazybridge.exporters import EventExporter

    hub = EventHub()
    exp = HubExporter(hub)
    assert isinstance(exp, EventExporter)


def test_hub_exporter_normalises_and_publishes():
    import datetime as dt

    hub = EventHub()
    q = hub.subscribe()
    HubExporter(hub).export({"event_type": "agent_start", "ts": dt.datetime(2026, 1, 1)})
    out = q.get(timeout=1.0)
    assert isinstance(out["ts"], str)  # normalised
