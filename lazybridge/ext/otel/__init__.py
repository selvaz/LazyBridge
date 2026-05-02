"""OpenTelemetry exporter for LazyBridge sessions.

Install with::

    pip install lazybridge[otel]

Then drop the exporter into a :class:`lazybridge.Session`::

    from lazybridge import Agent, Session
    from lazybridge.ext.otel import OTelExporter

    sess = Session(exporters=[OTelExporter(endpoint="http://jaeger:4318")])
    agent = Agent("claude-opus-4-7", session=sess)
    agent("...")

Hierarchy: each agent run becomes a span; tool / model events become
child spans inside it.
"""

#: Stability tag — see ``docs/guides/core-vs-ext.md``.
__stability__ = "stable"
__lazybridge_min__ = "1.0.0"

from lazybridge.ext.otel.exporter import OTelExporter

__all__ = ["OTelExporter"]
