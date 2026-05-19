"""HIL composition examples — the same primitive in four structural roles.

This package is a design showcase, not a production chat-app builder.
It demonstrates that LazyBridge's ``HumanEngine`` composes into surprising
shapes without any framework modification: leaf clarifier, pipeline
entrypoint, cyclic chat, and pluggable custom UI.

For a production-grade web UI, implement ``_UIProtocol`` and pass it to
``HumanEngine(ui=your_ui)`` — see ``04_custom_ui.py``.
"""
