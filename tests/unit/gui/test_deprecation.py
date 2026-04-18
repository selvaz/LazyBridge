"""Verify the legacy ``lazybridge.gui.human`` import emits a DeprecationWarning."""

from __future__ import annotations

import importlib
import sys
import warnings


def test_importing_lazybridge_gui_human_emits_deprecation_warning():
    # Force re-import so the __init__ module body runs again.
    for mod in list(sys.modules):
        if mod == "lazybridge.gui.human" or mod.startswith("lazybridge.gui.human."):
            del sys.modules[mod]
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        importlib.import_module("lazybridge.gui.human")
    texts = [str(w.message) for w in captured if issubclass(w.category, DeprecationWarning)]
    assert any("panel_input_fn" in t for t in texts), captured
