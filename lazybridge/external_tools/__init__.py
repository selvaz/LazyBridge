"""lazybridge.external_tools — deprecated namespace (moved to ``lazytools``).

The domain tool kits that lived here moved to the sibling ``lazytoolkit``
package in 0.8::

    lazybridge.external_tools.read_docs   →  lazytools.documents   (pip install 'lazytoolkit[docs]')
    lazybridge.external_tools.doc_skills  →  lazytools.skills       (pip install 'lazytoolkit[docs]')

The old import paths still work via lazy deprecation shims (see the
``read_docs`` / ``doc_skills`` submodules) that emit a
:class:`DeprecationWarning` and re-export from ``lazytools``. The shims
are removed in 0.9.

The HTML/PDF report assembler moved earlier, to the sibling
``lazybridge-reports`` package in 0.7.9 (see
https://github.com/selvaz/LazyReport).

These shims may only import from public ``lazybridge.*`` and (lazily,
inside ``__getattr__``) from ``lazytools`` — never from internal
``lazybridge.core.*``. The import boundary is enforced by
``tests/unit/test_ext_core_boundary.py``.
"""
