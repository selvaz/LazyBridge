"""lazybridge.external_tools — domain tool packages.

Pre-1.0 (alpha): each module here exposes a factory that returns
``list[Tool]`` for an Agent to call. These are *worked examples* of what
you can build on top of LazyBridge — not framework primitives.

Available packages::

    lazybridge.external_tools.read_docs        Multi-format document reader
    lazybridge.external_tools.doc_skills       BM25 local doc skill runtime
    lazybridge.external_tools.data_downloader  Yahoo / FRED / ECB ingestion
    lazybridge.external_tools.stat_runtime     Econometrics & time-series sandbox
    lazybridge.external_tools.veo              Google Veo video generation
    lazybridge.external_tools.report_builder   HTML/PDF report assembler

Each package ships its own optional-deps extra in ``pyproject.toml``.

These packages may only import from public ``lazybridge.*`` — never
from internal ``lazybridge.core.*`` or other private submodules
(enforced by ``tools/check_ext_imports.py``).
"""
