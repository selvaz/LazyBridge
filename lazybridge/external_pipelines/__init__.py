"""lazybridge.external_pipelines — pre-wired agent compositions.

Pre-1.0 (alpha): each module here is a worked example of a complete
agent pipeline built on top of the core runtime and one or more
``lazybridge.external_tools`` packages.

Available pipelines::

    lazybridge.external_pipelines.quant_agent  Quant analysis agent over
                                                data_downloader + stat_runtime

These packages may only import from public ``lazybridge.*`` and from
``lazybridge.external_tools.*`` — never from internal ``lazybridge.core.*``
or other private submodules.
"""
