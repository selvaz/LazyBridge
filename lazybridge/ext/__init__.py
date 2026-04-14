"""lazybridge.ext — domain-specific extensions and ready-made tools.

Extensions depend on the core framework (lazybridge.*) but the core
never imports from ext/.  Each extension is self-contained and may
require its own optional dependencies.

Available extensions::

    lazybridge.ext.stat_runtime      Econometrics & time-series analysis
    lazybridge.ext.data_downloader   Market data ingestion (Yahoo, FRED, ECB)
    lazybridge.ext.quant_agent       Pre-configured quantitative analysis agent
    lazybridge.ext.doc_skills        BM25 local documentation skill runtime
    lazybridge.ext.read_docs         Multi-format document reader
"""
