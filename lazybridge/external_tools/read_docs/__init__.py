"""lazybridge.external_tools.read_docs — Multi-format document reader (domain example).

Domain example shipped with LazyBridge — not part of the framework
contract. Pin to a specific lazybridge release if you depend on it.

Reads .txt, .md, .pdf, .docx, .html files from a folder or a single file
and returns their text content in a format ready for LLM consumption.

Quick start::

    from lazybridge.external_tools.read_docs import read_docs_tools
    from lazybridge import Agent

    agent = Agent("anthropic", tools=read_docs_tools())
    resp = agent("Read /path/to/reports and summarise the Q4 outlook.")

Optional dependencies::

    pip install lazybridge[tools]   # installs pypdf, python-docx, trafilatura
"""

from lazybridge.external_tools.read_docs.read_docs import read_docs_tools, read_folder_docs

__all__ = ["read_docs_tools", "read_folder_docs"]
