#: Stability tag — see ``docs/guides/core-vs-ext.md``.
__stability__ = "alpha"
__lazybridge_min__ = "1.0.0"

"""lazybridge.ext.read_docs — Multi-format document reader.

Reads .txt, .md, .pdf, .docx, .html files from a folder or a single file
and returns their text content in a format ready for LLM consumption.

Quick start::

    from lazybridge.ext.read_docs import read_folder_docs

    text = read_folder_docs("/path/to/reports", extensions="pdf,docx")

Optional dependencies::

    pip install lazybridge[tools]   # installs pypdf, python-docx, trafilatura
"""

from lazybridge.ext.read_docs.read_docs import read_folder_docs

__all__ = ["read_folder_docs"]
