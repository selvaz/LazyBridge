"""LazyBridge skill documentation — Claude Skill artifact + site renderer.

This package ships with the library so that ``pip install lazybridge``
places the skill on disk next to the code.  To open it programmatically::

    import importlib.resources, lazybridge.skill_docs
    (importlib.resources.files(lazybridge.skill_docs) / "SKILL.md").read_text()

To regenerate the rendered ``0X_<tier>.md`` files from the authoritative
``fragments/`` pool::

    python -m lazybridge.skill_docs._build
"""
