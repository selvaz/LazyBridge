"""``EncryptedStoreAdapter`` — at-rest encryption wrapper for :class:`Store`.

Wraps any :class:`lazybridge.store.Store` instance and encrypts values
before they're written, decrypting on read.  Keys are NOT encrypted —
the Store API treats keys as opaque routing strings, and encrypting
them would break iteration / ``__contains__`` semantics.

Why a separate adapter (not a Store subclass)?
  Encryption is an opt-in concern; pulling ``cryptography`` into the
  base import path would force every LazyBridge user to install a
  C-extension dependency.  The adapter pattern keeps the base Store
  cheap and lets encryption ride on the side under the ``[encryption]``
  optional extra.

Cipher choice: :class:`cryptography.fernet.Fernet`.  Fernet is the
high-level authenticated-encryption primitive from
``cryptography`` — AES-128-CBC + HMAC-SHA256, key rotation via
:class:`MultiFernet`, ciphertext stable across versions.  It is the
right default for "encrypt JSON blobs at rest" and matches the threat
model (a stolen SQLite file is unreadable).

Usage::

    from cryptography.fernet import Fernet
    from lazybridge.store import Store
    from lazybridge.store.encryption import EncryptedStoreAdapter

    key = Fernet.generate_key()  # persist this somewhere safe
    base = Store(db="state.sqlite")
    store = EncryptedStoreAdapter(base, key=key)

    store.write("agent.notes", {"draft": "secret thoughts"})
    # → SQLite row contains the Fernet token, not the JSON.

    store.read("agent.notes")
    # → {"draft": "secret thoughts"}

Key rotation::

    # Rotate to a new key while keeping the old one for decryption.
    store = EncryptedStoreAdapter(base, key=[new_key, old_key])
    # Every write uses the FIRST key; reads succeed for any listed key.

Threat model & non-goals:
  * Protects against an attacker who reads ``state.sqlite`` off disk.
  * Does NOT protect against an attacker with live process memory
    (the in-flight value is plaintext).
  * Does NOT encrypt keys, ``written_at``, or ``agent_id``.  An
    attacker with file access still sees the access pattern.
  * Not a substitute for OS-level disk encryption — defence in depth.

The adapter forwards every Store public method (``write``, ``read``,
``read_entry``, ``read_all``, ``delete``, ``clear``, ``keys``,
``items``, ``compare_and_swap``, ``to_text``, ``__iter__``,
``__contains__``, ``__len__``, ``close``).  ``compare_and_swap`` is
implemented at the adapter layer (NOT delegated raw) because the
underlying Store would otherwise compare ciphertext tokens — and Fernet
ciphertexts include a nonce, so two encryptions of the same plaintext
don't compare equal.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    from cryptography.fernet import Fernet, MultiFernet

    from lazybridge.store import Store, StoreEntry

# Stable token prefix written before the Fernet ciphertext so an
# adapter pointed at a Store with mixed plaintext / encrypted rows
# can tell them apart and raise instead of silently decrypting
# garbage (audit posture: errors always raise).
_TOKEN_PREFIX = "lb-enc-v1::"


class EncryptedStoreAdapter:
    """Encrypt values written to an inner :class:`Store`.

    Parameters
    ----------
    inner:
        The Store to wrap.  Any Store flavour works (in-memory or
        SQLite-backed); the adapter doesn't care about persistence.
    key:
        A Fernet key (``bytes``) or list of keys.  A list activates
        :class:`cryptography.fernet.MultiFernet` for rotation —
        encryption uses the first key, decryption tries each in
        order until one succeeds.

    Raises
    ------
    ImportError:
        If ``cryptography`` isn't installed.  Install
        ``lazybridge[encryption]`` to add it.
    ValueError:
        If ``key`` is empty or not bytes / list of bytes.
    """

    def __init__(self, inner: Store, *, key: bytes | list[bytes]) -> None:
        try:
            from cryptography.fernet import Fernet, MultiFernet
        except ImportError as e:
            raise ImportError(
                "EncryptedStoreAdapter requires the 'cryptography' package.\n"
                "  Install it via the encryption extra:\n"
                "    pip install 'lazybridge[encryption]'\n"
                "  Or directly:\n"
                "    pip install cryptography"
            ) from e

        if isinstance(key, bytes):
            self._fernet: Fernet | MultiFernet = Fernet(key)
        elif isinstance(key, list):
            if not key:
                raise ValueError("EncryptedStoreAdapter: key list is empty.")
            if not all(isinstance(k, bytes) for k in key):
                raise ValueError(
                    f"EncryptedStoreAdapter: every key in the rotation list must be bytes; "
                    f"got types {[type(k).__name__ for k in key]}."
                )
            self._fernet = MultiFernet([Fernet(k) for k in key])
        else:
            raise ValueError(f"EncryptedStoreAdapter: key must be bytes or list[bytes], got {type(key).__name__}.")

        self._inner = inner

    # --- value codec --------------------------------------------------

    def _encrypt(self, value: Any) -> str:
        """Serialise ``value`` to JSON, encrypt with Fernet, return the
        prefixed token as a ``str`` so the inner Store can persist it
        through its normal JSON path without further escaping.

        We route through :func:`lazybridge.store._to_jsonable` first so
        Pydantic models (and nested containers of them) are normalised
        to their ``model_dump(mode='json')`` shape — exactly what the
        base :class:`Store` persists.  Without this, ``default=str`` in
        ``json.dumps`` would fall back to ``repr()`` for any non-trivial
        object, encrypting a string like ``"x=1 name='hello'"`` instead
        of the round-trippable JSON shape.  That breaks structured
        outputs AND :meth:`compare_and_swap` (the plaintext we compare
        against on read wouldn't match the value the user passed).
        """
        from lazybridge.store import _to_jsonable

        plaintext = json.dumps(_to_jsonable(value), default=str).encode("utf-8")
        token = self._fernet.encrypt(plaintext).decode("ascii")
        return f"{_TOKEN_PREFIX}{token}"

    def _decrypt(self, stored: Any) -> Any:
        """Reverse :meth:`_encrypt`.  Raises :class:`ValueError` when
        the stored payload doesn't carry the v1 prefix — that means a
        row was written by a non-encrypted Store or an incompatible
        adapter version, which is a configuration error the caller
        needs to see, not silently mishandle."""
        if not isinstance(stored, str) or not stored.startswith(_TOKEN_PREFIX):
            raise ValueError(
                "EncryptedStoreAdapter: stored value is not an lb-enc-v1 token.  "
                "The underlying Store contains a plaintext row — either this adapter "
                "is pointed at a non-encrypted Store, or the row was written before "
                "encryption was enabled.  Re-encrypt the affected keys explicitly to "
                "migrate, or unwrap the adapter to read legacy rows."
            )
        token = stored[len(_TOKEN_PREFIX) :].encode("ascii")
        plaintext = self._fernet.decrypt(token).decode("utf-8")
        return json.loads(plaintext)

    # --- Store passthrough -------------------------------------------

    def write(self, key: str, value: Any, *, agent_id: str | None = None) -> None:
        self._inner.write(key, self._encrypt(value), agent_id=agent_id)

    def read(self, key: str, default: Any = None) -> Any:
        raw = self._inner.read(key, default=None)
        if raw is None:
            return default
        return self._decrypt(raw)

    def read_entry(self, key: str) -> StoreEntry | None:
        entry = self._inner.read_entry(key)
        if entry is None:
            return None
        from lazybridge.store import StoreEntry as _StoreEntry

        return _StoreEntry(
            key=entry.key,
            value=self._decrypt(entry.value),
            written_at=entry.written_at,
            agent_id=entry.agent_id,
        )

    def read_all(self) -> dict[str, Any]:
        return {k: self._decrypt(v) for k, v in self._inner.read_all().items()}

    def delete(self, key: str) -> None:
        self._inner.delete(key)

    def clear(self) -> None:
        self._inner.clear()

    def keys(self) -> list[str]:
        return self._inner.keys()

    def items(self, *, prefix: str | None = None) -> list[tuple[str, Any]]:
        """Return ``(key, decrypted-value)`` pairs, optionally restricted to
        keys starting with ``prefix``."""
        return [(k, self._decrypt(v)) for k, v in self._inner.items(prefix=prefix)]

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __contains__(self, key: object) -> bool:
        return key in self._inner

    def __len__(self) -> int:
        return len(self._inner)

    def to_text(self, keys: list[str] | None = None) -> str:
        data = self.read_all()
        if keys:
            data = {k: v for k, v in data.items() if k in keys}
        return "\n".join(f"{k}: {json.dumps(v, default=str)}" for k, v in data.items())

    def compare_and_swap(
        self,
        key: str,
        expected: Any,
        new: Any,
        *,
        agent_id: str | None = None,
    ) -> bool:
        """CAS over the decrypted value.

        Fernet ciphertexts embed a nonce, so two encryptions of the
        same plaintext are never equal.  Delegating to the inner
        Store's CAS would therefore always fail.  We decrypt the
        current ciphertext, compare against ``expected`` in plaintext
        space, then write the new encrypted value — *as long as* the
        ciphertext hasn't changed since the read.

        This double-check is what makes the adapter race-safe: another
        writer may have updated the value between our decrypt and the
        inner CAS attempt, in which case the inner CAS returns
        ``False`` and we propagate that.
        """
        current_token = self._inner.read(key)
        if current_token is None:
            current_plain = None
        else:
            current_plain = self._decrypt(current_token)

        if not _plain_eq(current_plain, expected):
            return False

        return self._inner.compare_and_swap(
            key,
            expected=current_token,
            new=self._encrypt(new),
            agent_id=agent_id,
        )

    def close(self) -> None:
        self._inner.close()

    # Expose the inner Store for callers that need to drop down (e.g.
    # backup / migration tooling).  Read-only by convention — mutating
    # ``adapter.inner`` directly bypasses encryption.
    @property
    def inner(self) -> Store:
        return self._inner


def _plain_eq(a: Any, b: Any) -> bool:
    """JSON-shape equality for the plaintext path.

    Mirrors :func:`lazybridge.store._json_eq` semantics so the
    adapter's CAS plays the same equality rules the base Store does:
    a Pydantic model and the dict it round-trips to compare equal
    (Codex P2 regression — without ``_to_jsonable`` we'd ``repr()``
    the model and never match the persisted dict shape).
    """
    from lazybridge.store import _to_jsonable

    try:
        sa = json.dumps(_to_jsonable(a), sort_keys=True, default=str)
        sb = json.dumps(_to_jsonable(b), sort_keys=True, default=str)
    except TypeError:
        return False
    return sa == sb
