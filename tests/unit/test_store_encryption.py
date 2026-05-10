"""Tests for ``EncryptedStoreAdapter`` — the Phase-5 at-rest encryption
wrapper around :class:`lazybridge.store.Store`.

Coverage axes:
* Round-trip — write → read returns the original value.
* On-disk shape — the inner Store NEVER sees plaintext.
* Key rotation — ``key=[new, old]`` (MultiFernet) decrypts both.
* CAS — succeeds, fails, propagates inner-CAS race losses.
* Error surface — missing dependency, bad key type, plaintext row
  raises (not silent decode garbage).
* Adapter passthrough — keys / __contains__ / __len__ / delete /
  clear / read_entry / read_all preserve Store semantics.

The tests skip gracefully when ``cryptography`` isn't installed so the
encryption extra stays genuinely optional.
"""

from __future__ import annotations

import pytest

cryptography = pytest.importorskip("cryptography.fernet")
from cryptography.fernet import Fernet, InvalidToken

from lazybridge.store import Store, StoreEntry
from lazybridge.store.encryption import EncryptedStoreAdapter


@pytest.fixture
def fresh_key() -> bytes:
    return Fernet.generate_key()


@pytest.fixture
def adapter(fresh_key: bytes) -> EncryptedStoreAdapter:
    return EncryptedStoreAdapter(Store(), key=fresh_key)


# ---------------------------------------------------------------------------
# Round-trip across types
# ---------------------------------------------------------------------------


def test_write_read_round_trip_for_primitive_value(adapter: EncryptedStoreAdapter):
    adapter.write("greeting", "hello world")
    assert adapter.read("greeting") == "hello world"


def test_write_read_round_trip_for_dict_value(adapter: EncryptedStoreAdapter):
    payload = {"name": "agent-1", "score": 42, "tags": ["a", "b"]}
    adapter.write("agent", payload)
    assert adapter.read("agent") == payload


def test_read_missing_key_returns_default(adapter: EncryptedStoreAdapter):
    assert adapter.read("absent", default="fallback") == "fallback"
    assert adapter.read("absent") is None


def test_pydantic_model_round_trips_as_json_shape(adapter: EncryptedStoreAdapter):
    """Codex P2 regression: ``_encrypt`` must route values through
    ``_to_jsonable`` so Pydantic models serialise as their JSON shape,
    not as ``repr()`` strings via ``default=str``.  Without the
    ``_to_jsonable`` hop the round-trip returned ``"x=1 name='hello'"``
    instead of ``{"x": 1, "name": "hello"}``."""
    pytest.importorskip("pydantic")
    from pydantic import BaseModel

    class M(BaseModel):
        x: int
        name: str

    adapter.write("pyd", M(x=1, name="hello"))
    out = adapter.read("pyd")
    assert out == {"x": 1, "name": "hello"}


def test_pydantic_model_compare_and_swap_works_through_adapter(adapter: EncryptedStoreAdapter):
    """The same Codex P2 fallout for CAS: comparing against the
    plaintext shape must succeed when the user passes either the model
    instance OR the equivalent dict.  Both round-trip to the same JSON
    shape, so equality holds across the adapter boundary."""
    pytest.importorskip("pydantic")
    from pydantic import BaseModel

    class M(BaseModel):
        x: int

    adapter.write("cfg", M(x=1))
    # The stored value is the JSON-shape dict; CAS against the model
    # (which normalises to the same dict) must succeed.
    assert adapter.compare_and_swap("cfg", expected=M(x=1), new=M(x=2)) is True
    assert adapter.read("cfg") == {"x": 2}


# ---------------------------------------------------------------------------
# The inner Store never sees plaintext
# ---------------------------------------------------------------------------


def test_inner_store_only_holds_lb_enc_v1_tokens(fresh_key: bytes):
    """Verifying the threat-model contract: anyone who reads the
    underlying Store gets ciphertext, not the user's payload."""
    inner = Store()
    adapter = EncryptedStoreAdapter(inner, key=fresh_key)
    adapter.write("secret", {"pin": "1234"})

    raw = inner.read("secret")
    assert isinstance(raw, str)
    assert raw.startswith("lb-enc-v1::")
    assert "1234" not in raw  # plaintext nowhere in the stored token


def test_inner_store_with_sqlite_backend_round_trips(tmp_path, fresh_key: bytes):
    """SQLite path — the file on disk must be ciphertext too.  This is
    the canonical "stolen db file" defence."""
    db = tmp_path / "state.sqlite"
    inner = Store(db=str(db))
    adapter = EncryptedStoreAdapter(inner, key=fresh_key)
    adapter.write("k", {"secret": "rosebud-1234"})
    assert adapter.read("k") == {"secret": "rosebud-1234"}

    # Close the inner Store so SQLite checkpoints the WAL to the main
    # file — otherwise the freshly-written row sits in .sqlite-wal and
    # the main file looks empty.
    adapter.close()

    # And the file on disk really doesn't have the plaintext.
    blob = db.read_bytes()
    assert b"lb-enc-v1::" in blob
    assert b"rosebud-1234" not in blob


# ---------------------------------------------------------------------------
# Key rotation via MultiFernet
# ---------------------------------------------------------------------------


def test_multi_fernet_key_rotation_decrypts_old_and_new(fresh_key: bytes):
    old = fresh_key
    new = Fernet.generate_key()

    # Old adapter writes under the old key.
    Store_inner = Store()
    old_adapter = EncryptedStoreAdapter(Store_inner, key=old)
    old_adapter.write("k", "old-value")

    # New adapter rotates: new key first, old key kept for reads.
    rotating = EncryptedStoreAdapter(Store_inner, key=[new, old])
    assert rotating.read("k") == "old-value"  # decrypts via old key

    # New writes go out under the new key.
    rotating.write("k", "new-value")

    # A reader holding ONLY the new key can still read.
    new_only = EncryptedStoreAdapter(Store_inner, key=new)
    assert new_only.read("k") == "new-value"

    # A reader holding only the old key can no longer decrypt — the
    # new write is invisible to them.  Fernet raises InvalidToken
    # specifically; we pin it so a future cipher-swap surfaces the
    # behaviour change instead of silently passing.
    old_only = EncryptedStoreAdapter(Store_inner, key=old)
    with pytest.raises(InvalidToken):
        old_only.read("k")


# ---------------------------------------------------------------------------
# Compare-and-swap over the decrypted value
# ---------------------------------------------------------------------------


def test_compare_and_swap_succeeds_when_expected_matches(adapter: EncryptedStoreAdapter):
    adapter.write("counter", 0)
    assert adapter.compare_and_swap("counter", expected=0, new=1) is True
    assert adapter.read("counter") == 1


def test_compare_and_swap_fails_when_expected_mismatches(adapter: EncryptedStoreAdapter):
    adapter.write("counter", 0)
    assert adapter.compare_and_swap("counter", expected=99, new=1) is False
    assert adapter.read("counter") == 0  # untouched


def test_compare_and_swap_creates_key_when_expected_none(adapter: EncryptedStoreAdapter):
    """Mirrors the inner Store's contract: ``expected=None`` means
    "must not currently exist"."""
    assert adapter.compare_and_swap("new", expected=None, new="first") is True
    assert adapter.read("new") == "first"

    # Now the key exists; a second CAS-from-None must fail.
    assert adapter.compare_and_swap("new", expected=None, new="conflict") is False
    assert adapter.read("new") == "first"


def test_compare_and_swap_round_trips_dict_values(adapter: EncryptedStoreAdapter):
    """CAS uses JSON-shape equality, so two equivalent dicts should
    compare equal even if Python re-ordered the keys.  Mirrors the base
    Store's _json_eq contract."""
    adapter.write("cfg", {"a": 1, "b": 2})
    # Different insertion order — must still match.
    assert adapter.compare_and_swap("cfg", expected={"b": 2, "a": 1}, new={"a": 3}) is True
    assert adapter.read("cfg") == {"a": 3}


# ---------------------------------------------------------------------------
# Error-surface contracts
# ---------------------------------------------------------------------------


def test_decrypt_of_plaintext_row_raises(fresh_key: bytes):
    """A bare row from a non-encrypted Store must raise on read — the
    "errors always raise" posture is the only safe behaviour here.
    Silently returning the plaintext would mask a deployment mistake
    (e.g. an EncryptedStoreAdapter pointed at the wrong Store)."""
    inner = Store()
    inner.write("legacy", "not-encrypted")
    adapter = EncryptedStoreAdapter(inner, key=fresh_key)
    with pytest.raises(ValueError, match="lb-enc-v1"):
        adapter.read("legacy")


def test_bad_key_type_raises_at_construction():
    with pytest.raises(ValueError, match="key must be bytes"):
        EncryptedStoreAdapter(Store(), key="not-bytes")  # type: ignore[arg-type]


def test_empty_key_list_raises_at_construction():
    with pytest.raises(ValueError, match="empty"):
        EncryptedStoreAdapter(Store(), key=[])


def test_mixed_key_list_types_raises(fresh_key: bytes):
    with pytest.raises(ValueError, match="must be bytes"):
        EncryptedStoreAdapter(Store(), key=[fresh_key, "string-key"])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Passthrough — delete / clear / keys / __contains__ / read_all / read_entry
# ---------------------------------------------------------------------------


def test_delete_removes_key(adapter: EncryptedStoreAdapter):
    adapter.write("k", "v")
    adapter.delete("k")
    assert adapter.read("k") is None
    assert "k" not in adapter


def test_clear_empties_store(adapter: EncryptedStoreAdapter):
    adapter.write("a", 1)
    adapter.write("b", 2)
    adapter.clear()
    assert adapter.keys() == []
    assert len(adapter) == 0


def test_keys_returns_plaintext_key_names(adapter: EncryptedStoreAdapter):
    """Keys are intentionally NOT encrypted — Store iteration must
    keep working.  This is part of the documented threat model."""
    adapter.write("alpha", 1)
    adapter.write("beta", 2)
    assert set(adapter.keys()) == {"alpha", "beta"}


def test_contains_uses_inner_membership(adapter: EncryptedStoreAdapter):
    adapter.write("present", 1)
    assert "present" in adapter
    assert "absent" not in adapter


def test_read_all_returns_decrypted_dict(adapter: EncryptedStoreAdapter):
    adapter.write("a", {"x": 1})
    adapter.write("b", {"y": 2})
    assert adapter.read_all() == {"a": {"x": 1}, "b": {"y": 2}}


def test_read_entry_returns_decrypted_entry(adapter: EncryptedStoreAdapter):
    adapter.write("k", "v", agent_id="agent-1")
    entry = adapter.read_entry("k")
    assert isinstance(entry, StoreEntry)
    assert entry.key == "k"
    assert entry.value == "v"
    assert entry.agent_id == "agent-1"


def test_read_entry_missing_returns_none(adapter: EncryptedStoreAdapter):
    assert adapter.read_entry("absent") is None


# ---------------------------------------------------------------------------
# Adapter property — inner is exposed read-only for backup tooling
# ---------------------------------------------------------------------------


def test_inner_property_exposes_underlying_store(fresh_key: bytes):
    base = Store()
    adapter = EncryptedStoreAdapter(base, key=fresh_key)
    assert adapter.inner is base


# ---------------------------------------------------------------------------
# Close — propagates to inner so SQLite handles aren't leaked
# ---------------------------------------------------------------------------


def test_close_propagates_to_inner(tmp_path, fresh_key: bytes):
    inner = Store(db=str(tmp_path / "x.sqlite"))
    adapter = EncryptedStoreAdapter(inner, key=fresh_key)
    adapter.write("k", "v")
    adapter.close()
    # Subsequent operations on the inner store should reflect closed state.
    with pytest.raises(RuntimeError):
        inner.read("k")
