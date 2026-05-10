# State primitives

`Memory` is the per-agent conversation history layer. `Store` is the
shared, optionally-persistent key-value blackboard.

For narrative usage see [Guides → Mid → Memory](../guides/mid/memory.md)
and [Guides → Mid → Store](../guides/mid/store.md). For wiring data
between Plan steps see [Sentinels & predicates](sentinels.md).

::: lazybridge.Memory

::: lazybridge.Store

## At-rest encryption

`EncryptedStoreAdapter` wraps any `Store` and encrypts values before
they hit the inner store; reads decrypt transparently. Keys are
**not** encrypted — `Store` treats keys as opaque routing strings,
and encrypting them would break iteration / `__contains__`.

Install the optional extra:

```bash
pip install 'lazybridge[encryption]'
```

The extra pulls [`cryptography`](https://cryptography.io); the
default install stays pure-Python.

```python
from cryptography.fernet import Fernet
from lazybridge.store import Store
from lazybridge.store.encryption import EncryptedStoreAdapter

key = Fernet.generate_key()  # persist somewhere safe (KMS, env var, etc.)
store = EncryptedStoreAdapter(Store(db="state.sqlite"), key=key)

store.write("agent.notes", {"draft": "secret thoughts"})
# SQLite row holds an `lb-enc-v1::` Fernet token, not the JSON.

store.read("agent.notes")
# {"draft": "secret thoughts"}
```

For key rotation use `MultiFernet` semantics by passing a list — the
first key is used for new writes, every key is tried on read:

```python
store = EncryptedStoreAdapter(Store(...), key=[new_key, old_key])
```

::: lazybridge.store.encryption.EncryptedStoreAdapter
