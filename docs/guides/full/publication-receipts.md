# Publication receipts

A publication receipt records the candidate commit, the writer and reviewer
permissions and admission results, the approval target, the required checks,
and the publication outcome. The issuer signs these fields with Ed25519.

```bash
pip install "lazybridge[receipt]"
```

## Sign and export a receipt

Load the issuer key from a dedicated path. `IssuerKey` creates a private PEM
file on first use and reuses it on later calls. New files use mode `0o600`.
On Windows the key inherits its ACL, so use a directory with a current-user-only
ACL.

```python
from pathlib import Path

from lazybridge.ext.receipt import IssuerKey, PublicationReceipt, sign


def export_receipt(evidence: dict, private_key_path: Path) -> None:
    key = IssuerKey(private_key_path)
    receipt = PublicationReceipt.model_validate({
        **evidence,
        "issuer": {"key_id": key.key_id, "public_key_fingerprint": key.fingerprint()},
    })
    signed = sign(receipt, key)
    Path("receipt.json").write_text(signed.model_dump_json(), encoding="utf-8")
    Path("issuer.pub.pem").write_bytes(key.public_key_pem())
```

Supply evidence as redacted metadata. Use repository-relative placeholders for
folders and omit prompts, model output, file contents, environment values,
tokens, and absolute home paths. The signature covers every field except
`signature` using sorted JSON keys, UTF-8, and compact separators. The
fingerprint is the hex SHA-256 digest of the raw public key. `sign` raises
`ValueError` if the issuer fingerprint or key ID differs from the signing key.

## Verify a receipt

Use a public key obtained from a trusted source. Supply `--expected-head` when
you have the current PR head SHA.

```bash
python -m lazybridge.ext.receipt verify receipt.json --public-key issuer.pub.pem
python -m lazybridge.ext.receipt verify receipt.json --public-key issuer.pub.pem --expected-head COMMIT_SHA
```

The CLI prints `valid` and exits `0` on success. A signed refusal also verifies
as `valid` when its evidence matches the candidate commit. Read
`outcome.refusal_reason` for the reason publication was refused. On failure, the
CLI prints the verification reason and exits `1`. In Python, use
`verify(receipt, public_key_pem, expected_head_sha=...)` and read `valid` and
`reason` from the result.

| Reason | Meaning |
| --- | --- |
| `bad_signature` | The supplied key could not authenticate the receipt. |
| `unknown_version` | The signature is valid and the receipt version is unsupported. |
| `issuer_mismatch` | The signed issuer fingerprint or key ID differs from the verification key. |
| `review_head_mismatch` | The review refers to a different commit. |
| `approval_head_mismatch` | The approval target refers to a different commit. |
| `check_head_mismatch` | A required check refers to a different commit. |
| `published_without_pr` | The receipt claims publication and has a null PR number. |
| `live_head_mismatch` | The supplied head differs from the signed publication head. |
| `invalid_receipt` | The CLI could not read the file or parse its receipt fields. |
