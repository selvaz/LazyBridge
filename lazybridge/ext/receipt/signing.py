"""Ed25519 signing and verification for publication receipts."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from lazybridge.ext.receipt.models import PublicationReceipt

try:
    from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
except ImportError as exc:
    raise ImportError(
        'Publication receipts require cryptography. Install it with pip install "lazybridge[receipt]".'
    ) from exc


class IssuerKey:
    """Load an Ed25519 private key, creating it at ``path`` on first use.

    New files are opened with mode ``0o600``. On Windows the key inherits
    its ACL regardless of this mode. Use a directory whose ACL already
    limits access to the current user.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        path = Path(path)
        try:
            pem = path.read_bytes()
        except FileNotFoundError:
            self._create(path)
            pem = path.read_bytes()

        key = serialization.load_pem_private_key(pem, password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise ValueError("The publication receipt issuer key must be an Ed25519 private key")
        self._key = key

    @staticmethod
    def _create(path: Path) -> None:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        pem = Ed25519PrivateKey.generate().private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        # Publish the complete file atomically without replacing a concurrent winner.
        with TemporaryDirectory(dir=path.parent) as directory:
            candidate = Path(directory) / "issuer.pem"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
            with os.fdopen(os.open(candidate, flags, 0o600), "wb") as key_file:
                key_file.write(pem)
                key_file.flush()
                os.fsync(key_file.fileno())
            try:
                os.link(candidate, path)
            except FileExistsError:
                pass

    @property
    def key_id(self) -> str:
        """Stable identifier derived from the public key."""
        return f"ed25519:{self.fingerprint()}"

    def public_key_pem(self) -> bytes:
        """Export the public key in SubjectPublicKeyInfo PEM format."""
        return self._key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    def fingerprint(self) -> str:
        """Return the lowercase hex SHA-256 digest of the raw public key."""
        return hashlib.sha256(self._key.public_key().public_bytes_raw()).hexdigest()


@dataclass(frozen=True)
class VerificationResult:
    valid: bool
    reason: str | None = None


def _canonical_json(fields: dict[str, Any]) -> bytes:
    return json.dumps(fields, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def canonical_bytes(receipt: PublicationReceipt) -> bytes:
    """Encode every receipt field except the signature as canonical UTF-8 JSON."""
    return _canonical_json(receipt.model_dump(mode="json", exclude={"signature"}, warnings=False))


def sign(receipt: PublicationReceipt, key: IssuerKey) -> PublicationReceipt:
    """Return a copy signed by the issuer key."""
    snapshot = receipt.model_copy(deep=True)
    signature = base64.b64encode(key._key.sign(canonical_bytes(snapshot))).decode("ascii")
    return snapshot.model_copy(update={"signature": signature})


def _signature_valid(body: bytes, signature: object, public_key: bytes | str | Ed25519PublicKey) -> bool:
    if not isinstance(signature, str):
        return False
    try:
        if isinstance(public_key, str):
            public_key = public_key.encode("ascii")
        key = serialization.load_pem_public_key(public_key) if isinstance(public_key, bytes) else public_key
        if not isinstance(key, Ed25519PublicKey):
            return False
        key.verify(base64.b64decode(signature, validate=True), body)
    except (InvalidSignature, UnsupportedAlgorithm, ValueError):
        return False
    return True


def verify(
    receipt: PublicationReceipt,
    public_key: bytes | str | Ed25519PublicKey,
    *,
    expected_head_sha: str | None = None,
) -> VerificationResult:
    """Verify the signature and head bindings using a trusted issuer public key."""
    try:
        body = canonical_bytes(receipt)
    except ValueError:
        return VerificationResult(False, "bad_signature")
    if not _signature_valid(body, receipt.signature, public_key):
        return VerificationResult(False, "bad_signature")
    return _verify_claims(receipt, expected_head_sha)


def _verify_claims(receipt: PublicationReceipt, expected_head_sha: str | None) -> VerificationResult:
    if receipt.receipt_version != 1:
        return VerificationResult(False, "unknown_version")
    head_sha = receipt.publication.head_sha
    if receipt.reviewer.reviewed_sha != head_sha:
        return VerificationResult(False, "review_head_mismatch")
    if receipt.approval.target.head_sha != head_sha:
        return VerificationResult(False, "approval_head_mismatch")
    if any(check.head_sha != head_sha for check in receipt.checks):
        return VerificationResult(False, "check_head_mismatch")
    if receipt.outcome.status == "published" and receipt.publication.pr_number is None:
        return VerificationResult(False, "published_without_pr")
    if expected_head_sha is not None and expected_head_sha != head_sha:
        return VerificationResult(False, "live_head_mismatch")
    return VerificationResult(True)
