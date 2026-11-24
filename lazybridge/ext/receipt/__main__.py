"""Verify a publication receipt from the command line."""

import argparse
import json
from pathlib import Path
from typing import Any

from lazybridge.ext.receipt import PublicationReceipt, VerificationResult
from lazybridge.ext.receipt.signing import (
    _canonical_json,
    _load_public_key,
    _signature_valid,
    _verify_claims,
    canonical_bytes,
)


def _unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    fields = dict(pairs)
    if len(fields) != len(pairs):
        raise ValueError("Duplicate receipt field")
    return fields


def _verify_json(data: str, public_key: bytes, expected_head_sha: str | None) -> VerificationResult:
    key = _load_public_key(public_key)
    if key is None:
        return VerificationResult(False, "invalid_public_key")
    try:
        fields = json.loads(data, object_pairs_hook=_unique_fields)
        if not isinstance(fields, dict):
            return VerificationResult(False, "invalid_receipt")
        signature = fields.pop("signature", None)
        body = _canonical_json(fields)
    except (ValueError, RecursionError):
        return VerificationResult(False, "invalid_receipt")

    # Authenticate the wire fields before parsing them into a typed model.
    if not _signature_valid(body, signature, key):
        return VerificationResult(False, "bad_signature")
    if type(fields.get("receipt_version")) is not int or fields["receipt_version"] != 1:
        return VerificationResult(False, "unknown_version")
    try:
        receipt = PublicationReceipt.model_validate({**fields, "signature": signature})
    except ValueError:
        return VerificationResult(False, "invalid_receipt")
    if body != canonical_bytes(receipt):
        return VerificationResult(False, "invalid_receipt")
    return _verify_claims(receipt, key, expected_head_sha)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify_parser = commands.add_parser("verify", help="verify a signed receipt file")
    verify_parser.add_argument("file", type=Path)
    verify_parser.add_argument("--public-key", type=Path, required=True)
    verify_parser.add_argument("--expected-head")
    args = parser.parse_args()

    try:
        data = args.file.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        print("invalid_receipt")
        return 1
    try:
        public_key = args.public_key.read_bytes()
    except OSError:
        print("invalid_public_key")
        return 1

    result = _verify_json(data, public_key, args.expected_head)
    print("valid" if result.valid else result.reason)
    return 0 if result.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
