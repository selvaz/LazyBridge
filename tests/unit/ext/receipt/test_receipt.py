"""Publication receipt encoding, signatures, and key persistence."""

import base64
import errno
import hashlib
import json
import os
import stat
import subprocess
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Event
from typing import Any

import pytest
from cryptography.hazmat.primitives.serialization import load_pem_public_key

from lazybridge.ext.receipt import IssuerKey, PublicationReceipt, VerificationResult, canonical_bytes, sign, verify
from lazybridge.ext.receipt.__main__ import _verify_json

HEAD = "a" * 40
OTHER_HEAD = "b" * 40
TIMESTAMP = "2026-11-24T10:11:12Z"


@pytest.fixture
def key(tmp_path: Path) -> IssuerKey:
    return IssuerKey(tmp_path / "keys" / "issuer.pem")


@pytest.fixture
def fields(key: IssuerKey) -> dict[str, Any]:
    writer = {
        "task_id": "write-123",
        "attempt": 1,
        "engine": "codex",
        "model": "writer-model",
        "launch_mode": "ungoverned-legacy",
        "admission": {"eligible": False, "reasons": ["release check failed"]},
        "policy": {
            "profile": "writer",
            "revision": "rev-1",
            "digest": "writer-digest",
            "folders": ["./café"],
            "network": "off",
            "tool_classes": ["read", "write"],
        },
    }
    return {
        "receipt_version": 1,
        "receipt_id": "0144f889-2f5a-4131-adc0-775736213825",
        "issued_at": TIMESTAMP,
        "issuer": {"key_id": key.key_id, "public_key_fingerprint": key.fingerprint(), "algorithm": "ed25519"},
        "repository": {"owner": "sample", "name": "test", "base_branch": "main"},
        "publication": {
            "head_branch": "topic",
            "head_sha": HEAD,
            "base_sha": OTHER_HEAD,
            "pr_number": 42,
            "pr_url": "https://example.test/sample/test/pull/42",
            "draft": True,
        },
        "writer": writer,
        "reviewer": {
            **deepcopy(writer),
            "task_id": "review-123",
            "engine": "claude-code",
            "model": "reviewer-model",
            "independence_mode": "opposite-engine",
            "reviewed_sha": HEAD,
            "verdict": "accepted",
        },
        "delegation": {"parent_digest": "parent-digest", "child_within_parent": True},
        "approval": {
            "ticket_id": "ticket-123",
            "decided_by": "owner",
            "decided_at": TIMESTAMP,
            "consumed_at": TIMESTAMP,
            "request_digest": "approval-digest",
            "target": {
                "action": "open_draft_pr",
                "repository": "sample/test",
                "head_branch": "topic",
                "head_sha": HEAD,
            },
        },
        "checks": [{"name": "unit", "command": "pytest", "status": "passed", "head_sha": HEAD}],
        "outcome": {"status": "published", "refusal_reason": None},
    }


def _verify_signed(fields: dict[str, Any], key: IssuerKey, expected_head: str | None = None) -> VerificationResult:
    return verify(
        sign(PublicationReceipt.model_validate(fields), key), key.public_key_pem(), expected_head_sha=expected_head
    )


def test_sign_and_verify_round_trip(fields: dict[str, Any], key: IssuerKey) -> None:
    receipt = PublicationReceipt.model_validate(fields)
    signed = sign(receipt, key)
    loaded = PublicationReceipt.model_validate_json(signed.model_dump_json())
    assert not receipt.signature
    assert signed.signature
    assert canonical_bytes(loaded) == canonical_bytes(receipt)
    public_key = load_pem_public_key(key.public_key_pem())
    assert hashlib.sha256(public_key.public_bytes_raw()).hexdigest() == key.fingerprint()
    assert verify(loaded, public_key, expected_head_sha=HEAD) == VerificationResult(True)


def test_bad_signature(fields: dict[str, Any], key: IssuerKey, tmp_path: Path) -> None:
    signed = sign(PublicationReceipt.model_validate(fields), key)
    modified = signed.model_copy(update={"signature": "!invalid base64!"})
    assert verify(modified, key.public_key_pem()) == VerificationResult(False, "bad_signature")
    wrong_key = IssuerKey(tmp_path / "other.pem")
    assert verify(signed, wrong_key.public_key_pem()) == VerificationResult(False, "bad_signature")


@pytest.mark.parametrize("issuer_field", ["public_key_fingerprint", "key_id"])
def test_issuer_mismatch(fields: dict[str, Any], key: IssuerKey, issuer_field: str) -> None:
    fields["issuer"][issuer_field] = "wrong-issuer"
    receipt = PublicationReceipt.model_validate(fields)
    with pytest.raises(ValueError, match="issuer does not match the signing key"):
        sign(receipt, key)

    signature = base64.b64encode(key._key.sign(canonical_bytes(receipt))).decode("ascii")
    signed = receipt.model_copy(update={"signature": signature})
    public_key = key.public_key_pem()
    assert verify(signed, public_key) == VerificationResult(False, "issuer_mismatch")
    assert _verify_json(signed.model_dump_json(), public_key, None) == VerificationResult(False, "issuer_mismatch")


def test_unknown_version(fields: dict[str, Any], key: IssuerKey) -> None:
    fields["receipt_version"] = 2
    assert _verify_signed(fields, key) == VerificationResult(False, "unknown_version")


def test_review_head_mismatch(fields: dict[str, Any], key: IssuerKey) -> None:
    fields["reviewer"]["reviewed_sha"] = OTHER_HEAD
    assert _verify_signed(fields, key) == VerificationResult(False, "review_head_mismatch")


def test_approval_head_mismatch(fields: dict[str, Any], key: IssuerKey) -> None:
    fields["approval"]["target"]["head_sha"] = OTHER_HEAD
    assert _verify_signed(fields, key) == VerificationResult(False, "approval_head_mismatch")


def test_check_head_mismatch(fields: dict[str, Any], key: IssuerKey) -> None:
    fields["checks"].append({**fields["checks"][0], "name": "second check", "head_sha": OTHER_HEAD})
    assert _verify_signed(fields, key) == VerificationResult(False, "check_head_mismatch")


def test_published_without_pr(fields: dict[str, Any], key: IssuerKey) -> None:
    fields["publication"]["pr_number"] = None
    assert _verify_signed(fields, key) == VerificationResult(False, "published_without_pr")


def test_live_head_mismatch(fields: dict[str, Any], key: IssuerKey) -> None:
    assert _verify_signed(fields, key, OTHER_HEAD) == VerificationResult(False, "live_head_mismatch")


def test_refused_after_review_head_drift(fields: dict[str, Any], key: IssuerKey) -> None:
    fields["publication"].update(head_sha=OTHER_HEAD, pr_number=None, pr_url=None)
    fields["approval"].update(ticket_id=None, decided_by=None, decided_at=None, consumed_at=None, request_digest=None)
    fields["outcome"] = {"status": "refused", "refusal_reason": "head_changed_after_review"}
    assert _verify_signed(fields, key, HEAD) == VerificationResult(True)


def test_refused_after_approval_head_drift(fields: dict[str, Any], key: IssuerKey) -> None:
    fields["publication"].update(head_sha=OTHER_HEAD, pr_number=None, pr_url=None)
    fields["reviewer"]["reviewed_sha"] = OTHER_HEAD
    fields["approval"]["consumed_at"] = None
    fields["outcome"] = {"status": "refused", "refusal_reason": "approval_target_mismatch"}
    signed = sign(PublicationReceipt.model_validate(fields), key)
    public_key = key.public_key_pem()
    assert verify(signed, public_key, expected_head_sha=HEAD) == VerificationResult(True)
    assert _verify_json(signed.model_dump_json(), public_key, HEAD) == VerificationResult(True)


@pytest.mark.parametrize(("pr_number", "reason"), [(42, "head_changed_after_review"), (None, ""), (None, None)])
def test_verify_rejects_malformed_refusals(
    fields: dict[str, Any], key: IssuerKey, pr_number: int | None, reason: str | None
) -> None:
    fields["publication"].update(pr_number=None, pr_url=None)
    fields["outcome"] = {"status": "refused", "refusal_reason": "head_changed_after_review"}
    receipt = PublicationReceipt.model_validate(fields)
    malformed = receipt.model_copy(
        update={
            "publication": receipt.publication.model_copy(update={"pr_number": pr_number}),
            "outcome": receipt.outcome.model_copy(update={"refusal_reason": reason}),
        }
    )
    signed = sign(malformed, key)
    public_key = key.public_key_pem()
    assert verify(signed, public_key) == VerificationResult(False, "invalid_receipt")
    assert _verify_json(signed.model_dump_json(), public_key, None) == VerificationResult(False, "invalid_receipt")


def _reverse_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _reverse_keys(value[key]) for key in reversed(value)}
    if isinstance(value, list):
        return [_reverse_keys(item) for item in value]
    return value


def test_canonical_bytes_are_order_independent(fields: dict[str, Any], key: IssuerKey) -> None:
    signed = sign(PublicationReceipt.model_validate(fields), key)
    reordered = _reverse_keys(signed.model_dump(mode="json"))
    expected = json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    assert canonical_bytes(PublicationReceipt.model_validate(reordered)) == canonical_bytes(signed) == expected
    assert "./café".encode() in expected
    assert _verify_json(json.dumps(reordered, indent=2), key.public_key_pem(), HEAD) == VerificationResult(True)


def _signed_field_paths(value: Any, path: tuple[str | int, ...] = ()) -> Iterator[tuple[str | int, ...]]:
    if path:
        yield path
    if isinstance(value, dict):
        for field, item in value.items():
            yield from _signed_field_paths(item, (*path, field))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _signed_field_paths(item, (*path, index))


def test_tampering_any_signed_field_breaks_signature(fields: dict[str, Any], key: IssuerKey) -> None:
    signed = sign(PublicationReceipt.model_validate(fields), key)
    for path in _signed_field_paths(fields):
        tampered = signed.model_dump(mode="json")
        parent: Any = tampered
        for part in path[:-1]:
            parent = parent[part]
        del parent[path[-1]]
        result = _verify_json(json.dumps(tampered), key.public_key_pem(), None)
        assert result == VerificationResult(False, "bad_signature"), path


def test_changed_signed_values_fail_before_claim_checks(fields: dict[str, Any], key: IssuerKey) -> None:
    signed = sign(PublicationReceipt.model_validate(fields), key)
    tampered = signed.model_dump(mode="json")
    tampered["reviewer"]["reviewed_sha"] = OTHER_HEAD
    tampered["receipt_version"] = 2
    assert verify(PublicationReceipt.model_validate(tampered), key.public_key_pem()) == VerificationResult(
        False, "bad_signature"
    )


@pytest.mark.parametrize(
    ("expected_head", "code", "message"),
    [(None, 0, "valid"), (HEAD, 0, "valid"), (OTHER_HEAD, 1, "live_head_mismatch")],
)
def test_cli_exit_codes(
    fields: dict[str, Any], key: IssuerKey, tmp_path: Path, expected_head: str | None, code: int, message: str
) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(sign(PublicationReceipt.model_validate(fields), key).model_dump_json(), encoding="utf-8")
    public_path = tmp_path / "issuer.pub.pem"
    public_path.write_bytes(key.public_key_pem())
    command = [
        sys.executable,
        "-m",
        "lazybridge.ext.receipt",
        "verify",
        str(receipt_path),
        "--public-key",
        str(public_path),
    ]
    if expected_head is not None:
        command += ["--expected-head", expected_head]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode == code, result.stderr
    assert result.stdout.strip() == message


def test_key_file_created_once_and_reused(tmp_path: Path) -> None:
    path = tmp_path / "keys" / "issuer.pem"
    first = IssuerKey(path)
    original = path.read_bytes()
    mtime = path.stat().st_mtime_ns
    second = IssuerKey(path)
    assert path.read_bytes() == original
    assert path.stat().st_mtime_ns == mtime
    assert first.key_id == second.key_id
    assert first.public_key_pem() == second.public_key_pem()
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_concurrent_key_creation_reuses_one_key(tmp_path: Path) -> None:
    path = tmp_path / "keys" / "issuer.pem"
    with ThreadPoolExecutor(max_workers=8) as executor:
        key_ids = {key.key_id for key in executor.map(IssuerKey, [path] * 8)}
    assert key_ids == {IssuerKey(path).key_id}
    assert list(path.parent.iterdir()) == [path]


def test_key_creation_without_hard_links(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "keys" / "issuer.pem"
    open_key = os.open
    link_attempts = []

    def unsupported_link(source: Path, target: Path) -> None:
        link_attempts.append(target)
        raise OSError(errno.EOPNOTSUPP, "Hard links are unavailable")

    def check_exclusive_create(file: Path, flags: int, mode: int = 0o777) -> int:
        if file == path:
            assert flags & os.O_CREAT
            assert flags & os.O_EXCL
            assert mode == 0o600
        return open_key(file, flags, mode)

    monkeypatch.setattr(os, "link", unsupported_link)
    monkeypatch.setattr(os, "open", check_exclusive_create)
    first = IssuerKey(path)
    original = path.read_bytes()
    assert IssuerKey(path).key_id == first.key_id
    assert path.read_bytes() == original
    assert link_attempts == [path]
    assert list(path.parent.iterdir()) == [path]


def test_exclusive_key_creation_preserves_concurrent_winner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    winner_path = tmp_path / "winner.pem"
    winner = IssuerKey(winner_path)
    winner_pem = winner_path.read_bytes()
    path = tmp_path / "issuer.pem"

    def unsupported_link(source: Path, target: Path) -> None:
        target.write_bytes(winner_pem)
        raise OSError(errno.EOPNOTSUPP, "Hard links are unavailable")

    monkeypatch.setattr(os, "link", unsupported_link)
    assert IssuerKey(path).key_id == winner.key_id
    assert path.read_bytes() == winner_pem


def test_concurrent_reader_waits_for_exclusive_key_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "issuer.pem"
    open_key = os.open
    read_bytes = Path.read_bytes
    created = Event()
    read_empty = Event()

    def unsupported_link(source: Path, target: Path) -> None:
        raise OSError(errno.EOPNOTSUPP, "Hard links are unavailable")

    def pause_after_create(file: Path, flags: int, mode: int = 0o777) -> int:
        fd = open_key(file, flags, mode)
        if file == path:
            created.set()
            assert read_empty.wait(timeout=5)
        return fd

    def track_empty_read(file: Path) -> bytes:
        data = read_bytes(file)
        if file == path and not data:
            read_empty.set()
        return data

    monkeypatch.setattr(os, "link", unsupported_link)
    monkeypatch.setattr(os, "open", pause_after_create)
    monkeypatch.setattr(Path, "read_bytes", track_empty_read)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(IssuerKey, path)
        assert created.wait(timeout=5)
        second = executor.submit(IssuerKey, path)
        assert first.result(timeout=5).key_id == second.result(timeout=5).key_id


def test_missing_cryptography_names_the_extra() -> None:
    script = 'import sys; sys.modules["cryptography"] = None; import lazybridge.ext.receipt'
    result = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True)
    assert result.returncode == 1
    assert (
        'ImportError: Publication receipts require cryptography. Install it with pip install "lazybridge[receipt]".'
        in (result.stderr)
    )
