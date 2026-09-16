"""INACTIVE / MOCK announcement bot for local dry runs only.

This module validates a local draft and writes a local simulation record. It
cannot send a post, makes no network requests, and reads no credentials. Do not
wire it to a real API without explicit operator approval.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PlatformRule:
    """A local-only validation rule for a simulated destination."""

    label: str
    character_limit: int


# INACTIVE / MOCK: this table changes validation and labels only. It does not
# select an API, account, endpoint, or credential.
PLATFORM_RULES = {
    "twitter": PlatformRule(label="Twitter/X", character_limit=280),
    "x": PlatformRule(label="X/Twitter", character_limit=280),
}

REQUIRED_METADATA_FIELDS = ("status", "campaign")


@dataclass(frozen=True)
class Draft:
    """Validated draft content and its non-secret descriptive metadata."""

    text: str
    metadata: dict[str, Any]


def _parse_markdown(path: Path) -> Draft:
    """Parse Markdown with a small, flat metadata front matter block."""

    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("Markdown drafts must start with a '---' metadata block")

    try:
        closing_index = next(
            index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"
        )
    except StopIteration as exc:
        raise ValueError("Markdown metadata block is missing its closing '---'") from exc

    metadata: dict[str, Any] = {}
    for line_number, line in enumerate(lines[1:closing_index], start=2):
        if not line.strip():
            continue
        key, separator, value = line.partition(":")
        if not separator or not key.strip() or not value.strip():
            raise ValueError(
                f"Invalid metadata on line {line_number}; use 'key: value'"
            )
        metadata[key.strip()] = value.strip()

    text = "\n".join(lines[closing_index + 1 :]).strip()
    return Draft(text=text, metadata=metadata)


def _parse_json(path: Path) -> Draft:
    """Parse a JSON object containing text and a metadata object."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc.msg} at line {exc.lineno}") from exc

    if not isinstance(payload, dict):
        raise ValueError("JSON draft must be an object")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("JSON draft field 'metadata' must be an object")
    return Draft(text=payload.get("text", ""), metadata=metadata)


def load_draft(path: Path) -> Draft:
    """Load a local Markdown or JSON draft without consulting external data."""

    if not path.is_file():
        raise ValueError(f"Draft file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".md":
        return _parse_markdown(path)
    if suffix == ".json":
        return _parse_json(path)
    raise ValueError("Draft must use the .md or .json extension")


def validate_draft(draft: Draft, rule: PlatformRule) -> None:
    """Validate required fields and the simulated platform character limit."""

    if not isinstance(draft.text, str) or not draft.text.strip():
        raise ValueError("Required field 'text' must be a non-empty string")

    missing = [field for field in REQUIRED_METADATA_FIELDS if not draft.metadata.get(field)]
    if missing:
        raise ValueError(f"Missing required metadata: {', '.join(missing)}")

    status = draft.metadata["status"]
    if not isinstance(status, str) or status.strip().upper() != "DRAFT":
        raise ValueError("Metadata field 'status' must be DRAFT")

    campaign = draft.metadata["campaign"]
    if not isinstance(campaign, str) or not campaign.strip():
        raise ValueError("Metadata field 'campaign' must be a non-empty string")

    character_count = len(draft.text)
    if character_count > rule.character_limit:
        raise ValueError(
            f"Post is {character_count} characters; {rule.label} allows "
            f"at most {rule.character_limit}"
        )


def write_dry_run_record(
    draft: Draft, source_path: Path, platform: str, rule: PlatformRule
) -> Path:
    """INACTIVE / MOCK: record locally what would be posted; never send it."""

    simulated_at = datetime.now(timezone.utc)
    timestamp = simulated_at.strftime("%Y%m%dT%H%M%S.%fZ")
    output_directory = Path(__file__).resolve().parent / "dry_run_output"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{timestamp}_{platform}.json"

    record = {
        "mode": "INACTIVE / MOCK - DRY RUN ONLY",
        "would_have_posted_to": rule.label,
        "platform_key": platform,
        "would_have_posted_at_utc": simulated_at.isoformat(),
        "source_draft": str(source_path.resolve()),
        "character_count": len(draft.text),
        "character_limit": rule.character_limit,
        "text": draft.text,
        "metadata": draft.metadata,
    }
    output_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the local-only simulator."""

    parser = argparse.ArgumentParser(
        description=(
            "INACTIVE / MOCK: validate a local announcement draft and write a "
            "local record of what would have been posted. No network calls or "
            "credentials are used."
        )
    )
    parser.add_argument(
        "--draft",
        required=True,
        type=Path,
        help="path to a local .md or .json draft",
    )
    parser.add_argument(
        "--platform",
        required=True,
        choices=sorted(PLATFORM_RULES),
        help="simulated platform; changes local validation and output labeling only",
    )
    return parser


def main() -> int:
    """Run the INACTIVE / MOCK local simulation."""

    parser = build_parser()
    args = parser.parse_args()
    rule = PLATFORM_RULES[args.platform]
    try:
        draft = load_draft(args.draft)
        validate_draft(draft, rule)
        output_path = write_dry_run_record(draft, args.draft, args.platform, rule)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))

    preview = " ".join(draft.text.split())
    print(f"[DRY RUN] Would have posted to {rule.label}: {preview}")
    print(f"[DRY RUN] Local record: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
