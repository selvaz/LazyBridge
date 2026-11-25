"""Provision approver credentials for a local approval registry."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from lazybridge import Store

from .queue import DEFAULT_PREFIX, ApprovalQueue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="Path to the operator-controlled approval database")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="Queue prefix in that database")
    commands = parser.add_subparsers(dest="command", required=True)
    issue = commands.add_parser("issue-approver-credential", help="Write a new approver credential to a private file")
    issue.add_argument("principal")
    issue.add_argument("--role", action="append", required=True, help="Role to grant, repeatable")
    issue.add_argument(
        "--output", type=Path, required=True, help="New credential file in an operator-controlled directory"
    )
    args = parser.parse_args()
    with Store(db=args.db) as store:
        queue = ApprovalQueue(store, prefix=args.prefix, allow_credential_issuance=True)
        try:
            fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            parser.error("the credential output file already exists")
        else:
            try:
                with os.fdopen(fd, "w", encoding="ascii") as output:
                    output.write(queue.issue_approver_credential(args.principal, args.role) + "\n")
            except Exception:
                args.output.unlink(missing_ok=True)
                raise


if __name__ == "__main__":
    main()
