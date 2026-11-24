"""Provision approver credentials for a local approval registry."""

from __future__ import annotations

import argparse

from lazybridge import Store

from .queue import DEFAULT_PREFIX, ApprovalQueue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="Path to the operator-controlled approval database")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="Queue prefix in that database")
    commands = parser.add_subparsers(dest="command", required=True)
    issue = commands.add_parser("issue-approver-credential", help="Print a new approver secret once")
    issue.add_argument("principal")
    issue.add_argument("--role", action="append", required=True, help="Role to grant, repeatable")
    args = parser.parse_args()
    with Store(db=args.db) as store:
        queue = ApprovalQueue(store, prefix=args.prefix, allow_credential_issuance=True)
        print(queue.issue_approver_credential(args.principal, args.role))


if __name__ == "__main__":
    main()
