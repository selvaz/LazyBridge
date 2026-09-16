# LazyBridge announcement bot — INACTIVE / MOCK

> **DRY RUN ONLY. This tooling must remain inactive until an operator has
> explicitly answered both questions in
> `docs/marketing/GATE-QUESTIONS.md`: who owns and controls the credentials,
> and what level of approval is required before publishing.**

This directory contains a local announcement simulator for LazyBridge's
possible future social presence. It validates illustrative drafts, applies a
platform-specific character limit, and records what *would* have been posted.
It never publishes, makes no HTTP request, reads no environment variable, and
uses no credential.

There is deliberately no live flag, endpoint, account integration, networking
library, or credential-loading path. The generated record is a timestamped JSON
file under `dry_run_output/`.

## Draft format

The bot accepts UTF-8 Markdown (`.md`) and JSON (`.json`) files. Both formats
require:

- non-empty post text;
- `status` metadata set to `DRAFT`;
- non-empty `campaign` metadata.

Markdown places flat `key: value` metadata between opening and closing `---`
lines, followed by the post text. JSON uses an object with a string `text`
field and a `metadata` object. The included `example_draft.md` demonstrates the
Markdown format and is intentionally fictional placeholder copy.

Run the simulator from the repository root:

```text
python tools/marketing/dry_run_bot.py --draft tools/marketing/example_draft.md --platform twitter
```

`twitter` and `x` are labels for the same 280-character local validation rule.
Additional platforms can later be added explicitly to `PLATFORM_RULES`; an
unknown platform is rejected.

## What would have to change to go live

This script itself must stay dry-run only. After both gate questions are
answered and an operator separately approves implementation, live publishing
would require a distinct, reviewed entry point and publisher adapter rather
than a switch in this CLI. For an X account, the operator would have to approve
the exact account and secret store, then provision environment variables named
`LAZYBRIDGE_X_ACCOUNT_ID` and `LAZYBRIDGE_X_ACCESS_TOKEN`. Those names are
illustrative requirements only: this code does not inspect them, and no real
value should be added to this directory or its drafts.

The separate live adapter would need an explicitly approved HTTP dependency
and one authenticated HTTP POST operation to the platform's create-post
resource. That call would submit the approved text for the approved account,
check the response status and returned post identifier, and handle retry and
audit behavior according to the operator's approval policy. Before any such
code exists, an operator must also confirm the platform's current API rules,
choose the real resource, approve the request fields and authentication
scheme, and define who may trigger the call. No endpoint URL or ready-to-run
SDK example is included here, so this scaffold cannot be turned live by adding
a token.

INACTIVE / MOCK — do not wire this script, its output writer, or any similarly
named publishing function to a real API without explicit operator approval.
