# Local Codex release check

The [recorded run](codex-release-check.json) used Codex CLI `0.167.0-alpha.4.1`
on Windows build 26100 at 01:46 UTC on November 24, 2026. It used the bundled
profile document at revision `2026-11-23.1` and the
`CodexAppServerClient` stdio launch path. The JSON includes the runtime revision
and individual probe results.

| Profile | Network off | Network on |
| --- | --- | --- |
| `read` | **Red.** The standalone shell read the sibling fixture, and the model had no verified scoped file-read tool. | **Red.** The same read failures apply, and the granted MCP host reached an undeclared endpoint. |
| `edit` | **Red.** Codex reported a read-only thread, denied the scratch write, and showed the same read failures. | **Red.** The same file failures apply, and the granted MCP host reached an undeclared endpoint. |
| `edit-and-run` | **Red.** The file failures apply, and the model exposed command tools without a verified command allowlist. | **Red.** The same file and shell failures apply. Shell could not reach the declared endpoint, and the MCP host reached the undeclared endpoint. |

The native process ignored both ambient MCP servers and disabled the hostile
project config. With networking off, the shell could reach neither endpoint and
the disabled MCP server rejected direct calls. With networking on, the granted
MCP tool appeared in discovery and the hidden tool rejected direct calls.
Model-facing shell denial passed for `read` and `edit`. Native thread resume
returned the original ID in all six rows, and a changed policy revision was
refused before spawning another app server.

## Run the check

```powershell
python -m lazybridge.policy.release_check --fixture-parent . --output docs/codex-release-check.json
```

Use `--profiles PATH` to check a deployment's pinned profile TOML. The module
starts the native CLI against disposable sibling folders and uses loopback
canaries and a scripted Responses endpoint. It requires no model credentials.
Exit code 1 means at least one row is red. Any missing required probe makes its
row red.

For governed launches, supply the executor-owned report through
`CodexAppServerClient(policy_release_report=...)`. The client matches evidence
to the profile, granted tool classes, network setting, policy revision, and
runtime revision before checking the effective configuration and thread
settings. A new governed home starts with the compiled TOML. Durable threads
retain their native state in the executor-owned `policy_state_dir`, where the
launcher binds each ID to its admission, policy digest, and credential
identity. All six combinations in this report are ineligible.
