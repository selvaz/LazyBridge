# Local Codex release check

The [recorded run](codex-release-check.json) used Codex CLI `0.167.0-alpha.4.1`
on Windows build 26100 at 02:06 UTC on November 24, 2026. It used the bundled
profile document at revision `2026-11-23.1` and the
`CodexAppServerClient` stdio launch path. The JSON includes the runtime revision
and individual probe results.

| Profile | Network off | Network on |
| --- | --- | --- |
| `read` | **Red.** The standalone command RPC wrote to scratch and the sibling and reached both endpoints with networking off. | **Red.** The standalone command RPC wrote to both folders. Both shell and MCP reached the undeclared endpoint. |
| `edit` | **Red.** The standalone command RPC wrote to the sibling and reached both endpoints. The model's thread reported read-only and its scratch write failed. | **Red.** The file failures persisted. Both shell and MCP reached the undeclared endpoint. |
| `edit-and-run` | **Red.** The standalone command RPC wrote to the sibling and reached both endpoints. The model exposed command tools without a verified allowlist. | **Red.** The file and command-scope failures persisted. Both shell and MCP reached the undeclared endpoint. |

The write probe verified both payloads on disk in every row. The shell probe
received the canary response from both endpoints after the Windows bootstrap
variables were restored. The standalone command RPC also read the sibling in
every row. The scripted model exposed no verified scoped file-read tool, and
model-facing shell denial still passed for `read` and `edit`.

The native process excluded both ambient MCP servers and disabled the hostile
project config. With networking off, the disabled MCP server rejected direct
calls. With networking on, discovery exposed the granted MCP tool and direct
calls to the hidden tool were rejected. Native resume returned the original ID
in all six rows, and a changed policy revision was refused before spawning
another app server. The repeated enforcement check on resume was red in every row.

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
retain their native state in the executor-owned `policy_state_dir`. For
API-key sessions, the executor supplies `policy_credential_id` as a UUID
assigned to its credential record. The resume record stores that identifier
alongside the admission and policy digest. Resolve the UUID and key together
in the executor and retain the UUID when rotating a key for the same identity.
All six combinations in this report are ineligible.

## ChatGPT subscription authentication

Existing Codex login reuse is unavailable in the governed launcher. On the
tested CLI, `codex login status` in a fresh isolated home returned exit 1
with the file, keyring, and ephemeral stores. The documented file store
resolves `auth.json` under `CODEX_HOME`, and Codex refreshes managed ChatGPT
credentials during use. The documentation gives no separate cache pointer.
A regular filesystem link would leave the target subject to writes.
[Codex authentication](https://learn.chatgpt.com/docs/auth#credential-storage)
· [Codex environment variables](https://learn.chatgpt.com/docs/config-file/environment-variables)

The no-copy options are a separate ChatGPT sign-in for an executor-owned
home, or host-managed OAuth with in-memory tokens and a refresh handler.
Eligible Enterprise workspaces can also provision a Codex access token for
the app-server child environment. Each option needs its own executor
integration and credential binding.
[Codex App Server authentication](https://learn.chatgpt.com/docs/app-server#authentication)
· [Codex access tokens](https://learn.chatgpt.com/docs/enterprise/access-tokens)
