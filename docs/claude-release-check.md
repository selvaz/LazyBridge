# Local Claude Code release check

The [recorded run](claude-release-check.json) used Claude Code `2.1.283`
bundled with Claude Agent SDK `0.2.129` on Windows build 26100 at 02:47 UTC on
November 24, 2026. It used policy revision `2026-11-23.1` and the
`AgentSdkClient/stdio` launch path. The JSON records the runtime fingerprint
and individual start and resume results.

| Profile | Network off | Network on |
| --- | --- | --- |
| `read` | **Red.** Native file and shell-denial checks passed. Worker isolation remains unverified. | **Red.** The ungranted MCP tool appeared in discovery. The granted tool host reached the undeclared endpoint. Worker isolation remains unverified. |
| `edit` | **Red.** In-folder writes succeeded and sibling writes failed. Worker isolation remains unverified. | **Red.** The ungranted MCP tool appeared in discovery. The granted tool host reached the undeclared endpoint. Worker isolation remains unverified. |
| `edit-and-run` | **Red.** Bash was absent and its direct call failed. A scoped shell worker with a granted command positive control is missing. | **Red.** The scoped shell worker is missing. The ungranted MCP tool appeared in discovery and the granted tool host reached the undeclared endpoint. |

All six rows rejected reads and writes to the disposable sibling folder.
In-folder reads returned the expected canary. In-folder writes succeeded for
both edit profiles and were denied for `read`. The hostile user and project
MCP servers were absent from discovery and rejected direct invocation. The
direct call to the ungranted MCP tool failed even in rows where the tool was
visible. Networking off removed the MCP server and its operations.

Each resume returned the original native session ID and repeated the
enforcement checks. The SDK client rejected a changed policy revision before
launching another process. The launcher inherited the parent process
environment, and the check supplied no independent confinement for
managed settings or worker processes. Every row remains ineligible.

## Run the check

```powershell
python -m lazybridge.policy.release_check --engine claude --fixture-parent . --output docs/claude-release-check.json
```

Install the `claude-code` extra to run the check. Use `--profiles PATH` for
a pinned profile document and `--claude-bin PATH` to select a specific CLI.
Each row starts and resumes the native CLI against a scripted loopback
Messages endpoint using a fixture key. The check uses no subscription quota.
The network-on grant covers the
fixture MCP server and one loopback route. The model sees native file tools
and the granted MCP configuration. Exit code 1 indicates at least one red
row. A missing probe makes its row red.

For admission, an executor can supply a trusted report located outside the
assigned folder through `AgentSdkClient(policy_release_report=...)`.
Admission requires a complete green row for the policy version and revision,
profile, granted tool classes, network value, and runtime fingerprint. The
client pins the matched executable for the launch. Shell grants and
network-on settings still require a confined worker. Governed session resume
requires a session bound to the same policy revision and grant by that SDK
client.
