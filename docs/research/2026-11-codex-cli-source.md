# Codex CLI cloud/environment source audit (2026-11-17)

Scope: upstream `openai/codex`, pinned to released tag `rust-v0.167.0` where possible. **VERIFIED** means directly present in source; **ASSUMED** means inference/field report. No credential or private API was used.

## 1. Calls and repository resolution

**VERIFIED.** There are two path styles (ChatGPT base is normalized to `/backend-api`):

| Operation | Codex-API path / ChatGPT path |
|---|---|
| environments, all | `GET /api/codex/environments` / `GET /wham/environments` |
| environments for repository | `GET /api/codex/environments/by-repo/github/{owner}/{repo}` / `GET /wham/environments/by-repo/github/{owner}/{repo}` |
| tasks, list | `GET /api/codex/tasks/list` / `GET /wham/tasks/list` (`limit`, `task_filter=current`, `cursor`, `environment_id`) |
| task, create | `POST /api/codex/tasks` / `POST /wham/tasks` |
| task, get/status/diff/messages | `GET /api/codex/tasks/{id}` / `GET /wham/tasks/{id}` |
| sibling attempts | `GET .../tasks/{task}/turns/{turn}/sibling_turns` |
| apply | **no backend endpoint**: fetches task detail/diff, then runs local `git apply` |
| environment create/update/get-by-id | **none in this CLI** |

Task routes are constructed [here](https://github.com/openai/codex/blob/rust-v0.167.0/codex-rs/backend-client/src/client.rs#L351-L510); create sends `environment_id`, branch, QA flag and input items [here](https://github.com/openai/codex/blob/rust-v0.167.0/codex-rs/cloud-tasks-client/src/http.rs#L310-L367), while apply is local [here](https://github.com/openai/codex/blob/rust-v0.167.0/codex-rs/cloud-tasks-client/src/http.rs#L403-L536).

**VERIFIED.** The by-repo endpoint exists and both autodetection and the TUI use it. They read every local Git remote, parse GitHub owner/repo, try by-repo first, then global list; selection is label, sole result, pinned, then task-count/first ([lookup/selection](https://github.com/openai/codex/blob/rust-v0.167.0/codex-rs/cloud-tasks/src/env_detect.rs#L24-L151), [remote parsing](https://github.com/openai/codex/blob/rust-v0.167.0/codex-rs/cloud-tasks/src/env_detect.rs#L206-L284), [TUI union](https://github.com/openai/codex/blob/rust-v0.167.0/codex-rs/cloud-tasks/src/env_detect.rs#L286-L382)). Thus repo auto-selection already exists, but only for environments returned by this legacy catalogue. **ASSUMED (reported):** current `environment-configs` environments may be invisible; see [#50182](https://github.com/openai/codex/issues/50182).

## 2. Creation

**VERIFIED.** Neither `codex cloud exec` nor its TUI creates or updates an environment: the client trait exposes task operations only ([trait](https://github.com/openai/codex/blob/rust-v0.167.0/codex-rs/cloud-tasks-client/src/api.rs#L126-L166)), environment code performs GETs only, and the CLI command enum contains no environment lifecycle command. **ASSUMED:** creation is web/desktop-only because the repository contains no supported CLI surface or create route; the actual closed-source service/UI implementation cannot be verified from this repository.

## 3. Authentication and callable surfaces

**VERIFIED.** Cloud commands load the normal Codex auth manager and let the CLI issue calls: the backend client obtains auth headers from `SharedAuthProvider`, also adding `ChatGPT-Account-Id` when present ([client/header construction](https://github.com/openai/codex/blob/rust-v0.167.0/codex-rs/backend-client/src/client.rs#L126-L247)). Default credential storage is `$CODEX_HOME/auth.json`; supported configuration also permits OS keyring, auto fallback, or process-only ephemeral storage ([store modes](https://github.com/openai/codex/blob/rust-v0.167.0/codex-rs/config/src/types.rs#L1350-L1376)). ChatGPT OAuth is refreshed and persisted by `AuthManager`; on 401 it reloads shared storage, then refreshes, while API-key auth is not refreshed ([manager](https://github.com/openai/codex/blob/rust-v0.167.0/codex-rs/login/src/auth/manager.rs#L1960-L2185)). A bridge can therefore spawn `codex cloud ...` without reading any credential.

**VERIFIED at v0.167.0 (earliest release ASSUMED; upstream changelog does not record per-flag provenance).** Complete cloud surface: bare `codex cloud` TUI; `exec QUERY --env ENV_ID [--attempts 1..4] [--branch BRANCH]`; `status TASK_ID`; `list [--env ENV_ID] [--limit 1..20] [--cursor CURSOR] [--json]`; `diff TASK_ID [--attempt 1..4]`; `apply TASK_ID [--attempt 1..4]` ([Clap definitions](https://github.com/openai/codex/blob/rust-v0.167.0/codex-rs/cloud-tasks/src/cli.rs#L13-L110)). There are no hidden/unreleased repo/environment flags on current main either. **VERIFIED:** app-server protocol has no cloud-environment/task lifecycle method; MCP server is a model-tool server, not a privileged cloud administration API. Hence the supported programmatic surface is the CLI subprocess only.

## 4. Upstream requests and smallest change

**VERIFIED.** [#24777](https://github.com/openai/codex/issues/24777) explicitly requests `cloud env list/get/create/update`, repo resolution and lifecycle automation; [#50182](https://github.com/openai/codex/issues/50182) tracks failure to discover current environments. Related current-catalogue failure: [#56350](https://github.com/openai/codex/issues/56350).

**Draft (ASSUMED backend capability).** Smallest useful patch: add optional `--repo owner/name` to `ExecCommand` in `cloud-tasks/src/cli.rs`; expose a public `resolve_environment_for_repo` in `env_detect.rs`; in `cloud-tasks/src/lib.rs`, resolve via the existing by-repo GET and require exactly one match before task creation. Tests belong in `env_detect_tests.rs` and CLI parsing tests. That yields zero-registration only where a legacy environment already exists. True “resolve-or-create” additionally needs a supported server create contract and new client method—absent today—and must not guess an undocumented `environment-configs` API.

## 5. Ranked recommendation today

1. **Supported/robust:** invoke the authenticated CLI as a subprocess. Run inside the checkout and omit custom token handling; discover through the interactive TUI/manual onboarding once, persist the returned opaque legacy env ID keyed by canonical `owner/repo`, then use `codex cloud exec --env ID --branch ...`, `status/diff/apply`. This is not zero-registration.
2. **Supported but manual per repo:** have the operator create/publish in the official UI, then verify it appears in `codex cloud`; fail clearly if by-repo discovery/catalogue cannot see it. Never scrape auth or call undocumented endpoints.
3. **Not recommended:** reverse-engineer web `environment-configs` or copy tokens. It is unsupported, couples to a split catalogue, and violates credential isolation.

**Missing for genuine zero-registration:** supported environment list JSON, current-catalogue repo lookup, create/update/publish/delete, deterministic ambiguity handling, and a documented app-server/JSON-RPC equivalent. Until upstream supplies those, “ANY GitHub repo, as local” cannot be achieved using only supported surfaces.
