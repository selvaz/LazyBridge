# DRAFT / PROPOSAL ONLY — NOT EXECUTED

> **Operator approval gate:** This document proposes changes to the live public
> metadata of `selvaz/LazyBridge`. Nothing below has been executed. Do not run
> any command in this document, edit the GitHub repository metadata, push a
> branch, or open/merge a pull request without explicit operator sign-off.

Date prepared: 2026-09-16

Source audit: [`audit-2026-09-15.md`](audit-2026-09-15.md)

## Scope and verified facts

The audit found two public metadata inconsistencies:

1. GitHub reports the repository license as **Other / `NOASSERTION`**.
2. The repository has no GitHub Topics.

The intended license is not ambiguous inside the repository:

- the root `LICENSE` identifies itself as **Apache License, Version 2.0**;
- `pyproject.toml` declares `license = { text = "Apache-2.0" }` and the
  `License :: OSI Approved :: Apache Software License` classifier; and
- the README badge and Licence section both say Apache 2.0.

The root `LICENSE` is nevertheless not the canonical Apache-2.0 template. After
`END OF TERMS AND CONDITIONS`, it substitutes a project-specific copyright and
application block for the canonical appendix. GitHub explains that its Licensee
detector compares `LICENSE` with known licenses and recommends simplifying a
license file, with extra complexity documented elsewhere, when a known license
is not detected. There is no `gh repo edit` flag that sets a detected license;
the repository content must be corrected and GitHub must re-detect it.

References:

- [GitHub: Licensing a repository](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository#detecting-a-license)
- [GitHub CLI: `gh repo edit`](https://cli.github.com/manual/gh_repo_edit)
- [Apache License 2.0 canonical text](https://www.apache.org/licenses/LICENSE-2.0.txt)

## Proposal 1 — make the Apache-2.0 license detectable

### Proposed change

After approval, use a **clean checkout or worktree** and a review branch. Replace
the root `LICENSE` with the unmodified canonical text published by Apache. Do
not add project-specific prose to that file. The existing copyright statement
does not need to be moved into `NOTICE` merely to make detection work: creating
a `NOTICE` file can impose downstream notice-retention obligations and should
be a separate legal/project decision.

Proposed commands from the repository root (PowerShell on Windows):

```powershell
git status --short
git switch -c metadata/apache-license-detection
curl.exe --fail --location "https://www.apache.org/licenses/LICENSE-2.0.txt" --output LICENSE
git diff --check
git diff -- LICENSE
git add -- LICENSE
git commit -m "Normalize Apache-2.0 license text"
git push -u origin metadata/apache-license-detection
```

Stop if `git status --short` is not empty, if the download fails, or if the diff
changes anything except `LICENSE`. Review and merge the resulting one-file
change through the repository's normal protected workflow; branch creation,
push, review, and merge all remain subject to explicit operator approval.

After the approved change is merged into the default branch, verify without
mutating repository state:

```powershell
gh api repos/selvaz/LazyBridge/license --jq '{key: .license.key, name: .license.name, spdx_id: .license.spdx_id}'
gh repo view selvaz/LazyBridge --json licenseInfo --jq .licenseInfo
```

Expected SPDX result: `Apache-2.0`, rather than `NOASSERTION`. GitHub's display
may update asynchronously; a temporary stale result is not a reason to make a
second license change immediately.

### Why this is worth fixing

A visitor currently sees an unexplained conflict: the README and PyPI metadata
promise Apache 2.0, while GitHub's prominent repository metadata says “Other.”
That creates avoidable doubt during a fast adoption, procurement, or compliance
review. Correct detection makes the permission model legible from the repository
landing page and enables GitHub's `license:Apache-2.0` search/filter behavior.

### Real risk and reversibility

- Even though the intended license stays Apache 2.0, replacing a public license
  file can look like a relicense. The diff and commit message must make clear
  that this normalizes the already-declared license rather than changes terms.
- Removing the project-specific copyright line from `LICENSE` reduces its
  visibility in that particular file. Copyright is not lost by removing that
  line, but the operator should decide whether attribution belongs elsewhere;
  that is outside this narrowly scoped fix.
- Fetching a remote file at execution time is a supply-chain boundary. The
  operator must inspect the complete diff and confirm that it is the Apache
  License 2.0 text before committing.
- GitHub detection is heuristic and may be delayed, so normalization improves
  the input but cannot promise an immediate UI refresh.
- The Git change is mechanically reversible with a later revert, but a public
  revert remains in history and cannot retract copies already fetched by users.
  For that reason, review before merge is more important than relying on
  rollback.

## Proposal 2 — add focused GitHub Topics

### Proposed topic list

The following 13 topics describe features explicitly present in `README.md` or
package metadata, rather than aspirational marketing terms:

| Topic | Repository evidence |
| --- | --- |
| `python` | Python 3.11+ library/package |
| `llm` | Multi-provider LLM framework |
| `ai-agents` | The primary public abstraction is `Agent` |
| `agent-framework` | Framework for building and composing agents |
| `multi-agent-systems` | Agents can be tools of other agents |
| `tool-calling` | Unified typed tool contract and parallel tool calls |
| `workflow-engine` | Deterministic `Plan` pipelines and routing |
| `dag` | Compile-time-validated plan DAGs |
| `human-in-the-loop` | `HumanEngine` approval gates |
| `opentelemetry` | OTel GenAI export is a documented feature |
| `openai` | Supported LLM provider extra |
| `anthropic` | Supported LLM provider extra |
| `google-gemini` | Supported Google GenAI provider extra |

This stays below GitHub's limit of 20. All names use only lowercase letters and
hyphens and are below the 50-character per-topic limit. The list deliberately
omits `mcp`: the README documents MCP composition through the sibling
`lazytoolkit` package, while the connector itself no longer ships in this repo.

After approval, first run the read-only check so any topics added since the
audit are preserved:

```powershell
gh repo view selvaz/LazyBridge --json repositoryTopics --jq '[.repositoryTopics[].name]'
```

Then add the proposed topics. `--add-topic` is the supported `gh repo edit`
option and adds rather than replaces topics:

```powershell
gh repo edit selvaz/LazyBridge --add-topic python --add-topic llm --add-topic ai-agents --add-topic agent-framework --add-topic multi-agent-systems --add-topic tool-calling --add-topic workflow-engine --add-topic dag --add-topic human-in-the-loop --add-topic opentelemetry --add-topic openai --add-topic anthropic --add-topic google-gemini
```

Verify with the same read-only command:

```powershell
gh repo view selvaz/LazyBridge --json repositoryTopics --jq '[.repositoryTopics[].name]'
```

GitHub also supports the equivalent UI operation: on the repository main page,
select the gear beside **About**, enter the topics in **Topics**, and select
**Save changes**.

References:

- [GitHub: Classifying a repository with topics](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/classifying-your-repository-with-topics)
- [GitHub CLI: `gh repo edit`](https://cli.github.com/manual/gh_repo_edit)

### Why this is worth fixing

Topics are visible on the repository landing page, are clickable entry points
to related projects, and participate in GitHub topic browsing and repository
search. A focused list tells a new visitor in seconds that LazyBridge is a
Python agent framework with tool calling, deterministic workflows,
human-in-the-loop controls, observability, and multiple model providers. The
empty field currently provides none of those discovery or trust cues.

### Real risk and reversibility

- Topics are public positioning. Over-broad or inaccurate topics can attract
  mismatched users and weaken trust. Provider topics can also imply a depth of
  integration that must remain true as the project evolves.
- Too many topics dilute the strongest discovery signals. The proposed list is
  intentionally focused, but the operator may approve a smaller subset.
- Topic changes take effect immediately on the live repository; there is no PR
  review layer and no repository commit recording the change.
- The change is fully reversible in GitHub metadata. To remove exactly this
  proposed set without disturbing unrelated topics, use the following command
  only after separate operator approval:

```powershell
gh repo edit selvaz/LazyBridge --remove-topic python --remove-topic llm --remove-topic ai-agents --remove-topic agent-framework --remove-topic multi-agent-systems --remove-topic tool-calling --remove-topic workflow-engine --remove-topic dag --remove-topic human-in-the-loop --remove-topic opentelemetry --remove-topic openai --remove-topic anthropic --remove-topic google-gemini
```

## Approval checklist

- [ ] Operator approves normalizing `LICENSE` to the canonical Apache-2.0 text.
- [ ] Operator approves the exact topic set (or records an edited subset).
- [ ] Execution happens from a clean checkout/worktree with the full license
      diff reviewed before commit.
- [ ] The license content change is reviewed and merged through the normal
      repository workflow.
- [ ] The topic mutation is run by an authenticated repository administrator.
- [ ] Post-change read-only checks show `Apache-2.0` and the approved topics.

Until every applicable approval is recorded, this document remains a proposal
only and **none of its mutating commands may be run**.
