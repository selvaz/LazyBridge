<!--
  Thanks for sending a PR.  The checklist below mirrors the gates in
  `CONTRIBUTING.md` and the CI workflows in `.github/workflows/`.
  Tick what applies; strike-through (`~~~~`) what genuinely doesn't.
-->

## Summary

<!-- One paragraph: what the PR changes and why.  Link to the issue
     or audit finding it closes. -->

## Tier

<!-- core / ext / external_tools / docs / build — see
     docs/guides/core-vs-ext.md for the import-boundary policy. -->

## Checklist

- [ ] `pytest` green locally
- [ ] `ruff check lazybridge tests` clean
- [ ] `ruff format --check lazybridge tests` clean
- [ ] `mypy lazybridge` clean (default tier and any touched strict tier)
- [ ] If touching `SKILL.md`: `python -m lazybridge.skill_docs._build --check`
- [ ] If touching `docs/`: `mkdocs build --strict`
- [ ] CHANGELOG entry under `[Unreleased]` (or active version)
- [ ] If adding a new doc page: added to `mkdocs.yml` nav
- [ ] If touching a public symbol: confirmed it's still in `lazybridge.__all__`

## Notes for review

<!-- Anything reviewer-facing: trade-offs taken, alternatives rejected,
     follow-up tickets you intend to file. -->
