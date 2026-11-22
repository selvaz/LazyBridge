# Delegation audit end-to-end tests

Run the offline scenarios serially with `python tests/e2e_audit/run_all.py`.
Scratch data defaults to `<temp>/lb-e2e`; set `LB_E2E_WORK` to choose a
different scratch root and `LB_E2E_EVIDENCE` to choose the report directory.
Set `LB_E2E_CODE_ROOT` to a checkout whose LazyBridge code should be tested.
The running Python interpreter is inherited by all child processes.

To opt into the individual pytest cases, run
`python -m pytest -m e2e tests/e2e_audit/test_e2e_audit.py`. They are excluded
from the default test run. Each case uses an isolated pytest temporary folder.
