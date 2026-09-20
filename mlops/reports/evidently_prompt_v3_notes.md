# Evidently regression notes — `prompt_v3`

- **pct_tests_passed:** 100.00%
- **promoted:** True (threshold=0.8)
- **per-check:** {'correctness_rate': 1.0, 'refusal_rate': 1.0}
- **suite tests:** 2/2 passed

## Failing cases

None — all scored checks passed.

## Judge sanity

Scripted heuristic judges (token overlap + refusal phrases). Categories should match human reading for golden scripted answers; `--bad-prompt-demo` should fail both checks.
