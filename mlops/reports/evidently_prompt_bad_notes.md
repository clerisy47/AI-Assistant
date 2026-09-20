# Evidently regression notes — `prompt_bad` (deliberate failure demo)

Generated with `--bad-prompt-demo` (no MLflow).

- **pct_tests_passed:** 0.00%
- **promoted:** False (threshold=0.8)
- **failing cases:** all golden ids (`sufficient_first_search`, `needs_re_search`,
  `verify_fail_then_revise`, `clarification`, `kb_unavailable_recognized`)

## Judge sanity

Incorrect / fabricated currents fail both checks as expected. Regenerate HTML:

```bash
uv run --extra mlops python -m mlops.evidently_regression \
  --bad-prompt-demo --prompt-version prompt_bad --no-mlflow
```
