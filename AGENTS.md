
# InferenceX PCA Demo

Streamlit analysis of SemiAnalysis InferenceX benchmark data.

## Invariants

- PCA inputs are configuration and workload fields only.
- Outcome metrics are overlays or supervised targets.
- Random Forest is the current supervised baseline.
- TabFM and VAE are future work unless explicitly requested.
- Do not infer or document the later investment product.
- Never commit local datasets, dumps, exports, logs, environments, caches, or secrets.
- Preserve grouped evaluation by config_id.
- Do not change analytical behavior merely to match README results.

## Validation

- Run Python syntax and import checks.
- Start Streamlit headlessly when data and dependencies are available.
- Report any checks that could not run.
- Use git status, add exact paths, commit, and push.
- Never force-push or use git reset --hard.
