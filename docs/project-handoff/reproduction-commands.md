# Reproduction commands

Normal work uses `.venv-streamlit/bin/python`; verify the frozen July data with
`test -d .data/inferencex-db-dump-2026-07-20`. Launch: `./run_dashboard.sh`.
Epoch raw hashes: `.venv-streamlit/bin/python -c 'from modeling.integration import verify_epoch_raw_hashes; print(verify_epoch_raw_hashes())'`.

Build Epoch integration: `.venv-streamlit/bin/python scripts/build_hardware_integration.py --data-dir .data/inferencex-db-dump-2026-07-20`.
Run conventional validation (reconciliation, A/B/C/D RF/CatBoost ladder,
holdout/support diagnostics): `.venv-streamlit/bin/python scripts/run_hardware_validation.py`.

Optional / expensive historical TabFM reproduction uses `.venv-tabfm/bin/python`:
`.venv-tabfm/bin/python scripts/run_hardware_validation.py --include-tabfm --historical-only`.
It is not needed for integration/DC Bench and should reproduce R² 0.961979 ±
0.008605 and MAE 338.540 tokens/s/GPU.

DC Bench: `.venv-streamlit/bin/python scripts/build_dcbench_inference_questions.py`;
score `responses.jsonl` with `.venv-streamlit/bin/python scripts/evaluate_dcbench_inference.py responses.jsonl`.
Focused tests: `.venv-streamlit/bin/python -m unittest tests.test_integration tests.test_hardware_validation tests.test_dcbench -v`.
Full: `.venv-streamlit/bin/python -m unittest discover -s tests -v`.
Syntax: `.venv-streamlit/bin/python -m py_compile modeling/dcbench.py scripts/build_dcbench_inference_questions.py scripts/evaluate_dcbench_inference.py`.
Check: `git diff --check`.
