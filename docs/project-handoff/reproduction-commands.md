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

DC Bench clean-CI rebuild uses the tracked immutable source slice:
`.venv-streamlit/bin/python scripts/build_dcbench_inference_questions.py` and
`.venv-streamlit/bin/python scripts/build_dcbench_inference_v0_2_candidates.py`.
When the excluded local full enriched view is available, regenerate the exact
slice with `.venv-streamlit/bin/python scripts/build_dcbench_source_slice.py`;
then prove full/slice equivalence with
`.venv-streamlit/bin/python scripts/verify_dcbench_source_slice_equivalence.py`.
The `--source full` option is an explicit local provenance check, not a normal
CI requirement. Score `responses.jsonl` with
`.venv-streamlit/bin/python scripts/evaluate_dcbench_inference.py responses.jsonl`.
Focused tests: `.venv-streamlit/bin/python -m unittest tests.test_integration tests.test_hardware_validation tests.test_dcbench tests.test_dcbench_v0_2_candidates -v`.
Full: `.venv-streamlit/bin/python -m unittest discover -s tests -v`.
Syntax: `.venv-streamlit/bin/python -m py_compile modeling/dcbench.py modeling/dcbench_source.py modeling/dcbench_v0_2_candidates.py scripts/build_dcbench_source_slice.py scripts/build_dcbench_inference_questions.py scripts/build_dcbench_inference_v0_2_candidates.py scripts/evaluate_dcbench_inference.py`.
Check: `git diff --check`.
