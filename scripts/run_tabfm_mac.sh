#!/usr/bin/env bash
# Sleep prevention and reproducible CPU settings for one sequential TabFM run.
set -uo pipefail

threads=""
log_path=""
forwarded=()
while (($#)); do
  case "$1" in
    --threads) threads="$2"; shift 2 ;;
    --log) log_path="$2"; shift 2 ;;
    --help) echo "Usage: $0 [--threads N] [--log PATH] -- diagnostic-script arguments"; exit 0 ;;
    *) forwarded+=("$1"); shift ;;
  esac
done

detected=""
if [[ "$(uname -s)" == "Darwin" ]]; then
  detected="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || true)"
  [[ -n "$detected" && "$detected" != "0" ]] || detected="$(sysctl -n hw.physicalcpu 2>/dev/null || true)"
fi
if [[ -z "$threads" ]]; then
  # The current supported default remains PyTorch's own choice; pass an explicit
  # count only after the bounded benchmark records a win.
  threads="${TABFM_THREADS:-}"
fi
if [[ -n "$threads" ]]; then
  export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads" VECLIB_MAXIMUM_THREADS="$threads" NUMEXPR_NUM_THREADS="$threads"
fi
export PYTHONUNBUFFERED=1
echo "TabFM Mac launcher: detected_performance_cores=${detected:-unknown} requested_threads=${threads:-pytorch-default} sequential=true"
command=(.venv-tabfm/bin/python scripts/model_diagnostics.py "${forwarded[@]}")
if [[ -n "$log_path" ]]; then
  mkdir -p "$(dirname "$log_path")"
  if command -v caffeinate >/dev/null 2>&1; then caffeinate -dimsu "${command[@]}" 2>&1 | tee "$log_path"; else "${command[@]}" 2>&1 | tee "$log_path"; fi
  exit "${PIPESTATUS[0]}"
fi
if command -v caffeinate >/dev/null 2>&1; then exec caffeinate -dimsu "${command[@]}"; else exec "${command[@]}"; fi
