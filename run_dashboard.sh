#!/usr/bin/env bash
# Launch the portable, immutable InferenceX dashboard checkpoint on macOS/Linux.
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python3.11}"
venv_dir="$repo_root/.venv-dashboard"
lock_file="$repo_root/requirements-lock.txt"
lock_marker="$venv_dir/.inferencex-requirements-sha256"

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Unsupported Python setup: $python_bin was not found. Install CPython 3.11 and retry." >&2
  exit 2
fi

python_version="$($python_bin -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$python_version" != "3.11" ]]; then
  echo "Unsupported Python version: found $python_version; this dashboard launcher requires CPython 3.11." >&2
  exit 2
fi

if [[ ! -x "$venv_dir/bin/python" ]]; then
  echo "Creating dashboard environment with $python_bin..."
  "$python_bin" -m venv "$venv_dir"
fi

lock_digest="$($venv_dir/bin/python -c 'import hashlib, pathlib; print(hashlib.sha256(pathlib.Path("requirements-lock.txt").read_bytes()).hexdigest())')"
if [[ ! -f "$lock_marker" ]] || [[ "$(<"$lock_marker")" != "$lock_digest" ]]; then
  echo "Installing locked dashboard dependencies..."
  "$venv_dir/bin/python" -m pip install --disable-pip-version-check -r "$lock_file"
  printf '%s\n' "$lock_digest" > "$lock_marker"
fi

echo "Validating frozen dashboard data..."
"$venv_dir/bin/python" scripts/bootstrap_dashboard_data.py
echo "Validating active research artifacts..."
"$venv_dir/bin/python" scripts/bootstrap_dashboard_artifacts.py

export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
exec "$venv_dir/bin/streamlit" run apps/inferencex_pca_demo.py "$@"
