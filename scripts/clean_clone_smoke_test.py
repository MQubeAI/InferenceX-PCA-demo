#!/usr/bin/env python3
"""Network-backed acceptance smoke test for a fresh dashboard checkout.

Run this only from a clean clone after the pinned dashboard-data release asset is
published. It intentionally refuses an existing `.data/` directory so a cached
checkpoint cannot hide a bootstrap failure.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from modeling.artifact_checkpoint import ArtifactIndexError, bootstrap_active_artifacts  # noqa: E402
from modeling.dataset_checkpoint import (  # noqa: E402
    DatasetBootstrapError,
    VERIFIED_CHECKPOINT,
    artifact_matches_active_dataset,
    bootstrap_dataset,
    default_data_dir,
    load_data_manifest,
)


def unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def wait_for_health(port: int, process: subprocess.Popen[str], timeout: float = 45.0) -> None:
    endpoint = f"http://127.0.0.1:{port}/_stcore/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout else ""
            raise RuntimeError(f"Streamlit exited before becoming healthy:\n{output}")
        try:
            with urllib.request.urlopen(endpoint, timeout=2) as response:
                if response.read().decode("utf-8") == "ok":
                    return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError("Streamlit health endpoint did not become ready within 45 seconds.")


def main() -> int:
    manifest = load_data_manifest()
    checkpoint_dir = default_data_dir(manifest)
    if checkpoint_dir.exists():
        print(
            f"Clean-clone smoke test refused existing checkpoint: {checkpoint_dir}. "
            "Run it from a fresh checkout with no .data directory.",
            file=sys.stderr,
        )
        return 2

    try:
        bootstrap_dataset()
        bootstrap_active_artifacts(manifest)
    except (ArtifactIndexError, DatasetBootstrapError) as exc:
        print(f"Clean-clone bootstrap failed: {exc}", file=sys.stderr)
        return 2
    from apps import inferencex_pca_demo as app  # noqa: E402

    _status, source = app.data_source_status(str(checkpoint_dir))
    active = app.build_dataset_manifest(source)
    if active["verification"]["status"] != VERIFIED_CHECKPOINT:
        raise RuntimeError("Bootstrap returned without a verified checkpoint.")

    pca = app.load_pca_target_artifact()
    if not artifact_matches_active_dataset(active, pca):
        raise RuntimeError("The verified checkpoint was not accepted by the July PCA artifact.")
    for path, method in (
        (app.AE_REPRESENTATION_ARTIFACT_PATH, "autoencoder"),
        (app.VAE_REPRESENTATION_ARTIFACT_PATH, "variational_autoencoder"),
    ):
        artifact, _embedding = app.load_neural_representation_artifact(str(path), method)
        if not artifact_matches_active_dataset(active, artifact):
            raise RuntimeError(f"{method} artifact is incompatible with the verified checkpoint.")
    for loader in (
        app.load_representation_comparison_artifact,
        app.load_representation_validation_artifact,
    ):
        artifact = loader()
        if not artifact_matches_active_dataset(active, artifact):
            raise RuntimeError("A representation artifact is incompatible with the verified checkpoint.")

    from streamlit.testing.v1 import AppTest

    tested = AppTest.from_file(
        str(REPOSITORY_ROOT / "apps" / "inferencex_pca_demo.py"), default_timeout=120
    ).run()
    if tested.exception or tested.error:
        raise RuntimeError(
            "Dashboard AppTest reported an exception: "
            + "; ".join(element.value for element in [*tested.exception, *tested.error])
        )
    labels = {tab.label for tab in tested.tabs}
    expected_tabs = {*app.MAIN_TAB_LABELS, *app.REPRESENTATION_SUBPAGE_LABELS}
    if not expected_tabs.issubset(labels):
        raise RuntimeError("Dashboard did not render every expected research page.")

    port = unused_port()
    environment = {**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT)}
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "apps/inferencex_pca_demo.py",
            "--server.headless=true",
            f"--server.port={port}",
            "--browser.gatherUsageStats=false",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        wait_for_health(port, process)
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    print("Clean-clone smoke test passed: bootstrap, artifacts, and Streamlit health are verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
