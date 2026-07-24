#!/usr/bin/env python3
"""Add artifact-selected cluster labels to an existing final Parquet companion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    companion = artifact["embedding_companion"]
    path = args.artifact.parent / companion["filename"]
    frame = pd.read_parquet(path)
    latent_columns = sorted(
        (column for column in frame if column.startswith("z")),
        key=lambda value: int(value[1:]),
    )
    selected_k = int(artifact["summary"]["clustering"]["selected_k"])
    labeled = []
    for seed, seed_frame in frame.groupby("seed", sort=False):
        seed_frame = seed_frame.copy()
        seed_frame["cluster"] = KMeans(
            n_clusters=selected_k,
            random_state=int(seed),
            n_init=20,
        ).fit_predict(seed_frame[latent_columns]).astype("int16")
        labeled.append(seed_frame)
    updated = pd.concat(labeled, ignore_index=True)
    updated.to_parquet(path, index=False, compression="zstd")
    companion["columns"] = updated.columns.tolist()
    companion["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    companion["bytes"] = path.stat().st_size
    artifact["computational_cost"]["companion_size_bytes"] = path.stat().st_size
    args.artifact.write_text(
        json.dumps(artifact, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    artifact["computational_cost"]["json_artifact_size_bytes"] = args.artifact.stat().st_size
    args.artifact.write_text(
        json.dumps(artifact, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Updated {path} and {args.artifact}")


if __name__ == "__main__":
    main()

