#!/usr/bin/env python3
"""Add decoder traversal summaries to an existing final representation artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from modeling.final_representation_training import _latent_traversal_summary
from modeling.representation_analysis import write_json_artifact


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    weights_path = args.artifact.parent / artifact["weights"]["filename"]
    states = torch.load(weights_path, map_location="cpu", weights_only=True)
    artifact["summary"]["latent_traversals"] = _latent_traversal_summary(
        states, artifact["encoded_feature_order"]
    )
    write_json_artifact(args.artifact, artifact)
    artifact["computational_cost"]["json_artifact_size_bytes"] = args.artifact.stat().st_size
    write_json_artifact(args.artifact, artifact)
    print(f"Updated {args.artifact}")


if __name__ == "__main__":
    main()

