"""Safe entry point for the future Stage 5 manifold/neighborhood study.

Without ``--execute`` this script only reports the frozen plan and, when the
verified dataset is available, validates the canonical cohort.  It never imports
UMAP or fits UMAP/t-SNE on that default path and never writes an artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from apps import inferencex_pca_demo as app
from modeling.manifold_analysis import (
    TSNE_2_GRID,
    UMAP_15_GRID,
    UMAP_2_GRID,
    canonical_stage5_data,
    fit_stage5_tsne,
    fit_stage5_umap,
    stage5_plan,
)
from scripts.build_july_pca_artifact import load_aggregate


PLANNED_EXECUTION_ORDER = (
    "canonical cohort validation",
    "original-space reference neighbor graphs",
    "UMAP-2 sensitivity",
    "hierarchy analysis",
    "workload/configuration ablations",
    "mixed-distance sensitivity",
    "t-SNE sensitivity",
    "later outcome overlays",
    "UMAP-15 evaluation",
    "grouped held-out UMAP transform evaluation",
    "cross-method consensus",
)


def _load_canonical_data(data_dir: str) -> Any:
    _raw, aggregate, _metadata = load_aggregate(data_dir)
    return canonical_stage5_data(aggregate)


def _execute_frozen_projection_fits(data: Any) -> dict[str, int]:
    """Fit frozen projections in memory only; PR 2 supplies evaluation/output logic.

    This deliberately has no artifact-writing side effect.  It is protected by
    the explicit command-line flag because it performs the real, costly fitting.
    """

    for config in UMAP_2_GRID:
        fit_stage5_umap(
            data.matrix,
            config,
            encoded_feature_names=data.encoded_feature_names,
        )
    for config in UMAP_15_GRID:
        fit_stage5_umap(
            data.matrix,
            config,
            encoded_feature_names=data.encoded_feature_names,
        )
    for config in TSNE_2_GRID:
        fit_stage5_tsne(
            data.matrix,
            config,
            encoded_feature_names=data.encoded_feature_names,
        )
    return {
        "umap_2_fits": len(UMAP_2_GRID),
        "umap_15_fits": len(UMAP_15_GRID),
        "tsne_2_fits": len(TSNE_2_GRID),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan or explicitly execute frozen Stage 5 manifold projection fits."
    )
    parser.add_argument("--data-dir", default=app.DEFAULT_DATA_DIR)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly run the real frozen UMAP/t-SNE fits in memory (no artifact is written).",
    )
    args = parser.parse_args(argv)
    plan = stage5_plan()
    plan["planned_execution_order"] = list(PLANNED_EXECUTION_ORDER)
    print(json.dumps(plan, indent=2, sort_keys=True))

    data_path = Path(args.data_dir)
    if not data_path.exists():
        if args.execute:
            parser.error(f"--execute requires an available verified dataset: {data_path}")
        print("Validation-only mode: no UMAP/t-SNE fit and no research artifact written.")
        print(f"Dataset not available; skipped canonical validation: {data_path}")
        return 0

    data = _load_canonical_data(args.data_dir)
    print(
        "Canonical Stage 5 cohort validated: "
        f"{len(data.cohort):,} rows, {data.cohort['config_id'].nunique():,} configurations."
    )
    if not args.execute:
        print("Validation-only mode: no UMAP/t-SNE fit and no research artifact written.")
        return 0

    fitted = _execute_frozen_projection_fits(data)
    print(
        "Explicit projection fitting completed in memory only: "
        + ", ".join(f"{name}={count}" for name, count in fitted.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
