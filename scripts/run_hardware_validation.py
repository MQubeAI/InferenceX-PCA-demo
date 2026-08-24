"""Run the controlled Epoch hardware-capability validation study.

The default environment runs conventional baselines.  Use the dedicated
``.venv-tabfm`` environment and ``--include-tabfm`` only for explicit TabFM
experiments; that dependency is deliberately not imported during normal app
startup.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from modeling.hardware_validation import (  # noqa: E402
    VALIDATION_ROOT,
    reproduce_historical_tabfm_result,
    run_primary_tabfm_ablation,
    run_controlled_validation,
    write_validation_json,
    write_controlled_validation_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-tabfm", action="store_true",
        help="Run TabFM on the primary controlled cohort (requires the dedicated TabFM environment).",
    )
    parser.add_argument(
        "--reproduce-historical-tabfm", action="store_true",
        help="Attempt the exact historical 4,096-row TabFM protocol (requires --include-tabfm).",
    )
    parser.add_argument(
        "--historical-only", action="store_true",
        help="Run only the exact historical TabFM reproduction, without the Epoch benchmark.",
    )
    parser.add_argument(
        "--primary-tabfm-only", action="store_true",
        help="Run only TabFM A/B/C/D on the fixed primary Epoch-resolved cohort.",
    )
    parser.add_argument(
        "--tabfm-context-cap", type=int,
        help="Optional fixed TabFM context cap shared by all primary representations.",
    )
    args = parser.parse_args()
    if (args.reproduce_historical_tabfm or args.historical_only or args.primary_tabfm_only) and not args.include_tabfm:
        parser.error("TabFM-only modes require --include-tabfm")
    if args.historical_only and args.primary_tabfm_only:
        parser.error("Choose only one of --historical-only or --primary-tabfm-only")
    if args.historical_only:
        result = reproduce_historical_tabfm_result()
        write_validation_json(VALIDATION_ROOT / "historical_tabfm_reproduction.json", result)
        print("Wrote historical TabFM reproduction artifact under data/derived/integration/reports/validation")
        return
    if args.primary_tabfm_only:
        result = run_primary_tabfm_ablation(args.tabfm_context_cap)
        write_validation_json(VALIDATION_ROOT / "primary_tabfm_ablation.json", result)
        print("Wrote primary TabFM ablation artifact under data/derived/integration/reports/validation")
        return
    validation = run_controlled_validation(
        include_tabfm=args.include_tabfm,
        reproduce_historical_tabfm=args.reproduce_historical_tabfm,
    )
    write_controlled_validation_outputs(validation)
    print("Wrote controlled hardware validation artifacts under data/derived/integration/reports/validation")


if __name__ == "__main__":
    main()
