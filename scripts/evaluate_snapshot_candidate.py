"""Generate a candidate-vs-active machine report and review-required policy result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from modeling.snapshot_refresh import audit_candidate_against_active, evaluate_refresh_policy, refresh_report_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--active-data-dir", required=True)
    parser.add_argument("--candidate-data-dir", required=True)
    parser.add_argument("--research-gates", type=Path, help="JSON status for pca/ae/vae/comparison/stage4/tabfm/uncertainty")
    parser.add_argument("--clean-clone-passed", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
    audit = audit_candidate_against_active(args.active_data_dir, args.candidate_data_dir)
    research_payload = json.loads(args.research_gates.read_text(encoding="utf-8")) if args.research_gates else None
    research = research_payload.get("gates") if isinstance(research_payload, dict) else None
    policy = evaluate_refresh_policy(audit, research=research, clean_clone_passed=args.clean_clone_passed)
    payload = {"candidate_manifest": manifest, "audit": audit, "research": research_payload, "policy": policy}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(refresh_report_markdown(manifest, audit, policy), encoding="utf-8")
    print(json.dumps({"promotion_status": policy["promotion_status"], "hard_failures": policy["hard_failures"]}, indent=2))
    return 0 if policy["promotion_status"] == "REVIEW_REQUIRED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
