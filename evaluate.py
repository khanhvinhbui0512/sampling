"""Recompute QM9/toy reports from explicit saved attempts, without generation."""

import argparse
import uuid

from src.evaluation.aggregate import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", default="", help="Comma-separated manifest paths")
    parser.add_argument("--run-ids", default="", help="Comma-separated IDs; includes ALL attempts")
    parser.add_argument("--comparison-id", default="qm9-evaluation")
    parser.add_argument("--attempt-id", default="")
    parser.add_argument("--config", default="configs/evaluation.yaml")
    args = parser.parse_args()
    try:
        result = evaluate(manifests=tuple(filter(None, args.manifests.split(","))),
                          run_ids=tuple(filter(None, args.run_ids.split(","))),
                          comparison_id=args.comparison_id, attempt_id=args.attempt_id or uuid.uuid4().hex,
                          config_path=args.config)
    except Exception as exc:
        print(f"Evaluation failed: {type(exc).__name__}: {exc}", flush=True)
        raise
    if not result["selection_complete"]:
        raise SystemExit("Report saved; selection contains unusable or duplicate-seed attempts. Review audit.jsonl and select explicit attempts.")


if __name__ == "__main__":
    main()
