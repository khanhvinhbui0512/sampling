"""Train or resume one QM9 verifier target from the repository root."""

import argparse
import uuid

from src.io import report_result
from src.training.verifier import train_verifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-manifests", required=True, help="One path or comma-separated shard manifest paths")
    parser.add_argument("--policy", required=True, choices=("native", "rho", "native-conditional"))
    parser.add_argument("--config", default="configs/verifier_training.yaml")
    parser.add_argument("--run-id", default="qm9-verifier")
    parser.add_argument("--attempt-id", default="")
    parser.add_argument("--resume-manifest", default="")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    try:
        result = train_verifier(
            args.rollout_manifests.split(","), policy=args.policy,
            attempt_id=args.attempt_id or uuid.uuid4().hex, config_path=args.config,
            run_id=args.run_id, device=args.device, resume_manifest=args.resume_manifest or None,
        )
    except Exception as exc:
        report_result(f"QM9 verifier training failed: {type(exc).__name__}: {exc}")
        raise
    report_result(f"Trained {args.policy} verifier: epoch={result['epoch']}, parameters={result['parameter_count']}; "
                  f"checkpoint={result['checkpoint_manifest']}")
