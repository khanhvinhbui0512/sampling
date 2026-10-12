"""Collect upstream-native, rho or independent-mask native QM9 verifier labels."""

import argparse
import uuid

from src.io import report_result
from src.qm9.rollouts import collect_rollouts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-manifest", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--policy", required=True, choices=("native", "rho", "native-conditional"))
    parser.add_argument("--config", default="configs/rollouts.yaml")
    parser.add_argument("--run-id", default="qm9-rollouts")
    parser.add_argument("--attempt-id", default="")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    try:
        result = collect_rollouts(
            args.assets_manifest, args.checkpoint_manifest, args.calibration_manifest,
            policy=args.policy, attempt_id=args.attempt_id or uuid.uuid4().hex,
            config_path=args.config, run_id=args.run_id, shard_index=args.shard_index,
            num_shards=args.num_shards, device=args.device,
        )
    except Exception as exc:
        report_result(f"QM9 rollout collection failed: {type(exc).__name__}: {exc}")
        raise
    report_result(f"Collected {args.policy}: {result['rollouts']} rollouts, {result['snapshots']} snapshots; "
                  f"manifest={result['manifest']}")
