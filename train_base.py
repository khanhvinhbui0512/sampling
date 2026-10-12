"""Train or resume the QM9 MDLM backbone from prepared local assets."""

import argparse
import uuid

from src.io import report_result
from src.training.base import train_base


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-manifest", required=True)
    parser.add_argument("--config", default="configs/base_training.yaml")
    parser.add_argument("--run-id", default="mdlm-qm9")
    parser.add_argument("--attempt-id", default="")
    parser.add_argument("--resume-manifest", default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    try:
        result = train_base(args.assets_manifest, config_path=args.config, run_id=args.run_id,
                            attempt_id=args.attempt_id or uuid.uuid4().hex, device=args.device,
                            resume_manifest=args.resume_manifest)
    except Exception as exc:
        report_result(f"QM9 training failed: {type(exc).__name__}: {exc}")
        raise
    report_result(f"Trained QM9 MDLM: step={result['step']}, parameters={result['parameter_count']}; "
                  f"checkpoint={result['checkpoint_manifest']}")
