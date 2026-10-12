"""Generate independent QM9 BASE calibration and lock the Pass@95 threshold."""

import argparse
import uuid

from src.io import report_result
from src.qm9.calibration import calibrate_qm9


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-manifest", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--config", default="configs/reference_checks.yaml")
    parser.add_argument("--calibration-id", default="qm9-base-pass95")
    parser.add_argument("--attempt-id", default="")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    try:
        result = calibrate_qm9(
            args.assets_manifest, args.checkpoint_manifest, config_path=args.config,
            calibration_id=args.calibration_id, attempt_id=args.attempt_id or uuid.uuid4().hex,
            device=args.device,
        )
    except Exception as exc:
        report_result(f"QM9 calibration failed: {type(exc).__name__}: {exc}")
        raise
    report_result(f"Calibrated QM9: n={result['sample_count']}, threshold={result['threshold']:.6f}; "
                  f"manifest={result['manifest']}")
