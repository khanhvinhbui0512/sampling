"""Run QM9 asset preparation locally from the repository root."""

import argparse
import uuid

from src.io import report_result
from src.qm9.dataset import prepare_qm9_assets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qm9.yaml")
    parser.add_argument("--asset-id", default="qm9-v1", help="Logical asset group; each attempt is immutable.")
    parser.add_argument("--attempt-id", default=None, help="Defaults to a new UUID; never overwrites an attempt.")
    args = parser.parse_args()
    try:
        assets = prepare_qm9_assets(args.config, asset_id=args.asset_id,
                                    attempt_id=args.attempt_id or uuid.uuid4().hex,
                                    execution_metadata={"backend": "local"})
    except Exception as exc:
        report_result(f"QM9 preparation failed: {type(exc).__name__}: {exc}")
        return 1
    result = assets.summary()
    report_result(f"Prepared QM9: train={result['train_rows']}, test={result['test_rows']}, "
                  f"vocab={result['vocab_size']}; manifest={result['manifest']}; no base weights")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
