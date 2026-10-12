"""Run offline QM9 verification; use --gpu-smoke only for an explicit CUDA check."""

import argparse
import uuid

from src.experiments.verification import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", dest="config_path", default="configs/verification.yaml")
    parser.add_argument("--selection", default="", help="Comma-separated test paths or pytest node IDs")
    parser.add_argument("--gpu-smoke", action="store_true")
    parser.add_argument("--run-id", default="qm9-verify")
    parser.add_argument("--attempt-id", default="")
    arguments = vars(parser.parse_args())
    arguments["attempt_id"] = arguments["attempt_id"] or uuid.uuid4().hex
    try:
        result = verify(**arguments)
    except Exception as exc:
        print(f"Verification failed: {type(exc).__name__}: {exc}", flush=True)
        raise
    raise SystemExit(result["exit_code"])


if __name__ == "__main__":
    main()
