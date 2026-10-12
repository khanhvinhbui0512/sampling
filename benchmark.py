"""Run a locked QM9 pilot/final matrix, or resume explicitly unfinished units."""

import argparse
import uuid

from src.experiments.benchmark import benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", dest="config_path", default="configs/benchmark.yaml")
    parser.add_argument("--benchmark-id", default="qm9-benchmark")
    parser.add_argument("--attempt-id", default="")
    parser.add_argument("--resume-manifest")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    arguments = vars(parser.parse_args())
    arguments["attempt_id"] = arguments["attempt_id"] or uuid.uuid4().hex
    try:
        result = benchmark(**arguments)
    except Exception as exc:
        print(f"Benchmark failed: {type(exc).__name__}: {exc}", flush=True)
        raise
    raise SystemExit(result["exit_code"])


if __name__ == "__main__":
    main()
