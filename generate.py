"""Generate one QM9 run; execute from the repository root."""

import argparse
import uuid

from src.experiments.run import METHODS, VALUE_LAWS, generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-manifest", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--verifier", choices=("none", "neural", "mc", "cars"), default="none")
    parser.add_argument("--value-law", choices=VALUE_LAWS, default="native-conditional")
    parser.add_argument("--verifier-manifest")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--config", dest="config_path", default="configs/mdm_vgb.yaml")
    parser.add_argument("--cars-config", default="configs/cars.yaml")
    parser.add_argument("--reference-config", default="configs/reference_checks.yaml")
    parser.add_argument("--cache-mode", choices=("cold", "warm"), default="cold")
    parser.add_argument("--warm-cache-manifest")
    parser.add_argument("--run-id", default="qm9-generate")
    parser.add_argument("--attempt-id", default="")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = vars(parser.parse_args())
    args["attempt_id"] = args["attempt_id"] or uuid.uuid4().hex
    generate(**args)


if __name__ == "__main__":
    main()
