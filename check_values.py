"""Check exact finite toy values on CPU or uncertain QM9 values on a locked panel."""

import argparse
import uuid

from src.experiments.toy import check_toy_values
from src.experiments.value_checks import check_values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("toy", "qm9"), default="qm9")
    parser.add_argument("--assets-manifest")
    parser.add_argument("--checkpoint-manifest")
    parser.add_argument("--calibration-manifest")
    parser.add_argument("--panel-manifest")
    parser.add_argument("--estimators", default="mc,cars", help="Comma-separated mc,neural,cars; toy uses mc,cars")
    parser.add_argument("--value-law", choices=("native-conditional", "rho"), default="native-conditional")
    parser.add_argument("--verifier-manifest")
    parser.add_argument("--config", dest="config_path", default="configs/value_checks.yaml")
    parser.add_argument("--inference-config", default="configs/mdm_vgb.yaml")
    parser.add_argument("--cars-config", default="configs/cars.yaml")
    parser.add_argument("--reference-config", default="configs/reference_checks.yaml")
    parser.add_argument("--cache-mode", choices=("cold", "warm"), default="cold")
    parser.add_argument("--warm-cache-manifest")
    parser.add_argument("--run-id", default="qm9-check-values")
    parser.add_argument("--attempt-id", default="")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = vars(parser.parse_args())
    mode = args.pop("mode")
    args["attempt_id"] = args["attempt_id"] or uuid.uuid4().hex
    if mode == "toy":
        if (any(args[key] for key in ("assets_manifest", "checkpoint_manifest", "calibration_manifest",
                                     "panel_manifest", "verifier_manifest", "warm_cache_manifest"))
                or args["cache_mode"] != "cold" or args["estimators"] != "mc,cars"):
            parser.error("Toy mode uses its finite fixture and MC/CARS; omit QM9 manifests and warm cache")
        check_toy_values(**{key: args[key] for key in ("attempt_id", "value_law", "config_path", "run_id")})
    else:
        if any(not args[key] for key in ("assets_manifest", "checkpoint_manifest", "calibration_manifest", "panel_manifest")):
            parser.error("QM9 mode requires assets, checkpoint, calibration and panel manifests")
        args["estimators"] = tuple(args["estimators"].split(","))
        check_values(**args)


if __name__ == "__main__":
    main()
