"""Lock a QM9 state panel independently of compared estimators."""

import argparse
import uuid

from src.experiments.value_checks import prepare_value_panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-manifest", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--source-manifests", default="", help="Comma-separated completed BASE/baseline VGB runs")
    parser.add_argument("--value-law", choices=("native-conditional", "rho"), default="native-conditional")
    parser.add_argument("--config", dest="config_path", default="configs/value_checks.yaml")
    parser.add_argument("--inference-config", default="configs/mdm_vgb.yaml")
    parser.add_argument("--cars-config", default="configs/cars.yaml")
    parser.add_argument("--reference-config", default="configs/reference_checks.yaml")
    parser.add_argument("--run-id", default="qm9-value-panel")
    parser.add_argument("--attempt-id", default="")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = vars(parser.parse_args())
    args["attempt_id"] = args["attempt_id"] or uuid.uuid4().hex
    args["source_manifests"] = tuple(args["source_manifests"].split(",")) if args["source_manifests"] else ()
    prepare_value_panel(**args)


if __name__ == "__main__":
    main()
