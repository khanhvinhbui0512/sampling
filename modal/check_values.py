"""Check toy values on CPU or a locked QM9 panel on GPU with Modal."""

import argparse
import uuid

import modal
from modal import App, Image, Volume

app = App("nam-cars-vgb-qm9", include_source=False)
image = (
    Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .pip_install(["numpy", "PyYAML", "torch", "transformers", "datasets", "huggingface-hub",
                  "safetensors", "einops", "rdkit", "tqdm", "modal"])
    .add_local_dir("src", "/root/src", copy=True, ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_dir("configs", "/root/configs", copy=True)
    .add_local_file("tests/fixtures/toy_joint.json", "/root/tests/fixtures/toy_joint.json", copy=True)
    .add_local_file("check_values.py", "/root/check_values.py", copy=True)
    .add_local_file("modal/check_values.py", "/root/modal/check_values.py", copy=True)
    .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/README.md", copy=True)
    .add_local_file("THIRD_PARTY_NOTICES.md", "/root/THIRD_PARTY_NOTICES.md", copy=True)
    .env({"PYTHONPATH": "/root", "TOKENIZERS_PARALLELISM": "false", "HF_HOME": "workspace/cache/huggingface"})
    .workdir("/root")
)
volume = Volume.from_name("nam-cars-vgb-qm9-artifacts", create_if_missing=True)


@app.function(image=image, gpu="RTX-PRO-6000:1", cpu=4, memory=32768, timeout=3600 * 24,
              max_containers=1, retries=0, volumes={"/root/workspace": volume},
              serialized=True, include_source=False)
def run_command(arguments: dict) -> dict:
    from modal import Volume
    from src.config import load_config, project_path
    from src.experiments.value_checks import check_values

    paths = [arguments.get(key) for key in ("assets_manifest", "checkpoint_manifest", "calibration_manifest",
                                           "panel_manifest", "verifier_manifest", "warm_cache_manifest")]
    paths.append(load_config(arguments["config_path"])["check_output_dir"])
    if any(path and not project_path(path).startswith("workspace/") for path in paths):
        raise ValueError("Modal inputs/outputs must be under workspace/")
    stored = Volume.from_name("nam-cars-vgb-qm9-artifacts")
    stored.reload()
    try:
        result = check_values(**arguments, device="cuda", execution_metadata={"backend": "modal"})
    except Exception as exc:
        try:
            stored.commit()
        except Exception as commit_error:
            exc.add_note(f"Could not persist failed value checks: {commit_error}")
        raise
    stored.commit()
    return result


@app.function(image=image, gpu=None, cpu=2, memory=8192, timeout=3600 * 6,
              max_containers=1, retries=0, volumes={"/root/workspace": volume},
              serialized=True, include_source=False)
def run_toy(arguments: dict) -> dict:
    from modal import Volume
    from src.config import load_config, project_path
    from src.experiments.toy import check_toy_values

    if not project_path(load_config(arguments["config_path"])["check_output_dir"]).startswith("workspace/"):
        raise ValueError("Modal outputs must be under workspace/")
    stored = Volume.from_name("nam-cars-vgb-qm9-artifacts")
    stored.reload()
    try:
        result = check_toy_values(**arguments, execution_metadata={"backend": "modal"})
    except Exception as exc:
        try:
            stored.commit()
        except Exception as commit_error:
            exc.add_note(f"Could not persist failed toy checks: {commit_error}")
        raise
    stored.commit()
    return result


def _run_task(arguments):
    mode = arguments.pop("mode")
    arguments["attempt_id"] = arguments.get("attempt_id") or uuid.uuid4().hex
    try:
        if mode == "toy":
            if (any(arguments.get(key) for key in ("assets_manifest", "checkpoint_manifest", "calibration_manifest",
                                                  "panel_manifest", "verifier_manifest", "warm_cache_manifest"))
                    or arguments["cache_mode"] != "cold" or arguments["estimators"] != "mc,cars"):
                raise ValueError("Toy mode uses its finite fixture and MC/CARS; omit QM9 manifests and warm cache")
            result = run_toy.remote({key: arguments[key] for key in ("attempt_id", "value_law", "config_path", "run_id")})
        elif mode == "qm9":
            if any(not arguments.get(key) for key in ("assets_manifest", "checkpoint_manifest", "calibration_manifest", "panel_manifest")):
                raise ValueError("QM9 checks require assets, checkpoint, calibration and panel manifests")
            arguments["estimators"] = tuple(arguments["estimators"].split(","))
            result = run_command.remote(arguments)
        else:
            raise ValueError("Mode must be toy or qm9")
    except Exception as exc:
        print(f"Value checks failed: {type(exc).__name__}: {exc}", flush=True)
        raise
    print(f"Checked {result['states']} states; manifest={result['manifest']}", flush=True)


@app.local_entrypoint()
def main(mode: str = "qm9", assets_manifest: str = "", checkpoint_manifest: str = "",
         calibration_manifest: str = "", panel_manifest: str = "", estimators: str = "mc,cars",
         value_law: str = "native-conditional", verifier_manifest: str = "",
         config: str = "configs/value_checks.yaml", inference_config: str = "configs/mdm_vgb.yaml",
         cars_config: str = "configs/cars.yaml", reference_config: str = "configs/reference_checks.yaml",
         cache_mode: str = "cold", warm_cache_manifest: str = "", run_id: str = "qm9-check-values", attempt_id: str = ""):
    _run_task(dict(mode=mode, assets_manifest=assets_manifest, checkpoint_manifest=checkpoint_manifest,
                   calibration_manifest=calibration_manifest, panel_manifest=panel_manifest, estimators=estimators,
                   value_law=value_law, verifier_manifest=verifier_manifest, config_path=config,
                   inference_config=inference_config, cars_config=cars_config, reference_config=reference_config,
                   cache_mode=cache_mode, warm_cache_manifest=warm_cache_manifest, run_id=run_id, attempt_id=attempt_id))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="qm9", choices=("toy", "qm9"))
    for name in ("assets-manifest", "checkpoint-manifest", "calibration-manifest", "panel-manifest", "verifier-manifest"):
        parser.add_argument("--" + name, default="")
    parser.add_argument("--estimators", default="mc,cars")
    parser.add_argument("--value-law", default="native-conditional")
    parser.add_argument("--config", dest="config_path", default="configs/value_checks.yaml")
    parser.add_argument("--inference-config", default="configs/mdm_vgb.yaml")
    parser.add_argument("--cars-config", default="configs/cars.yaml")
    parser.add_argument("--reference-config", default="configs/reference_checks.yaml")
    parser.add_argument("--cache-mode", default="cold")
    parser.add_argument("--warm-cache-manifest", default="")
    parser.add_argument("--run-id", default="qm9-check-values")
    parser.add_argument("--attempt-id", default="")
    with modal.enable_output(), app.run():
        _run_task(vars(parser.parse_args()))
