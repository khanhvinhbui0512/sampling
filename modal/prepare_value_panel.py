"""Prepare and lock one QM9 value panel on Modal."""

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
    .add_local_file("prepare_value_panel.py", "/root/prepare_value_panel.py", copy=True)
    .add_local_file("modal/prepare_value_panel.py", "/root/modal/prepare_value_panel.py", copy=True)
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
    from src.experiments.value_checks import prepare_value_panel

    paths = [arguments[key] for key in ("assets_manifest", "checkpoint_manifest", "calibration_manifest")]
    paths.extend(arguments["source_manifests"])
    paths.append(load_config(arguments["config_path"])["output_dir"])
    if any(not project_path(path).startswith("workspace/") for path in paths):
        raise ValueError("Modal inputs/outputs must be under workspace/")
    stored = Volume.from_name("nam-cars-vgb-qm9-artifacts")
    stored.reload()
    try:
        result = prepare_value_panel(**arguments, device="cuda", execution_metadata={"backend": "modal"})
    except Exception as exc:
        try:
            stored.commit()
        except Exception as commit_error:
            exc.add_note(f"Could not persist failed panel: {commit_error}")
        raise
    stored.commit()
    return result


def _run_task(arguments):
    arguments["attempt_id"] = arguments.get("attempt_id") or uuid.uuid4().hex
    if isinstance(arguments["source_manifests"], str):
        arguments["source_manifests"] = tuple(arguments["source_manifests"].split(",")) if arguments["source_manifests"] else ()
    try:
        result = run_command.remote(arguments)
    except Exception as exc:
        print(f"QM9 panel preparation failed: {type(exc).__name__}: {exc}", flush=True)
        raise
    print(f"Locked {result['states']} states; manifest={result['manifest']}", flush=True)


@app.local_entrypoint()
def main(assets_manifest: str, checkpoint_manifest: str, calibration_manifest: str,
         source_manifests: str = "", value_law: str = "native-conditional",
         config: str = "configs/value_checks.yaml", inference_config: str = "configs/mdm_vgb.yaml",
         cars_config: str = "configs/cars.yaml", reference_config: str = "configs/reference_checks.yaml",
         run_id: str = "qm9-value-panel", attempt_id: str = ""):
    _run_task(dict(assets_manifest=assets_manifest, checkpoint_manifest=checkpoint_manifest,
                   calibration_manifest=calibration_manifest, source_manifests=source_manifests,
                   value_law=value_law, config_path=config, inference_config=inference_config,
                   cars_config=cars_config, reference_config=reference_config, run_id=run_id, attempt_id=attempt_id))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets-manifest", "checkpoint-manifest", "calibration-manifest"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--source-manifests", default="")
    parser.add_argument("--value-law", default="native-conditional")
    parser.add_argument("--config", dest="config_path", default="configs/value_checks.yaml")
    parser.add_argument("--inference-config", default="configs/mdm_vgb.yaml")
    parser.add_argument("--cars-config", default="configs/cars.yaml")
    parser.add_argument("--reference-config", default="configs/reference_checks.yaml")
    parser.add_argument("--run-id", default="qm9-value-panel")
    parser.add_argument("--attempt-id", default="")
    with modal.enable_output(), app.run():
        _run_task(vars(parser.parse_args()))
