"""Generate QM9 outputs on Modal; run from the repository root."""

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
    .add_local_file("generate.py", "/root/generate.py", copy=True)
    .add_local_file("modal/generate.py", "/root/modal/generate.py", copy=True)
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
    from src.config import project_path
    from src.experiments.run import generate

    for key in ("assets_manifest", "checkpoint_manifest", "calibration_manifest", "verifier_manifest", "warm_cache_manifest"):
        if arguments.get(key) and not project_path(arguments[key]).startswith("workspace/"):
            raise ValueError("Modal artifact paths must be under workspace/")
    stored = Volume.from_name("nam-cars-vgb-qm9-artifacts")
    stored.reload()
    try:
        result = generate(**arguments, device="cuda", execution_metadata={"backend": "modal"})
    except Exception as exc:
        try:
            stored.commit()
        except Exception as commit_error:
            exc.add_note(f"Could not persist failed generation: {commit_error}")
        raise
    stored.commit()
    return result


def _run_task(arguments):
    arguments["attempt_id"] = arguments.get("attempt_id") or uuid.uuid4().hex
    try:
        result = run_command.remote(arguments)
    except Exception as exc:
        print(f"QM9 generation failed: {type(exc).__name__}: {exc}", flush=True)
        raise
    print(f"Generated {result['outputs']} outputs; manifest={result['manifest']}", flush=True)


@app.local_entrypoint()
def main(assets_manifest: str, checkpoint_manifest: str, calibration_manifest: str, method: str,
         verifier: str = "none", value_law: str = "native-conditional", verifier_manifest: str = "",
         num_samples: int = 1, seed: int = 1, config: str = "configs/mdm_vgb.yaml",
         cars_config: str = "configs/cars.yaml", reference_config: str = "configs/reference_checks.yaml",
         cache_mode: str = "cold", warm_cache_manifest: str = "", run_id: str = "qm9-generate", attempt_id: str = ""):
    arguments = dict(assets_manifest=assets_manifest, checkpoint_manifest=checkpoint_manifest,
                     calibration_manifest=calibration_manifest, method=method, verifier=verifier,
                     value_law=value_law, verifier_manifest=verifier_manifest, num_samples=num_samples,
                     seed=seed, config_path=config, cars_config=cars_config, reference_config=reference_config,
                     cache_mode=cache_mode, warm_cache_manifest=warm_cache_manifest, run_id=run_id, attempt_id=attempt_id)
    _run_task(arguments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets-manifest", "checkpoint-manifest", "calibration-manifest", "method"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--verifier", default="none")
    parser.add_argument("--value-law", default="native-conditional")
    parser.add_argument("--verifier-manifest", default="")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--config", dest="config_path", default="configs/mdm_vgb.yaml")
    parser.add_argument("--cars-config", default="configs/cars.yaml")
    parser.add_argument("--reference-config", default="configs/reference_checks.yaml")
    parser.add_argument("--cache-mode", default="cold")
    parser.add_argument("--warm-cache-manifest", default="")
    parser.add_argument("--run-id", default="qm9-generate")
    parser.add_argument("--attempt-id", default="")
    with modal.enable_output(), app.run():
        _run_task(vars(parser.parse_args()))
