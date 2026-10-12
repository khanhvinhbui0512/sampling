"""Generate QM9 BASE calibration on Modal; run from the repository root."""

import argparse
import uuid

import modal
from modal import App, Image, Volume

app = App("nam-cars-vgb-qm9", include_source=False)
image = (
    Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .pip_install([
        "numpy", "PyYAML", "torch", "transformers", "datasets", "huggingface-hub",
        "safetensors", "einops", "rdkit", "tqdm", "modal",
    ])
    .add_local_dir("src", "/root/src", copy=True, ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_dir("configs", "/root/configs", copy=True)
    .add_local_file("calibrate.py", "/root/calibrate.py", copy=True)
    .add_local_file("modal/calibrate.py", "/root/modal/calibrate.py", copy=True)
    .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/README.md", copy=True)
    .add_local_file("THIRD_PARTY_NOTICES.md", "/root/THIRD_PARTY_NOTICES.md", copy=True)
    .env({"PYTHONPATH": "/root", "TOKENIZERS_PARALLELISM": "false",
          "HF_HOME": "workspace/cache/huggingface"})
    .workdir("/root")
)
volume = Volume.from_name("nam-cars-vgb-qm9-artifacts", create_if_missing=True)


@app.function(
    image=image,
    gpu="RTX-PRO-6000:1",
    cpu=4,
    memory=32768,
    timeout=3600 * 6,
    max_containers=1,
    retries=0,
    volumes={"/root/workspace": volume},
    serialized=True,
    include_source=False,
)
def run_command(assets_manifest: str, checkpoint_manifest: str, config: str,
                calibration_id: str, attempt_id: str) -> dict:
    from modal import Volume

    from src.config import load_config, project_path
    from src.qm9.calibration import calibrate_qm9

    for path in (assets_manifest, checkpoint_manifest):
        if not project_path(path).startswith("workspace/"):
            raise ValueError("Modal asset/checkpoint paths must be under workspace/")
    output_dir = load_config(config)["qm9"]["calibration"]["output_dir"]
    if not project_path(output_dir).startswith("workspace/"):
        raise ValueError("Modal calibration outputs must be under workspace/")
    stored_calibration = Volume.from_name("nam-cars-vgb-qm9-artifacts")
    stored_calibration.reload()
    try:
        result = calibrate_qm9(
            assets_manifest, checkpoint_manifest, config_path=config,
            calibration_id=calibration_id, attempt_id=attempt_id, device="cuda",
            execution_metadata={"backend": "modal"},
        )
    except Exception as exc:
        try:
            stored_calibration.commit()
        except Exception as commit_error:
            exc.add_note(f"Could not persist the failed attempt: {commit_error}")
        raise
    stored_calibration.commit()
    return result


def _run_task(assets_manifest: str, checkpoint_manifest: str, config: str,
              calibration_id: str, attempt_id: str) -> None:
    try:
        result = run_command.remote(assets_manifest, checkpoint_manifest, config,
                                    calibration_id, attempt_id or uuid.uuid4().hex)
    except Exception as exc:
        print(f"QM9 calibration failed: {type(exc).__name__}: {exc}", flush=True)
        raise
    print(f"Calibrated QM9: n={result['sample_count']}, threshold={result['threshold']:.6f}; "
          f"manifest={result['manifest']}", flush=True)


@app.local_entrypoint()
def main(assets_manifest: str, checkpoint_manifest: str, config: str = "configs/reference_checks.yaml",
         calibration_id: str = "qm9-base-pass95", attempt_id: str = "") -> None:
    _run_task(assets_manifest, checkpoint_manifest, config, calibration_id, attempt_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-manifest", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--config", default="configs/reference_checks.yaml")
    parser.add_argument("--calibration-id", default="qm9-base-pass95")
    parser.add_argument("--attempt-id", default="")
    args = parser.parse_args()
    with modal.enable_output(), app.run():
        _run_task(args.assets_manifest, args.checkpoint_manifest, args.config,
                  args.calibration_id, args.attempt_id)
