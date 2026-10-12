"""Prepare QM9 assets on Modal; run from the repository root."""

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
    .add_local_file("prepare_assets.py", "/root/prepare_assets.py", copy=True)
    .add_local_file("modal/prepare_assets.py", "/root/modal/prepare_assets.py", copy=True)
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
    gpu=None,  # Asset preparation does not use a neural model.
    cpu=2,
    memory=8192,
    timeout=3600,
    max_containers=1,
    retries=0,
    volumes={"/root/workspace": volume},
    serialized=True,  # Supports both modal run and the Python __main__ entrypoint.
    include_source=False,
)
def run_command(config: str, asset_id: str, attempt_id: str) -> dict:
    from modal import Volume

    from src.qm9.dataset import prepare_qm9_assets

    stored_assets = Volume.from_name("nam-cars-vgb-qm9-artifacts")
    stored_assets.reload()
    try:
        assets = prepare_qm9_assets(config, asset_id=asset_id, attempt_id=attempt_id,
                                    execution_metadata={"backend": "modal"})
    except Exception as exc:
        try:
            stored_assets.commit()  # Keep the failed attempt's manifest too.
        except Exception as commit_error:
            exc.add_note(f"Could not persist the failed attempt: {commit_error}")
        raise
    stored_assets.commit()
    return assets.summary()


def _run_task(asset_id: str, config: str, attempt_id: str) -> None:
    try:
        # Poll the same call through the SDK's retryable result endpoint.
        call = run_command.spawn(config, asset_id, attempt_id or uuid.uuid4().hex)
        print(f"QM9 preparation call: {call.object_id}", flush=True)
        result = call.get()
    except Exception as exc:
        print(f"QM9 preparation failed: {type(exc).__name__}: {exc}", flush=True)
        raise
    print(f"Prepared QM9: train={result['train_rows']}, test={result['test_rows']}, "
          f"vocab={result['vocab_size']}; manifest={result['manifest']}; no base weights", flush=True)


@app.local_entrypoint()
def main(asset_id: str = "qm9-v1", config: str = "configs/qm9.yaml", attempt_id: str = "") -> None:
    _run_task(asset_id, config, attempt_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qm9.yaml")
    parser.add_argument("--asset-id", default="qm9-v1")
    parser.add_argument("--attempt-id", default="")
    args = parser.parse_args()
    with modal.enable_output(), app.run():
        _run_task(args.asset_id, args.config, args.attempt_id)
