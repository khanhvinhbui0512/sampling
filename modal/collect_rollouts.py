"""Collect QM9 verifier rollout shards on Modal; run from the repository root."""

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
    .add_local_file("collect_rollouts.py", "/root/collect_rollouts.py", copy=True)
    .add_local_file("modal/collect_rollouts.py", "/root/modal/collect_rollouts.py", copy=True)
    .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/README.md", copy=True)
    .add_local_file("THIRD_PARTY_NOTICES.md", "/root/THIRD_PARTY_NOTICES.md", copy=True)
    .env({"PYTHONPATH": "/root", "TOKENIZERS_PARALLELISM": "false",
          "HF_HOME": "workspace/cache/huggingface"})
    .workdir("/root")
)
volume = Volume.from_name("nam-cars-vgb-qm9-artifacts", create_if_missing=True)


@app.function(
    image=image, gpu="RTX-PRO-6000:1", cpu=4, memory=32768, timeout=3600 * 24,
    max_containers=1, retries=0, volumes={"/root/workspace": volume},
    serialized=True, include_source=False,
)
def run_command(assets_manifest: str, checkpoint_manifest: str, calibration_manifest: str,
                policy: str, config: str, run_id: str, attempt_id: str,
                shard_index: int, num_shards: int) -> dict:
    from modal import Volume

    from src.config import load_config, project_path
    from src.qm9.rollouts import collect_rollouts

    for path in (assets_manifest, checkpoint_manifest, calibration_manifest, load_config(config)["output_dir"]):
        if not project_path(path).startswith("workspace/"):
            raise ValueError("Modal inputs/outputs must be under workspace/")
    stored_rollouts = Volume.from_name("nam-cars-vgb-qm9-artifacts")
    stored_rollouts.reload()
    try:
        result = collect_rollouts(
            assets_manifest, checkpoint_manifest, calibration_manifest, policy=policy,
            config_path=config, run_id=run_id, attempt_id=attempt_id, shard_index=shard_index,
            num_shards=num_shards, device="cuda", execution_metadata={"backend": "modal"},
        )
    except Exception as exc:
        try:
            stored_rollouts.commit()
        except Exception as commit_error:
            exc.add_note(f"Could not persist failed rollout attempt: {commit_error}")
        raise
    stored_rollouts.commit()
    return result


def _run_task(assets_manifest, checkpoint_manifest, calibration_manifest, policy,
              config, run_id, attempt_id, shard_index, num_shards):
    try:
        result = run_command.remote(assets_manifest, checkpoint_manifest, calibration_manifest,
                                    policy, config, run_id, attempt_id or uuid.uuid4().hex,
                                    shard_index, num_shards)
    except Exception as exc:
        print(f"QM9 rollout collection failed: {type(exc).__name__}: {exc}", flush=True)
        raise
    print(f"Collected {policy}: {result['rollouts']} rollouts, {result['snapshots']} snapshots; "
          f"manifest={result['manifest']}", flush=True)


@app.local_entrypoint()
def main(assets_manifest: str, checkpoint_manifest: str, calibration_manifest: str,
         policy: str, config: str = "configs/rollouts.yaml", run_id: str = "qm9-rollouts",
         attempt_id: str = "", shard_index: int = 0, num_shards: int = 1):
    _run_task(assets_manifest, checkpoint_manifest, calibration_manifest, policy,
              config, run_id, attempt_id, shard_index, num_shards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-manifest", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--policy", required=True, choices=("native", "rho", "native-conditional"))
    parser.add_argument("--config", default="configs/rollouts.yaml")
    parser.add_argument("--run-id", default="qm9-rollouts")
    parser.add_argument("--attempt-id", default="")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    with modal.enable_output(), app.run():
        _run_task(args.assets_manifest, args.checkpoint_manifest, args.calibration_manifest,
                  args.policy, args.config, args.run_id, args.attempt_id, args.shard_index, args.num_shards)
