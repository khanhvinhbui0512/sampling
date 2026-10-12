"""Train or resume one QM9 verifier target on Modal from the repository root."""

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
    .add_local_file("train_verifier.py", "/root/train_verifier.py", copy=True)
    .add_local_file("modal/train_verifier.py", "/root/modal/train_verifier.py", copy=True)
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
def run_command(rollout_manifests: str, policy: str, config: str, run_id: str,
                attempt_id: str, resume_manifest: str) -> dict:
    from modal import Volume

    from src.config import load_config, project_path
    from src.training.verifier import train_verifier

    manifests = rollout_manifests.split(",")
    for path in (*manifests, resume_manifest, load_config(config)["output_dir"]):
        if path and not project_path(path).startswith("workspace/"):
            raise ValueError("Modal inputs/outputs must be under workspace/")
    stored_verifiers = Volume.from_name("nam-cars-vgb-qm9-artifacts")
    stored_verifiers.reload()
    try:
        result = train_verifier(
            manifests, policy=policy, config_path=config, run_id=run_id, attempt_id=attempt_id,
            device="cuda", resume_manifest=resume_manifest or None,
            on_checkpoint=stored_verifiers.commit, execution_metadata={"backend": "modal"},
        )
    except Exception as exc:
        try:
            stored_verifiers.commit()
        except Exception as commit_error:
            exc.add_note(f"Could not persist failed verifier attempt: {commit_error}")
        raise
    stored_verifiers.commit()
    return result


def _run_task(rollout_manifests, policy, config, run_id, attempt_id, resume_manifest):
    try:
        result = run_command.remote(rollout_manifests, policy, config, run_id,
                                    attempt_id or uuid.uuid4().hex, resume_manifest)
    except Exception as exc:
        print(f"QM9 verifier training failed: {type(exc).__name__}: {exc}", flush=True)
        raise
    print(f"Trained {policy} verifier: epoch={result['epoch']}, parameters={result['parameter_count']}; "
          f"checkpoint={result['checkpoint_manifest']}", flush=True)


@app.local_entrypoint()
def main(rollout_manifests: str, policy: str, config: str = "configs/verifier_training.yaml",
         run_id: str = "qm9-verifier", attempt_id: str = "", resume_manifest: str = ""):
    _run_task(rollout_manifests, policy, config, run_id, attempt_id, resume_manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-manifests", required=True, help="One path or comma-separated shard manifest paths")
    parser.add_argument("--policy", required=True, choices=("native", "rho", "native-conditional"))
    parser.add_argument("--config", default="configs/verifier_training.yaml")
    parser.add_argument("--run-id", default="qm9-verifier")
    parser.add_argument("--attempt-id", default="")
    parser.add_argument("--resume-manifest", default="")
    args = parser.parse_args()
    with modal.enable_output(), app.run():
        _run_task(args.rollout_manifests, args.policy, args.config,
                  args.run_id, args.attempt_id, args.resume_manifest)
