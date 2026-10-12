"""Run offline tests on Modal and check committed logs from a second function."""

import argparse
import uuid

import modal
from modal import App, Image, Volume

app = App("nam-cars-vgb-qm9", include_source=False)
image = (
    Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .pip_install(["torch", "numpy", "PyYAML", "transformers", "tokenizers", "datasets",
                  "huggingface-hub", "safetensors", "einops", "rdkit", "tqdm", "pytest",
                  "lark", "matplotlib", "setuptools", "wheel", "modal"])
    .add_local_dir("src", "/root/src", copy=True, ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_dir("configs", "/root/configs", copy=True)
    .add_local_dir("tests", "/root/tests", copy=True, ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_dir("modal", "/root/modal", copy=True, ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_file("run.py", "/root/run.py", copy=True)
    .add_local_file("prepare_assets.py", "/root/prepare_assets.py", copy=True)
    .add_local_file("train_base.py", "/root/train_base.py", copy=True)
    .add_local_file("calibrate.py", "/root/calibrate.py", copy=True)
    .add_local_file("collect_rollouts.py", "/root/collect_rollouts.py", copy=True)
    .add_local_file("train_verifier.py", "/root/train_verifier.py", copy=True)
    .add_local_file("generate.py", "/root/generate.py", copy=True)
    .add_local_file("prepare_value_panel.py", "/root/prepare_value_panel.py", copy=True)
    .add_local_file("check_values.py", "/root/check_values.py", copy=True)
    .add_local_file("evaluate.py", "/root/evaluate.py", copy=True)
    .add_local_file("verify.py", "/root/verify.py", copy=True)
    .add_local_file("benchmark.py", "/root/benchmark.py", copy=True)
    .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/README.md", copy=True)
    .add_local_file("THIRD_PARTY_NOTICES.md", "/root/THIRD_PARTY_NOTICES.md", copy=True)
    .env({"PYTHONPATH": "/root", "TOKENIZERS_PARALLELISM": "false"})
    .workdir("/root")
)
volume = Volume.from_name("nam-cars-vgb-qm9-artifacts", create_if_missing=True)


def _execute(arguments, execution):
    from modal import Volume
    from src.experiments.verification import verify

    stored = Volume.from_name("nam-cars-vgb-qm9-artifacts")
    stored.reload()
    try:
        result = verify(**arguments, execution_metadata=execution)
    except Exception as exc:
        try:
            stored.commit()
        except Exception as commit_error:
            exc.add_note(f"Could not persist verification failure: {commit_error}")
        raise
    stored.commit()  # Also commit tests.txt when pytest returned a failure code.
    return result


@app.function(image=image, gpu=None, cpu=2, memory=8192, timeout=3600 * 6,
              max_containers=1, retries=0, volumes={"/root/workspace": volume},
              serialized=True, include_source=False)
def run_command(arguments: dict) -> dict:
    from src.config import load_config

    if arguments["gpu_smoke"] or load_config(arguments["config_path"])["gpu_smoke"]:
        raise ValueError("GPU smoke must use run_gpu_smoke")
    return _execute(arguments, {"backend": "modal", "cpu": 2, "memory_mib": 8192, "gpu": None})


@app.function(image=image, gpu="RTX-PRO-6000:1", cpu=4, memory=32768, timeout=3600 * 6,
              max_containers=1, retries=0, volumes={"/root/workspace": volume},
              serialized=True, include_source=False)
def run_gpu_smoke(arguments: dict) -> dict:
    from src.config import load_config

    if not (arguments["gpu_smoke"] or load_config(arguments["config_path"])["gpu_smoke"]):
        raise ValueError("GPU smoke requires an explicit config/CLI request")
    return _execute(arguments, {"backend": "modal", "cpu": 4, "memory_mib": 32768,
                                "gpu": "RTX-PRO-6000:1"})


@app.function(image=image, gpu=None, cpu=2, memory=8192, timeout=600,
              max_containers=1, retries=0, volumes={"/root/workspace": volume},
              serialized=True, include_source=False)
def read_committed(manifest_path: str) -> dict:
    from modal import Volume
    from src.experiments.verification import read_verification

    Volume.from_name("nam-cars-vgb-qm9-artifacts").reload()
    return read_verification(manifest_path)


def _run_task(arguments):
    from src.config import load_config

    arguments["attempt_id"] = arguments["attempt_id"] or uuid.uuid4().hex
    use_gpu = arguments["gpu_smoke"] or load_config(arguments["config_path"])["gpu_smoke"]
    try:
        result = (run_gpu_smoke if use_gpu else run_command).remote(arguments)
        if read_committed.remote(result["manifest"]) != result:
            raise ValueError("Committed verification artifacts differ from the producing task")
    except Exception as exc:
        print(f"Verification failed: {type(exc).__name__}: {exc}", flush=True)
        raise
    print(f"Verification exit={result['exit_code']}; committed logs readable; manifest={result['manifest']}", flush=True)
    return result["exit_code"]


@app.local_entrypoint()
def main(config: str = "configs/verification.yaml", selection: str = "", gpu_smoke: bool = False,
         run_id: str = "qm9-verify", attempt_id: str = ""):
    raise SystemExit(_run_task(dict(config_path=config, selection=selection, gpu_smoke=gpu_smoke,
                                    run_id=run_id, attempt_id=attempt_id)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", dest="config_path", default="configs/verification.yaml")
    parser.add_argument("--selection", default="")
    parser.add_argument("--gpu-smoke", action="store_true")
    parser.add_argument("--run-id", default="qm9-verify")
    parser.add_argument("--attempt-id", default="")
    with modal.enable_output(), app.run():
        exit_code = _run_task(vars(parser.parse_args()))
    raise SystemExit(exit_code)
