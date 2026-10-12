"""CPU benchmark coordinator with one GPU worker at a time and persistent attempts."""

import argparse
import uuid

import modal
from modal import App, Image, Volume

app = App("nam-cars-vgb-qm9", include_source=False)
image = (
    Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .pip_install(["torch", "numpy", "PyYAML", "transformers", "tokenizers", "datasets",
                  "huggingface-hub", "safetensors", "einops", "rdkit", "tqdm", "matplotlib", "modal"])
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


@app.function(image=image, gpu="RTX-PRO-6000:1", cpu=4, memory=32768, timeout=3600 * 24,
              max_containers=1, retries=0, volumes={"/root/workspace": volume},
              serialized=True, include_source=False)
def run_worker(job: dict) -> dict:
    from modal import Volume
    from src.experiments.benchmark import run_unit

    stored = Volume.from_name("nam-cars-vgb-qm9-artifacts")
    stored.reload()
    try:
        result = run_unit(job, device="cuda", execution_metadata={"backend": "modal"})
    except Exception as exc:
        try:
            stored.commit()
        except Exception as commit_error:
            exc.add_note(f"Could not persist failed benchmark worker: {commit_error}")
        raise
    stored.commit()
    return result


@app.function(image=image, gpu=None, cpu=2, memory=8192, timeout=3600 * 24,
              max_containers=1, retries=0, volumes={"/root/workspace": volume},
              serialized=True, include_source=False)
def run_command(arguments: dict) -> dict:
    from modal import Volume
    from src.experiments.benchmark import benchmark

    stored = Volume.from_name("nam-cars-vgb-qm9-artifacts")
    stored.reload()

    def dispatch(job):
        stored.commit()  # Publish the plan/config snapshots and dispatch intent first.
        try:
            return run_worker.remote(job)
        finally:
            stored.reload()  # Include failed worker artifacts in the retry ledger too.

    try:
        result = benchmark(**arguments, device="cuda", worker=dispatch,
                           execution_metadata={"backend": "modal"})
    except Exception as exc:
        try:
            stored.commit()
        except Exception as commit_error:
            exc.add_note(f"Could not persist failed benchmark coordinator: {commit_error}")
        raise
    stored.commit()
    return result


def _run_task(arguments):
    arguments["attempt_id"] = arguments["attempt_id"] or uuid.uuid4().hex
    try:
        result = run_command.remote(arguments)
    except Exception as exc:
        print(f"Benchmark failed: {type(exc).__name__}: {exc}", flush=True)
        raise
    print(f"Benchmark {result['completed_units']}/{result['total_units']} units; pilot_ready={result['pilot_ready']}; manifest={result['manifest']}", flush=True)
    return result["exit_code"]


@app.local_entrypoint()
def main(config: str = "configs/benchmark.yaml", benchmark_id: str = "qm9-benchmark",
         attempt_id: str = "", resume_manifest: str = ""):
    raise SystemExit(_run_task(dict(config_path=config, benchmark_id=benchmark_id,
                                    attempt_id=attempt_id, resume_manifest=resume_manifest or None)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", dest="config_path", default="configs/benchmark.yaml")
    parser.add_argument("--benchmark-id", default="qm9-benchmark")
    parser.add_argument("--attempt-id", default="")
    parser.add_argument("--resume-manifest")
    with modal.enable_output(), app.run():
        exit_code = _run_task(vars(parser.parse_args()))
    raise SystemExit(exit_code)
