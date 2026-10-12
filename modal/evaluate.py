"""Evaluate saved QM9 artifacts on a Modal CPU and optionally download reports."""

import argparse
import hashlib
import os
import uuid

import modal
from modal import App, Image, Volume

app = App("nam-cars-vgb-qm9", include_source=False)
image = (
    Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .pip_install(["numpy", "PyYAML", "transformers", "rdkit", "tqdm", "matplotlib", "modal"])
    .add_local_dir("src", "/root/src", copy=True, ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_dir("configs", "/root/configs", copy=True)
    .add_local_file("evaluate.py", "/root/evaluate.py", copy=True)
    .add_local_file("modal/evaluate.py", "/root/modal/evaluate.py", copy=True)
    .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/README.md", copy=True)
    .add_local_file("THIRD_PARTY_NOTICES.md", "/root/THIRD_PARTY_NOTICES.md", copy=True)
    .env({"PYTHONPATH": "/root", "TOKENIZERS_PARALLELISM": "false"})
    .workdir("/root")
)
volume = Volume.from_name("nam-cars-vgb-qm9-artifacts", create_if_missing=True)


@app.function(image=image, gpu=None, cpu=2, memory=8192, timeout=3600 * 6,
              max_containers=1, retries=0, volumes={"/root/workspace": volume},
              serialized=True, include_source=False)
def run_command(arguments: dict) -> dict:
    from modal import Volume
    from src.config import load_config, project_path
    from src.evaluation.aggregate import evaluate

    paths = [*arguments["manifests"], load_config(arguments["config_path"])["output_dir"]]
    if any(not project_path(path).startswith("workspace/") for path in paths):
        raise ValueError("Modal inputs/outputs must be under workspace/")
    stored = Volume.from_name("nam-cars-vgb-qm9-artifacts")
    stored.reload()
    try:
        result = evaluate(**arguments, execution_metadata={"backend": "modal", "cpu": 2, "memory_mib": 8192})
    except Exception as exc:
        try:
            stored.commit()
        except Exception as commit_error:
            exc.add_note(f"Could not persist failed evaluation: {commit_error}")
        raise
    stored.commit()
    return result


def download_reports(result, selected, output_dir):
    """Only registered report files are downloadable; never overwrite local files."""
    from src.config import project_path

    names = list(dict.fromkeys(["manifest.json", *selected]))
    if any(name not in result["artifacts"] for name in names):
        raise ValueError("Download names must appear in this evaluation manifest")
    source = os.path.dirname(project_path(result["manifest"]))
    if not source.startswith("workspace/"):
        raise ValueError("Remote report is outside the mounted workspace")
    destination = project_path(os.path.join(output_dir, os.path.basename(os.path.dirname(os.path.dirname(source))), os.path.basename(source)))
    for name in names:
        target = project_path(os.path.join(destination, name))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        digest = hashlib.sha256()
        try:
            with open(target, "xb") as handle:
                for block in volume.read_file(os.path.join(source.removeprefix("workspace/"), name)):
                    digest.update(block)
                    handle.write(block)
            if digest.hexdigest() != result["artifacts"][name]["sha256"]:
                raise ValueError(f"Downloaded report hash mismatch: {name}")
        except FileExistsError:
            raise
        except Exception:
            if os.path.isfile(target):
                os.unlink(target)
            raise
    print(f"Downloaded {len(names)} selected report files to {destination}", flush=True)


def _run_task(arguments):
    selected = tuple(filter(None, arguments.pop("download").split(",")))
    output_dir = arguments.pop("download_dir")
    arguments["manifests"] = tuple(filter(None, arguments["manifests"].split(",")))
    arguments["run_ids"] = tuple(filter(None, arguments["run_ids"].split(",")))
    arguments["attempt_id"] = arguments.get("attempt_id") or uuid.uuid4().hex
    try:
        result = run_command.remote(arguments)
        if selected:
            download_reports(result, selected, output_dir)
    except Exception as exc:
        print(f"Evaluation failed: {type(exc).__name__}: {exc}", flush=True)
        raise
    print(f"Evaluated {result['records']} method records; complete={result['selection_complete']}; manifest={result['manifest']}", flush=True)
    if not result["selection_complete"]:
        raise RuntimeError("Report saved; review unusable/duplicate-seed attempts in audit.jsonl and select explicit manifests")


@app.local_entrypoint()
def main(manifests: str = "", run_ids: str = "", comparison_id: str = "qm9-evaluation",
         attempt_id: str = "", config: str = "configs/evaluation.yaml", download: str = "",
         download_dir: str = "workspace/downloads/evaluation"):
    _run_task(dict(manifests=manifests, run_ids=run_ids, comparison_id=comparison_id,
                   attempt_id=attempt_id, config_path=config, download=download, download_dir=download_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", default="")
    parser.add_argument("--run-ids", default="")
    parser.add_argument("--comparison-id", default="qm9-evaluation")
    parser.add_argument("--attempt-id", default="")
    parser.add_argument("--config", dest="config_path", default="configs/evaluation.yaml")
    parser.add_argument("--download", default="")
    parser.add_argument("--download-dir", default="workspace/downloads/evaluation")
    with modal.enable_output(), app.run():
        _run_task(vars(parser.parse_args()))
