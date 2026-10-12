"""Run T04 on Modal with ``python modal/fetch_models.py`` from repo root."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import modal
from modal import App
from tqdm.auto import tqdm

from src.autoregressive import validate_models
from src.config import load_config


def _load_helper():
    if not os.path.isfile("pyproject.toml") or not os.path.isfile("src/__init__.py"):
        raise RuntimeError("Run Modal commands from the repository root")
    name = "smiles_modal_common"
    spec = importlib.util.spec_from_file_location(name, "modal/common.py")
    if spec is None or spec.loader is None:
        raise ImportError("Cannot read modal/common.py; run from the repository root")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


try:
    COMMON = _load_helper()
    SETTINGS = COMMON.load_settings()
    app = App(SETTINGS["app_name"] + "-fetch-models", include_source=False)
    WORKER_OPTIONS = COMMON.worker_options(SETTINGS, "fetch_models")
    image = WORKER_OPTIONS.pop("image")
    volumes = WORKER_OPTIONS.pop("volumes")
except Exception as exc:
    tqdm.write(f"T04 Modal: failed; {type(exc).__name__}: {exc}")
    raise


@app.function(image=image, volumes=volumes, **WORKER_OPTIONS)
def worker(argv: list[str], artifact_root: str) -> dict:
    from src.tasks.fetch_models import execute

    started = time.perf_counter()
    try:
        with COMMON.volume_session(SETTINGS, artifact_root):
            record = execute(argv, execution="modal")
    except Exception as exc:
        tqdm.write(f"T04 Modal worker: failed; {type(exc).__name__}: {exc}")
        raise
    return {"record": record.to_dict(), "worker_seconds": time.perf_counter() - started}


def _parse_overrides(value: str) -> list[str]:
    if not value:
        return []
    if value.lstrip().startswith("["):
        items = json.loads(value)
        if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
            raise ValueError("--set must be one KEY=VALUE string or a JSON list")
        return items
    return [value]


@app.local_entrypoint()
def main(config: str = "configs/models.yaml", set: str = "", run_id: str = "",
         resume: bool = False, smoke: bool = False, artifact_root: str = "") -> None:
    try:
        COMMON.require_packaged_config(config)
        overrides = _parse_overrides(set)
        resolved_overrides = [*overrides, "runtime.execution=modal"]
        if smoke:
            resolved_overrides.append("suite=smoke")
        if artifact_root:
            resolved_overrides.append("paths.artifact_root=" + json.dumps(artifact_root))
        resolved = load_config(config, overrides=resolved_overrides, validator=validate_models)
        root = resolved["paths"]["artifact_root"]
        if root == "." and not artifact_root:
            root = SETTINGS["artifact_root"]
        COMMON.require_volume_path(SETTINGS, root)
        if resolved["runtime"]["modal_volume"] != SETTINGS["volumes"]["artifacts"]["name"]:
            raise ValueError("runtime.modal_volume differs from the configured Volume")
        if resume and not run_id:
            raise ValueError("--resume requires the original --run-id")
        argv = COMMON.shared_arguments(config=config, overrides=overrides, run_id=run_id,
                                       resume=resume, smoke=smoke, artifact_root=root)
        started = time.perf_counter()
        with tqdm(total=1, desc="Modal pretrained models", unit="job", disable=not resolved["runtime"]["progress"]) as progress:
            result = worker.remote(argv, root)
            progress.update(1)
        record = result["record"]
        elapsed = time.perf_counter() - started
        tqdm.write(f"T04 Modal: {record['status']}; {record['summary']}; RPC={elapsed:.2f}s")
        if record["status"] != "completed":
            raise SystemExit(1)
    except Exception as exc:
        tqdm.write(f"T04 Modal: failed; {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    try:
        arguments = COMMON.script_arguments("Fetch pinned pretrained models on Modal.", default_config="configs/models.yaml")
        with modal.enable_output(), app.run():
            main(**arguments)
    except KeyboardInterrupt:
        tqdm.write("T04 Modal: interrupted")
        raise SystemExit(130) from None
    except Exception as exc:
        tqdm.write(f"T04 Modal: failed; {type(exc).__name__}: {exc}")
        raise SystemExit(1) from None
