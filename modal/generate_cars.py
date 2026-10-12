"""Run T09 on Modal with ``python modal/generate_cars.py`` from repo root."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import modal
from modal import App
from tqdm.auto import tqdm

from src.config import load_config
from src.tasks.common import build_parser
from src.tasks.generate_cars import _arguments, model_overrides, validate_cars


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
    app = App(SETTINGS["app_name"] + "-generate-cars", include_source=False)
    WORKER_OPTIONS = COMMON.worker_options(SETTINGS, "generate_cars")
    image = WORKER_OPTIONS.pop("image")
    volumes = WORKER_OPTIONS.pop("volumes")
except Exception as exc:
    tqdm.write(f"T09 Modal: failed; {type(exc).__name__}: {exc}")
    raise


@app.function(image=image, volumes=volumes, **WORKER_OPTIONS)
def worker(argv: list[str], artifact_root: str) -> dict:
    from src.tasks.generate_cars import execute

    started = time.perf_counter()
    try:
        with COMMON.volume_session(SETTINGS, artifact_root):
            record = execute(argv, execution="modal", on_checkpoint=COMMON.artifact_volume(SETTINGS).commit)
    except Exception as exc:
        tqdm.write(f"T09 Modal worker: failed; {type(exc).__name__}: {exc}")
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
def main(config: str = "configs/cars.yaml", set: str = "", run_id: str = "", resume: bool = False,
         smoke: bool = False, artifact_root: str = "", seed: int | None = None, group: str = "",
         mdlm_report: str = "", checkpoint: str = "") -> None:
    try:
        COMMON.require_packaged_config(config)
        overrides = _parse_overrides(set)
        resolved_overrides = [*overrides, *model_overrides(training_report=mdlm_report, checkpoint=checkpoint),
                              "runtime.execution=modal"]
        if group:
            resolved_overrides.append("group=" + group)
        if seed is not None:
            resolved_overrides.append(f"seed={seed}")
        if smoke:
            resolved_overrides.append("suite=smoke")
        if artifact_root:
            resolved_overrides.append("paths.artifact_root=" + json.dumps(artifact_root))
        resolved = load_config(config, overrides=resolved_overrides, validator=validate_cars)
        root = resolved["paths"]["artifact_root"]
        if root == "." and not artifact_root:
            root = SETTINGS["artifact_root"]
        COMMON.require_volume_path(SETTINGS, root)
        if resolved["runtime"]["modal_volume"] != SETTINGS["volumes"]["artifacts"]["name"]:
            raise ValueError("runtime.modal_volume differs from the configured Volume")
        if resume and not run_id:
            raise ValueError("--resume requires the original --run-id")
        argv = COMMON.shared_arguments(config=config, overrides=overrides, run_id=run_id, resume=resume,
                                       smoke=smoke, artifact_root=root, seed=seed, group=group)
        if mdlm_report:
            argv.extend(["--mdlm-report", mdlm_report])
        if checkpoint:
            argv.extend(["--checkpoint", checkpoint])
        started = time.perf_counter()
        with tqdm(total=1, desc="Modal CARS", unit="job", disable=not resolved["runtime"]["progress"]) as progress:
            result = worker.remote(argv, root)
            progress.update(1)
        record = result["record"]
        elapsed = time.perf_counter() - started
        tqdm.write(f"T09 Modal: {record['status']}; {record['summary']}; RPC={elapsed:.2f}s")
        if record["status"] != "completed":
            raise SystemExit(1)
    except Exception as exc:
        tqdm.write(f"T09 Modal: failed; {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    try:
        parser = build_parser("Generate SMILES with CARS on Modal.", default_config="configs/cars.yaml",
                              supports_group=True, add_arguments=_arguments)
        arguments = parser.parse_args()
        with modal.enable_output(), app.run():
            main(config=arguments.config, set=json.dumps(arguments.overrides), run_id=arguments.run_id or "",
                 resume=arguments.resume, smoke=arguments.smoke, artifact_root=arguments.artifact_root or "",
                 seed=arguments.seed, group=arguments.group or "", mdlm_report=arguments.mdlm_report or "",
                 checkpoint=arguments.checkpoint or "")
    except KeyboardInterrupt:
        tqdm.write("T09 Modal: interrupted")
        raise SystemExit(130) from None
    except Exception as exc:
        tqdm.write(f"T09 Modal: failed; {type(exc).__name__}: {exc}")
        raise SystemExit(1) from None
