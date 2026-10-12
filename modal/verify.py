"""Run T12 remotely with ``python modal/verify.py`` from repo root."""

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
from src.tasks.verify import PHASES, validate_verification


def _load_helper():
    if not os.path.isfile("pyproject.toml") or not os.path.isfile("src/__init__.py"):
        raise RuntimeError("Run Modal commands from the repository root")
    # A unique module name keeps the project folder separate from the Modal SDK.
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
    app = App(SETTINGS["app_name"] + "-verify", include_source=False)
    WORKER_OPTIONS = COMMON.worker_options(SETTINGS, "verify")
    image = WORKER_OPTIONS.pop("image")
    volumes = WORKER_OPTIONS.pop("volumes")
except Exception as exc:
    tqdm.write(f"T12 Modal: failed; {type(exc).__name__}: {exc}")
    raise


@app.function(image=image, volumes=volumes, **WORKER_OPTIONS)
def worker(argv: list[str], artifact_root: str) -> dict:
    from src.tasks.verify import execute

    started = time.perf_counter()
    try:
        with COMMON.volume_session(SETTINGS, artifact_root):
            record = execute(argv, execution="modal")
    except Exception as exc:
        tqdm.write(f"T12 Modal worker: failed; {type(exc).__name__}: {exc}")
        raise
    return {"record": record.to_dict(), "worker_seconds": time.perf_counter() - started}


def _parse_overrides(value: str) -> list[str]:
    """Modal primitive CLI arguments encode repeated --set as a JSON list."""
    if not value:
        return []
    if value.lstrip().startswith("["):
        items = json.loads(value)
        if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
            raise ValueError("--set must be one KEY=VALUE string or a JSON list of those strings")
        return items
    return [value]


@app.local_entrypoint()
def main(
    config: str = "configs/verification.yaml", phase: str = "", set: str = "",
    run_id: str = "", resume: bool = False, smoke: bool = False, artifact_root: str = "",
) -> None:
    """Forward config, phase, overrides, run ID, resume, smoke, and artifact root."""
    try:
        COMMON.require_packaged_config(config)
        overrides = _parse_overrides(set)
        if phase and phase not in PHASES:
            raise ValueError(f"Unknown phase {phase}; choose one of {', '.join(PHASES)}")
        resolved_overrides = list(overrides)
        if phase:
            resolved_overrides.append("verification.phase=" + phase)
        if smoke:
            resolved_overrides.append("suite=smoke")
        if artifact_root:
            resolved_overrides.append("paths.artifact_root=" + json.dumps(artifact_root))
        resolved_overrides.append("runtime.execution=modal")
        resolved = load_config(config, overrides=resolved_overrides, validator=validate_verification)
        # The local default '.' maps to the persistent remote artifact subtree.
        root = resolved["paths"]["artifact_root"]
        if root == "." and not artifact_root:
            root = SETTINGS["artifact_root"]
        COMMON.require_volume_path(SETTINGS, root)
        configured_volume = resolved["runtime"]["modal_volume"]
        if configured_volume != SETTINGS["volumes"]["artifacts"]["name"]:
            raise ValueError("runtime.modal_volume differs from the configured Modal Volume")
        if resume and not run_id:
            raise ValueError("--resume requires the original --run-id")
        argv = COMMON.shared_arguments(
            config=config, overrides=overrides, run_id=run_id, resume=resume,
            smoke=smoke, artifact_root=root,
        )
        if phase:
            argv.extend(("--phase", phase))
        started = time.perf_counter()
        with tqdm(total=1, desc="Modal verification", unit="job", disable=not resolved["runtime"]["progress"]) as progress:
            result = worker.remote(argv, root)
            progress.update(1)
        rpc_seconds = time.perf_counter() - started
        record = result["record"]
        # Keep transport/startup time distinct from algorithm compute accounting.
        overhead = max(0.0, rpc_seconds - result["worker_seconds"])
        tqdm.write(
            f"T12 Modal: {record['status']}; {record['summary']}; "
            f"RPC={rpc_seconds:.2f}s; startup/transport={overhead:.2f}s",
        )
        if record["status"] != "completed":
            raise SystemExit(1)
    except Exception as exc:
        tqdm.write(f"T12 Modal: failed; {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    try:
        arguments = COMMON.script_arguments(
            "Verify the SMILES project on Modal.", default_config="configs/verification.yaml", phases=PHASES,
        )
        with modal.enable_output(), app.run():
            main(**arguments)
    except KeyboardInterrupt:
        tqdm.write("T12 Modal: interrupted")
        raise SystemExit(130) from None
    except Exception as exc:
        tqdm.write(f"T12 Modal: failed; {type(exc).__name__}: {exc}")
        raise SystemExit(1) from None
