"""T14: evaluate checked generation artifacts on the shared Modal Volume."""

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
from src.tasks.evaluate import add_arguments, execute, validate_evaluation


def _load_helper():
    if not os.path.isfile("pyproject.toml") or not os.path.isfile("src/__init__.py"):
        raise RuntimeError("Run Modal commands from the repository root")
    spec = importlib.util.spec_from_file_location("smiles_modal_common", "modal/common.py")
    if spec is None or spec.loader is None:
        raise ImportError("Cannot read modal/common.py")
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    return helper


try:
    COMMON = _load_helper()
    SETTINGS = COMMON.load_settings()
    app = App(SETTINGS["app_name"]+"-evaluate", include_source=False)
    WORKER_OPTIONS = COMMON.worker_options(SETTINGS, "evaluate")
except Exception as exc:
    tqdm.write(f"T14 Modal: failed; {type(exc).__name__}: {exc}")
    raise


@app.function(**WORKER_OPTIONS)
def worker(argv: list[str], artifact_root: str) -> dict:
    with COMMON.volume_session(SETTINGS, artifact_root):
        record = execute(argv, execution="modal")
    return record.to_dict()


@app.local_entrypoint()
def main(config: str = "configs/evaluation.yaml", set: str = "", run_id: str = "", resume: bool = False,
         smoke: bool = False, artifact_root: str = "", seed: int | None = None, generation_run: str = "") -> None:
    COMMON.require_packaged_config(config)
    overrides = json.loads(set) if set.lstrip().startswith("[") else ([set] if set else [])
    if not isinstance(overrides, list) or any(not isinstance(item, str) for item in overrides):
        raise ValueError("--set must be a KEY=VALUE string or JSON list")
    source = ["evaluation.source_run="+json.dumps(generation_run)] if generation_run else []
    extra = [*source, "runtime.execution=modal"]
    if smoke:
        extra.append("suite=smoke")
    if seed is not None:
        extra.append(f"seed={seed}")
    if artifact_root:
        extra.append("paths.artifact_root="+json.dumps(artifact_root))
    resolved = load_config(config, overrides=[*overrides, *extra], validator=validate_evaluation)
    root = resolved["paths"]["artifact_root"]
    if root == "." and not artifact_root:
        root = SETTINGS["artifact_root"]
    COMMON.require_volume_path(SETTINGS, root)
    if resolved["runtime"]["modal_volume"] != SETTINGS["volumes"]["artifacts"]["name"]:
        raise ValueError("runtime.modal_volume differs from the configured Volume")
    if resume and not run_id:
        raise ValueError("--resume requires the original --run-id")
    argv = COMMON.shared_arguments(config=config, overrides=overrides, run_id=run_id, resume=resume,
                                   smoke=smoke, artifact_root=root, seed=seed)
    if generation_run:
        argv.extend(["--generation-run", generation_run])
    started = time.perf_counter()
    with tqdm(total=1, desc="Modal molecular evaluation", unit="job", disable=not resolved["runtime"]["progress"]) as progress:
        record = worker.remote(argv, root)
        progress.update(1)
    tqdm.write(f"T14 Modal: {record['status']}; {record['summary']}; RPC={time.perf_counter()-started:.2f}s")
    if record["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        parser = build_parser("Evaluate molecular generation on Modal.", default_config="configs/evaluation.yaml", add_arguments=add_arguments)
        args = parser.parse_args()
        with modal.enable_output(), app.run():
            main(config=args.config, set=json.dumps(args.overrides), run_id=args.run_id or "", resume=args.resume,
                 smoke=args.smoke, artifact_root=args.artifact_root or "", seed=args.seed, generation_run=args.generation_run or "")
    except KeyboardInterrupt:
        tqdm.write("T14 Modal: interrupted")
        raise SystemExit(130) from None
    except Exception as exc:
        tqdm.write(f"T14 Modal: failed; {type(exc).__name__}: {exc}")
        raise SystemExit(1) from None
