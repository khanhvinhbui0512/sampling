"""T13: GPU value evaluation with committed state pools, shards and resume."""

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
from src.tasks.evaluate_value import _arguments, execute, validate_value_eval
from src.tasks.generation import source_overrides


def _load_helper():
    if not os.path.isfile("pyproject.toml") or not os.path.isfile("src/__init__.py"):
        raise RuntimeError("Run Modal commands from the repository root")
    name = "smiles_modal_common"
    spec = importlib.util.spec_from_file_location(name, "modal/common.py")
    if spec is None or spec.loader is None:
        raise ImportError("Cannot read modal/common.py")
    helper = importlib.util.module_from_spec(spec)
    sys.modules[name] = helper
    spec.loader.exec_module(helper)
    return helper


try:
    COMMON = _load_helper()
    SETTINGS = COMMON.load_settings()
    app = App(SETTINGS["app_name"] + "-evaluate-value", include_source=False)
    WORKER_OPTIONS = COMMON.worker_options(SETTINGS, "evaluate_value")
except Exception as exc:
    tqdm.write(f"T13 Modal: failed; {type(exc).__name__}: {exc}")
    raise


@app.function(**WORKER_OPTIONS)
def worker(argv: list[str], artifact_root: str) -> dict:
    started = time.perf_counter()
    with COMMON.volume_session(SETTINGS, artifact_root):
        record = execute(argv, execution="modal", on_checkpoint=COMMON.artifact_volume(SETTINGS).commit)
    return {"record": record.to_dict(), "worker_seconds": time.perf_counter()-started}


@app.local_entrypoint()
def main(config: str = "configs/value_eval.yaml", set: str = "", run_id: str = "", resume: bool = False,
         smoke: bool = False, artifact_root: str = "", seed: int | None = None,
         mdlm_report: str = "", checkpoint: str = "", verifier_report: str = "", verifier_checkpoint: str = "") -> None:
    COMMON.require_packaged_config(config)
    overrides = json.loads(set) if set.lstrip().startswith("[") else ([set] if set else [])
    if not isinstance(overrides, list) or any(not isinstance(item, str) for item in overrides):
        raise ValueError("--set must be one KEY=VALUE string or a JSON list")
    sources = source_overrides(mdlm_report=mdlm_report, checkpoint=checkpoint,
                               verifier_report=verifier_report, verifier_checkpoint=verifier_checkpoint)
    resolved_overrides = [*overrides, *sources, "runtime.execution=modal"]
    if seed is not None:
        resolved_overrides.append(f"seed={seed}")
    if smoke:
        resolved_overrides.append("suite=smoke")
    if artifact_root:
        resolved_overrides.append("paths.artifact_root=" + json.dumps(artifact_root))
    resolved = load_config(config, overrides=resolved_overrides, validator=validate_value_eval)
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
    for flag, value in (("--mdlm-report", mdlm_report), ("--checkpoint", checkpoint),
                        ("--verifier-report", verifier_report), ("--verifier-checkpoint", verifier_checkpoint)):
        if value:
            argv.extend([flag, value])
    started = time.perf_counter()
    with tqdm(total=1, desc="Modal value evaluation", unit="job", disable=not resolved["runtime"]["progress"]) as progress:
        result = worker.remote(argv, root)
        progress.update(1)
    record = result["record"]
    tqdm.write(f"T13 Modal: {record['status']}; {record['summary']}; RPC={time.perf_counter()-started:.2f}s")
    if record["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        parser = build_parser("Evaluate SMILES policy values on Modal.", default_config="configs/value_eval.yaml", add_arguments=_arguments)
        args = parser.parse_args()
        with modal.enable_output(), app.run():
            main(config=args.config, set=json.dumps(args.overrides), run_id=args.run_id or "", resume=args.resume,
                 smoke=args.smoke, artifact_root=args.artifact_root or "", seed=args.seed,
                 mdlm_report=args.mdlm_report or "", checkpoint=args.checkpoint or "",
                 verifier_report=args.verifier_report or "", verifier_checkpoint=args.verifier_checkpoint or "")
    except KeyboardInterrupt:
        tqdm.write("T13 Modal: interrupted")
        raise SystemExit(130) from None
    except Exception as exc:
        tqdm.write(f"T13 Modal: failed; {type(exc).__name__}: {exc}")
        raise SystemExit(1) from None
