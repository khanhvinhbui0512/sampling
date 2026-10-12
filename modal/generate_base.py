"""T08: run base on Modal with the same task, model inputs and recorder."""

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
from src.tasks.generation import TASKS, add_arguments, execute_generation, source_overrides, validate_generation

METHOD = "base"
TASK_ID = "T08"


def _load_helper():
    if not os.path.isfile("pyproject.toml") or not os.path.isfile("src/__init__.py"):
        raise RuntimeError("Run Modal commands from the repository root")
    spec = importlib.util.spec_from_file_location("smiles_modal_common", "modal/common.py")
    if spec is None or spec.loader is None:
        raise ImportError("Cannot read modal/common.py")
    helper = importlib.util.module_from_spec(spec)
    sys.modules["smiles_modal_common"] = helper
    spec.loader.exec_module(helper)
    return helper


try:
    COMMON = _load_helper()
    SETTINGS = COMMON.load_settings()
    app = App(SETTINGS["app_name"] + "-generate-" + METHOD.replace("_", "-"), include_source=False)
    worker_options = COMMON.worker_options(SETTINGS, "generate_" + METHOD)
except Exception as exc:
    tqdm.write(f"{TASK_ID} Modal: failed; {type(exc).__name__}: {exc}")
    raise


@app.function(**worker_options)
def worker(argv: list[str], artifact_root: str) -> dict:
    started = time.perf_counter()
    with COMMON.volume_session(SETTINGS, artifact_root):
        record = execute_generation(METHOD, argv, execution="modal",
                                    on_checkpoint=COMMON.artifact_volume(SETTINGS).commit)
    return {"record": record.to_dict(), "worker_seconds": time.perf_counter()-started}


@app.local_entrypoint()
def main(config: str = TASKS[METHOD][1], set: str = "", run_id: str = "", resume: bool = False,
         smoke: bool = False, artifact_root: str = "", seed: int | None = None, group: str = "",
         mdlm_report: str = "", checkpoint: str = "", verifier_report: str = "", verifier_checkpoint: str = "") -> None:
    COMMON.require_packaged_config(config)
    overrides = json.loads(set) if set.lstrip().startswith("[") else ([set] if set else [])
    if not isinstance(overrides, list) or any(not isinstance(item, str) for item in overrides):
        raise ValueError("--set must be one KEY=VALUE string or a JSON list")
    if METHOD != "mdm_vgb" and (verifier_report or verifier_checkpoint):
        raise ValueError("Only learned MDM-VGB accepts verifier inputs")
    sources = source_overrides(mdlm_report=mdlm_report, checkpoint=checkpoint,
                               verifier_report=verifier_report, verifier_checkpoint=verifier_checkpoint)
    resolved_overrides = [*overrides, *sources, "runtime.execution=modal"]
    if group:
        resolved_overrides.append("group=" + group)
    if seed is not None:
        resolved_overrides.append(f"seed={seed}")
    if smoke:
        resolved_overrides.append("suite=smoke")
    if artifact_root:
        resolved_overrides.append("paths.artifact_root=" + json.dumps(artifact_root))
    resolved = load_config(config, overrides=resolved_overrides, validator=lambda cfg: validate_generation(cfg, METHOD))
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
    for flag, value in (("--mdlm-report", mdlm_report), ("--checkpoint", checkpoint),
                        ("--verifier-report", verifier_report), ("--verifier-checkpoint", verifier_checkpoint)):
        if value:
            argv.extend([flag, value])
    started = time.perf_counter()
    with tqdm(total=1, desc="Modal " + METHOD, unit="job", disable=not resolved["runtime"]["progress"]) as progress:
        result = worker.remote(argv, root)
        progress.update(1)
    record = result["record"]
    tqdm.write(f"{TASK_ID} Modal: {record['status']}; {record['summary']}; RPC={time.perf_counter()-started:.2f}s")
    if record["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        parser = build_parser("Generate " + METHOD + " on Modal.", default_config=TASKS[METHOD][1],
                              supports_group=True, add_arguments=lambda item: add_arguments(item, METHOD))
        args = parser.parse_args()
        with modal.enable_output(), app.run():
            main(config=args.config, set=json.dumps(args.overrides), run_id=args.run_id or "", resume=args.resume,
                 smoke=args.smoke, artifact_root=args.artifact_root or "", seed=args.seed, group=args.group or "",
                 mdlm_report=args.mdlm_report or "", checkpoint=args.checkpoint or "",
                 verifier_report=getattr(args, "verifier_report", None) or "",
                 verifier_checkpoint=getattr(args, "verifier_checkpoint", None) or "")
    except KeyboardInterrupt:
        tqdm.write(f"{TASK_ID} Modal: interrupted")
        raise SystemExit(130) from None
    except Exception as exc:
        tqdm.write(f"{TASK_ID} Modal: failed; {type(exc).__name__}: {exc}")
        raise SystemExit(1) from None
