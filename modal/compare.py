"""T15: CPU coordinator with separate generation, evaluation and report workers."""

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
from src.tasks.compare import add_arguments, argument_overrides, execute, execute_spec, validate_comparison


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
    app = App(SETTINGS["app_name"] + "-compare", include_source=False)
except Exception as exc:
    tqdm.write(f"T15 Modal: failed; {type(exc).__name__}: {exc}")
    raise


def _child(stage: str, spec: dict, artifact_root: str) -> dict:
    # volume_session commits even when a task fails; coordinator reloads next.
    with COMMON.volume_session(SETTINGS, artifact_root):
        record = execute_spec(stage, spec, execution="modal",
                              on_checkpoint=COMMON.artifact_volume(SETTINGS).commit)
    return {"status": record.status, "run_id": record.identity.run_id}


@app.function(**COMMON.worker_options(SETTINGS, "generate_base"))
def base_worker(spec: dict, artifact_root: str) -> dict:
    return _child("generation", spec, artifact_root)


@app.function(**COMMON.worker_options(SETTINGS, "generate_cars"))
def cars_worker(spec: dict, artifact_root: str) -> dict:
    return _child("generation", spec, artifact_root)


@app.function(**COMMON.worker_options(SETTINGS, "generate_mdm_vgb"))
def learned_worker(spec: dict, artifact_root: str) -> dict:
    return _child("generation", spec, artifact_root)


@app.function(**COMMON.worker_options(SETTINGS, "generate_hybrid"))
def hybrid_worker(spec: dict, artifact_root: str) -> dict:
    return _child("generation", spec, artifact_root)


@app.function(**COMMON.worker_options(SETTINGS, "evaluate"))
def evaluation_worker(spec: dict, artifact_root: str) -> dict:
    return _child("evaluate", spec, artifact_root)


@app.function(**COMMON.worker_options(SETTINGS, "report"))
def report_worker(spec: dict, artifact_root: str) -> dict:
    return _child("report", spec, artifact_root)


@app.function(**COMMON.worker_options(SETTINGS, "compare", coordinator=True))
def coordinator(argv: list[str], artifact_root: str) -> dict:
    volume = COMMON.artifact_volume(SETTINGS)

    def dispatch(stage, spec):
        if stage == "generation":
            worker = {"base": base_worker, "cars": cars_worker, "cars_ar": cars_worker,
                      "mdm_vgb": learned_worker, "hybrid": hybrid_worker}[spec["method"]]
        else:
            worker = evaluation_worker if stage == "evaluate" else report_worker
        return worker.remote(spec, artifact_root)

    with COMMON.volume_session(SETTINGS, artifact_root):
        record = execute(argv, execution="modal", dispatch=dispatch,
                         on_checkpoint=volume.commit, reload_artifacts=volume.reload)
    return {"status": record.status, "summary": record.summary, "artifacts": list(record.artifacts)}


@app.local_entrypoint()
def main(config: str = "configs/comparison.yaml", set: str = "", run_id: str = "", resume: bool = False,
         smoke: bool = False, artifact_root: str = "", mdlm_report: str = "", checkpoint: str = "",
         verifier_report: str = "", verifier_checkpoint: str = "", purpose: str = "",
         lock_from: str = "", select_variant: str = "", lock_manifest: str = "") -> None:
    COMMON.require_packaged_config(config)
    overrides = json.loads(set) if set.lstrip().startswith("[") else ([set] if set else [])
    if not isinstance(overrides, list) or any(not isinstance(item, str) for item in overrides):
        raise ValueError("--set must be a KEY=VALUE string or JSON list")
    parser = build_parser("Compare on Modal", default_config=config, supports_seed=False, add_arguments=add_arguments)
    flags = []
    for flag, value in (("--mdlm-report", mdlm_report), ("--checkpoint", checkpoint),
                        ("--verifier-report", verifier_report), ("--verifier-checkpoint", verifier_checkpoint),
                        ("--purpose", purpose), ("--lock-from", lock_from), ("--select-variant", select_variant),
                        ("--lock-manifest", lock_manifest)):
        if value:
            flags.extend([flag, value])
    extra = [*argument_overrides(parser.parse_args(flags)), "runtime.execution=modal"]
    if smoke:
        extra.append("suite=smoke")
    if artifact_root:
        extra.append("paths.artifact_root=" + json.dumps(artifact_root))
    resolved = load_config(config, overrides=[*overrides, *extra], validator=validate_comparison)
    root = resolved["paths"]["artifact_root"]
    if root == "." and not artifact_root:
        root = SETTINGS["artifact_root"]
    COMMON.require_volume_path(SETTINGS, root)
    if resolved["runtime"]["modal_volume"] != SETTINGS["volumes"]["artifacts"]["name"]:
        raise ValueError("runtime.modal_volume differs from the configured Volume")
    if resume and not run_id:
        raise ValueError("--resume requires the original --run-id")
    argv = COMMON.shared_arguments(config=config, overrides=overrides, run_id=run_id, resume=resume,
                                   smoke=smoke, artifact_root=root) + flags
    started = time.perf_counter()
    with tqdm(total=1, desc="Modal comparison coordinator", unit="matrix", disable=not resolved["runtime"]["progress"]) as progress:
        result = coordinator.remote(argv, root)
        progress.update(1)
    tqdm.write(f"T15 Modal: {result['status']}; {result['summary']}; RPC={time.perf_counter()-started:.2f}s")
    tqdm.write("Artifacts: " + ", ".join(result["artifacts"]))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        parser = build_parser("Run a matched comparison on Modal.", default_config="configs/comparison.yaml",
                              supports_seed=False, add_arguments=add_arguments)
        args = parser.parse_args()
        with modal.enable_output(), app.run():
            main(config=args.config, set=json.dumps(args.overrides), run_id=args.run_id or "", resume=args.resume,
                 smoke=args.smoke, artifact_root=args.artifact_root or "", mdlm_report=args.mdlm_report or "",
                 checkpoint=args.checkpoint or "", verifier_report=args.verifier_report or "",
                 verifier_checkpoint=args.verifier_checkpoint or "", purpose=args.purpose or "",
                 lock_from=args.lock_from or "", select_variant=args.select_variant or "", lock_manifest=args.lock_manifest or "")
    except KeyboardInterrupt:
        tqdm.write("T15 Modal: interrupted")
        raise SystemExit(130) from None
    except Exception as exc:
        tqdm.write(f"T15 Modal: failed; {type(exc).__name__}: {exc}")
        raise SystemExit(1) from None
