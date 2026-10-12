"""T16: CPU report generation from checked metrics in the Modal Volume."""

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
from src.tasks.report import add_arguments, execute, validate_report


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
    app = App(SETTINGS["app_name"]+"-report", include_source=False)
    WORKER_OPTIONS = COMMON.worker_options(SETTINGS, "report")
except Exception as exc:
    tqdm.write(f"T16 Modal: failed; {type(exc).__name__}: {exc}")
    raise


@app.function(**WORKER_OPTIONS)
def worker(argv: list[str], artifact_root: str) -> dict:
    with COMMON.volume_session(SETTINGS, artifact_root):
        record = execute(argv, execution="modal")
    return {"status": record.status, "summary": record.summary, "artifacts": list(record.artifacts)}


@app.local_entrypoint()
def main(config: str = "configs/report.yaml", set: str = "", run_id: str = "", resume: bool = False,
         smoke: bool = False, artifact_root: str = "", seed: int | None = None,
         evaluation_manifest: str = "[]", value_manifest: str = "[]") -> None:
    COMMON.require_packaged_config(config)
    overrides = json.loads(set) if set.lstrip().startswith("[") else ([set] if set else [])
    if not isinstance(overrides, list) or any(not isinstance(item, str) for item in overrides):
        raise ValueError("--set must be a KEY=VALUE string or JSON list")
    molecular, value = json.loads(evaluation_manifest), json.loads(value_manifest)
    if any(not isinstance(paths, list) or any(not isinstance(path, str) for path in paths) for paths in (molecular, value)):
        raise ValueError("Manifest arguments must encode lists of paths")
    extra = ["runtime.execution=modal"]
    if molecular:
        extra.append("report.evaluation_manifests="+json.dumps(molecular))
    if value:
        extra.append("report.value_manifests="+json.dumps(value))
    if smoke:
        extra.append("suite=smoke")
    if seed is not None:
        extra.append(f"seed={seed}")
    if artifact_root:
        extra.append("paths.artifact_root="+json.dumps(artifact_root))
    resolved = load_config(config, overrides=[*overrides, *extra], validator=validate_report)
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
    for flag, paths in (("--evaluation-manifest", molecular), ("--value-manifest", value)):
        for path in paths:
            argv.extend([flag, path])
    started = time.perf_counter()
    with tqdm(total=1, desc="Modal report", unit="job", disable=not resolved["runtime"]["progress"]) as progress:
        result = worker.remote(argv, root)
        progress.update(1)
    tqdm.write(f"T16 Modal: {result['status']}; {result['summary']}; RPC={time.perf_counter()-started:.2f}s")
    tqdm.write("Artifacts: "+", ".join(result["artifacts"]))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        parser = build_parser("Publish a checked SMILES report on Modal.", default_config="configs/report.yaml", add_arguments=add_arguments)
        args = parser.parse_args()
        with modal.enable_output(), app.run():
            main(config=args.config, set=json.dumps(args.overrides), run_id=args.run_id or "", resume=args.resume,
                 smoke=args.smoke, artifact_root=args.artifact_root or "", seed=args.seed,
                 evaluation_manifest=json.dumps(args.evaluation_manifest), value_manifest=json.dumps(args.value_manifest))
    except KeyboardInterrupt:
        tqdm.write("T16 Modal: interrupted")
        raise SystemExit(130) from None
    except Exception as exc:
        tqdm.write(f"T16 Modal: failed; {type(exc).__name__}: {exc}")
        raise SystemExit(1) from None
