"""T04: fetch only configured pretrained artifacts; never execute upstream code."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from collections.abc import Sequence
from contextlib import contextmanager

from tqdm.auto import tqdm

from src.autoregressive import model_source, require_model_snapshot, validate_models
from src.config import config_digest
from src.schemas import TaskResult
from src.tasks.common import TaskContext, run_task
from src.tasks.prepare_corpus import commit_file, corpus_lock


def _published_file(path: str, detail: dict) -> bool:
    if not os.path.isfile(path) or os.path.getsize(path) != detail["size"]:
        return False
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == detail["md5"]


def _figshare(context: TaskContext, options: dict) -> None:
    request = urllib.request.Request(options["metadata_url"], headers={"User-Agent": "smiles-T04/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        article = json.load(response)
    if (article["id"], article["version"]) != (options["article_id"], options["article_version"]):
        raise ValueError("Figshare metadata differs from the pinned article version")
    official = {entry["id"]: entry for entry in article["files"]}
    for filename, detail in options["files"].items():
        entry = official.get(detail["file_id"], {})
        if (entry.get("name"), entry.get("size"), entry.get("computed_md5")) != (
            filename, detail["size"], detail["md5"],
        ):
            raise ValueError(f"Published RetroStar checksum/size differs: {filename}")
        target = context.store.path(os.path.join(options["directory"], filename))
        if _published_file(target, detail):
            context.log(f"RetroStar: verified cached {filename}")
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        temporary = target + ".part"
        request = urllib.request.Request(detail["url"], headers={"User-Agent": "smiles-T04/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response:
            with open(temporary, "wb") as output:
                with context.progress(total=detail["size"], desc=filename, unit="B", unit_scale=True) as progress:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        progress.update(len(chunk))
                    output.flush()
                    os.fsync(output.fileno())
        if not _published_file(temporary, detail):
            raise ValueError(f"RetroStar download has wrong size/checksum: {filename}")
        commit_file(temporary, target)


def _huggingface(context: TaskContext, options: dict) -> None:
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    token = os.environ.get("HF_TOKEN") or None
    try:
        info = HfApi().model_info(options["model_id"], revision=options["revision"],
                                 token=token, files_metadata=True)
        if info.sha != options["revision"]:
            raise ValueError("Hugging Face resolved a different commit")
        siblings = {file.rfilename: file for file in info.siblings}
        if any(filename not in siblings for filename in options["required_files"]):
            raise ValueError("Pinned snapshot is missing required files")

        class DownloadProgress(tqdm):
            def __init__(self, *args, **kwargs):
                kwargs["disable"] = not context.config["runtime"]["progress"]
                super().__init__(*args, **kwargs)

        snapshot_download(
            repo_id=options["model_id"], revision=options["revision"], token=token,
            local_dir=context.store.path(options["directory"]),
            allow_patterns=options["required_files"], tqdm_class=DownloadProgress,
        )
        # Large HF files publish SHA-256 in their LFS metadata; small files use
        # the Git blob SHA-1, including Git's length-prefixed blob header.
        for filename in options["required_files"]:
            physical = context.store.require_file(os.path.join(options["directory"], filename))
            sibling = siblings[filename]
            if sibling.size is not None and os.path.getsize(physical) != sibling.size:
                raise ValueError(f"Snapshot file has the wrong size: {filename}")
            if sibling.lfs is not None:
                from src.artifacts import sha256_file

                if sha256_file(physical) != sibling.lfs.sha256:
                    raise ValueError(f"Snapshot LFS checksum mismatch: {filename}")
            else:
                digest = hashlib.sha1(usedforsecurity=False)
                digest.update(f"blob {os.path.getsize(physical)}\0".encode("ascii"))
                with open(physical, "rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                if not sibling.blob_id or digest.hexdigest() != sibling.blob_id:
                    raise ValueError(f"Snapshot Git blob checksum mismatch: {filename}")
    except GatedRepoError:
        raise RuntimeError("Llama access denied. Accept its HF license and provide HF_TOKEN; no fallback model.") from None
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        raise RuntimeError(f"Pinned Llama download failed (HTTP {status}); check HF access/network. No fallback model.") from None


@contextmanager
def _model_lock(context: TaskContext):
    # Reuse the existing artifact-root writer lock: T02/T03/T04 run sequentially.
    with corpus_lock(context.store, execution=context.config["runtime"]["execution"]):
        yield


def fetch_models(context: TaskContext) -> TaskResult:
    state = context.resume_state or {"completed": {}}
    artifacts = []
    with _model_lock(context):
        for name in context.progress(context.config["models"]["download"], desc="Pretrained models", unit="model"):
            options = context.config["models"][name]
            path = options["manifest"]
            if name in state["completed"]:
                context.store.require_file(path, checksum=state["completed"][name])
                require_model_snapshot(context.store, options)
            elif os.path.isfile(context.store.path(path)):
                require_model_snapshot(context.store, options)
                context.log(f"{name}: verified existing snapshot")
            else:
                with context.tracker.measure("model_download"):
                    if options["backend"] == "huggingface":
                        _huggingface(context, options)
                    else:
                        _figshare(context, options)
                    source = model_source(options)
                    context.store.create_manifest(
                        path, kind="pretrained_model", config_hash=context.identity.config_hash,
                        files=[os.path.join(options["directory"], file) for file in options["required_files"]],
                        metadata={"source": source, "source_hash": config_digest(source)},
                    )
            from src.artifacts import sha256_file

            state["completed"][name] = sha256_file(context.store.require_file(path))
            context.save_checkpoint(state)
            artifacts.append(path)
            context.log(f"{name}: ready; manifest={path}")
    report = context.output_path("models.json")
    context.store.write_json(report, {"models": state["completed"], "mdlm_initialization": "scratch"})
    return TaskResult(f"pretrained models={len(artifacts)}; report={report}", artifacts=tuple(artifacts + [report]))


def execute(argv: Sequence[str] | None = None, *, execution: str = "local"):
    return run_task("T04", fetch_models, argv=argv, execution=execution,
                    description="Fetch pinned Llama and RetroStar snapshots.",
                    default_config="configs/models.yaml", supports_seed=False, validator=validate_models)


if __name__ == "__main__":
    try:
        record = execute()
        if record.status != "completed":
            raise SystemExit(1)
    except KeyboardInterrupt:
        tqdm.write("T04: interrupted")
        raise SystemExit(130) from None
    except Exception as exc:
        tqdm.write(f"T04: failed; {type(exc).__name__}: {exc}")
        raise SystemExit(1) from None
