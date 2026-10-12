"""T05: train the shared class-conditioned MDLM, selecting only on validation.

Uses bounded shard caching, gradient accumulation, fixed validation randomness,
atomic safetensors exports, and the shared optimizer/RNG resume journal. Test
tensor shards are never opened by this trainer.
"""

from __future__ import annotations

import bisect
import math
import os
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from contextlib import nullcontext
from typing import Any

import torch
from safetensors.torch import save_file
from tqdm.auto import tqdm

from src.artifacts import ArtifactStore, Dependency, capture_rng_state, restore_rng_state, sha256_file
from src.config import Config, config_digest
from src.mdlm import GroupMDLM, corrupt_tokens, masked_objective, model_specification, validate_mdlm
from src.schemas import GROUPS, ArtifactManifest, TaskResult
from src.tasks.common import TaskContext, run_task
from src.tasks.prepare_corpus import commit_file
from src.tokenizer import SmilesTokenizer

TRAINER_VERSION = "smiles-mdlm-trainer-v1"
SHARD_VERSION = "smiles-token-shards-v1"


def training_dependencies(config: Config, store: ArtifactStore) -> dict[str, Dependency]:
    options = config["data"]["tokenizer"]
    metadata = store.read_json(options["metadata_path"])
    tokenizer = SmilesTokenizer.load(store, options["metadata_path"])
    if (tokenizer.max_length != config["mdlm"]["architecture"]["max_length"]
            or metadata["vocabulary_path"] != options["vocabulary_path"]
            or metadata.get("group_order") != list(GROUPS)
            or metadata["provenance"]["training_split"] != "train"
            or metadata["training_only_vocabulary"] is not True):
        raise ValueError("T03 tokenizer width, vocabulary path, groups, or training provenance differs")
    corpus_path = store.directory("data", "corpus", "manifest.json")
    corpus_hash = metadata["provenance"]["corpus_sha256"]
    store.require_file(corpus_path, checksum=corpus_hash)
    corpus = ArtifactManifest.from_dict(store.read_json(corpus_path))
    train_path = store.directory("data", "corpus", "train.parquet")
    if corpus.kind != "corpus" or corpus.files.get(train_path) != metadata["provenance"]["training_split_sha256"]:
        raise ValueError("Tokenizer does not originate from the declared training corpus")
    dependencies = {
        "corpus": Dependency(corpus_path, checksum=corpus_hash),
        "tokenizer": Dependency(options["metadata_path"]),
        "vocabulary": Dependency(options["vocabulary_path"], checksum=metadata["vocabulary_sha256"]),
    }
    for split in ("train", "validation"):
        path = store.directory("data", "tokenized", split, "manifest.json")
        reference = metadata["tokenized_manifests"][split]
        if reference["path"] != path:
            raise ValueError(f"Unexpected {split} tokenized manifest path")
        manifest = store.read_json(path)
        source = store.directory("data", "corpus", split + ".parquet")
        if manifest["metadata"]["source_sha256"] != corpus.files.get(source):
            raise ValueError(f"{split} token shards do not originate from this corpus split")
        dependencies[split] = Dependency(path, checksum=reference["sha256"], manifest=True, kind="tokenized", metadata={
            "split": split, "version": SHARD_VERSION, "corpus_sha256": corpus_hash,
            "tokenizer_fingerprint": tokenizer.fingerprint, "max_length": tokenizer.max_length,
            "vocab_size": tokenizer.vocab_size, "group_order": list(GROUPS),
            "tokenization_fingerprint": metadata["tokenization_fingerprint"],
        })
    return dependencies


class TokenShards:
    """Random access by global row, with at most cache_size resident shards."""

    def __init__(self, store: ArtifactStore, split: str, tokenizer: SmilesTokenizer, *, cache_size: int) -> None:
        self.store, self.split, self.tokenizer = store, split, tokenizer
        self.manifest = store.require_manifest(store.directory("data", "tokenized", split, "manifest.json"), kind="tokenized")
        self.metadata = self.manifest.metadata
        self.shards = self.metadata["shards"]
        self.ends = []
        count = 0
        for shard in self.shards:
            if (self.manifest.files.get(shard["path"]) != shard["sha256"]
                    or type(shard["rows"]) is not int or shard["rows"] < 0):
                raise ValueError("Shard list disagrees with its manifest")
            count += shard["rows"]
            self.ends.append(count)
        if (count != self.metadata["rows"] or count == 0
                or any(self.metadata["group_rows"]["kept"].get(group, 0) == 0 for group in GROUPS)):
            raise ValueError(f"{split} is empty or lacks a molecular group; inspect T03 lengths.json")
        self.rows, self.cache_size = count, cache_size
        self.cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

    def load_shard(self, index: int) -> dict[str, Any]:
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        detail = self.shards[index]
        payload = torch.load(self.store.require_file(detail["path"], checksum=detail["sha256"]),
                             map_location="cpu", weights_only=True)
        ids, attention, groups = payload["token_ids"], payload["attention_mask"], payload["group_mask"]
        expected_shape = (detail["rows"], self.tokenizer.max_length)
        if (payload["version"] != SHARD_VERSION or payload["split"] != self.split
                or payload["tokenizer_fingerprint"] != self.tokenizer.fingerprint
                or payload["corpus_sha256"] != self.metadata["corpus_sha256"]
                or payload["group_order"] != list(GROUPS)
                or payload["special_tokens"] != self.tokenizer.special_tokens.to_dict()
                or ids.shape != expected_shape or attention.shape != expected_shape
                or ids.dtype != torch.int64 or attention.dtype != torch.bool
                or groups.shape != (detail["rows"], len(GROUPS)) or groups.dtype != torch.bool
                or payload["lengths"].shape != (detail["rows"],)):
            raise ValueError(f"Malformed or mismatched {self.split} shard")
        special = self.tokenizer.special_tokens
        lengths = attention.sum(-1)
        if (torch.any(ids < 0) or torch.any(ids >= self.tokenizer.vocab_size)
                or torch.any(ids == special.mask_id) or torch.any(groups.sum(-1) == 0)
                or torch.any(lengths < 1) or torch.any(lengths != payload["lengths"])
                or torch.any(attention != (ids != special.pad_id))
                or torch.any((ids == special.eos_id).sum(-1) != 1)
                or torch.any(attention != (torch.arange(ids.shape[1])[None] < lengths[:, None]))
                or torch.any(ids[torch.arange(ids.shape[0]), lengths - 1] != special.eos_id)):
            raise ValueError(f"Invalid EOS/PAD, vocabulary, length, or labels in {self.split} shard")
        self.cache[index] = payload
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return payload

    def batch(self, indices: Sequence[int], *, generator: torch.Generator | None = None):
        # Read each shard once per batch, even if the bounded cache cannot hold
        # every randomly selected shard. Allocate only the requested tensor rows.
        locations: dict[int, list[tuple[int, int]]] = {}
        for output, global_index in enumerate(indices):
            if not 0 <= global_index < self.rows:
                raise IndexError("Training row is out of range")
            shard = bisect.bisect_right(self.ends, global_index)
            previous = self.ends[shard - 1] if shard else 0
            locations.setdefault(shard, []).append((output, global_index - previous))
        ids = torch.empty((len(indices), self.tokenizer.max_length), dtype=torch.long)
        attention = torch.empty_like(ids, dtype=torch.bool)
        group_mask = torch.empty((len(indices), len(GROUPS)), dtype=torch.bool)
        for shard, rows in locations.items():
            payload = self.load_shard(shard)
            destination, source = zip(*rows, strict=True)
            ids[list(destination)] = payload["token_ids"][list(source)]
            attention[list(destination)] = payload["attention_mask"][list(source)]
            group_mask[list(destination)] = payload["group_mask"][list(source)]
        groups = torch.multinomial(group_mask.float(), 1, generator=generator).flatten()
        return ids, attention, groups

    def validation_batches(self, size: int, max_rows: int | None, *, generator) -> Iterator[tuple]:
        count = self.rows if max_rows is None else min(self.rows, max_rows)
        for start in range(0, count, size):
            yield self.batch(range(start, min(start + size, count)), generator=generator)


class WarmupCosine:
    """Step-indexed LR schedule; first optimizer update is warmup step one."""

    def __init__(self, optimizer: torch.optim.Optimizer, training: dict[str, Any]) -> None:
        self.optimizer = optimizer
        self.settings = {key: training[key] for key in (
            "steps", "warmup_steps", "learning_rate", "minimum_learning_rate",
        )}
        self.completed_steps = 0

    def prepare_next(self) -> float:
        step = self.completed_steps + 1
        settings = self.settings
        if step > settings["steps"]:
            raise ValueError("Scheduler exceeds configured training steps")
        warmup, peak, minimum = settings["warmup_steps"], settings["learning_rate"], settings["minimum_learning_rate"]
        if warmup and step <= warmup:
            rate = peak * step / warmup
        else:
            fraction = (step - warmup) / (settings["steps"] - warmup)
            rate = minimum + (peak - minimum) * (1 + math.cos(math.pi * fraction)) / 2
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        return rate

    def state_dict(self) -> dict[str, Any]:
        return {"settings": self.settings, "completed_steps": self.completed_steps}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["settings"] != self.settings or not 0 <= state["completed_steps"] <= self.settings["steps"]:
            raise ValueError("Resume LR scheduler differs from the configured schedule")
        self.completed_steps = state["completed_steps"]


def _device_precision(training: dict) -> tuple[torch.device, torch.dtype | None]:
    requested = training["device"]
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device("cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu")
    precision = None
    if training["dtype"] == "bfloat16" and device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("GPU does not support requested BF16; explicitly configure float32")
        precision = torch.bfloat16
    # CPU uses float32; this actual precision is recorded in the resume state.
    return device, precision


def _precision_context(device: torch.device, precision: torch.dtype | None):
    return nullcontext() if precision is None else torch.autocast(device_type=device.type, dtype=precision)


def _corrupted_batch(batch, special, diffusion, *, generator=None):
    ids, molecular_attention, groups = batch
    # T03's mask describes molecular length; it is not the diffusion attention
    # mask. As in QM9, clean PAD is corrupted and supervised so the model can
    # learn padded tails when sampling from an entirely MASKed fixed-width root.
    attention = torch.ones_like(molecular_attention, dtype=torch.bool)
    times = torch.rand(ids.shape[0], generator=generator).clamp_min(diffusion["minimum_time"])
    corrupted, masked = corrupt_tokens(ids, attention, special, times, diffusion["epsilon"], generator=generator)
    return ids, attention, groups, times, corrupted, masked


def _batch_loss(context: TaskContext, model: GroupMDLM, batch: tuple, device, precision) -> torch.Tensor:
    targets, attention, groups, times, corrupted, masked = [tensor.to(device) for tensor in batch]
    with context.tracker.model_call("mdlm", batch_size=targets.shape[0], gpu=device.type == "cuda", device=device):
        with _precision_context(device, precision):
            logits = model(corrupted, times, groups, attention)
    loss = masked_objective(logits, targets, masked, attention, times, model.special_tokens)
    if not torch.isfinite(loss):
        raise FloatingPointError("MDLM loss is not finite")
    return loss


@torch.inference_mode()
def _validate(context: TaskContext, model: GroupMDLM, shards: TokenShards, device, precision) -> dict[str, Any]:
    options = context.config["mdlm"]["validation"]
    generator = torch.Generator(device="cpu").manual_seed(options["seed"])
    count = shards.rows if options["max_rows"] is None else min(shards.rows, options["max_rows"])
    total, rows, masked_tokens = 0.0, 0, 0
    group_counts = dict.fromkeys(GROUPS, 0)
    was_training = model.training
    model.eval()
    try:
        with context.progress(total=count, desc="MDLM validation", unit="row", leave=False) as progress:
            for batch in shards.validation_batches(options["batch_size"], options["max_rows"], generator=generator):
                corrupted = _corrupted_batch(batch, model.special_tokens, context.config["mdlm"]["diffusion"], generator=generator)
                loss = _batch_loss(context, model, corrupted, device, precision)
                size = batch[0].shape[0]
                total += float(loss.item()) * size
                rows += size
                masked_tokens += int(corrupted[-1].sum())
                for index, group in enumerate(GROUPS):
                    group_counts[group] += int((batch[2] == index).sum())
                progress.update(size)
    finally:
        model.train(was_training)
    if rows == 0 or masked_tokens == 0:
        raise ValueError("Validation has no rows or sampled masks; use a larger validation budget")
    return {"loss": total / rows, "rows": rows, "masked_tokens": masked_tokens,
            "group_counts": group_counts, "seed": options["seed"], "split": "validation",
            "objective": context.config["mdlm"]["diffusion"]["objective"]}


def _export(context: TaskContext, model: GroupMDLM, step: int, validation: dict[str, Any] | None,
            actual: dict[str, Any]) -> str:
    directory = context.store.directory("checkpoints", "mdlm", context.identity.suite,
                                        f"seed_{context.identity.seed}", context.identity.run_id, f"step_{step:08d}")
    weights, specification = os.path.join(directory, "model.safetensors"), os.path.join(directory, "config.json")
    physical = context.store.path(weights)
    os.makedirs(os.path.dirname(physical), exist_ok=True)
    save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, physical + ".part")
    commit_file(physical + ".part", physical)
    context.store.write_json(specification, model.specification)
    inputs = os.path.join(directory, "inputs.json")
    context.store.write_json(inputs, {"dependency_hashes": dict(context.identity.dependency_hashes),
                                    "groups": list(GROUPS), "suite": context.identity.suite,
                                    "seed": context.identity.seed, "actual_runtime": actual})
    path = os.path.join(directory, "manifest.json")
    context.store.create_manifest(path, kind="mdlm_checkpoint", config_hash=context.identity.config_hash,
        files=[weights, specification, inputs], metadata={
            "version": TRAINER_VERSION, "step": step, "suite": context.identity.suite,
            "seed": context.identity.seed, "groups": list(GROUPS), "weights_path": weights,
            "specification_path": specification, "specification_hash": config_digest(model.specification),
            "tokenizer_fingerprint": model.specification["tokenizer_fingerprint"],
            "corpus_sha256": context.identity.dependency_hashes["corpus"],
            "dependency_hashes": dict(context.identity.dependency_hashes),
            "validation": validation, "actual_runtime": actual,
        })
    return path


def _checkpoint(context, model, optimizer, scheduler, state, actual, on_checkpoint):
    context.save_checkpoint(state, training_state={
        "version": TRAINER_VERSION, "specification_hash": config_digest(model.specification),
        "model": {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()},
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "rng": capture_rng_state(), "actual_runtime": actual,
    })
    if on_checkpoint is not None:
        on_checkpoint()


def train_mdlm(context: TaskContext, *, on_checkpoint: Callable[[], None] | None = None) -> TaskResult:
    config, store = context.config, context.store
    options, training = config["mdlm"], config["mdlm"]["training"]
    tokenizer = SmilesTokenizer.load(store, config["data"]["tokenizer"]["metadata_path"])
    train = TokenShards(store, "train", tokenizer, cache_size=training["cache_shards"])
    validation = TokenShards(store, "validation", tokenizer, cache_size=training["cache_shards"])
    specification = model_specification(config, tokenizer)
    device, precision = _device_precision(training)
    model = GroupMDLM(specification).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], betas=tuple(training["betas"]),
                                 eps=training["adam_epsilon"], weight_decay=training["weight_decay"])
    scheduler = WarmupCosine(optimizer, training)
    actual = {"device": device.type, "precision": str(precision or torch.float32),
              "torch_version": str(torch.__version__), "cuda_devices": torch.cuda.device_count() if device.type == "cuda" else 0,
              "parameters": sum(parameter.numel() for parameter in model.parameters()),
              "global_batch_size": training["global_batch_size"], "micro_batch_size": training["micro_batch_size"]}
    state = context.resume_state or {"version": TRAINER_VERSION, "step": 0, "best_loss": None,
                                     "best_manifest": None, "best_manifest_sha256": None,
                                     "last_manifest": None, "last_manifest_sha256": None, "last_validation": None}
    if state["version"] != TRAINER_VERSION:
        raise ValueError("Unsupported trainer resume state")
    saved = context.training_state
    if saved is not None:
        if (saved["version"] != TRAINER_VERSION or saved["specification_hash"] != config_digest(specification)
                or saved["actual_runtime"] != actual):
            raise ValueError("Resume architecture, precision, device, or training runtime differs")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        if scheduler.completed_steps != state["step"]:
            raise ValueError("Resume step differs from the optimizer schedule")
        # The framework restored RNG before model construction, which consumes
        # random draws. Restore again here, after all initialization and loading.
        restore_rng_state(saved["rng"])
        for name in ("best", "last"):
            if state[name + "_manifest"]:
                store.require_file(state[name + "_manifest"], checksum=state[name + "_manifest_sha256"])
                store.require_manifest(state[name + "_manifest"], kind="mdlm_checkpoint")
        context.training_state = None
        saved = None  # Release the large CPU resume payload after restoration.
    elif state["step"] != 0:
        raise ValueError("Resume is missing model/optimizer/scheduler/RNG tensors")
    else:
        _checkpoint(context, model, optimizer, scheduler, state, actual, on_checkpoint)
    context.log(f"MDLM: parameters={actual['parameters']}; {device.type}/{actual['precision']}; train={train.rows}; validation={validation.rows}")
    context.tracker.event("training_settings", specification_hash=config_digest(specification), actual=actual,
                          dependencies=dict(context.identity.dependency_hashes), test_used_for_selection=False)
    accumulation = training["global_batch_size"] // training["micro_batch_size"]
    model.train()
    with context.progress(range(state["step"] + 1, training["steps"] + 1), initial=state["step"],
                          total=training["steps"], desc="Train MDLM", unit="step") as progress:
        for step in progress:
            rate = scheduler.prepare_next()
            optimizer.zero_grad(set_to_none=True)
            total_loss, masked_tokens = 0.0, 0
            with context.tracker.measure("training_batch"):
                indices = torch.randint(train.rows, (training["global_batch_size"],)).tolist()
                global_batch = train.batch(indices)
            for offset in range(0, training["global_batch_size"], training["micro_batch_size"]):
                with context.tracker.measure("training_batch"):
                    batch = tuple(tensor[offset:offset + training["micro_batch_size"]] for tensor in global_batch)
                    corrupted = _corrupted_batch(batch, model.special_tokens, options["diffusion"])
                loss = _batch_loss(context, model, corrupted, device, precision)
                total_loss += float(loss.detach().item()) / accumulation
                masked_tokens += int(corrupted[-1].sum())
                with context.tracker.measure("backward", gpu=device.type == "cuda", device=device):
                    (loss / accumulation).backward()
            with context.tracker.measure("optimizer", gpu=device.type == "cuda", device=device):
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
            scheduler.completed_steps += 1
            state["step"] = step
            if step % training["log_every"] == 0 or step == training["steps"]:
                context.tracker.event("training_step", step=step, loss=total_loss, learning_rate=rate,
                                      gradient_norm=float(norm), masked_tokens=masked_tokens)
                progress.set_postfix(loss=f"{total_loss:.4f}", lr=f"{rate:.2g}")
                context.log(f"step={step}; loss={total_loss:.4f}; grad={float(norm):.3f}; lr={rate:.3g}")
            exported = None
            if step % training["validate_every"] == 0 or step == training["steps"]:
                result = _validate(context, model, validation, device, precision)
                state["last_validation"] = {"step": step, **result}
                context.tracker.event("validation", step=step, **result)
                context.log(f"validation step={step}; loss={result['loss']:.4f}; rows={result['rows']}")
                if state["best_loss"] is None or result["loss"] < state["best_loss"]:
                    exported = _export(context, model, step, state["last_validation"], actual)
                    state["best_loss"], state["best_manifest"] = result["loss"], exported
                    state["best_manifest_sha256"] = sha256_file(store.require_file(exported))
            if step % training["save_every"] == 0 or step == training["steps"]:
                same_step_validation = state["last_validation"] if state["last_validation"] and state["last_validation"]["step"] == step else None
                exported = exported or _export(context, model, step, same_step_validation, actual)
                state["last_manifest"] = exported
                state["last_manifest_sha256"] = sha256_file(store.require_file(exported))
                _checkpoint(context, model, optimizer, scheduler, state, actual, on_checkpoint)
                context.log(f"checkpoint step={step}; manifest={exported}")
    if not state["best_manifest"] or not state["last_manifest"]:
        raise ValueError("Training did not produce validation-selected and final checkpoints")
    report = context.output_path("training.json")
    store.write_json(report, {"version": TRAINER_VERSION, "suite": context.identity.suite,
                             "steps": state["step"], "best_validation_loss": state["best_loss"],
                             "best_manifest": state["best_manifest"], "last_manifest": state["last_manifest"],
                             "dependency_hashes": dict(context.identity.dependency_hashes),
                             "actual_runtime": actual, "test_used_for_selection": False,
                             "checkpoint_selection": "minimum fixed-seed validation masked ELBO",
                             "validation": state["last_validation"]})
    artifacts = tuple(dict.fromkeys([report, state["best_manifest"], state["last_manifest"]]))
    return TaskResult(f"steps={state['step']}; best validation={state['best_loss']:.4f}; report={report}", artifacts=artifacts)


def execute(argv: Sequence[str] | None = None, *, execution: str = "local",
            on_checkpoint: Callable[[], None] | None = None):
    return run_task("T05", lambda context: train_mdlm(context, on_checkpoint=on_checkpoint), argv=argv, execution=execution,
                    description="Train the shared group-conditioned ChEMBL MDLM.",
                    default_config="configs/mdlm.yaml", validator=validate_mdlm, dependencies=training_dependencies)


if __name__ == "__main__":
    try:
        record = execute()
        if record.status != "completed":
            raise SystemExit(1)
    except KeyboardInterrupt:
        tqdm.write("T05: interrupted")
        raise SystemExit(130) from None
    except Exception as exc:
        tqdm.write(f"T05: failed; {type(exc).__name__}: {exc}")
        raise SystemExit(1) from None
