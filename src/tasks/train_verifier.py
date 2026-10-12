"""T07: fit binary rollout returns, selecting and calibrating on locked validation.

Every snapshot is one regression row, but partition membership is determined by
the terminal identity database from T06. No test data or future test states are
loaded. Model, optimizer, epoch cursor and RNG are saved at update boundaries.
"""

from __future__ import annotations

import argparse
import bisect
import math
import sqlite3
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence

import torch
from tqdm.auto import tqdm

from src.artifacts import ArtifactStore, Dependency, capture_rng_state, restore_rng_state, sha256_file
from src.config import Config, config_digest
from src.schemas import GROUPS, ArtifactManifest, TaskResult, require_count
from src.tasks.build_rollouts import ROLLOUT_VERSION
from src.tasks.common import TaskContext, run_task
from src.tasks.prepare_corpus import benchmark_dependency, checker_versions
from src.tasks.train_mdlm import _device_precision, _precision_context
from src.tokenizer import SmilesTokenizer
from src.verifier import StateVerifier, data_overrides, save_verifier, validate_verifier, verifier_loss, verifier_specification

TRAINER_VERSION = "smiles-verifier-trainer-v1"


def training_dependencies(config: Config, store: ArtifactStore) -> dict[str, Dependency]:
    path = config["verifier"]["data"]["manifest"]
    tokenizer_path = config["data"]["tokenizer"]["metadata_path"]
    tokenizer = SmilesTokenizer.load(store, tokenizer_path)
    token_metadata = store.read_json(tokenizer_path)
    dataset = store.require_manifest(path, kind="rollout_dataset", metadata={
        "version": ROLLOUT_VERSION, "suite": config["suite"], "groups": list(GROUPS),
        "reward_version": config["reward"]["version"], "reward_mode": config["reward"]["mode"],
        "tokenizer_fingerprint": tokenizer.fingerprint, "vocab_size": tokenizer.vocab_size,
        "max_length": tokenizer.max_length, "special_tokens": tokenizer.special_tokens.to_dict(),
    })
    source = dataset.metadata
    hashes = source["dependency_hashes"]
    benchmark = benchmark_dependency(config, store)
    policy_name = source["policy"]["policy"]
    expected_target = {"absorbing_reverse-v1": "reference_rollout_return", "left_to_right-v1": "q_z_rollout_return"}
    if (source["checker_versions"] != checker_versions() or source["ideal_value_claim"] is not False
            or config_digest(source["policy"]) != source["policy_hash"]
            or source["target_kind"] != expected_target.get(policy_name)
            or sha256_file(store.require_file(benchmark.path)) != hashes["benchmark"]
            or token_metadata["vocabulary_path"] != config["data"]["tokenizer"]["vocabulary_path"]):
        raise ValueError("Rollout target, reward/checker, policy or tokenizer provenance differs")
    checkpoint = store.require_manifest(source["source_model_manifest"], kind="mdlm_checkpoint", metadata={
        "suite": config["suite"], "groups": list(GROUPS), "tokenizer_fingerprint": tokenizer.fingerprint,
    })
    if (checkpoint.metadata["dependency_hashes"]["tokenizer"] != hashes["tokenizer"]
            or checkpoint.metadata["dependency_hashes"]["vocabulary"] != hashes["vocabulary"]
            or source["source_mdlm_sha256"] != hashes["mdlm"]):
        raise ValueError("Rollout source MDLM does not use this tokenizer/vocabulary")
    partition = source["partition"]
    if (partition["path"] not in dataset.files or partition["algorithm"] != "seeded_identity_rank"
            or set(source["splits"]) != {"train", "validation"}):
        raise ValueError("T07 needs the locked T06 identity partitions")
    return {
        "rollouts": Dependency(path, manifest=True, kind="rollout_dataset"),
        "mdlm": Dependency(source["source_model_manifest"], checksum=source["source_mdlm_sha256"],
                           manifest=True, kind="mdlm_checkpoint"),
        "tokenizer": Dependency(tokenizer_path, checksum=hashes["tokenizer"]),
        "vocabulary": Dependency(token_metadata["vocabulary_path"], checksum=hashes["vocabulary"]),
        "benchmark": benchmark,
    }


class SnapshotShards:
    """Bounded row access with shard schema and locked identity checks."""

    def __init__(self, store: ArtifactStore, dataset: ArtifactManifest, split: str, tokenizer: SmilesTokenizer,
                 *, cache_size: int) -> None:
        if split not in ("train", "validation"):
            raise ValueError("Verifier may load only train/validation")
        self.store, self.dataset, self.split, self.tokenizer = store, dataset, split, tokenizer
        self.details = dataset.metadata["splits"][split]
        self.shards, self.ends = self.details["shards"], []
        self.k = dataset.metadata["snapshots_per_rollout"]
        require_count(self.k, "rollout snapshots_per_rollout", positive=True)
        require_count(cache_size, "snapshot cache_size", positive=True)
        total = 0
        for shard in self.shards:
            if (dataset.files.get(shard["path"]) != shard["sha256"] or type(shard["rows"]) is not int
                    or shard["rows"] <= 0 or shard["rows"] % self.k):
                raise ValueError("Snapshot shards must contain complete rollouts covered by the manifest")
            total += shard["rows"]
            self.ends.append(total)
        if (not total or total != self.details["rows"] or total != self.details["rollouts"] * self.k
                or not 0 <= self.details["positive_rows"] <= total
                or sum(self.details["group_rows"].values()) != total):
            raise ValueError(f"Empty or inconsistent {split} partition; generate more T06 identities")
        self.rows, self.cache_size = total, cache_size
        self.cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        partition = dataset.metadata["partition"]["path"]
        # The published database is immutable. Hash it once for this reader,
        # rather than scanning the entire identity index for every shard load.
        self.partition_file = store.require_file(partition, checksum=dataset.files[partition])

    def load_shard(self, index: int) -> dict[str, Any]:
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        detail, source = self.shards[index], self.dataset.metadata
        payload = torch.load(self.store.require_file(detail["path"], checksum=detail["sha256"]),
                             map_location="cpu", weights_only=True)
        expected = {"version": ROLLOUT_VERSION, "split": self.split, "groups": list(GROUPS),
                    "special_tokens": self.tokenizer.special_tokens.to_dict(), "tokenizer_fingerprint": self.tokenizer.fingerprint,
                    **{key: source[key] for key in ("source_mdlm_sha256", "policy_hash", "reward_version")}}
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError("Snapshot shard provenance differs from the T06 manifest")
        n, width = detail["rows"], self.tokenizer.max_length
        dtypes = {"state_ids": torch.long, "final_ids": torch.long, "mask": torch.bool, "locked": torch.bool,
                  "group_ids": torch.long, "rewards": torch.float32, "snapshot_indices": torch.int32,
                  "snapshot_steps": torch.int32, "snapshot_times": torch.float32}
        for name, dtype in dtypes.items():
            tensor = payload[name]
            shape = (n, width) if name in ("state_ids", "final_ids", "mask", "locked") else (n,)
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != dtype or tensor.shape != shape:
                raise ValueError(f"Invalid snapshot tensor {name}")
        ids, final, mask, groups, rewards = [payload[key] for key in ("state_ids", "final_ids", "mask", "group_ids", "rewards")]
        if (torch.any(ids < 0) or torch.any(ids >= self.tokenizer.vocab_size) or torch.any(final < 0)
                or torch.any(final >= self.tokenizer.vocab_size) or torch.any(mask != (ids == self.tokenizer.special_tokens.mask_id))
                or torch.any(payload["locked"] & mask) or torch.any(ids[~mask] != final[~mask])
                or torch.any(groups < 0) or torch.any(groups >= len(GROUPS)) or not torch.isfinite(rewards).all()
                or torch.any((rewards != 0) & (rewards != 1)) or not torch.isfinite(payload["snapshot_times"]).all()
                or torch.any(payload["snapshot_indices"] != torch.arange(self.k).repeat(n // self.k))):
            raise ValueError("Snapshot tokens, known coordinates, labels or indices are inconsistent")
        for name in ("rollout_ids", "identity_keys"):
            values = payload[name]
            if (not isinstance(values, list) or len(values) != n or any(not isinstance(value, str) or not value for value in values)
                    or any(len(set(values[start:start + self.k])) != 1 for start in range(0, n, self.k))):
                raise ValueError("Every rollout must keep all its snapshots and one terminal identity")
        for tensor in (final, groups, rewards):
            grouped = tensor.reshape(n // self.k, self.k, *tensor.shape[1:])
            if torch.any(grouped != grouped[:, :1]):
                raise ValueError("Snapshots from a rollout disagree about its final state/group/reward")
        connection = sqlite3.connect(self.partition_file)
        try:
            connection.execute("PRAGMA query_only=ON")
            keys = sorted(set(payload["identity_keys"]))
            for start in range(0, len(keys), 900):
                batch = keys[start:start + 900]
                rows = connection.execute("SELECT identity_key, partition FROM identities WHERE identity_key IN ("
                                          + ",".join("?" for _ in batch) + ")", batch).fetchall()
                if len(rows) != len(batch) or any(part != self.split for _, part in rows):
                    raise ValueError("Snapshot identity violates its locked train/validation partition")
        finally:
            connection.close()
        self.cache[index] = payload
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return payload

    def batch(self, indices: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        locations: dict[int, list[tuple[int, int]]] = {}
        for destination, row in enumerate(indices):
            if not 0 <= row < self.rows:
                raise IndexError("Snapshot row out of range")
            shard = bisect.bisect_right(self.ends, row)
            locations.setdefault(shard, []).append((destination, row - (self.ends[shard - 1] if shard else 0)))
        ids = torch.empty((len(indices), self.tokenizer.max_length), dtype=torch.long)
        groups, rewards = torch.empty(len(indices), dtype=torch.long), torch.empty(len(indices), dtype=torch.float32)
        for shard, mapping in locations.items():
            payload = self.load_shard(shard)
            destinations, rows = zip(*mapping, strict=True)
            ids[list(destinations)] = payload["state_ids"][list(rows)]
            groups[list(destinations)] = payload["group_ids"][list(rows)]
            rewards[list(destinations)] = payload["rewards"][list(rows)]
        return ids, groups, rewards

    def validation_batches(self, size: int) -> Iterator[tuple]:
        for start in range(0, self.rows, size):
            yield self.batch(range(start, min(start + size, self.rows)))


class EpochSampler:
    """Shuffle shards and then rows; visit every row once with resumable cursors."""

    def __init__(self, shards: SnapshotShards) -> None:
        self.sizes = [detail["rows"] for detail in shards.shards]
        self.offsets = [0, *shards.ends[:-1]]
        self.shard_order: list[int] = []
        self.row_order = torch.empty(0, dtype=torch.long)
        self.shard_cursor = self.row_cursor = 0

    def next_indices(self, count: int) -> list[int]:
        if not self.shard_order:
            self.shard_order = torch.randperm(len(self.sizes)).tolist()
        chosen: list[int] = []
        while len(chosen) < count and self.shard_cursor < len(self.shard_order):
            shard = self.shard_order[self.shard_cursor]
            if self.row_order.numel() == 0:
                self.row_order = torch.randperm(self.sizes[shard])
            size = min(count - len(chosen), self.row_order.numel() - self.row_cursor)
            chosen.extend((self.row_order[self.row_cursor:self.row_cursor + size] + self.offsets[shard]).tolist())
            self.row_cursor += size
            if self.row_cursor == self.row_order.numel():
                self.shard_cursor += 1
                self.row_cursor, self.row_order = 0, torch.empty(0, dtype=torch.long)
        return chosen

    def reset(self) -> None:
        self.shard_order, self.shard_cursor, self.row_cursor = [], 0, 0
        self.row_order = torch.empty(0, dtype=torch.long)

    def state_dict(self) -> dict:
        return {"sizes": self.sizes, "shard_order": self.shard_order, "shard_cursor": self.shard_cursor,
                "row_order": self.row_order, "row_cursor": self.row_cursor}

    def load_state_dict(self, saved: dict, expected_rows: int) -> None:
        order, row_order = saved["shard_order"], saved["row_order"]
        cursor, row_cursor = saved["shard_cursor"], saved["row_cursor"]
        if (saved["sizes"] != self.sizes or (order and sorted(order) != list(range(len(self.sizes))))
                or not 0 <= cursor <= len(order) or row_order.dtype != torch.long or row_order.ndim != 1
                or not 0 <= row_cursor <= row_order.numel()):
            raise ValueError("Resume epoch sampler has invalid shard/row cursors")
        if row_order.numel() and (cursor == len(order) or row_order.numel() != self.sizes[order[cursor]]
                                 or not torch.equal(row_order.sort().values, torch.arange(row_order.numel()))):
            raise ValueError("Resume sampler row order is not a complete shard permutation")
        if row_order.numel() and row_cursor == row_order.numel():
            raise ValueError("A consumed shard must advance the resume shard cursor")
        consumed = sum(self.sizes[shard] for shard in order[:cursor]) + row_cursor
        if consumed != expected_rows:
            raise ValueError("Resume sampler differs from the epoch row counter")
        self.shard_order, self.row_order, self.shard_cursor, self.row_cursor = order, row_order, cursor, row_cursor


class CalibrationMeter:
    """Snapshot-weighted binary regression/reliability statistics, without fitting."""

    def __init__(self, bins: int) -> None:
        self.bins, self.rows = bins, 0
        self.prediction = self.reward = self.squared_error = self.absolute_error = 0.0
        self.counts = torch.zeros(bins, dtype=torch.long)
        self.predicted = torch.zeros(bins, dtype=torch.float64)
        self.observed = torch.zeros(bins, dtype=torch.float64)

    def update(self, predicted: torch.Tensor, rewards: torch.Tensor) -> None:
        predicted, rewards = predicted.detach().cpu().double(), rewards.detach().cpu().double()
        if predicted.numel() == 0:
            return
        verifier_loss(predicted, rewards)  # Validate sigmoid/binary inputs.
        self.rows += predicted.numel()
        self.prediction += float(predicted.sum())
        self.reward += float(rewards.sum())
        self.squared_error += float((predicted - rewards).square().sum())
        self.absolute_error += float((predicted - rewards).abs().sum())
        indices = (predicted * self.bins).long().clamp(max=self.bins - 1)
        self.counts.scatter_add_(0, indices, torch.ones_like(indices))
        self.predicted.scatter_add_(0, indices, predicted)
        self.observed.scatter_add_(0, indices, rewards)

    def report(self) -> dict:
        reliability, error = [], 0.0
        for index in range(self.bins):
            count = int(self.counts[index])
            predicted = float(self.predicted[index]) / count if count else None
            observed = float(self.observed[index]) / count if count else None
            if count:
                error += count * abs(predicted - observed)
            reliability.append({"lower": index / self.bins, "upper": (index + 1) / self.bins, "rows": count,
                                "mean_prediction": predicted, "success_rate": observed})
        return {"rows": self.rows, "positive_rows": int(self.reward),
                "mse": self.squared_error / self.rows if self.rows else None,
                "brier": self.squared_error / self.rows if self.rows else None,
                "mae": self.absolute_error / self.rows if self.rows else None,
                "ece": error / self.rows if self.rows else None,
                "mean_prediction": self.prediction / self.rows if self.rows else None,
                "success_rate": self.reward / self.rows if self.rows else None, "reliability_bins": reliability}


def _forward(context: TaskContext, model: StateVerifier, ids, groups, device, precision):
    with context.tracker.model_call("verifier", batch_size=ids.shape[0], gpu=device.type == "cuda", device=device):
        with _precision_context(device, precision):
            return model(ids.to(device), groups.to(device))


@torch.inference_mode()
def _validate(context: TaskContext, model: StateVerifier, shards: SnapshotShards, device, precision) -> dict:
    options = context.config["verifier"]["validation"]
    bins, edges = options["calibration_bins"], options["mask_fraction_edges"]
    overall = CalibrationMeter(bins)
    groups = {group: CalibrationMeter(bins) for group in GROUPS}
    depths = [CalibrationMeter(bins) for _ in range(len(edges) - 1)]
    was_training = model.training
    model.eval()
    try:
        with context.progress(total=shards.rows, desc="Verifier validation", unit="snapshot", leave=False) as progress:
            for ids, group_ids, rewards in shards.validation_batches(options["batch_size"]):
                predictions = _forward(context, model, ids, group_ids, device, precision).cpu()
                overall.update(predictions, rewards)
                for index, group in enumerate(GROUPS):
                    selected = group_ids == index
                    groups[group].update(predictions[selected], rewards[selected])
                fractions = (ids == model.special_tokens.mask_id).double().mean(-1)
                for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
                    selected = (fractions >= lower) & ((fractions <= upper) if index == len(depths) - 1 else (fractions < upper))
                    depths[index].update(predictions[selected], rewards[selected])
                progress.update(ids.shape[0])
    finally:
        model.train(was_training)
    result = overall.report()
    if (result["rows"] != shards.rows or result["positive_rows"] != shards.details["positive_rows"]
            or any(groups[group].rows != shards.details["group_rows"][group] for group in GROUPS)):
        raise ValueError("Validation counts disagree with the locked rollout manifest")
    return {"split": "validation", "unit": "snapshot", "overall": result,
            "groups": {group: meter.report() for group, meter in groups.items()},
            "mask_fraction": [{"lower": edges[index], "upper": edges[index + 1],
                               "upper_inclusive": index == len(depths) - 1, **meter.report()}
                              for index, meter in enumerate(depths)],
            "calibration_fitted": False, "test_used": False}


def _checkpoint(context, model, optimizer, sampler, state, actual, on_checkpoint):
    context.save_checkpoint(state, training_state={
        "version": TRAINER_VERSION, "specification_hash": config_digest(model.specification),
        "model": {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()},
        "optimizer": optimizer.state_dict(), "sampler": sampler.state_dict(),
        "rng": capture_rng_state(), "actual_runtime": actual,
        "scheduler": {"kind": "constant", "completed_updates": state["step"]},
    })
    if on_checkpoint is not None:
        on_checkpoint()


def _export(context, model, state, actual) -> str:
    directory = context.store.directory("checkpoints", "verifier", context.identity.suite,
                                        f"seed_{context.identity.seed}", context.identity.run_id, f"step_{state['step']:08d}")
    return save_verifier(context.store, directory, model, config_hash=context.identity.config_hash,
                         metadata={"version": TRAINER_VERSION, "step": state["step"], "epoch": state["completed_epochs"],
                                   "suite": context.identity.suite, "seed": context.identity.seed,
                                   "rollout_manifest": context.config["verifier"]["data"]["manifest"],
                                   "dependency_hashes": dict(context.identity.dependency_hashes),
                                   "actual_runtime": actual, "checkpoint_selection": "minimum validation MSE",
                                   "test_used_for_selection": False}, calibration=state["last_validation"])


def train_verifier(context: TaskContext, *, on_checkpoint: Callable[[], None] | None = None) -> TaskResult:
    config, store = context.config, context.store
    options, training = config["verifier"], config["verifier"]["training"]
    dataset = store.require_manifest(options["data"]["manifest"], kind="rollout_dataset")
    tokenizer = SmilesTokenizer.load(store, config["data"]["tokenizer"]["metadata_path"])
    train = SnapshotShards(store, dataset, "train", tokenizer, cache_size=training["cache_shards"])
    validation = SnapshotShards(store, dataset, "validation", tokenizer, cache_size=training["cache_shards"])
    specification = verifier_specification(config, tokenizer, dataset)
    device, precision = _device_precision(training)
    model = StateVerifier(specification).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], betas=tuple(training["betas"]),
                                 eps=training["adam_epsilon"], weight_decay=training["weight_decay"])
    sampler = EpochSampler(train)
    actual = {"device": device.type, "precision": str(precision or torch.float32), "torch_version": str(torch.__version__),
              "cuda_devices": torch.cuda.device_count() if device.type == "cuda" else 0,
              "parameters": sum(parameter.numel() for parameter in model.parameters()),
              "global_batch_size": training["global_batch_size"], "micro_batch_size": training["micro_batch_size"]}
    state = context.resume_state or {"version": TRAINER_VERSION, "stage": "training", "step": 0, "completed_epochs": 0,
                                    "epoch_rows": 0, "epoch_loss_sum": 0.0, "epoch_positive_rows": 0,
                                    "best_loss": None, "best_manifest": None, "best_manifest_sha256": None,
                                    "last_manifest": None, "last_manifest_sha256": None,
                                    "last_validation": None, "bad_epochs": 0}
    if (state["version"] != TRAINER_VERSION or state["stage"] not in ("training", "complete")
            or not 0 <= state["completed_epochs"] <= training["epochs"] or not 0 <= state["epoch_rows"] <= train.rows):
        raise ValueError("Unsupported or inconsistent verifier resume state")
    saved = context.training_state
    if saved is not None:
        if (saved["version"] != TRAINER_VERSION or saved["specification_hash"] != config_digest(specification)
                or saved["actual_runtime"] != actual
                or saved["scheduler"] != {"kind": "constant", "completed_updates": state["step"]}):
            raise ValueError("Resume verifier architecture, optimizer schedule or runtime differs")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        sampler.load_state_dict(saved["sampler"], state["epoch_rows"])
        # Restore after model initialization, which consumes random draws.
        restore_rng_state(saved["rng"])
        for name in ("best", "last"):
            if state[name + "_manifest"]:
                store.require_file(state[name + "_manifest"], checksum=state[name + "_manifest_sha256"])
                store.require_manifest(state[name + "_manifest"], kind="verifier_checkpoint")
        context.training_state, saved = None, None
    elif state["step"] or state["completed_epochs"] or state["epoch_rows"]:
        raise ValueError("Resume is missing verifier model/optimizer/sampler/RNG tensors")
    else:
        _checkpoint(context, model, optimizer, sampler, state, actual, on_checkpoint)
    context.log(f"Verifier: parameters={actual['parameters']}; {device.type}/{actual['precision']}; train={train.rows}; validation={validation.rows}")
    for split, shards in (("train", train), ("validation", validation)):
        if shards.details["positive_rows"] in (0, shards.rows):
            context.log(f"{split}: only one reward class is present; calibration and usefulness require reviewing rollout diversity", level="WARNING")
    context.tracker.event("verifier_settings", target_kind=specification["target_kind"], actual=actual,
                          dependencies=dict(context.identity.dependency_hashes), test_used_for_selection=False)
    model.train()
    with context.progress(total=training["epochs"], initial=state["completed_epochs"], desc="Train verifier", unit="epoch") as epochs:
        while state["stage"] == "training" and state["completed_epochs"] < training["epochs"]:
            epoch = state["completed_epochs"] + 1
            with context.progress(total=train.rows, initial=state["epoch_rows"], desc=f"Verifier epoch {epoch}", unit="snapshot", leave=False) as progress:
                while indices := sampler.next_indices(training["global_batch_size"]):
                    optimizer.zero_grad(set_to_none=True)
                    with context.tracker.measure("verifier_training_batch"):
                        ids, groups, rewards = train.batch(indices)
                    batch_loss, size = 0.0, len(indices)
                    for offset in range(0, size, training["micro_batch_size"]):
                        end = min(offset + training["micro_batch_size"], size)
                        predictions = _forward(context, model, ids[offset:end], groups[offset:end], device, precision)
                        loss = verifier_loss(predictions, rewards[offset:end].to(device))
                        weight = (end - offset) / size
                        batch_loss += float(loss.detach()) * weight
                        with context.tracker.measure("verifier_backward", gpu=device.type == "cuda", device=device):
                            (loss * weight).backward()
                    with context.tracker.measure("verifier_optimizer", gpu=device.type == "cuda", device=device):
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip"], error_if_nonfinite=True)
                        optimizer.step()
                    state["step"] += 1
                    state["epoch_rows"] += size
                    state["epoch_loss_sum"] += batch_loss * size
                    state["epoch_positive_rows"] += int(rewards.sum())
                    if state["step"] % training["log_every_updates"] == 0 or state["epoch_rows"] == train.rows:
                        context.tracker.event("verifier_training_step", step=state["step"], epoch=epoch,
                                              loss=batch_loss, gradient_norm=float(norm), learning_rate=training["learning_rate"])
                        context.log(f"epoch={epoch}; update={state['step']}; MSE={batch_loss:.4f}; grad={float(norm):.3f}")
                        progress.set_postfix(mse=f"{batch_loss:.4f}")
                    if state["step"] % training["save_every_updates"] == 0:
                        _checkpoint(context, model, optimizer, sampler, state, actual, on_checkpoint)
                    progress.update(size)
            if state["epoch_rows"] != train.rows or state["epoch_positive_rows"] != train.details["positive_rows"]:
                raise ValueError("Epoch did not visit every train snapshot exactly once with the declared labels")
            calibrated = _validate(context, model, validation, device, precision)
            score = calibrated["overall"]["mse"]
            if score is None or not math.isfinite(score):
                raise FloatingPointError("Verifier validation MSE is not finite")
            state["completed_epochs"] = epoch
            state["last_validation"] = {"epoch": epoch, "step": state["step"], **calibrated}
            context.tracker.event("verifier_validation", epoch=epoch, step=state["step"], mse=score,
                                  ece=calibrated["overall"]["ece"], train_mse=state["epoch_loss_sum"] / train.rows)
            exported = _export(context, model, state, actual)
            digest = sha256_file(store.require_file(exported))
            state["last_manifest"], state["last_manifest_sha256"] = exported, digest
            if state["best_loss"] is None or score < state["best_loss"]:
                state["best_loss"], state["best_manifest"], state["best_manifest_sha256"] = score, exported, digest
                state["bad_epochs"] = 0
            else:
                state["bad_epochs"] += 1
            sampler.reset()
            state["epoch_rows"], state["epoch_loss_sum"], state["epoch_positive_rows"] = 0, 0.0, 0
            patience = training["early_stopping_patience"]
            if epoch == training["epochs"] or (patience is not None and state["bad_epochs"] >= patience):
                state["stage"] = "complete"
            _checkpoint(context, model, optimizer, sampler, state, actual, on_checkpoint)
            epochs.update(1)
            epochs.set_postfix(mse=f"{score:.4f}", best=f"{state['best_loss']:.4f}")
            context.log(f"validation epoch={epoch}; MSE={score:.4f}; ECE={calibrated['overall']['ece']:.4f}; checkpoint={exported}")
    if state["stage"] != "complete" or not state["best_manifest"] or not state["last_manifest"]:
        raise ValueError("Verifier training did not produce a validation-selected checkpoint")
    report = context.output_path("training.json")
    best = store.require_manifest(state["best_manifest"], kind="verifier_checkpoint")
    store.write_json(report, {"version": TRAINER_VERSION, "suite": context.identity.suite, "seed": context.identity.seed,
                             "epochs": state["completed_epochs"], "updates": state["step"],
                             "best_validation_mse": state["best_loss"], "best_manifest": state["best_manifest"],
                             "last_manifest": state["last_manifest"], "best_calibration": best.metadata["calibration_path"],
                             "last_validation": state["last_validation"], "actual_runtime": actual,
                             "target_kind": specification["target_kind"], "target_definition": specification["target_definition"],
                             "dependency_hashes": dict(context.identity.dependency_hashes), "test_used_for_selection": False,
                             "checkpoint_selection": "minimum locked validation MSE", "scheduler": "constant",
                             "partition": dataset.metadata["partition"], "splits": dataset.metadata["splits"]})
    artifacts = tuple(dict.fromkeys((report, state["best_manifest"], state["last_manifest"])))
    return TaskResult(f"epochs={state['completed_epochs']}; best validation MSE={state['best_loss']:.4f}; report={report}", artifacts=artifacts)


def _arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rollouts", help="Project-relative T06 rollout dataset manifest.")


def execute(argv: Sequence[str] | None = None, *, execution: str = "local", on_checkpoint: Callable[[], None] | None = None):
    return run_task("T07", lambda context: train_verifier(context, on_checkpoint=on_checkpoint), argv=argv, execution=execution,
                    description="Train and calibrate the grouped SMILES process verifier.", default_config="configs/verifier.yaml",
                    validator=validate_verifier, dependencies=training_dependencies, add_arguments=_arguments,
                    argument_overrides=lambda arguments: data_overrides(arguments.rollouts or ""))


if __name__ == "__main__":
    try:
        record = execute()
        if record.status != "completed":
            raise SystemExit(1)
    except KeyboardInterrupt:
        tqdm.write("T07: interrupted")
        raise SystemExit(130) from None
    except Exception as exc:
        tqdm.write(f"T07: failed; {type(exc).__name__}: {exc}")
        raise SystemExit(1) from None
