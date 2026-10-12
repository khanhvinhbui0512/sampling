"""Pinned, offline Llama adapter for the CARS AR replication arm.

The prompt text is supplied by the benchmark loader. Chat formatting uses the
snapshot's template exactly once. No sampling filters or grammar masks are
applied here; the CARS sampler will own those operations.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.artifacts import ArtifactStore
from src.config import Config, config_digest, project_path
from src.schemas import require_identifier
from src.tracking import CostTracker


def model_source(options: Mapping[str, Any]) -> dict[str, Any]:
    """Content identity, independent of which other models T04 downloads."""
    return {key: value for key, value in options.items()
            if key not in {"dtype", "device_map", "temperature", "top_p", "top_k"}}


def validate_models(config: Config) -> None:
    models = config.get("models", {})
    selected = models.get("download")
    if (not isinstance(selected, list) or not selected or len(set(selected)) != len(selected)
            or any(name not in ("llama", "retrostar") for name in selected)):
        raise ValueError("models.download must select llama and/or retrostar once each")
    directories = []
    for name in ("llama", "retrostar"):
        options = models[name]
        directory = project_path(options["directory"])
        manifest = project_path(options["manifest"])
        if manifest != os.path.join(directory, "manifest.json"):
            raise ValueError(f"{name}: manifest must be directory/manifest.json")
        directories.append(directory)
        files = options["required_files"]
        if not isinstance(files, list) or not files or len(set(files)) != len(files):
            raise ValueError(f"{name}: required_files must be a nonempty unique list")
        for filename in files:
            require_identifier(filename, "model filename")
            if filename == "manifest.json":
                raise ValueError("A model manifest cannot include itself")
    if any(a == b or a.startswith(b + os.sep) or b.startswith(a + os.sep)
           for a, b in [(directories[0], directories[1])]):
        raise ValueError("Pretrained model directories must be separate")
    llama = models["llama"]
    if (llama["backend"] != "huggingface" or not re.fullmatch(r"[0-9a-f]{40}", llama["revision"])
            or llama["tokenizer_id"] != llama["model_id"]
            or llama["tokenizer_revision"] != llama["revision"]):
        raise ValueError("Pin Llama and its tokenizer to the same full commit")
    if (llama["temperature"], llama["top_p"], llama["top_k"]) != (1.0, 1.0, 0):
        raise ValueError("The replication adapter requires temperature=1, top_p=1, top_k=0")
    if llama["dtype"] != "bfloat16" or llama["chat_template"] != "tokenizer":
        raise ValueError("Llama replication requires BF16 and the pinned tokenizer chat template")
    retro = models["retrostar"]
    if (retro["backend"] != "figshare" or retro["tokenizer"] is not None
            or set(retro["files"]) != set(retro["required_files"])
            or not re.fullmatch(r"[0-9a-f]{40}", retro["source_revision"])):
        raise ValueError("RetroStar must pin its source revision and both classifier artifacts")
    if retro["revision"] != f"figshare-{retro['article_id']}-v{retro['article_version']}":
        raise ValueError("RetroStar revision differs from its article/version")
    for detail in retro["files"].values():
        if (type(detail["size"]) is not int or detail["size"] <= 0
                or not re.fullmatch(r"[0-9a-f]{32}", detail["md5"])
                or detail["url"] != f"https://ndownloader.figshare.com/files/{detail['file_id']}"):
            raise ValueError("RetroStar files need immutable IDs, sizes, and published MD5 hashes")
    if models["mdlm"]["initialization"] != "scratch" or models["mdlm"]["pretrained_model"] is not None:
        raise ValueError("The primary MDLM is trained from scratch")


def require_model_snapshot(store: ArtifactStore, options: Mapping[str, Any]):
    source = model_source(options)
    manifest = store.require_manifest(options["manifest"], kind="pretrained_model", metadata={
        "source": source, "source_hash": config_digest(source),
    })
    expected = {os.path.join(options["directory"], filename) for filename in options["required_files"]}
    if set(manifest.files) != expected:
        raise ValueError("Pretrained snapshot has a different file set")
    return manifest


class LlamaAdapter:
    """One forward per full-prefix batch, with explicit padding attention masks."""

    def __init__(self, store: ArtifactStore, options: Mapping[str, Any], *,
                 tracker: CostTracker | None = None, inner: bool = False) -> None:
        self.manifest = require_model_snapshot(store, options)
        self.options = dict(options)
        self.model_id = options["model_id"] + "@" + options["revision"]
        self.tracker, self.inner = tracker, inner
        directory = store.path(options["directory"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False,
        )
        if not self.tokenizer.chat_template:
            raise ValueError("Pinned Llama tokenizer has no chat template")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, device_map=options["device_map"],
        ).eval()
        self.device = self.model.get_input_embeddings().weight.device
        self.vocab_size = self.model.config.vocab_size
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        if not eos or any(type(token) is not int or not 0 <= token < self.vocab_size for token in eos):
            raise ValueError("Pinned generation config lacks valid EOS token IDs")
        self.eos_token_ids = tuple(eos)
        self.metadata = {
            "model_id": self.model_id, "revision": options["revision"],
            "tokenizer_revision": options["tokenizer_revision"], "dtype": str(self.model.dtype),
            "temperature": 1.0, "top_p": 1.0, "top_k": 0,
            "eos_token_ids": list(self.eos_token_ids),
        }

    def format_prompt(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("The benchmark prompt must be a nonempty string")
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
        )

    def encode_prompt(self, prompt: str) -> tuple[int, ...]:
        # Template text already contains the BOS token; do not add another one.
        text = self.format_prompt(prompt)
        return tuple(self.tokenizer.encode(text, add_special_tokens=False))

    @torch.inference_mode()
    def next_token_logits(self, prefixes: Sequence[Sequence[int]]) -> torch.Tensor:
        if not prefixes or any(not prefix for prefix in prefixes):
            raise ValueError("AR prefixes must be nonempty")
        width = max(map(len, prefixes))
        if width > self.model.config.max_position_embeddings:
            raise ValueError("AR prefix exceeds model context; truncation is forbidden")
        ids = torch.full((len(prefixes), width), self.tokenizer.pad_token_id,
                         dtype=torch.long, device=self.device)
        attention = torch.zeros_like(ids, dtype=torch.bool)
        for row, prefix in enumerate(prefixes):
            if any(type(token) is not int or not 0 <= token < self.vocab_size for token in prefix):
                raise ValueError("AR prefix contains an out-of-vocabulary token")
            ids[row, -len(prefix):] = torch.tensor(prefix, device=self.device)
            attention[row, -len(prefix):] = True
        positions = (attention.long().cumsum(-1) - 1).clamp_min(0)
        timing = nullcontext() if self.tracker is None else self.tracker.model_call(
            "ar", batch_size=len(prefixes), inner=self.inner,
            gpu=self.device.type == "cuda", device=self.device,
        )
        with timing:
            # Request only the final token logits to avoid B*L*V memory use.
            output = self.model(input_ids=ids, attention_mask=attention,
                                position_ids=positions, use_cache=False, logits_to_keep=1)
        logits = output.logits[:, -1].float()
        if not torch.isfinite(logits).all():
            raise ValueError("Llama produced nonfinite logits")
        return logits

    def next_token_probabilities(self, prefixes: Sequence[Sequence[int]]) -> torch.Tensor:
        return self.next_token_logits(prefixes).softmax(-1)
