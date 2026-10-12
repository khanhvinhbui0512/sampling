"""CARS' one-step RetroStar proxy, using only a checked T04 snapshot."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Sequence
from typing import Any

from src.artifacts import ArtifactStore, sha256_file
from src.autoregressive import require_model_snapshot


class RetroStarProxy:
    def __init__(self, predict: Callable[[list[str]], Sequence[float | None]] | None,
                 *, reason: str | None = None, provenance: dict | None = None,
                 threshold: float = 0.4, batch_size: int = 32) -> None:
        self.predict, self.reason = predict, reason
        self.provenance = provenance or {}
        if threshold != 0.4 or type(batch_size) is not int or batch_size <= 0:
            raise ValueError("The CARS proxy requires probability > 0.4 and a positive batch size")
        self.threshold, self.batch_size = threshold, batch_size
        self.cache: dict[str, float | None] = {}

    def score(self, canonical: Sequence[str | None]) -> dict[str, Any]:
        total, valid = len(canonical), [text for text in canonical if text is not None]
        result = {"status": "unavailable", "value": None, "numerator": None,
                  "denominator": total, "valid_denominator": len(valid),
                  "valid_only_value": None, "threshold": self.threshold,
                  "interpretation": "One-step reaction proxy; no multi-step route claim",
                  "provenance": self.provenance}
        if self.predict is None:
            return {**result, "reason": self.reason or "reaction_model_unavailable"}
        if total == 0:
            return {**result, "reason": "empty_sample_set"}
        missing = list(dict.fromkeys(text for text in valid if text not in self.cache))
        try:
            for start in range(0, len(missing), self.batch_size):
                batch = missing[start:start+self.batch_size]
                probabilities = list(self.predict(batch))
                if len(probabilities) != len(batch):
                    raise ValueError("Reaction model output count differs from its input")
                for text, probability in zip(batch, probabilities, strict=True):
                    if probability is not None and (isinstance(probability, bool)
                            or not math.isfinite(probability) or not 0 <= probability <= 1):
                        raise ValueError("Invalid top-1 reaction probability")
                    self.cache[text] = probability
        except Exception as exc:
            return {**result, "reason": f"reaction_inference_failed: {type(exc).__name__}: {exc}"}
        successes = sum(self.cache[text] is not None and self.cache[text] > self.threshold for text in valid)
        return {**result, "status": "available", "value": successes/total,
                "numerator": successes, "valid_only_value": successes/len(valid) if valid else None,
                "no_reaction_outputs": sum(self.cache[text] is None for text in valid)}


def load_retrostar(store: ArtifactStore, model: dict, options: dict) -> RetroStarProxy:
    if not options["enabled"]:
        return RetroStarProxy(None, reason="disabled_by_config")
    # Missing artifacts are unavailable; a present but corrupt manifest must fail validation.
    if not os.path.isfile(store.path(model["manifest"])):
        return RetroStarProxy(None, reason="missing_T04_manifest")
    try:
        manifest = require_model_snapshot(store, model)
    except FileNotFoundError as exc:
        return RetroStarProxy(None, reason=f"missing_T04_model_file: {exc}")
    provenance = {"manifest": model["manifest"], "manifest_sha256": sha256_file(store.require_file(model["manifest"])),
                  "source": manifest.metadata["source"], "expansion_topk": options["expansion_topk"]}
    try:
        import torch
        from syntheseus import Molecule
        from syntheseus_retro_star_benchmark import RetroStarReactionModel

        device = 0 if options["device"] == "auto" and torch.cuda.is_available() else -1
        if options["device"] == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            device = 0
        # Explicit paths suppress upstream's implicit model download behavior.
        reaction_model = RetroStarReactionModel(
            model_checkpoint=store.require_file(os.path.join(model["directory"], "pretrained-reaction-model.ckpt")),
            template_file=store.require_file(os.path.join(model["directory"], "template_rules_1.dat")),
            expansion_topk=options["expansion_topk"], device=device, shuffle=False,
        )

        def predict(texts: list[str]) -> list[float | None]:
            with torch.inference_mode():
                reactions = reaction_model._get_reactions([Molecule(text) for text in texts], num_results=1)
            return [float(rows[0].metadata["probability"]) if rows else None for rows in reactions]

        return RetroStarProxy(predict, provenance={**provenance, "device": device}, batch_size=options["batch_size"])
    except (ImportError, OSError, RuntimeError) as exc:
        return RetroStarProxy(None, reason=f"reaction_model_load_failed: {type(exc).__name__}: {exc}", provenance=provenance)
