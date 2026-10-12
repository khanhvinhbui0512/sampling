"""Compose the shared MDM-VGB kernel with a CARS value provider."""

from src.mdm_vgb import MDMVGBSampler
from src.values import CARSValueProvider


def build_hybrid(model, inner_model, tokenizer, checker, search_options, value_options, *, seed=0, tracker=None):
    if model.model_id != inner_model.model_id or model.special_tokens != inner_model.special_tokens:
        raise ValueError("Hybrid outer and inner MDLM must share the same checkpoint and precision")
    if tracker is not None and (getattr(inner_model, "tracker", None) is not tracker or not getattr(inner_model, "inner", False)):
        raise ValueError("Hybrid inner forwards must use the shared tracker and inner accounting")
    provider = CARSValueProvider.from_mdlm(inner_model, tokenizer, checker, m=value_options["m"],
        max_nodes=value_options["trie_max_nodes"], cache_capacity=value_options["cache_capacity"], seed=seed, tracker=tracker)
    return MDMVGBSampler(model, provider, tokenizer, checker, search_options, seed=seed, tracker=tracker)
