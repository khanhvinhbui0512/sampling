"""Build a standard fast tokenizer from the published QM9 vocabulary and regex.

Reference: yairschiff/qm9-tokenizer at 7e50af1babe9455a42aa64cb4ebbe43851990ea3.
The published vocab.json already includes five special tokens. Use its IDs
directly: the old Python class adds five again, leaving gaps in its IDs.
Save a standard tokenizer.json for local loading by every later task.
"""

import os
import re

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from src.io import read_json, write_json

SMILES_PATTERN = (
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)


def prepare_qm9_tokenizer(directory: str) -> PreTrainedTokenizerFast:
    """Keep published IDs, upstream regex splitting, concatenation and CLS/SEP framing."""
    config = read_json(os.path.join(directory, "tokenizer_config.json"))
    if config["tokenizer_class"] != "QM9Tokenizer":
        raise ValueError("Expected the upstream QM9Tokenizer asset")
    vocabulary = read_json(os.path.join(directory, "vocab.json"))
    special_ids = {"<bos>": 0, "<eos>": 1, "<mask>": 2, "<pad>": 3, "<unk>": 4}
    if (set(vocabulary.values()) != set(range(len(vocabulary)))
            or any(vocabulary.get(token) != index for token, index in special_ids.items())):
        raise ValueError("Expected contiguous QM9 vocabulary IDs with special tokens at 0..4")
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Split(Regex(SMILES_PATTERN), behavior="removed", invert=True)
    backend.decoder = decoders.Fuse()
    backend.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>", pair="<bos> $A <eos> $B:1 <eos>:1",
        special_tokens=[("<bos>", 0), ("<eos>", 1)],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="<bos>", eos_token="<eos>",
        cls_token="<bos>", sep_token="<eos>", mask_token="<mask>",
        pad_token="<pad>", unk_token="<unk>",
        clean_up_tokenization_spaces=config["clean_up_tokenization_spaces"],
    )
    # Retain the original Hub metadata alongside the portable tokenizer files.
    write_json(os.path.join(directory, "source_tokenizer_config.json"), config)
    tokenizer.save_pretrained(directory)
    return tokenizer


def reference_qm9_ids(text: str, vocabulary: dict[str, int], *, add_special_tokens: bool) -> list[int]:
    """Check fast encoding against the original regex and published vocabulary IDs."""
    ids = [vocabulary.get(token, vocabulary["<unk>"]) for token in re.findall(SMILES_PATTERN, text)]
    return [vocabulary["<bos>"], *ids, vocabulary["<eos>"]] if add_special_tokens else ids
