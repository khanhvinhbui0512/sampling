"""Named RNG streams independent of creation order, with serializable state.

Separate streams isolate sampler, discovery and estimation draws. Within one
stream, changing batching can still change draw order; use a stable query/sample
ID in the stream name when that independence is required. Torch is imported
only when a Torch generator or global training RNG is explicitly requested.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import random
from typing import Any

import numpy as np

from src.types import require_count, require_id


def derive_seed(root_seed: int, *labels: str) -> int:
    require_count(root_seed, "root_seed")
    for label in labels:
        require_id(label, "seed label")
    payload = json.dumps(["cars-vgb-rng-v1", root_seed, list(labels)], ensure_ascii=True)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _tuples(value: Any) -> Any:
    return tuple(_tuples(item) for item in value) if isinstance(value, (list, tuple)) else value


def _encode_torch_state(state: Any) -> str:
    return base64.b64encode(bytes(state.cpu().tolist())).decode("ascii")


def _decode_torch_state(state: str, torch: Any) -> Any:
    raw = base64.b64decode(state, validate=True)
    return torch.tensor(list(raw), dtype=torch.uint8, device="cpu")


class RNGStream:
    def __init__(self, seed: int) -> None:
        require_count(seed, "seed")
        self.seed = seed
        self.python = random.Random(seed)
        self.numpy = np.random.default_rng(seed)
        self._torch: dict[str, Any] = {}

    def torch_generator(self, device: str = "cpu") -> Any:
        import torch

        requested = torch.device(device)
        if requested.type == "cuda" and requested.index is None:
            requested = torch.device("cuda", torch.cuda.current_device())
        key = str(requested)
        if key not in self._torch:
            generator = torch.Generator(device=requested)
            generator.manual_seed(derive_seed(self.seed, "torch", key))
            self._torch[key] = generator
        return self._torch[key]

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "seed": self.seed,
            "python": self.python.getstate(),
            "numpy": copy.deepcopy(self.numpy.bit_generator.state),
            "torch": {device: _encode_torch_state(rng.get_state())
                      for device, rng in self._torch.items()},
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> RNGStream:
        if state.get("version") != 1:
            raise ValueError("Unsupported RNG stream state version")
        stream = cls(state["seed"])
        stream.python.setstate(_tuples(state["python"]))
        stream.numpy.bit_generator.state = copy.deepcopy(state["numpy"])
        for device, encoded in state["torch"].items():
            import torch

            stream.torch_generator(device).set_state(_decode_torch_state(encoded, torch))
        return stream


class RNGPool:
    """One pool per run; names such as estimation/query-001 identify stable streams."""

    def __init__(self, root_seed: int, run_id: str) -> None:
        require_count(root_seed, "root_seed")
        require_id(run_id, "run_id")
        self.root_seed = root_seed
        self.run_id = run_id
        self._streams: dict[str, RNGStream] = {}

    def stream(self, name: str) -> RNGStream:
        require_id(name, "stream name")
        if name not in self._streams:
            self._streams[name] = RNGStream(derive_seed(self.root_seed, self.run_id, name))
        return self._streams[name]

    @property
    def sampler(self) -> RNGStream:
        return self.stream("sampler")

    @property
    def discovery(self) -> RNGStream:
        return self.stream("discovery")

    @property
    def estimation(self) -> RNGStream:
        return self.stream("estimation")

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "root_seed": self.root_seed,
            "run_id": self.run_id,
            "streams": {name: stream.state_dict() for name, stream in self._streams.items()},
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> RNGPool:
        if state.get("version") != 1:
            raise ValueError("Unsupported RNG pool state version")
        pool = cls(state["root_seed"], state["run_id"])
        for name, saved in state["streams"].items():
            require_id(name, "stream name")
            expected = derive_seed(pool.root_seed, pool.run_id, name)
            if saved["seed"] != expected:
                raise ValueError(f"RNG seed identity mismatch for stream {name}")
            pool._streams[name] = RNGStream.from_state_dict(saved)
        return pool


def seed_training_rng(seed: int, *, deterministic: bool) -> None:
    """Explicit global setup for libraries that do not accept independent RNGs.

    Determinism across library releases, devices or CUDA versions is not promised.
    The caller must persist both this flag and the resolved software environment.
    """
    import torch

    require_count(seed, "seed")
    random.seed(seed)
    np.random.seed(derive_seed(seed, "training-numpy") % (2**32))
    torch.manual_seed(derive_seed(seed, "training-torch"))
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def capture_training_rng(*, include_cuda: bool = False) -> dict[str, Any]:
    import torch

    np_state = np.random.get_state()
    if include_cuda and not torch.cuda.is_available():
        raise ValueError("CUDA RNG capture requested without an available CUDA device")
    return {
        "version": 1,
        "python": random.getstate(),
        "numpy": [np_state[0], np_state[1].tolist(), *np_state[2:]],
        "torch_cpu": _encode_torch_state(torch.get_rng_state()),
        "torch_cuda": [_encode_torch_state(item) for item in torch.cuda.get_rng_state_all()]
        if include_cuda else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }


def restore_training_rng(state: dict[str, Any]) -> None:
    import torch

    if state.get("version") != 1:
        raise ValueError("Unsupported training RNG state version")
    cuda_states = state["torch_cuda"]
    if cuda_states is not None and len(cuda_states) != torch.cuda.device_count():
        raise ValueError("CUDA device count differs from the saved RNG state")
    numpy_state = state["numpy"]
    random.setstate(_tuples(state["python"]))
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4])))
    torch.set_rng_state(_decode_torch_state(state["torch_cpu"], torch))
    if cuda_states is not None:
        torch.cuda.set_rng_state_all([_decode_torch_state(item, torch) for item in cuda_states])
    torch.use_deterministic_algorithms(state["deterministic_algorithms"])
    torch.backends.cudnn.benchmark = state["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = state["cudnn_deterministic"]
