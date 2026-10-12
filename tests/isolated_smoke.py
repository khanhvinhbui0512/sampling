"""Helper invoked only inside the disposable copy created by Step 12 tests."""

import importlib
import os
import pkgutil
import sys
import zipfile

sys.path.insert(0, os.getcwd())


def main():
    assert not os.path.exists("resource"), "Run this helper in the test's isolated copy"
    from tqdm.auto import tqdm
    import src
    from setuptools.build_meta import build_wheel
    from src.config import load_config
    from src.evaluation.aggregate import evaluate
    from src.experiments.toy import check_toy_values
    from src.io import read_completed_manifest, write_yaml
    from src.models.configuration_udlm import UDLMConfig

    root = os.path.realpath(".") + os.sep
    for info in tqdm(list(pkgutil.walk_packages(src.__path__, prefix="src.")), desc="Isolated imports"):
        module = importlib.import_module(info.name)
        assert os.path.realpath(module.__file__).startswith(root), info.name
    config = UDLMConfig(vocab_size=3, model_length=3, hidden_dim=16, cond_dim=8,
                        n_blocks=1, n_heads=2, auto_map={"AutoModel": "resource.models.UDLM"})
    assert not config.to_dict().get("auto_map")
    settings = load_config("configs/value_checks.yaml")
    settings["check_output_dir"] = "workspace/results/value_checks"
    settings["toy"].update(discovery_attempts=1, numerator_draws=8, denominator_draws=8,
                            mc_draws=8, min_denominator_hits=1, max_nodes=500)
    write_yaml("workspace/toy.yaml", settings)
    checked = check_toy_values(attempt_id="isolated", config_path="workspace/toy.yaml")
    read_completed_manifest(checked["manifest"])
    evaluation = load_config("configs/evaluation.yaml")
    evaluation["uncertainty"]["bootstrap_replicates"] = 10
    write_yaml("workspace/evaluation.yaml", evaluation)
    result = evaluate(manifests=[checked["manifest"]], attempt_id="isolated", config_path="workspace/evaluation.yaml")
    assert result["selection_complete"] and result["records"] == 2
    read_completed_manifest(result["manifest"])
    wheel_name = build_wheel("workspace/wheels")
    with zipfile.ZipFile(os.path.join("workspace/wheels", wheel_name)) as wheel:
        names = wheel.namelist()
    assert any(name.startswith("src/models/") for name in names)
    assert all(not name.startswith(("resource/", "tests/", "modal/")) for name in names)
    print("Isolated smoke passed: imports, model config, toy values, evaluation and wheel", flush=True)


if __name__ == "__main__":
    main()
