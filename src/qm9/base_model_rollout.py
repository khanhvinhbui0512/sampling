#!/usr/bin/env python
import argparse
import random
from pathlib import Path

if __package__:
    from . import task as task_module
    from .harness import load as load_harness
    from .utils import load_config, log_metrics, repo_root, setup_logging, set_seed, write_rollout
else:
    import task as task_module
    from harness import load as load_harness
    from utils import load_config, log_metrics, repo_root, setup_logging, set_seed, write_rollout


def select_rollout_examples(examples, config, seed, task_module):
    if getattr(task_module, "HANDLES_ROLLOUT_REPEATS", False):
        return examples
    num_rollouts = config.get("rollout", {}).get("num_rollouts", -1)
    if num_rollouts is None:
        return examples
    num_rollouts = int(num_rollouts)
    if num_rollouts < 0 or num_rollouts >= len(examples):
        return examples
    indices = list(range(len(examples)))
    random.Random(0 if seed is None else int(seed)).shuffle(indices)
    return [examples[idx] for idx in indices[:num_rollouts]]


def main(args: argparse.Namespace):
    config_path = Path(args.config) if args.config else Path("config/qm9/qm9_rollout.yaml")
    if not config_path.is_absolute():
        config_path = repo_root() / config_path
    config = load_config(config_path)
    base_seed = config.get("seed")
    set_seed(base_seed)
    logger = setup_logging(config.get("logging", {}).get("level", "INFO"))

    task_module.configure(config)
    examples = task_module.load_examples(split=config["data"]["split"])
    examples = select_rollout_examples(examples, config, base_seed, task_module)
    model_config = {key: value for key, value in config.get("model", {}).items() if value is not None}
    for key in ("model_name", "tokenizer_path"):
        model_path = Path(model_config[key])
        if not model_path.is_absolute():
            model_path = repo_root() / model_path
        model_config[key] = str(model_path)
    harness = load_harness(device=args.device, **model_config)
    rows = task_module.collect_rollout_rows(harness, examples, config, rank=0)
    output = config.get("output")
    if output:
        output = Path(str(output).format(algorithm="base", algorithm_name="Base", stage="rollout", task=args.task))
        if not output.is_absolute():
            output = repo_root() / output
        write_rollout(output, rows, task_module=task_module, config=config)
        logger.info("wrote %d snapshot rows to %s", len(rows), output)
    final_rows = {}
    for row in rows:
        final_rows.setdefault(row["rollout_id"], row)
    metrics = task_module.metrics(list(final_rows.values()))
    num_rollouts = len(final_rows)
    snapshots_per_rollout = int(config.get("rollout", {}).get("snapshots_per_rollout", 1))
    metrics.update(
        {
            "num_rollouts": num_rollouts,
            "num_snapshots": num_rollouts * snapshots_per_rollout,
            "snapshots_per_rollout": snapshots_per_rollout,
        }
    )
    log_metrics(metrics, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="qm9", choices=["qm9"], help="QM9 rollout task.")
    parser.add_argument("--config", default=None, help="Config path (default: config/qm9/qm9_rollout.yaml).")
    parser.add_argument("--device", default="cuda:0", help="Device for single-GPU rollout (default: cuda:0).")
    main(parser.parse_args())
