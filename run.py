"""Select one existing local QM9 task; task arguments keep their original meaning."""

import argparse
import os
import subprocess
import sys


TASKS = (
    "prepare_assets", "train_base", "calibrate", "collect_rollouts", "train_verifier",
    "generate", "prepare_value_panel", "check_values", "verify", "benchmark", "evaluate",
)


def main(argv=None):
    # Parse only the selector. Each existing runner owns its flags and validation.
    parser = argparse.ArgumentParser(
        description=__doc__, add_help=False, allow_abbrev=False,
        usage="python run.py --task TASK [task arguments]",
        epilog="Use --task TASK --help for task flags. Run from the repository root.",
    )
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("-h", "--help", action="store_true", help="Show general or selected task help")
    args, task_arguments = parser.parse_known_args(argv)
    if args.task is None:
        if args.help:
            parser.print_help()
            return 0
        parser.error("--task is required")
    if not os.path.isfile("pyproject.toml") or not os.path.isdir("src"):
        parser.error("Run this task from the repository root")
    script = args.task + ".py"
    if not os.path.isfile(script):
        parser.error(f"Missing local runner: {script}")
    if args.help:
        task_arguments.append("--help")
    try:
        # Inherit output so the task's tqdm bars, result and errors stay visible.
        result = subprocess.run([sys.executable, script, *task_arguments], check=False)
    except KeyboardInterrupt:
        print(f"Task {args.task}: interrupted", flush=True)
        return 130
    except OSError as exc:
        print(f"Task {args.task}: could not start: {exc}", flush=True)
        return 1
    exit_code = result.returncode if result.returncode >= 0 else 128 - result.returncode
    if not args.help:
        print(f"Task {args.task}: exit_code={exit_code}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
