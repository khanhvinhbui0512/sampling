"""T11: the same MDM-VGB kernel with fixed-m CARS values."""

from tqdm.auto import tqdm

from src.tasks.generation import execute_generation


def execute(argv=None, *, execution="local", on_checkpoint=None):
    return execute_generation("hybrid", argv, execution=execution, on_checkpoint=on_checkpoint)


if __name__ == "__main__":
    try:
        record = execute()
        if record.status != "completed":
            raise SystemExit(1)
    except KeyboardInterrupt:
        tqdm.write("T11: interrupted")
        raise SystemExit(130) from None
    except Exception as exc:
        tqdm.write(f"T11: failed; {type(exc).__name__}: {exc}")
        raise SystemExit(1) from None
