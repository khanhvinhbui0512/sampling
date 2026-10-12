"""Run configured verification phases and persist their evidence and failures.

Run from the repository root with ``python src/tasks/verify.py``. Future tests
are named by YAML; this module imports no future model, sampler, or metric code.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import sys
import time
import tomllib
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from tqdm.auto import tqdm

from src import __version__
from src.artifacts import Dependency, sha256_file, utc_now
from src.config import Config, project_path
from src.schemas import RunRecord, TaskResult, require_identifier, require_sha256
from src.tasks.common import TaskContext, run_task

PHASES = ("environment", "benchmark", "corpus", "cars", "values", "integration", "metrics", "all")


def validate_verification(config: Config) -> None:
    verification = config.get("verification")
    if not isinstance(verification, dict) or verification.get("phase") not in PHASES:
        raise ValueError("verification.phase must name a configured verification phase")
    if not isinstance(verification.get("check_installed_dependencies"), bool):
        raise TypeError("check_installed_dependencies must be boolean")
    phases = verification.get("phases")
    if not isinstance(phases, dict) or set(phases) != set(PHASES):
        raise ValueError("Configure environment, benchmark, corpus, cars, values, integration, metrics, and all")
    args = verification.get("pytest_args")
    if not isinstance(args, list) or any(not isinstance(item, str) for item in args):
        raise TypeError("pytest_args must be a list of option strings")
    allowed = {"-q", "-v", "-vv", "-s", "-x", "--disable-warnings", "--strict-markers", "--strict-config"}
    for argument in args:
        if argument in allowed:
            continue
        if argument.startswith("--maxfail=") and argument.partition("=")[2].isdigit():
            if int(argument.partition("=")[2]) > 0:
                continue
        raise ValueError(f"Unsupported pytest option: {argument}; test paths come from the phase")
    for name in PHASES[:-1]:
        phase = phases[name]
        if not isinstance(phase, dict):
            raise TypeError(f"Phase {name} must be a mapping")
        for key in ("tests", "project_files"):
            paths = phase.get(key)
            if not isinstance(paths, list) or (key == "tests" and not paths):
                raise ValueError(f"Phase {name} requires a {key} list")
            for path in paths:
                project_path(path)
                if key == "tests" and (not path.startswith("tests/") or not path.endswith(".py")):
                    raise ValueError(f"Phase test must be a project tests/*.py path: {path}")
            if len(set(paths)) != len(paths):
                raise ValueError(f"Duplicate {key} paths in phase {name}")
        artifacts = phase.get("artifacts")
        if not isinstance(artifacts, dict):
            raise TypeError(f"Phase {name} requires an artifacts mapping")
        for key, artifact in artifacts.items():
            require_identifier(key, "verification artifact")
            if not isinstance(artifact, dict):
                raise TypeError(f"Artifact {key} must be a mapping")
            path = artifact.get("path")
            if path is not None:
                project_path(path)
            if not isinstance(artifact.get("manifest"), bool):
                raise TypeError(f"Artifact {key}.manifest must be boolean")
            if artifact.get("checksum") is not None:
                require_sha256(artifact["checksum"])
            if artifact.get("kind") is not None:
                require_identifier(artifact["kind"], "manifest kind")
            if not isinstance(artifact.get("metadata"), dict):
                raise TypeError(f"Artifact {key}.metadata must be a mapping")
            if not artifact["manifest"] and (artifact.get("kind") is not None or artifact["metadata"]):
                raise ValueError("Kind/metadata requirements apply only to artifact manifests")
            if not isinstance(artifact.get("run_record", False), bool):
                raise TypeError("run_record must be boolean")
            if artifact.get("run_record") and artifact["manifest"]:
                raise ValueError("Run records and artifact manifests have different schemas")
    aggregate = phases["all"]
    if not isinstance(aggregate, dict) or aggregate.get("includes") != list(PHASES[:-1]):
        raise ValueError("all must include each ordinary phase once, in workflow order")


def _environment_checks(check_dependencies: bool) -> tuple[dict[str, Any], list[str]]:
    evidence: dict[str, Any] = {
        "python": platform.python_version(), "platform": platform.platform(),
        "src_version": __version__, "packages": {},
    }
    errors = []
    if sys.version_info[:2] != (3, 12):
        errors.append(f"Expected Python 3.12; found {platform.python_version()}")
    with open("pyproject.toml", "rb") as handle:
        project = tomllib.load(handle)["project"]
    if project["version"] != __version__:
        errors.append("src version differs from pyproject.toml")
    try:
        installed_version = importlib.metadata.version(project["name"])
        evidence["installed_project_version"] = installed_version
        if installed_version != project["version"]:
            errors.append("Installed project version differs from pyproject.toml")
    except importlib.metadata.PackageNotFoundError:
        errors.append('Install the project from the repository root with pip install -e ".[dev]"')
    if check_dependencies:
        # packaging is a dependency of the installed development tools.
        try:
            from packaging.requirements import Requirement
        except ImportError:
            errors.append("Missing packaging; install the project's development dependencies")
            return evidence, errors
        requirements = [*project["dependencies"], *project["optional-dependencies"]["dev"]]
        for text in requirements:
            requirement = Requirement(text)
            if requirement.marker is not None and not requirement.marker.evaluate():
                continue
            try:
                version = importlib.metadata.version(requirement.name)
            except importlib.metadata.PackageNotFoundError:
                evidence["packages"][requirement.name] = None
                errors.append(f"Missing package: {requirement.name}")
                continue
            evidence["packages"][requirement.name] = version
            if not requirement.specifier.contains(version, prereleases=True):
                errors.append(f"{requirement.name} {version} does not satisfy {requirement.specifier}")
    return evidence, errors


def _preflight(context: TaskContext, name: str) -> dict[str, Any]:
    phase = context.config["verification"]["phases"][name]
    result: dict[str, Any] = {
        "phase": name, "status": "failed", "errors": [], "fingerprints": {},
        "environment": {}, "tests": [], "collection_errors": [], "exit_code": None,
        "counts": {"passed": 0, "failed": 0, "skipped": 0}, "reused": False,
    }
    files = set([*phase["project_files"], *phase["tests"], "tests/conftest.py", context.arguments.config])
    # Changes to any implementation invalidate a previously passed phase on resume.
    for directory in ("src", "modal"):
        for root, directories, filenames in os.walk(directory, followlinks=False):
            directories[:] = [part for part in directories if part != "__pycache__"]
            for filename in filenames:
                if filename.endswith(".py"):
                    files.add(os.path.join(root, filename))
    for path in sorted(files):
        try:
            result["fingerprints"]["project:" + path] = sha256_file(path)
        except (OSError, ValueError) as exc:
            result["errors"].append(f"Required project file {path}: {exc}")
    for key, artifact in phase["artifacts"].items():
        path = artifact["path"]
        if path is None:
            result["errors"].append(
                f"Required artifact {key} has no path; set verification.phases.{name}.artifacts.{key}.path",
            )
            continue
        try:
            dependency = Dependency(
                path, checksum=artifact["checksum"], manifest=artifact["manifest"],
                kind=artifact["kind"], metadata=artifact["metadata"],
            )
            digest = context.store.check_dependencies({key: dependency})[key]
            result["fingerprints"]["artifact:" + key] = digest
            if artifact.get("run_record"):
                record = RunRecord.from_dict(context.store.read_json(path))
                if record.status != "completed" or not record.artifacts:
                    raise ValueError("Generation run must be completed and declare its output files")
                if record.identity.task_id not in {"T08", "T09", "T10", "T11"}:
                    raise ValueError("Expected a generation run from T08, T09, T10, or T11")
                if record.identity.suite != context.identity.suite or record.identity.reward_version != context.identity.reward_version:
                    raise ValueError("Generation run uses a different suite or reward version")
                for output in record.artifacts:
                    result["fingerprints"]["run_file:" + output] = sha256_file(context.store.require_file(output))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            result["errors"].append(f"Required artifact {key} ({path}): {exc}")
    if name == "environment":
        try:
            evidence, errors = _environment_checks(context.config["verification"]["check_installed_dependencies"])
            result["environment"] = evidence
            result["errors"].extend(errors)
        except (OSError, ValueError, KeyError) as exc:
            result["errors"].append(f"Environment inspection failed: {exc}")
    return result


class _TestEvidence:
    """Keep pytest outcomes in the report without parsing console text."""

    def __init__(self, context: TaskContext, phase: str) -> None:
        self.rows: list[dict[str, Any]] = []
        self.collection_errors: list[str] = []
        self.progress = context.progress(total=None, desc=f"verify:{phase}", unit="test")

    def pytest_collection_finish(self, session: Any) -> None:
        self.progress.total = len(session.items)
        self.progress.refresh()

    def pytest_collectreport(self, report: Any) -> None:
        if report.failed:
            self.collection_errors.append(str(report.longrepr))

    def pytest_runtest_logreport(self, report: Any) -> None:
        self.rows.append({
            "nodeid": report.nodeid, "when": report.when, "outcome": report.outcome,
            "seconds": report.duration, "detail": str(report.longrepr) if report.longrepr else None,
        })
        if report.when == "teardown":
            self.progress.update(1)

    def counts(self) -> dict[str, int]:
        outcomes: dict[str, list[str]] = {}
        for row in self.rows:
            outcomes.setdefault(row["nodeid"], []).append(row["outcome"])
        counts = {"passed": 0, "failed": 0, "skipped": 0}
        for rows in outcomes.values():
            outcome = "failed" if "failed" in rows else "skipped" if "skipped" in rows else "passed"
            counts[outcome] += 1
        return counts


@contextmanager
def _test_environment(context: TaskContext, phase: str):
    values = {
        "SMILES_VERIFICATION_CONFIG": context.store.path(context.output_path("resolved_config.yaml")),
        "SMILES_ARTIFACT_ROOT": context.store.root,
        "SMILES_VERIFICATION_PHASE": phase,
    }
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _run_phase(context: TaskContext, result: dict[str, Any]) -> None:
    if result["errors"]:
        return
    try:
        import pytest
    except ImportError:
        result["errors"].append("Missing pytest; install the development dependencies")
        return
    phase = context.config["verification"]["phases"][result["phase"]]
    plugin = _TestEvidence(context, result["phase"])
    try:
        with _test_environment(context, result["phase"]), context.tracker.measure("verification"):
            exit_code = int(pytest.main([
                *phase["tests"], *context.config["verification"]["pytest_args"],
                "--rootdir=.", "-c", "pyproject.toml",
            ], plugins=[plugin]))
        result["exit_code"] = exit_code
        result["tests"] = plugin.rows
        result["collection_errors"] = plugin.collection_errors
        result["counts"] = plugin.counts()
        # An entirely skipped phase supplies no evidence of a working task.
        if exit_code == 0 and result["counts"]["passed"] > 0 and not result["counts"]["failed"]:
            result["status"] = "passed"
        else:
            result["errors"].append(f"pytest exit={exit_code}; counts={result['counts']}")
    except Exception as exc:
        result["errors"].append(f"pytest failed to execute: {type(exc).__name__}: {exc}")
    finally:
        plugin.progress.close()


def _save_report(context: TaskContext, selected: list[str], results: list[dict[str, Any]]) -> tuple[str, str]:
    complete = len(results) == len(selected)
    passed = sum(row["status"] == "passed" for row in results)
    status = "passed" if complete and passed == len(selected) else "failed" if complete else "running"
    report = {
        "schema_version": 1, "run_id": context.identity.run_id,
        "config_hash": context.identity.config_hash, "execution": context.config["runtime"]["execution"],
        "requested_phase": context.config["verification"]["phase"], "selected_phases": selected,
        "status": status, "updated_at": utc_now(), "phases": results,
    }
    json_path = context.output_path("verification.json")
    text_path = context.output_path("verification.txt")
    context.store.write_json(json_path, report)
    lines = [f"Verification: {status}; phases passed {passed}/{len(selected)}"]
    for row in results:
        lines.append(f"{row['phase']}: {row['status']}; counts={row['counts']}; reused={row['reused']}")
        lines.extend("  " + error for error in row["errors"])
    context.store.write_text(text_path, "\n".join(lines) + "\n")
    return json_path, text_path


def verify(context: TaskContext) -> TaskResult:
    requested = context.config["verification"]["phase"]
    selected = list(PHASES[:-1]) if requested == "all" else [requested]
    previous: Mapping[str, Any] = {}
    if context.resumed and context.resume_state is not None:
        state = context.resume_state
        if not isinstance(state, dict) or state.get("selected_phases") != selected:
            raise ValueError("Verification checkpoint has a different phase selection")
        previous = {row["phase"]: row for row in state["phase_reports"]}
    results = []
    report_paths = _save_report(context, selected, results)
    for name in context.progress(selected, desc="verification", unit="phase"):
        started = time.perf_counter()
        result = _preflight(context, name)
        old = previous.get(name)
        if (
            not result["errors"] and old is not None and old["status"] == "passed"
            and old["fingerprints"] == result["fingerprints"] and old["environment"] == result["environment"]
        ):
            result = dict(old, reused=True)
        else:
            _run_phase(context, result)
            result["seconds"] = time.perf_counter() - started
        results.append(result)
        report_paths = _save_report(context, selected, results)
        context.save_checkpoint({"selected_phases": selected, "phase_reports": results})
        context.log(f"{name}: {result['status']}; tests={result['counts']}")
        for error in result["errors"]:
            context.log(error, level="ERROR")
    passed = sum(result["status"] == "passed" for result in results)
    return TaskResult(
        f"verification {passed}/{len(selected)} phases passed; report={context.store.path(report_paths[0])}",
        status="completed" if passed == len(selected) else "failed", artifacts=report_paths,
    )


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--phase", choices=PHASES, default=None, help="Verification phase; default comes from YAML.")


def _argument_overrides(arguments: argparse.Namespace) -> list[str]:
    return [] if arguments.phase is None else ["verification.phase=" + arguments.phase]


def execute(argv: Sequence[str] | None = None, *, execution: str = "local") -> RunRecord:
    """Used by both entrypoints; a failed report is returned with failed status."""
    return run_task(
        "T12", verify, description="Verify configured SMILES project phases.", argv=argv,
        default_config="configs/verification.yaml", execution=execution, supports_seed=False,
        validator=validate_verification, add_arguments=_add_arguments, argument_overrides=_argument_overrides,
    )


def main() -> None:
    try:
        record = execute()
    except KeyboardInterrupt:
        tqdm.write("T12: interrupted")
        raise SystemExit(130) from None
    except Exception:
        # Keep the runner's short failure message as the CLI result. Programmatic
        # callers of execute() still receive the original exception.
        raise SystemExit(1) from None
    if record.status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
