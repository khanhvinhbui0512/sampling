#!/usr/bin/env bash
set -euo pipefail

TASK_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$TASK_REPO_ROOT"
exec modal run modal/generate.py::main "$@"
