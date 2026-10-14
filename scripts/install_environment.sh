#!/usr/bin/env bash
# Dependencies and versions live in environment.yml, not in this installer.
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
environment_name=pih3l-femto
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  cat <<'HELP'
Usage: bash scripts/install_environment.sh [new-environment-name]

Requires an existing Conda/Miniforge installation. Creates a new environment
from environment.yml, installs this checkout as a package, and checks its CLI
and Python dependencies. An existing environment is never updated implicitly.
Run from a Linux Farm checkout; the standard environment name is pih3l-femto.
HELP
  exit 0
fi
if [[ $# -gt 1 ]]; then
  echo 'Expected at most one new environment name.' >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then
  environment_name=$1
fi
if [[ ! $environment_name =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
  echo 'Environment name must contain only letters, digits, underscore, dot or dash.' >&2
  exit 2
fi
if ! command -v conda >/dev/null 2>&1; then
  echo 'Conda is missing. Install Miniforge first, then rerun this script.' >&2
  exit 1
fi

conda env create --file "$project_root/environment.yml" --name "$environment_name"
conda run --no-capture-output --name "$environment_name" \
  python -m pip install --no-deps --no-build-isolation "$project_root"
conda run --no-capture-output --name "$environment_name" python -m pip check
conda run --no-capture-output --name "$environment_name" pih3l --help
conda run --no-capture-output --name "$environment_name" pih3l doctor
printf '\nEnvironment ready. Activate with: conda activate %s\n' "$environment_name"
