#!/usr/bin/env bash
# Linux bootstrap for the original Meta backend. Reuses only an owned environment.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: bash chapter3_segmentation/setup_sam3.sh [SAM3_SOURCE_DIR [VENV_DIR]]
PYTHON_BIN may select a Python 3.12 interpreter for this pinned dependency set.
Defaults: external/sam3 and .venv-sam3 under this checkout.
An existing environment is reused only if this script created it for this checkout.
HELP
    exit 0
fi
[[ $# -le 2 ]] || { echo "Expected at most SAM3_SOURCE_DIR and VENV_DIR." >&2; exit 2; }
sam3_root="${1:-$project_root/external/sam3}"
environment_dir="${2:-$project_root/.venv-sam3}"
sam3_commit="eef6839bcfc3458dd52ea80535dc6bd62aea86b1"
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This setup requires Linux or WSL2." >&2
    exit 2
fi
command -v git >/dev/null || { echo "Install git first." >&2; exit 2; }
python_bin="${PYTHON_BIN:-}"
if [[ -z "$python_bin" ]]; then
    for candidate in python3.12 python3; do
        if command -v "$candidate" >/dev/null && "$candidate" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))'; then
            python_bin="$candidate"
            break
        fi
    done
fi
if [[ -z "$python_bin" ]] || ! command -v "$python_bin" >/dev/null || ! "$python_bin" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))'; then
    echo "This pinned dependency set requires Python 3.12 (numpy 1.26.4). Install it with venv support, or set PYTHON_BIN=/path/to/python3.12." >&2
    exit 2
fi
sam3_root="$("$python_bin" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$sam3_root")"
environment_dir="$("$python_bin" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$environment_dir")"
marker="$environment_dir/.phytea-sam3-venv"
if [[ -e "$environment_dir" ]]; then
    if [[ ! -f "$marker" || "$(cat "$marker")" != "$project_root" || ! -x "$environment_dir/bin/python" ]]; then
        echo "Refusing to modify an existing environment not created for this checkout: $environment_dir" >&2
        echo "Choose a new VENV_DIR; no existing files were replaced." >&2
        exit 2
    fi
    "$environment_dir/bin/python" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' || {
        echo "The existing environment needs Python 3.12 for this dependency set; choose a new VENV_DIR." >&2; exit 2;
    }
else
    "$python_bin" -m venv "$environment_dir"
    printf '%s\n' "$project_root" > "$marker"
fi
if [[ ! -e "$sam3_root" ]]; then
    mkdir -p "$(dirname "$sam3_root")"
    git clone --no-checkout https://github.com/facebookresearch/sam3.git "$sam3_root"
    git -C "$sam3_root" checkout --detach "$sam3_commit"
elif [[ ! -d "$sam3_root/.git" ]] || [[ "$(git -C "$sam3_root" rev-parse HEAD)" != "$sam3_commit" ]]; then
    echo "Existing SAM3 checkout is not at $sam3_commit. Use a new path or select that revision yourself." >&2
    exit 2
fi
if ! git -C "$sam3_root" diff --quiet || ! git -C "$sam3_root" diff --cached --quiet; then
    echo "SAM3 has tracked local changes. Choose a clean checkout; no source files were changed." >&2
    exit 2
fi
python="$environment_dir/bin/python"
"$python" -m pip install --upgrade pip
# The constraint also applies to upstream extras so they cannot replace this tested base.
constraints="$environment_dir/phytea-base-constraints.txt"
printf '%s\n' 'torch==2.7.1' 'torchvision==0.22.1' 'setuptools<82' 'numpy==1.26.4' > "$constraints"
"$python" -m pip install --constraint "$constraints" torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
"$python" -m pip install --constraint "$constraints" -r "$project_root/chapter3_segmentation/requirements-train.txt"
"$python" -m pip install --constraint "$constraints" -e "$sam3_root[train]"
"$python" -m pip check
"$python" -c 'import pkg_resources, torch, torchvision; print("PyTorch", torch.__version__, "CUDA", torch.version.cuda)'
echo "Activate with: source $environment_dir/bin/activate"
echo "Next: bash chapter3_segmentation/run_server.sh --stage check --sam3-root '$sam3_root' --python '$python' --data-dir /path/to/prepared-data --checkpoint /path/to/sam3.pt"
