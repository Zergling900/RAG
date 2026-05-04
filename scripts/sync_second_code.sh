#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${1:-/home/quanwei/second}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_ROOT="${2:-${REPO_ROOT}/documents/code/w_be_surface_md}"

if [[ ! -d "$SOURCE_ROOT" ]]; then
    echo "Source directory does not exist: $SOURCE_ROOT" >&2
    exit 1
fi

shopt -s globstar nullglob

copy_patterns() {
    local component="$1"
    shift

    for pattern in "$@"; do
        for src in "$SOURCE_ROOT"/$pattern; do
            [[ -f "$src" ]] || continue
            local rel="${src#"$SOURCE_ROOT"/}"
            local dest="${DEST_ROOT}/${component}/${rel}"
            install -d "$(dirname "$dest")"
            cp -p "$src" "$dest"
        done
    done
}

copy_patterns "w_be_abop_potential" \
    "potentials/abop/**/*.cpp" \
    "potentials/shared/**/*.inc" \
    "parameter_BeW_ABOP.pr2"

copy_patterns "w_eam2_potential" \
    "potentials/eam2/**/*.cpp" \
    "potentials/eam2/**/*.h" \
    "parameter_W_EAM.p2" \
    "parameter.p4" \
    "run.eam2.sh" \
    "run2.eam2.sh"

copy_patterns "md_simulation_core" \
    "common/**/*.cpp" \
    "common/**/*.h" \
    "docs/*.md" \
    "compile_flags.txt" \
    "Compilation.sh" \
    "parameter.p1" \
    "parameter.p3" \
    "parameter.p5" \
    "run.sh" \
    "run1.sh" \
    "run2.sh"

copy_patterns "neb_workflow" \
    "NEB/*.cpp" \
    "NEB/*.sh"

copy_patterns "least_squares_activation_fitting" \
    "plot/*.py" \
    "plot/diffusion_xy/*.py" \
    "plot/*.md" \
    "plot/diffusion_xy/*.md" \
    "plotE.py" \
    "plot.py" \
    "plot2.py"

copy_patterns "preprocessing_relax_tools" \
    "cut/*.cpp" \
    "cut/*.md" \
    "relax/*.cpp" \
    "relax/*.md" \
    "relax/*.sh"

echo "Synced code snapshot to $DEST_ROOT"
