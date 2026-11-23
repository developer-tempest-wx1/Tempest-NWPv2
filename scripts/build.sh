#!/usr/bin/env bash
set -euo pipefail

# Build helper for Tempest-NWP v2 (Rust + CUDA atmospheric/ocean-SI3 solver + Python bridge).
# Supports NVIDIA Turing/Ampere/Ada/Hopper and RTX 50xx Blackwell (sm_120, CUDA >= 12.8).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

ARCH="${TEMPEST_CUDA_ARCH:-auto}"
RUN_TESTS=0
EDITABLE_PY=0

usage() {
    cat <<'EOF'
Usage: ./scripts/build.sh [OPTIONS]

Options:
  --arch <spec>         CUDA target arch: auto (default), all, sm_120, sm_89, sm_86,sm_120, ...
  --test                Run `cargo test --release` and `pytest -q tests/` after building
  --editable-python     Run `pip install -e . --no-build-isolation` after building
  -h, --help            Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch)
            [[ $# -ge 2 ]] || { echo "Error: --arch requires a value" >&2; exit 2; }
            ARCH="$2"
            shift 2
            ;;
        --test)
            RUN_TESTS=1
            shift
            ;;
        --editable-python)
            EDITABLE_PY=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

# Locate nvcc
if ! command -v nvcc >/dev/null 2>&1; then
    for candidate in "${CUDA_HOME:-}/bin" "${CUDA_PATH:-}/bin" "/usr/local/cuda/bin"; do
        if [[ -x "${candidate}/nvcc" ]]; then
            export PATH="${candidate}:${PATH}"
            break
        fi
    done
fi

if ! command -v nvcc >/dev/null 2>&1; then
    echo "Error: nvcc not found. Install NVIDIA CUDA Toolkit (>= 12.8 for RTX 50xx sm_120) or set CUDA_HOME." >&2
    exit 1
fi

# Locate cargo
if ! command -v cargo >/dev/null 2>&1; then
    if [[ -x "${HOME}/.cargo/bin/cargo" ]]; then
        export PATH="${HOME}/.cargo/bin:${PATH}"
    else
        echo "Error: cargo not found. Install Rust via https://rustup.rs" >&2
        exit 1
    fi
fi

# Locate Python
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    fi
fi

echo "=== Tempest-NWP v2 Build ==="
echo "Repo    : ${REPO_ROOT}"
echo "NVCC    : $(command -v nvcc) ($(nvcc --version | grep -o 'release [0-9.]*' | head -n1 || echo 'unknown'))"
echo "Cargo   : $(cargo --version)"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU(s)  :"
    nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | sed 's/^/  - /'
else
    echo "GPU(s)  : none detected (headless build)"
fi
echo "Arch    : TEMPEST_CUDA_ARCH=${ARCH}"

export TEMPEST_CUDA_ARCH="${ARCH}"
cargo build --release

SO_PATH="${REPO_ROOT}/crates/tempest-cuda/libtempest_cuda.so"
if [[ ! -f "${SO_PATH}" ]]; then
    echo "Error: expected shared library not produced at ${SO_PATH}" >&2
    exit 1
fi
echo "Built   : ${SO_PATH} ($(du -h "${SO_PATH}" | cut -f1))"

if [[ -n "${PYTHON_BIN}" ]]; then
    PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" -c "
import ctypes
lib = ctypes.CDLL('${SO_PATH}')
assert hasattr(lib, 'nh_cuda_create'), 'Missing nh_cuda_create symbol'
assert hasattr(lib, 'nh_cuda_step_rk3'), 'Missing nh_cuda_step_rk3 symbol'
assert hasattr(lib, 'nh_ocean_si3_create'), 'Missing NEMO+SI3 ocean symbol'
print('ABI OK  : libtempest_cuda.so loaded (atmosphere + NEMO/SI3 CUDA symbols verified)')
"
fi

if [[ "${EDITABLE_PY}" -eq 1 ]]; then
    [[ -n "${PYTHON_BIN}" ]] || { echo "Error: python not found for --editable-python" >&2; exit 1; }
    "${PYTHON_BIN}" -m pip install -e . --no-build-isolation
fi

if [[ "${RUN_TESTS}" -eq 1 ]]; then
    echo "=== Running Rust/CUDA Release Tests ==="
    cargo test --release
    if [[ -n "${PYTHON_BIN}" ]]; then
        echo "=== Running Python Pytest Suite ==="
        "${PYTHON_BIN}" -m pytest -q tests/
    fi
fi

echo "=== Build Complete ==="
