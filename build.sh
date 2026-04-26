#!/bin/bash
# Build wrapper for the C++ solvers.
#
# Usage (from repo root):
#   ./build.sh src/solve_omp          # builds src/solve_omp.cpp -> src/solve_omp
#   ./build.sh src/solve_omp_opt      # builds src/solve_omp_opt.cpp -> src/solve_omp_opt
#   CXXFLAGS=-DUSE_ZSTD ./build.sh src/solve_omp_opt_zstd
#
# The argument is the output path WITHOUT the .cpp suffix. The script expects
# a file named "$1.cpp" to exist.
#
# Requirements:
#   - macOS: Homebrew libomp (brew install libomp)
#   - Linux: g++ with OpenMP support
#   - C++23 compiler

set -e

OS="$(uname -s)"
TARGET="$1"

if [[ -z "$TARGET" ]]; then
    echo "Usage: $0 <target-without-.cpp> (e.g. src/solve_omp)"
    exit 1
fi

if [[ ! -f "$TARGET.cpp" ]]; then
    echo "Error: $TARGET.cpp not found"
    exit 1
fi

echo "Building $TARGET.cpp -> $TARGET"

if [[ "$OS" == "Linux" ]]; then
    g++ "$TARGET.cpp" -fopenmp -O3 -march=native --std=gnu++23 -Wall $CXXFLAGS -o "$TARGET"
elif [[ "$OS" == "Darwin" ]]; then
    c++ -Xclang -fopenmp \
        -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include \
        -lomp \
        "$TARGET.cpp" \
        -Wall -O3 -march=native --std=gnu++23 $CXXFLAGS \
        -o "$TARGET"
else
    echo "System not supported"
    exit 1
fi

echo "OK"
