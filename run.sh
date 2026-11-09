#!/bin/bash
# Build with CMake and run. Extra arguments are passed to the program, e.g. `./run.sh test`.
set -e
cd "$(dirname "$0")"

# on the university cluster, load a recent gcc first (no-op elsewhere)
if command -v module >/dev/null 2>&1; then module add gcc; fi

echo "#################"
echo "    COMPILING    "
echo "#################"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

echo "#################"
echo "     RUNNING     "
echo "#################"
# nice lowers the priority, as required on shared machines (https://www.fi.muni.cz/tech/unix/computation.html.en)
# run from the repo root so the relative data/ paths resolve
nice -n 19 ./build/network "$@"
