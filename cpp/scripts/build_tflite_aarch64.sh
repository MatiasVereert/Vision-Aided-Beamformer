#!/usr/bin/env bash
# build_tflite_aarch64.sh -- libtensorflowlite_c.so para la Kria, compilada en la PC.
# ----------------------------------------------------------------------------
# Misma version y mismas fuentes que la build x86 ($TF_SRC, hoy v2.20.0): los
# headers con los que compila el OFB tienen que ser los de la .so que carga.
#
# Dos pasos, porque TFLite al compilar cruzado se niega a arrancar sin un
# `flatc` que corra EN LA PC (CMakeLists.txt de lite, TFLITE_HOST_TOOLS_DIR):
#   1. flatc nativo  -> $HOST_TOOLS   (una vez; despues se reusa)
#   2. lite/c cruzado -> $OUT/libtensorflowlite_c.so
#
# Requisitos: sudo apt install g++-aarch64-linux-gnu cmake
# Uso:        cpp/scripts/build_tflite_aarch64.sh   (desde cualquier directorio)
# Variables:  TF_SRC, OUT, HOST_TOOLS, JOBS
# ----------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLCHAIN="$HERE/../cmake/aarch64-linux-gnu.cmake"
TF_SRC=${TF_SRC:-$HOME/src/tensorflow}
OUT=${OUT:-$HOME/src/tflite_build_aarch64}
HOST_TOOLS=${HOST_TOOLS:-$HOME/src/tflite_host_tools}
# Acotado a proposito: la PC ya se quedo sin memoria una vez con -j alto.
JOBS=${JOBS:-8}

command -v aarch64-linux-gnu-g++ >/dev/null || {
  echo "ERROR: falta el compilador cruzado: sudo apt install g++-aarch64-linux-gnu"; exit 1; }
[ -f "$TF_SRC/tensorflow/lite/c/CMakeLists.txt" ] || {
  echo "ERROR: no encuentro las fuentes de tensorflow en $TF_SRC"; exit 1; }

# --- 1. flatc para la PC ----------------------------------------------------
if [ ! -x "$HOST_TOOLS/flatc" ]; then
  echo ">> 1/2 flatc nativo -> $HOST_TOOLS"
  cmake -S "$TF_SRC/tensorflow/lite/tools/cmake/native_tools/flatbuffers" \
        -B "$HOST_TOOLS/build" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$HOST_TOOLS/build" -j "$JOBS"
  cp "$(find "$HOST_TOOLS/build" -name flatc -type f -perm -u+x | head -1)" "$HOST_TOOLS/flatc"
else
  echo ">> 1/2 flatc nativo ya esta: $HOST_TOOLS/flatc"
fi

# --- 2. TFLite C API, cruzada -----------------------------------------------
# TF_MAJOR/MINOR/PATCH_VERSION las inyecta Bazel desde tf_version.bzl; el build
# de CMake no las define y release_version.h corta con #error. Se generan de
# ese mismo archivo y se fuerzan con -include, como en la build x86 (pasarlas
# con -D obliga a pelear con las comillas de TF_VERSION_SUFFIX).
TF_VER=$(sed -n 's/^TF_VERSION = "\([0-9.]*\)"/\1/p' "$TF_SRC/tensorflow/tf_version.bzl")
[ -n "$TF_VER" ] || { echo "ERROR: no pude leer TF_VERSION de tf_version.bzl"; exit 1; }
mkdir -p "$OUT"
MACROS="$OUT/tf_version_macros.h"
IFS=. read -r V_MAJ V_MIN V_PAT <<< "$TF_VER"
cat > "$MACROS" <<EOF
// Generado por build_tflite_aarch64.sh desde tf_version.bzl ($TF_VER).
#pragma once
#define TF_MAJOR_VERSION $V_MAJ
#define TF_MINOR_VERSION $V_MIN
#define TF_PATCH_VERSION $V_PAT
#define TF_VERSION_SUFFIX ""
EOF
# -D CMAKE_*_FLAGS pisa el CMAKE_*_FLAGS_INIT del toolchain: por eso el
# -mtune se repite aca.
FLAGS="-mtune=cortex-a53 -include $MACROS"

# Las mismas opciones que la build x86: Release + XNNPACK (que es el default).
echo ">> 2/2 libtensorflowlite_c.so aarch64 ($TF_VER) -> $OUT"
cmake -S "$TF_SRC/tensorflow/lite/c" -B "$OUT" \
      -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_FLAGS="$FLAGS" \
      -DCMAKE_CXX_FLAGS="$FLAGS" \
      -DTFLITE_ENABLE_XNNPACK=ON \
      -DTFLITE_HOST_TOOLS_DIR="$HOST_TOOLS"
cmake --build "$OUT" -j "$JOBS" --target tensorflowlite_c

echo "=================================================="
file "$OUT/libtensorflowlite_c.so"
ls -la "$OUT/libtensorflowlite_c.so"
echo "=================================================="
