# Toolchain para compilar EN LA PC y correr EN LA KRIA (Cortex-A53, aarch64).
#
#   cmake -S cpp -B cpp/build-aarch64 \
#         -DCMAKE_TOOLCHAIN_FILE=cpp/cmake/aarch64-linux-gnu.cmake \
#         -DTFLITE_BUILD=$HOME/src/tflite_build_aarch64 \
#         -DOFB_BLOCK_UPD=2
#
# Usa el g++-aarch64-linux-gnu de apt. Funciona SIN sysroot porque la PC y la
# Kria son las dos Ubuntu 24.04: la glibc del cruzado (2.39) es la misma que la
# de la placa. Si alguna de las dos cambia de version, un binario armado aca
# puede pedir simbolos GLIBC_2.xx que la placa no tiene.
set(CMAKE_SYSTEM_NAME      Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# -mtune y NO -mcpu: XNNPACK compila cada familia de microkernels con su propio
# -march (armv8.2-a+dotprod, +fp16...) y elige en tiempo de ejecucion; un
# -mcpu=cortex-a53 global choca con esos -march.
set(CMAKE_C_FLAGS_INIT   "-mtune=cortex-a53")
set(CMAKE_CXX_FLAGS_INIT "-mtune=cortex-a53")

# Bibliotecas y headers SOLO del lado aarch64; los programas, del host.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
