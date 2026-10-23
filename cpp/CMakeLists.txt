cmake_minimum_required(VERSION 3.16)

# C ademas de CXX: los dos archivos de KissFFT son C.
project(ofb C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# --- TFLite (LiteRT), API de C --------------------------------------------
# Rutas como CACHE PATH: se cambian con -DTFLITE_BUILD=... sin editar el archivo,
# que es lo que vas a necesitar cuando compiles para aarch64.
set(TFLITE_SRC   "$ENV{HOME}/src/tensorflow"   CACHE PATH "raiz del repo de tensorflow")
set(TFLITE_BUILD "$ENV{HOME}/src/tflite_build" CACHE PATH "donde quedo libtensorflowlite_c.so")

add_library(tflite_c SHARED IMPORTED)
set_target_properties(tflite_c PROPERTIES
    IMPORTED_LOCATION             "${TFLITE_BUILD}/libtensorflowlite_c.so"
    INTERFACE_INCLUDE_DIRECTORIES "${TFLITE_SRC}")

# --- KissFFT: fuente vendorizado, se compila con el proyecto ---------------
# Target propio para (a) compilarlo una vez y (b) no aplicarle -Wall -Wextra,
# que es codigo de terceros y tiene warnings que no son tuyos.
add_library(kissfft STATIC
    third_party/kissfft/kiss_fft.c
    third_party/kissfft/kiss_fftr.c)
target_include_directories(kissfft PUBLIC third_party/kissfft)

# --- flags de aviso, solo para TU codigo -----------------------------------
add_library(ofb_warnings INTERFACE)
target_compile_options(ofb_warnings INTERFACE -Wall -Wextra)

# --- tools/: produce algo que MIRAS ----------------------------------------
add_executable(ofb_offline tools/ofb_offline.cpp)
target_include_directories(ofb_offline PRIVATE .)
target_link_libraries(ofb_offline PRIVATE ofb_warnings)

# --- tests/: producen un VEREDICTO (exit 0 / 1) ----------------------------
# test_core no necesita NI TFLite NI KissFFT: solve.hpp es aritmetica pura.
# Esa independencia es deliberada -- un fallo de la biblioteca no te impide
# probar el nucleo.
add_executable(test_core tests/test_core.cpp)
target_include_directories(test_core PRIVATE .)
target_link_libraries(test_core PRIVATE ofb_warnings)

add_executable(test_ofb tests/test_ofb.cpp)
target_include_directories(test_ofb PRIVATE .)
target_link_libraries(test_ofb PRIVATE kissfft tflite_c ofb_warnings)
# Sin esto enlaza bien y FALLA AL ARRANCAR: el cargador dinamico no busca
# en TFLITE_BUILD.
set_target_properties(test_ofb PROPERTIES BUILD_RPATH "${TFLITE_BUILD}")

# --- ctest ------------------------------------------------------------------
enable_testing()
set(PARITY_DATA "${CMAKE_CURRENT_SOURCE_DIR}/../tests/cpp_parity/data")
set(DTLN_MODEL  "${CMAKE_CURRENT_SOURCE_DIR}/../src/dnn_denoise/models/model_quant_1.tflite")

add_test(NAME core COMMAND test_core "${PARITY_DATA}")
add_test(NAME ofb  COMMAND test_ofb  "${DTLN_MODEL}" "${PARITY_DATA}")
