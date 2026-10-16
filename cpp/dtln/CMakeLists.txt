cmake_minimum_required(VERSION 3.16)

project( dtln C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable( dtln_test_a 
                  tools/dtln_test_a.cpp
                  ../third_party/kissfft/kiss_fft.c
                  ../third_party/kissfft/kiss_fftr.c
)

target_include_directories(dtln_test_a PRIVATE
    ..                               # raiz de includes: para <dtln/config.hpp>
    /home/matias/src/tensorflow
    ../third_party/kissfft
    ../third_party/kissfft/tools)     # para <tensorflow/lite/c/c_api.h>

target_link_directories(dtln_test_a PRIVATE
    /home/matias/src/tflite_build)   # el DIRECTORIO, no el .so

target_link_libraries(dtln_test_a PRIVATE
    tensorflowlite_c)                # sin "lib", sin ".so"

set_target_properties(dtln_test_a PROPERTIES
    BUILD_RPATH /home/matias/src/tflite_build)

  



add_executable( dtln_stage_a
                  tools/dtln_stage_a.cpp
                  ../third_party/kissfft/kiss_fft.c
                  ../third_party/kissfft/kiss_fftr.c
)
target_include_directories(dtln_stage_a PRIVATE .. /home/matias/src/tensorflow ../third_party/kissfft)
target_link_directories(dtln_stage_a PRIVATE /home/matias/src/tflite_build)
target_link_libraries(dtln_stage_a PRIVATE tensorflowlite_c)
set_target_properties(dtln_stage_a PROPERTIES BUILD_RPATH /home/matias/src/tflite_build)
