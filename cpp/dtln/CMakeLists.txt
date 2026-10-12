cmake_minimum_required(VERSION 3.16)

project( dtln CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable( dtln_stage_a tools/dtln_test_a.cpp)



target_include_directories(dtln_stage_a PRIVATE
    .                                # para <dtln/config.hpp>, tus headers
    /home/matias/src/tensorflow)     # para <tensorflow/lite/c/c_api.h>

target_link_directories(dtln_stage_a PRIVATE
    /home/matias/src/tflite_build)   # el DIRECTORIO, no el .so

target_link_libraries(dtln_stage_a PRIVATE
    tensorflowlite_c)                # sin "lib", sin ".so"

set_target_properties(dtln_stage_a PROPERTIES
    BUILD_RPATH /home/matias/src/tflite_build)