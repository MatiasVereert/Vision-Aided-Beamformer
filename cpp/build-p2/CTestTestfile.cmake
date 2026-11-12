# CMake generated Testfile for 
# Source directory: /home/matias/Documents/Tesis/Vision-Aided-Beamformer/cpp
# Build directory: /home/matias/Documents/Tesis/Vision-Aided-Beamformer/cpp/build-p2
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(core "/home/matias/Documents/Tesis/Vision-Aided-Beamformer/cpp/build-p2/test_core" "/home/matias/Documents/Tesis/Vision-Aided-Beamformer/cpp/../tests/cpp_parity/data")
set_tests_properties(core PROPERTIES  _BACKTRACE_TRIPLES "/home/matias/Documents/Tesis/Vision-Aided-Beamformer/cpp/CMakeLists.txt;80;add_test;/home/matias/Documents/Tesis/Vision-Aided-Beamformer/cpp/CMakeLists.txt;0;")
add_test(ofb "/home/matias/Documents/Tesis/Vision-Aided-Beamformer/cpp/build-p2/test_ofb" "/home/matias/Documents/Tesis/Vision-Aided-Beamformer/cpp/../src/dnn_denoise/models/model_quant_1.tflite" "/home/matias/Documents/Tesis/Vision-Aided-Beamformer/cpp/../tests/cpp_parity/data")
set_tests_properties(ofb PROPERTIES  _BACKTRACE_TRIPLES "/home/matias/Documents/Tesis/Vision-Aided-Beamformer/cpp/CMakeLists.txt;81;add_test;/home/matias/Documents/Tesis/Vision-Aided-Beamformer/cpp/CMakeLists.txt;0;")
