cmake_minimum_required(VERSION 3.16)

project(ofb CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(ofb_offline tools/ofb_offline.cpp)

target_include_directories(ofb_offline PRIVATE .)