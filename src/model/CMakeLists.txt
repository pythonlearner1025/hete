cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(DeepCFRModel)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set the path to LibTorch
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/Users/minjunes/libtorch")

# Find LibTorch
find_package(Torch REQUIRED)

add_executable(model model.cpp)
target_link_libraries(model "${TORCH_LIBRARIES}")
set_property(TARGET model PROPERTY CXX_STANDARD 17)
