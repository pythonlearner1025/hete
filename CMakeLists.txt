# CMakeLists.txt at project_root/

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(PokerProject)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set build type to Release for optimization by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Find LibTorch
# Ensure that CMAKE_PREFIX_PATH includes the path to LibTorch
# You can set it externally when invoking CMake:
# cmake -DCMAKE_PREFIX_PATH=/Users/minjunes/libtorch ..
find_package(Torch REQUIRED)

# Include directories for OMPEval
# Correct the include path to point to OMPEval (parent of omp/)
include_directories(${CMAKE_SOURCE_DIR}/OMPEval)

# Link directories for OMPEval libraries
link_directories(${CMAKE_SOURCE_DIR}/OMPEval/lib)

# Find OMPEval library
# Assuming the library file is named libompeval.a or libompeval.dylib
find_library(OMPEVAL_LIB ompeval PATHS ${CMAKE_SOURCE_DIR}/OMPEval/lib NO_DEFAULT_PATH)

if(NOT OMPEVAL_LIB)
  message(FATAL_ERROR "OMPEval library not found in ${CMAKE_SOURCE_DIR}/OMPEval/lib")
endif()

# Manually specify OpenMP include and lib paths based on Homebrew installation
# Update these paths if your Homebrew installation is in a different prefix
set(OPENMP_INCLUDE_DIR /opt/homebrew/opt/libomp/include)
set(OPENMP_LIB_DIR /opt/homebrew/opt/libomp/lib)

# Add OpenMP include directories
include_directories(${OPENMP_INCLUDE_DIR})

# Link directories for OpenMP
link_directories(${OPENMP_LIB_DIR})

# Ensure that libomp.dylib exists
if(NOT EXISTS "${OPENMP_LIB_DIR}/libomp.dylib")
  message(FATAL_ERROR "libomp.dylib not found in ${OPENMP_LIB_DIR}")
endif()

# Gather non-model source files (exclude model directory)
file(GLOB_RECURSE NON_MODEL_SRCS "${CMAKE_SOURCE_DIR}/src/*.cpp" "${CMAKE_SOURCE_DIR}/src/*.cxx")

# Gather model source files
file(GLOB_RECURSE MODEL_SRCS "${CMAKE_SOURCE_DIR}/src/model/*.cpp" "${CMAKE_SOURCE_DIR}/src/model/*.cxx")

# Combine all source files
set(ALL_SRCS ${NON_MODEL_SRCS} ${MODEL_SRCS})

# Add executable named 'main'
add_executable(main ${ALL_SRCS})

# Link with OMPEval, LibTorch, and OpenMP
# Use the absolute path to libomp.dylib to avoid confusion
target_link_libraries(main PRIVATE ${OMPEVAL_LIB} "${TORCH_LIBRARIES}" "${OPENMP_LIB_DIR}/libomp.dylib")

# Include LibTorch headers
target_include_directories(main PRIVATE ${TORCH_INCLUDE_DIRS})

# Add LibTorch compile options
target_compile_options(main PRIVATE "${TORCH_CXX_FLAGS}")

# Set RPATH to include OpenMP and LibTorch libraries
# Ensure OpenMP's lib directory is searched first
set_target_properties(main PROPERTIES
    BUILD_RPATH "/opt/homebrew/opt/libomp/lib;/Users/minjunes/libtorch/lib"
    INSTALL_RPATH "/opt/homebrew/opt/libomp/lib;/Users/minjunes/libtorch/lib"
    CXX_VISIBILITY_PRESET hidden
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Specify runtime output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
