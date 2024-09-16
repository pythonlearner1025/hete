cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(PokerProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/home/minjune/libtorch")
find_package(Torch REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/OMPEval)
link_directories(${CMAKE_SOURCE_DIR}/OMPEval/lib)

find_library(OMPEVAL_LIB ompeval PATHS ${CMAKE_SOURCE_DIR}/OMPEval/lib NO_DEFAULT_PATH)
if(NOT OMPEVAL_LIB)
  message(FATAL_ERROR "OMPEval library not found in ${CMAKE_SOURCE_DIR}/OMPEval/lib")
endif()

# Check the system type
if(UNIX AND NOT APPLE)
  # Linux
  find_package(OpenMP REQUIRED)
  set(OPENMP_LIB OpenMP::OpenMP_CXX)
elseif(APPLE)
  # macOS
  set(OPENMP_INCLUDE_DIR /opt/homebrew/opt/libomp/include)
  set(OPENMP_LIB_DIR /opt/homebrew/opt/libomp/lib)
  include_directories(${OPENMP_INCLUDE_DIR})
  link_directories(${OPENMP_LIB_DIR})
  if(NOT EXISTS "${OPENMP_LIB_DIR}/libomp.dylib")
    message(FATAL_ERROR "libomp.dylib not found in ${OPENMP_LIB_DIR}")
  endif()
  set(OPENMP_LIB "${OPENMP_LIB_DIR}/libomp.dylib")
else()
  message(FATAL_ERROR "Unsupported operating system")
endif()

file(GLOB_RECURSE NON_MODEL_SRCS "${CMAKE_SOURCE_DIR}/src/*.cpp" "${CMAKE_SOURCE_DIR}/src/*.cxx")
file(GLOB_RECURSE MODEL_SRCS "${CMAKE_SOURCE_DIR}/src/model/*.cpp" "${CMAKE_SOURCE_DIR}/src/model/*.cxx")
list(APPEND MODEL_SRCS "${CMAKE_SOURCE_DIR}/src/model/model.h")
set(ALL_SRCS ${NON_MODEL_SRCS} ${MODEL_SRCS})

add_executable(main ${ALL_SRCS})

target_link_libraries(main PRIVATE ${OMPEVAL_LIB} "${TORCH_LIBRARIES}" ${OPENMP_LIB})
target_include_directories(main PRIVATE ${TORCH_INCLUDE_DIRS})
target_compile_options(main PRIVATE "${TORCH_CXX_FLAGS}")

if(APPLE)
  set_target_properties(main PROPERTIES
      BUILD_RPATH "${OPENMP_LIB_DIR};/Users/minjunes/libtorch/lib"
      INSTALL_RPATH "${OPENMP_LIB_DIR};/Users/minjunes/libtorch/lib"
      CXX_VISIBILITY_PRESET hidden
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
  )
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)