cmake_minimum_required(VERSION 3.20)
project(NeuralNet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# default to an optimized build - training is far too slow without it
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Off by default for now: the current pragmas spawn 16 threads even for tiny layers,
# which makes small networks much slower (see the OpenMP cleanup step in the plan).
option(NN_USE_OPENMP "Build with OpenMP multithreading" OFF)

# link-time optimization in Release, so moving function bodies into .cpp files costs no speed
include(CheckIPOSupported)
check_ipo_supported(RESULT NN_IPO_SUPPORTED OUTPUT NN_IPO_ERROR)

# ---- library: everything except main (header-only for now, gets .cpp files as code moves out of headers)
add_library(nn INTERFACE)
target_include_directories(nn INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(nn INTERFACE -Wall -Wextra -Wpedantic)
if(NN_USE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(nn INTERFACE OpenMP::OpenMP_CXX)
else()
    target_compile_options(nn INTERFACE -Wno-unknown-pragmas)   # the omp pragmas are expected to be ignored
endif()

# ---- executable
add_executable(network src/neural_net.cpp)
target_link_libraries(network PRIVATE nn)
# TEMPORARY: `./network test` relies on assert(), which Release's -DNDEBUG would disable.
# Remove once the tests move to GoogleTest.
target_compile_options(network PRIVATE -UNDEBUG)
if(NN_IPO_SUPPORTED)
    set_property(TARGET network PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()
