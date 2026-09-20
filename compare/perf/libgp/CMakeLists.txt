cmake_minimum_required(VERSION 3.16)
project(libgp-perf LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# Header-only. Do not add_subdirectory(Eigen): its CMake looks for Fortran / BLAS.
FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    SOURCE_SUBDIR nonexistent
)
FetchContent_MakeAvailable(eigen)
add_library(Eigen3::Eigen INTERFACE IMPORTED)
target_include_directories(Eigen3::Eigen INTERFACE "${eigen_SOURCE_DIR}")

# SOURCE_SUBDIR skips libgp's CMake (it calls find_package(Eigen3) and
# optionally builds Python bindings). We compile the C++ sources ourselves.
FetchContent_Declare(
    libgp
    GIT_REPOSITORY https://github.com/mblum/libgp.git
    GIT_TAG f4a2fb7d4e1de7de8f1555318f14750830913663
    SOURCE_SUBDIR nonexistent
)
FetchContent_MakeAvailable(libgp)

FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/gp_version.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/gp_version.h"
    COPYONLY
)

file(GLOB LIBGP_SOURCES CONFIGURE_DEPENDS "${libgp_SOURCE_DIR}/src/*.cc")
list(FILTER LIBGP_SOURCES EXCLUDE REGEX "bindings\\.cpp$")

# Upstream Windows shim is a non-static Utils::drand48; gp_utils.cc calls it
# without an object. Mark it static once after the fetch.
set(LIBGP_UTILS_H "${libgp_SOURCE_DIR}/include/gp_utils.h")
file(READ "${LIBGP_UTILS_H}" LIBGP_UTILS_H_TEXT)
if(NOT LIBGP_UTILS_H_TEXT MATCHES "static inline double drand48")
    string(REPLACE
        "inline double drand48()"
        "static inline double drand48()"
        LIBGP_UTILS_H_TEXT
        "${LIBGP_UTILS_H_TEXT}"
    )
    file(WRITE "${LIBGP_UTILS_H}" "${LIBGP_UTILS_H_TEXT}")
endif()

add_library(gp STATIC ${LIBGP_SOURCES})
target_include_directories(gp PUBLIC
    "${CMAKE_CURRENT_BINARY_DIR}/include"
    "${libgp_SOURCE_DIR}/include"
)
target_link_libraries(gp PUBLIC Eigen3::Eigen)
if(MSVC)
    target_compile_options(gp PRIVATE "/FI${CMAKE_CURRENT_SOURCE_DIR}/msvc_compat.h")
endif()

add_executable(libgp-perf src/main.cpp)
target_link_libraries(libgp-perf PRIVATE gp nlohmann_json::nlohmann_json)
if(WIN32)
    target_link_libraries(libgp-perf PRIVATE psapi)
endif()

add_executable(libgp-online src/online.cpp)
target_link_libraries(libgp-online PRIVATE gp nlohmann_json::nlohmann_json)
if(WIN32)
    target_link_libraries(libgp-online PRIVATE psapi)
endif()
