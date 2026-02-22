cmake_minimum_required(VERSION 3.27)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(tileon CXX C)

option(BUILD_BINARY "Build C++ binaries" OFF)
option(BUILD_WITH_CCACHE "Build with ccache (if available)" ON)
option(BUILD_WITH_LTO "Build with link-time optimization" ON)

if(BUILD_WITH_CCACHE)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "C compiler launcher")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "CXX compiler launcher")
  else()
    message(STATUS "Could not find ccache. Consider installing ccache to speed up compilation.")
  endif()
endif()

set(TILEON_BUILD_TYPE ${CMAKE_BUILD_TYPE})

if(NOT TILEON_BUILD_TYPE)
    set(TILEON_BUILD_TYPE "Release")
endif()

set(TILEON_PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/python/tileon")

find_package(Python REQUIRED COMPONENTS Development.Module)
find_program(
    PYBIND11_STUBGEN_EXECUTABLE
    NAMES pybind11-stubgen
    REQUIRED
)
add_subdirectory(third_party/pybind11)

set(PYTHON_MODULE_NAME "_C")
pybind11_add_module(${PYTHON_MODULE_NAME}
  python/src/init.cpp
  python/src/env.cpp
  python/src/ir.cpp
  python/src/interpreter.cpp
  python/src/specialize.cpp
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_C PRIVATE OpenMP::OpenMP_CXX)
endif()

set_target_properties(_C PROPERTIES UNITY_BUILD OFF)

set(STUB_NUMPY_ARRAY_MODE "use-type-var")
set(STUB_EXTENSION "pyi")

set(STUBGEN_ARGS "")
list(APPEND STUBGEN_ARGS "-o" "${TILEON_PACKAGE_DIR}")

if(STUB_NUMPY_ARRAY_MODE STREQUAL "wrap-with-annotated")
    list(APPEND STUBGEN_ARGS "--numpy-array-wrap-with-annotated")
elseif(STUB_NUMPY_ARRAY_MODE STREQUAL "use-type-var")
    list(APPEND STUBGEN_ARGS "--numpy-array-use-type-var")
elseif(STUB_NUMPY_ARRAY_MODE STREQUAL "remove-parameters")
    list(APPEND STUBGEN_ARGS "--numpy-array-remove-parameters")
endif()

if(STUB_EXTENSION)
    list(APPEND STUBGEN_ARGS "--stub-extension" "${STUB_EXTENSION}")
endif()

list(APPEND STUBGEN_ARGS "${PYTHON_MODULE_NAME}")

add_custom_command(
    OUTPUT "${TILEON_PACKAGE_DIR}/${PYTHON_MODULE_NAME}.pyi"
    COMMAND ${Python_EXECUTABLE} -m pybind11_stubgen ${STUBGEN_ARGS}
    COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/inject_numpy.py" "${TILEON_PACKAGE_DIR}/${PYTHON_MODULE_NAME}"
    DEPENDS ${PYTHON_MODULE_NAME}
    WORKING_DIRECTORY $<TARGET_FILE_DIR:${PYTHON_MODULE_NAME}>
    COMMENT "Generating stubs for ${PYTHON_MODULE_NAME} using pybind11-stubgen"
)

add_custom_target(
    generate_stubs
    ALL
    DEPENDS "${TILEON_PACKAGE_DIR}/${PYTHON_MODULE_NAME}.pyi"
)

install(TARGETS _C DESTINATION "${TILEON_PACKAGE_DIR}")
