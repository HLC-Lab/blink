cmake_minimum_required(VERSION 3.16)
project(blink C CXX)

# On HPC systems where CC/CXX are already MPI wrappers (mpicc/mpicxx),
# FindMPI's link-test often fails because LD_LIBRARY_PATH isn't set up for
# CMake's subprocess.  Pass -DBLINK_MPI_WRAPPER=ON to skip FindMPI and create
# stub INTERFACE targets instead — the wrapper already injects all flags.
option(BLINK_MPI_WRAPPER
    "Assume CC/CXX are MPI compiler wrappers; skip find_package(MPI)" OFF)

if(BLINK_MPI_WRAPPER)
    # Prefer $ENV{MPICC} (always correct on HPC modules) over CMAKE_C_COMPILER
    # (which may be the underlying gcc if the CMake cache is stale).
    if(DEFINED ENV{MPICC})
        set(_mpi_cc "$ENV{MPICC}")
    elseif(CMAKE_C_COMPILER MATCHES "mpi")
        set(_mpi_cc "${CMAKE_C_COMPILER}")
    else()
        find_program(_mpi_cc NAMES mpicc DOC "MPI C compiler wrapper")
    endif()
    if(NOT _mpi_cc)
        message(FATAL_ERROR
            "BLINK_MPI_WRAPPER=ON but no MPI wrapper found.\n"
            "Set the MPICC environment variable or put mpicc on PATH.")
    endif()
    message(STATUS "BLINK_MPI_WRAPPER=ON — querying '${_mpi_cc}' for MPI flags")
    execute_process(COMMAND "${_mpi_cc}" --showme:incdirs
                    OUTPUT_VARIABLE _mpi_incdirs OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    execute_process(COMMAND "${_mpi_cc}" --showme:libdirs
                    OUTPUT_VARIABLE _mpi_libdirs OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    execute_process(COMMAND "${_mpi_cc}" --showme:libs
                    OUTPUT_VARIABLE _mpi_libs    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(NOT _mpi_incdirs OR NOT _mpi_libdirs)
        message(FATAL_ERROR
            "MPI wrapper '${_mpi_cc}' did not return include/lib dirs.\n"
            "Check that it supports --showme:incdirs (OpenMPI wrapper required).")
    endif()
    string(REPLACE " " ";" _mpi_incdirs "${_mpi_incdirs}")
    string(REPLACE " " ";" _mpi_libdirs "${_mpi_libdirs}")
    string(REPLACE " " ";" _mpi_libs    "${_mpi_libs}")
    set(_mpi_lib_flags "")
    foreach(_l ${_mpi_libs})
        list(APPEND _mpi_lib_flags "-l${_l}")
    endforeach()
    message(STATUS "  includes : ${_mpi_incdirs}")
    message(STATUS "  libdirs  : ${_mpi_libdirs}")
    message(STATUS "  libs     : ${_mpi_lib_flags}")
    foreach(_tgt MPI::MPI_C MPI::MPI_CXX)
        if(NOT TARGET ${_tgt})
            add_library(${_tgt} INTERFACE IMPORTED)
        endif()
        set_target_properties(${_tgt} PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${_mpi_incdirs}"
            INTERFACE_LINK_DIRECTORIES    "${_mpi_libdirs}"
            INTERFACE_LINK_LIBRARIES      "${_mpi_lib_flags}")
    endforeach()
else()
    find_package(MPI REQUIRED COMPONENTS C CXX)
endif()

# All binaries land in <build>/bin/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Default to Release so -O3 is active unless the user overrides
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(BLINK_COVERAGE "Build with gcov coverage instrumentation (-O0 --coverage)" OFF)
if(BLINK_COVERAGE)
    # -O0 for accurate line attribution; atomic counter updates so the multiple
    # concurrent MPI ranks of a single benchmark binary don't corrupt the shared
    # .gcda files when they dump coverage on exit.
    add_compile_options(-O0 -g --coverage -fprofile-update=atomic)
    add_link_options(--coverage)
else()
    add_compile_options(-O3)
endif()
add_compile_definitions(_GNU_SOURCE)

# GPU backend selection.  AUTO probes CUDA, then HIP, then falls back to a
# host-only build.  Only the C runtime libraries are linked: no kernels exist,
# so neither nvcc nor hipcc is needed.
set(BLINK_GPU "AUTO" CACHE STRING "GPU backend: AUTO, OFF, CUDA or HIP")
set_property(CACHE BLINK_GPU PROPERTY STRINGS AUTO OFF CUDA HIP)
string(TOUPPER "${BLINK_GPU}" _blink_gpu_req)
if(NOT _blink_gpu_req MATCHES "^(AUTO|OFF|CUDA|HIP)$")
    message(FATAL_ERROR "BLINK_GPU must be AUTO, OFF, CUDA or HIP (got '${BLINK_GPU}')")
endif()
set(BLINK_GPU_BACKEND "NONE")
if(_blink_gpu_req STREQUAL "CUDA" AND CMAKE_VERSION VERSION_LESS 3.17)
    message(FATAL_ERROR "BLINK_GPU=CUDA needs CMake >= 3.17 (FindCUDAToolkit); this is ${CMAKE_VERSION}")
endif()
if((_blink_gpu_req STREQUAL "CUDA" OR _blink_gpu_req STREQUAL "AUTO") AND NOT CMAKE_VERSION VERSION_LESS 3.17)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(BLINK_GPU_BACKEND "CUDA")
    elseif(_blink_gpu_req STREQUAL "CUDA")
        message(FATAL_ERROR "BLINK_GPU=CUDA but no CUDA toolkit was found (set CUDAToolkit_ROOT)")
    endif()
endif()
if(BLINK_GPU_BACKEND STREQUAL "NONE" AND (_blink_gpu_req STREQUAL "HIP" OR _blink_gpu_req STREQUAL "AUTO"))
    find_package(hip QUIET HINTS /opt/rocm $ENV{ROCM_PATH})
    if(hip_FOUND)
        set(BLINK_GPU_BACKEND "HIP")
    elseif(_blink_gpu_req STREQUAL "HIP")
        message(FATAL_ERROR "BLINK_GPU=HIP but no ROCm/HIP was found (set CMAKE_PREFIX_PATH=/opt/rocm)")
    endif()
endif()
message(STATUS "BLINK_GPU: requested ${BLINK_GPU}, using ${BLINK_GPU_BACKEND}")

# The GPU tests need a GPU and a GPU-aware MPI.  A machine that merely has a toolkit installed (AUTO) must keep a
# green plain `ctest`, so they are opt-in there and on by default only when CUDA/HIP was requested explicitly.
if(_blink_gpu_req STREQUAL "CUDA" OR _blink_gpu_req STREQUAL "HIP")
    set(_blink_gpu_tests_default ON)
else()
    set(_blink_gpu_tests_default OFF)
endif()
option(BLINK_GPU_TESTS "Register the GPU tests (_gpu twins, test_gpu_layer); needs a GPU and a GPU-aware MPI"
       ${_blink_gpu_tests_default})

add_subdirectory(src)

option(BLINK_TESTS "Build correctness tests (requires GTest)" ON)

if(BLINK_TESTS)
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        message(STATUS "GTest not found via find_package — fetching with FetchContent")
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.14.0
        )
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
        if(NOT TARGET GTest::GTest)
            add_library(GTest::GTest ALIAS gtest)
            add_library(GTest::Main  ALIAS gtest_main)
        endif()
        set(GTest_FOUND TRUE)
    endif()

    if(GTest_FOUND)
        enable_testing()
        add_subdirectory(tests)
        message(STATUS "GTest ready — correctness tests enabled (run: ctest --test-dir build)")
    else()
        message(WARNING "GTest unavailable — correctness tests skipped (pass -DBLINK_TESTS=OFF to silence)")
    endif()
else()
    message(STATUS "BLINK_TESTS=OFF — correctness tests skipped")
endif()
