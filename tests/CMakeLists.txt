# ── Black-box benchmark test ──────────────────────────────────────────────────
# A regular single-process GTest that spawns mpirun <benchmark> -debug via
# popen() (see popen_helpers.h) and parses the structured debug output.  The
# benchmark paths and mpiexec command are baked in at configure time
# (overridable via env vars).
#
# `nprocs` arg is forwarded to ctest's PROCESSORS property so `ctest -j`
# accounts for each suite's actual MPI process count and avoids
# oversubscribing the host.
function(add_benchmark_test name nprocs)
    add_executable(${name} ${name}.cpp)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(${name} PRIVATE
        "BLINK_BIN_DIR=\"${CMAKE_BINARY_DIR}/bin\""
        "BLINK_MPIEXEC=\"${MPIEXEC_EXECUTABLE}\""
        "BLINK_NPROC_FLAG=\"${MPIEXEC_NUMPROC_FLAG}\"")
    target_link_libraries(${name} PRIVATE GTest::GTest GTest::Main)
    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES PROCESSORS ${nprocs})
endfunction()

# ── Registered tests ──────────────────────────────────────────────────────────
# nprocs is the MAXIMUM rank count any single test_*.cpp launches via mpirun.
add_benchmark_test(test_collectives 8)
add_benchmark_test(test_pairwise    8)
add_benchmark_test(test_ring        8)
add_benchmark_test(test_incast      8)
add_benchmark_test(test_pingpong    8)
add_benchmark_test(test_kpartners   8)
add_benchmark_test(test_stencil     8)
add_benchmark_test(test_tagmatch    4)
add_benchmark_test(test_misc        8)

# ── GPU wrapper unit test (GPU builds only) ───────────────────────────────────
if(NOT BLINK_GPU_BACKEND STREQUAL "NONE")
    add_executable(test_gpu_layer test_gpu_layer.cpp)
    target_include_directories(test_gpu_layer PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_gpu_layer PRIVATE GTest::GTest GTest::Main blink_common)
    set_target_properties(test_gpu_layer PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    add_test(NAME test_gpu_layer COMMAND test_gpu_layer)
    set_tests_properties(test_gpu_layer PROPERTIES RESOURCE_LOCK gpu)
endif()
