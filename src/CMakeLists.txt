# Auto-discover every .c and .cpp file under src/ and build one executable per file.
# CONFIGURE_DEPENDS re-triggers configure when files are added or removed.
# common.h is in this directory (src/) and is found via the include path below.
# All executable names are derived from the source filename stem, e.g.
#   alltoall/alltoall_nb.c  ->  bin/alltoall_nb

# The shared runtime (common.c) is compiled once into a static library and
# linked into every benchmark — cleaner and faster than re-instantiating a
# header of static-inline functions in each TU, and it yields a single .gcda so
# gcov coverage of the shared code is counted accurately.  MPI/m propagate to
# consumers via PUBLIC.  C++ benchmarks link it too (common.h uses extern "C").
add_library(blink_common STATIC common.c gpu.c gpu_mode.c)
target_include_directories(blink_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(blink_common PUBLIC MPI::MPI_C m)

# GPU backend (selected in the top-level CMakeLists.txt).  gpu.c is the only
# file that includes the vendor runtime headers; no kernels, so plain C compilers suffice.
if(BLINK_GPU_BACKEND STREQUAL "CUDA")
    target_compile_definitions(blink_common PUBLIC BLINK_GPU_CUDA)
    target_link_libraries(blink_common PUBLIC CUDA::cudart)
elseif(BLINK_GPU_BACKEND STREQUAL "HIP")
    target_compile_definitions(blink_common PUBLIC BLINK_GPU_HIP)
    target_link_libraries(blink_common PUBLIC hip::host)
endif()

file(GLOB_RECURSE c_srcs   CONFIGURE_DEPENDS "*.c")
file(GLOB_RECURSE cpp_srcs CONFIGURE_DEPENDS "*.cpp")
# common.c, gpu.c and gpu_mode.c are the library, not benchmarks — don't build an executable from it.
list(REMOVE_ITEM c_srcs "${CMAKE_CURRENT_SOURCE_DIR}/common.c"
                        "${CMAKE_CURRENT_SOURCE_DIR}/gpu.c"
                        "${CMAKE_CURRENT_SOURCE_DIR}/gpu_mode.c")

foreach(src IN LISTS c_srcs cpp_srcs)
    get_filename_component(name ${src} NAME_WE)
    get_filename_component(ext  ${src} EXT)

    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${name} PRIVATE blink_common)

    if(ext STREQUAL ".c")
        target_link_libraries(${name} PRIVATE MPI::MPI_C m)
    else()
        target_link_libraries(${name} PRIVATE MPI::MPI_CXX m)
    endif()
endforeach()
