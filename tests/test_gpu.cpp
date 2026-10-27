/* Black-box tests of the -gpu runtime (src/gpu_mode.c). */
#include <gtest/gtest.h>
#include <string>
#include "popen_helpers.h"

using blink::DebugRun;
using blink::run_debug_capture;

#if BLINK_GPU_BUILD && BLINK_GPU_TESTS
#include "gpu.h"
static int gpu_device_count_for_test() { return gpu_device_count(); }
#endif

static bool contains(const std::string &s, const char *needle) { return s.find(needle) != std::string::npos; }

/* A Buffer-less executable accepts -gpu, runs, and warns once (any build). */
TEST(GpuRuntime, BufferlessExecutableWarnsAndRuns) {
    DebugRun r = run_debug_capture("barrier_b", 2, "-gpu");
    EXPECT_TRUE(r.normal_exit);
    EXPECT_EQ(r.exit_code, 0) << r.stderr_raw;
    EXPECT_TRUE(contains(r.stderr_raw, "has no effect")) << r.stderr_raw;
    EXPECT_FALSE(contains(r.stdout_raw, "has no effect"));
}

/* The Tools null_dummy and dist_test ignore their arguments (CRAB documents null_dummy that way);
 * -gpu support must not make them parse, validate or reject the common flags. */
TEST(GpuRuntime, ToolsStillIgnoreTheirArguments) {
    for (const char *bin : {"null_dummy", "dist_test"}) {
        DebugRun r = run_debug_capture(bin, 2, "-grty 0 -mrank 7 -maxsamples 0");
        EXPECT_TRUE(r.normal_exit) << bin;
        EXPECT_EQ(r.exit_code, 0) << bin << ": " << r.stderr_raw;
    }
}

TEST(GpuRuntime, ToolsWarnThatGpuHasNoEffect) {
    for (const char *bin : {"null_dummy", "dist_test"}) {
        DebugRun r = run_debug_capture(bin, 2, "-gpu");
        EXPECT_EQ(r.exit_code, 0) << bin << ": " << r.stderr_raw;
        EXPECT_TRUE(contains(r.stderr_raw, "has no effect")) << bin << ": " << r.stderr_raw;
    }
}

TEST(GpuRuntime, HelpDocumentsFlag) {
    DebugRun r = run_debug_capture("alltoall_b", 1, "-h");
    EXPECT_TRUE(contains(r.stdout_raw, "-gpu")) << r.stdout_raw;
}

#if BLINK_GPU_BUILD && BLINK_GPU_TESTS
TEST(GpuRuntime, EachRankReportsGpuAndDevice) {
    DebugRun r = run_debug_capture("alltoall_b", 4, "-gpu");
    ASSERT_EQ(r.exit_code, 0) << r.stderr_raw;
    ASSERT_EQ((int)r.ranks.size(), 4);
    for (int k = 0; k < 4; k++) {
        EXPECT_EQ(blink::get_int(r.ranks, k, "gpu"), 1);
        EXPECT_GE(blink::get_int(r.ranks, k, "device"), 0);
        EXPECT_EQ(blink::get_str(r.ranks, k, "check"), "OK");
        /* send + recv buffer were handed to MPI as device mirrors */
        EXPECT_EQ(blink::get_int(r.ranks, k, "mirrors"), 2) << "alltoall_b rank " << k;
    }
}

/* Two ranks on this one-GPU machine really share a device: warned once, on stderr. */
TEST(GpuRuntime, RanksSharingAGpuAreWarnedOnce) {
    if (gpu_device_count_for_test() != 1) GTEST_SKIP() << "needs exactly one visible GPU";
    DebugRun r = run_debug_capture("alltoall_b", 2, "-gpu");
    ASSERT_EQ(r.exit_code, 0) << r.stderr_raw;
    size_t first = r.stderr_raw.find("share a GPU");
    EXPECT_NE(first, std::string::npos) << r.stderr_raw;
    EXPECT_EQ(r.stderr_raw.find("share a GPU", first + 1), std::string::npos) << r.stderr_raw;
}

TEST(GpuRuntime, ZeroByteMessagesWork) {
    DebugRun r = run_debug_capture("alltoall_b", 2, "-gpu -msgsize 0");
    EXPECT_EQ(r.exit_code, 0) << r.stderr_raw;
}

/* One-sided on device memory: either it works (all checks OK) or the run aborts
 * with an explicit message.  Silent wrong data / exit 0 with FAIL is the bug. */
TEST(GpuRuntime, OneSidedIsRejectedExplicitlyOrCorrect) {
    for (const char *bin : {"incast_put", "incast_get"}) {
        DebugRun r = run_debug_capture(bin, 4, "-gpu");
        if (r.normal_exit && r.exit_code == 0) {
            blink::check_all_ok(r.ranks, bin);
        } else {
            EXPECT_TRUE(contains(r.stderr_raw, "not supported by this MPI")) << bin << ": " << r.stderr_raw;
        }
    }
}

/* A non-GPU-aware MPI dereferences the device pointer on the CPU and segfaults.  Force that with Open MPI's
 * host-only components: the run must end with an explicit message, never a bare crash. */
TEST(GpuRuntime, NonGpuAwareMpiIsRejectedExplicitly) {
    setenv("OMPI_MCA_pml", "ob1", 1);
    setenv("OMPI_MCA_btl", "self,sm", 1);
    setenv("OMPI_MCA_accelerator", "null", 1);
    DebugRun r = run_debug_capture("alltoall_b", 2, "-gpu");
    unsetenv("OMPI_MCA_pml"); unsetenv("OMPI_MCA_btl"); unsetenv("OMPI_MCA_accelerator");
    if (r.normal_exit && r.exit_code == 0) {
        blink::check_all_ok(r.ranks, "alltoall_b");
    } else {
        EXPECT_TRUE(contains(r.stderr_raw, "not GPU-aware")) << r.stderr_raw;
    }
}

/* Nonblocking reductions (libnbc reduces on the CPU) crash on device pointers in some MPIs:
 * either they work or the run aborts with an explicit message, never a bare segfault. */
TEST(GpuRuntime, NonblockingReductionsAreRejectedExplicitlyOrCorrect) {
    for (const char *bin : {"allreduce_nb", "reduce_nb", "reduce_scatter_nb"}) {
        DebugRun r = run_debug_capture(bin, 4, "-gpu");
        if (r.normal_exit && r.exit_code == 0) {
            blink::check_all_ok(r.ranks, bin);
        } else {
            EXPECT_TRUE(contains(r.stderr_raw, "not supported by this MPI")) << bin << ": " << r.stderr_raw;
        }
    }
}
#elif !BLINK_GPU_BUILD
TEST(GpuRuntime, GpuFlagOnHostOnlyBuildFailsClearly) {
    DebugRun r = run_debug_capture("alltoall_b", 2, "-gpu");
    EXPECT_NE(r.exit_code, 0);
    EXPECT_TRUE(contains(r.stderr_raw, "built without GPU support")) << r.stderr_raw;
}
#endif
