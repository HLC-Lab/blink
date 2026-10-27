/* Black-box tests of the -gpu runtime (src/gpu_mode.c). */
#include <gtest/gtest.h>
#include <string>
#include "popen_helpers.h"

using blink::DebugRun;
using blink::run_debug_capture;

static bool contains(const std::string &s, const char *needle) { return s.find(needle) != std::string::npos; }

/* A Buffer-less executable accepts -gpu, runs, and warns once (any build). */
TEST(GpuRuntime, BufferlessExecutableWarnsAndRuns) {
    DebugRun r = run_debug_capture("barrier_b", 2, "-gpu");
    EXPECT_TRUE(r.normal_exit);
    EXPECT_EQ(r.exit_code, 0) << r.stderr_raw;
    EXPECT_TRUE(contains(r.stderr_raw, "has no effect")) << r.stderr_raw;
    EXPECT_FALSE(contains(r.stdout_raw, "has no effect"));
}

TEST(GpuRuntime, HelpDocumentsFlag) {
    DebugRun r = run_debug_capture("alltoall_b", 1, "-h");
    EXPECT_TRUE(contains(r.stdout_raw, "-gpu")) << r.stdout_raw;
}

#if BLINK_GPU_BUILD
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
#else
TEST(GpuRuntime, GpuFlagOnHostOnlyBuildFailsClearly) {
    DebugRun r = run_debug_capture("alltoall_b", 2, "-gpu");
    EXPECT_NE(r.exit_code, 0);
    EXPECT_TRUE(contains(r.stderr_raw, "built without GPU support")) << r.stderr_raw;
}
#endif
