/*
 * test_large.cpp — opt-in large-message regression tests (32-bit offset overflow).
 *
 * Each case runs one benchmark with the smallest configuration whose buffer
 * offset passes 2^31 bytes and checks `-debug` reports check=OK on every rank.
 * A pass is real evidence for the size_t offsets, but a pass of the OLD int
 * code at -O2 would not have proved it safe (signed overflow is UB, so the
 * compiler may widen the index).
 *
 * These need several GiB to tens of GiB of RAM, so they are meant for
 * Leonardo / the CRAB machine, not for CI or a laptop.  They run only when
 * BLINK_LARGE_TESTS=1; each case also skips if its estimated footprint
 * (ranks x per-rank buffers, host mode) exceeds 80% of MemAvailable.
 * The estimates count only the benchmark's own buffers, not MPI's shared-memory
 * transport, hence the 80% margin.
 */
#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "popen_helpers.h"

using blink::run_debug_capture;
using blink::DebugRun;

static const size_t GiB = (size_t)1 << 30;
static const size_t MiB = (size_t)1 << 20;

/* MemAvailable from /proc/meminfo in bytes, or 0 if it cannot be read. */
static size_t mem_available_bytes()
{
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    size_t kb = 0;
    while (fgets(line, sizeof(line), f))
        if (sscanf(line, "MemAvailable: %zu kB", &kb) == 1) break;
    fclose(f);
    return kb * 1024;
}

/* Run `binary` on `nprocs` ranks with `args` and require check=OK on every
 * rank.  Skips unless BLINK_LARGE_TESTS=1 and the footprint fits in memory. */
static void run_large_case(const std::string& binary, int nprocs,
                           const std::string& args, size_t footprint_bytes)
{
    const char *opt = std::getenv("BLINK_LARGE_TESTS");
    if (!opt || std::string(opt) != "1")
        GTEST_SKIP() << "set BLINK_LARGE_TESTS=1 to run the large-message tests";

    size_t avail = mem_available_bytes();
    if (avail == 0 || footprint_bytes > avail / 10 * 8)
        GTEST_SKIP() << "needs ~" << footprint_bytes / GiB << " GiB, MemAvailable is "
                     << avail / GiB << " GiB (limit is 80% of it)";

    DebugRun r = run_debug_capture(binary, nprocs, args);
    ASSERT_TRUE(r.normal_exit) << r.command << "\nstderr:\n" << r.stderr_raw;
    ASSERT_EQ(r.exit_code, 0)  << r.command << "\nstderr:\n" << r.stderr_raw;
    ASSERT_EQ((int)r.ranks.size(), nprocs) << "stdout:\n" << r.stdout_raw;
    for (auto& kv : r.ranks)
        EXPECT_EQ(kv.second["check"], "OK") << "rank " << kv.first << " in " << r.command;
}

/*
 * broadcast_b: 2 ranks, 1 GiB, -grty 3.
 * Per rank: buf = grty*msg = 3 GiB.  &buf[i*msg_size] reaches i=2 -> 2*2^30 = 2^31.
 * Footprint: 2 x 3 GiB = 6 GiB.
 */
TEST(LargeMessage, BroadcastB)
{
    run_large_case("broadcast_b", 2, "-msgsize 1073741824 -grty 3", 2 * 3 * GiB);
}

/*
 * ring_nb: 2 ranks, 768 MiB, -grty 2.
 * Per rank: send = 768 MiB, recv = 2*grty*msg = 4 x 768 MiB = 3 GiB, total 3.75 GiB.
 * &recv_dev[(2*i+1)*msg_size] at i=1 is 3*768 MiB = 2.25 GiB > 2^31.
 * Footprint: 2 x 3.75 GiB = 7.5 GiB.
 */
TEST(LargeMessage, RingNb)
{
    run_large_case("ring_nb", 2, "-msgsize 805306368 -grty 2", 2 * (768 * MiB + 3 * GiB));
}

/*
 * stencil_2d_nb: 4 ranks (2x2 grid), 768 MiB, -grty 1.
 * Per rank: send = 768 MiB, recv = 4*grty*msg = 3 GiB, total 3.75 GiB.
 * &recv_dev[(4*i+d)*msg_size] at d=3 is 3*768 MiB = 2.25 GiB > 2^31.
 * Footprint: 4 x 3.75 GiB = 15 GiB.
 */
TEST(LargeMessage, Stencil2dNb)
{
    run_large_case("stencil_2d_nb", 4, "-msgsize 805306368 -grty 1", 4 * (768 * MiB + 3 * GiB));
}

/*
 * incast_b: 4 ranks, 1 GiB.
 * Master: send = 1 GiB, recv = (w_size-1)*msg = 3 GiB; the other ranks only send (1 GiB).
 * &recv_dev[j*msg_size] at j=2 is 2*2^30 = 2^31.
 * Footprint: (1 + 3) + 3 x 1 = 7 GiB.
 */
TEST(LargeMessage, IncastB)
{
    run_large_case("incast_b", 4, "-msgsize 1073741824", (1 + 3) * GiB + 3 * GiB);
}

/*
 * kpartners_nb: 4 ranks, 1 GiB, -k 3.
 * Per rank: send = 1 GiB, recv = k*grty*msg = 3 GiB, total 4 GiB.
 * &recv_dev[(i*npartners+p)*msg_size] at p=2 is 2*2^30 = 2^31.
 * Footprint: 4 x 4 GiB = 16 GiB.
 */
TEST(LargeMessage, KpartnersNb)
{
    run_large_case("kpartners_nb", 4, "-msgsize 1073741824 -k 3", 4 * 4 * GiB);
}

/*
 * alltoall_man: 3 ranks, 1 GiB.
 * Per rank: send = w_size*msg = 3 GiB, recv = grty*w_size*msg = 3 GiB, total 6 GiB.
 * &send_dev[j*msg_size] and &recv_dev[(i*w_size+j)*msg_size] at j=2 are 2*2^30 = 2^31.
 * Footprint: 3 x 6 GiB = 18 GiB.
 */
TEST(LargeMessage, AlltoallMan)
{
    run_large_case("alltoall_man", 3, "-msgsize 1073741824", 3 * 6 * GiB);
}

/*
 * alltoall_comm_only: 3 ranks, 1 GiB.
 * Per rank: send = recv = w_size*msg = 3 GiB, total 6 GiB.
 * The Isend/Irecv offset i*datatype_size*count at i=2 is 2*2^30 = 2^31.
 * Footprint: 3 x 6 GiB = 18 GiB.
 */
TEST(LargeMessage, AlltoallCommOnly)
{
    run_large_case("alltoall_comm_only", 3, "-msgsize 1073741824", 3 * 6 * GiB);
}

/*
 * pairwise_b: 2 ranks, 1 GiB, -grty 3.
 * Per rank: send = 1 GiB, recv = grty*msg = 3 GiB, total 4 GiB.
 * &recv_dev[i*msg_size] at i=2 is 2*2^30 = 2^31.
 * Footprint: 2 x 4 GiB = 8 GiB.
 */
TEST(LargeMessage, PairwiseB)
{
    run_large_case("pairwise_b", 2, "-msgsize 1073741824 -grty 3", 2 * 4 * GiB);
}
