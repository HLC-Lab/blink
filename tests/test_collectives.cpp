/*
 * test_collectives.cpp — correctness tests for collective benchmarks.
 *
 * Strategy: spawn the actual benchmark binary with -debug and parse
 * "DEBUG rank=X nprocs=Y [root=Z] check=OK|FAIL" lines.
 * No MPI in this binary.
 *
 * Properties checked:
 *   Coverage       — every rank 0..N-1 appears exactly once.
 *   Nprocs         — every rank reports the correct communicator size.
 *   Root           — rooted collectives use the expected master_rank.
 *   NonDefaultRoot — -mrank flag correctly changes the root.
 *   DataIntegrity  — every rank reports check=OK (data-integrity verified
 *                    by the benchmark itself in debug mode).
 *   Agreement      — _nb variant reports the same result as the _b variant.
 *
 * Data filled in debug mode (inside the benchmark, not here):
 *   allgather/alltoall   fill send chunk with my_rank byte; verify recv[r*M+j]==r
 *   allreduce/reduce     fill send with my_rank (int); verify SUM == N*(N-1)/2
 *   reduce_scatter       same; scatter distributes one block to each rank
 *   broadcast            root fills with master_rank byte; all verify
 *   gather               fill send with my_rank byte; root verifies recv[r*M+j]==r
 *   scatter              root prepares chunk r=byte(r); each rank verifies recv[j]==my_rank
 *   barrier              reaches the debug print ⟹ barrier completed (deadlock check)
 *
 * Requires 8 MPI ranks.
 */
#include <gtest/gtest.h>
#include <string>
#include "popen_helpers.h"

using blink::DebugMap;
using blink::run_debug;
using blink::get_int;
using blink::get_str;
using blink::check_all_ok;
using blink::check_coverage;
using blink::check_nprocs;

/* ── shared helpers ─────────────────────────────────────────────────────────── */

static void check_root(const DebugMap& m, int expected_root, const std::string& ctx)
{
    for (const auto& [r, fields] : m) {
        auto it = fields.find("root");
        if (it == fields.end()) continue;
        EXPECT_EQ(std::stoi(it->second), expected_root)
            << ctx << ": rank " << r << " reports root=" << it->second;
    }
}

/* ── unrooted collectives ───────────────────────────────────────────────────── */

class UnrootedCollectiveTest : public ::testing::TestWithParam<std::string> {
protected:
    static constexpr int N = 8;
    void SetUp() override {
        if (blink::gpu_unsupported(GetParam()))
            GTEST_SKIP() << GetParam() << " on GPU memory is unsupported by this MPI";
    }
};

TEST_P(UnrootedCollectiveTest, Coverage) {
    auto m = run_debug(GetParam(), N);
    check_coverage(m, N, GetParam());
}
TEST_P(UnrootedCollectiveTest, Nprocs) {
    auto m = run_debug(GetParam(), N);
    check_nprocs(m, N, GetParam());
}
TEST_P(UnrootedCollectiveTest, DataIntegrity) {
    auto m = run_debug(GetParam(), N);
    check_coverage(m, N, GetParam());
    check_all_ok(m, GetParam());
}
TEST_P(UnrootedCollectiveTest, DataIntegrity_burst) {
    auto m = run_debug(GetParam(), N, "-blength 0.001");
    check_coverage(m, N, GetParam() + " burst");
    check_all_ok(m, GetParam() + " burst");
}
/* Granularity batching (-grty > 1): the _nb / manual variants issue grty
 * concurrent operations per timed window.  Each must write a DISJOINT receive
 * slot — a shared buffer would be an MPI overlap violation that can corrupt
 * data.  This exercises that path (default -grty 1 would never catch it).   */
TEST_P(UnrootedCollectiveTest, DataIntegrity_grty) {
    auto m = run_debug(GetParam(), N, "-grty 4");
    check_coverage(m, N, GetParam() + " grty=4");
    check_all_ok(m, GetParam() + " grty=4");
}
/* Randomised burst length (-bldist) + a pause (-bpause): exercises the
 * if(burst_length_rand) / if(burst_pause!=0) / if(burst_pause_rand) branches in
 * this benchmark's measurement loop, which the fixed -blength burst test misses. */
TEST_P(UnrootedCollectiveTest, DataIntegrity_burstdist) {
    auto m = run_debug(GetParam(), N, "-iter 3 -blength 0.0004 -bldist exp -bpause 0.0003 -bpdist exp");
    check_coverage(m, N, GetParam() + " burstdist");
    check_all_ok(m, GetParam() + " burstdist");
}

INSTANTIATE_TEST_SUITE_P(
    Collectives, UnrootedCollectiveTest,
    ::testing::Values(
        "allgather_b",       "allgather_nb",    "allgather_comm_only",
        "allreduce_b",       "allreduce_nb",
        "alltoall_b",        "alltoall_nb",     "alltoall_man",        "alltoall_comm_only",
        "barrier_b",         "barrier_nb",
        "reduce_scatter_b",  "reduce_scatter_nb"
    )
);

/* ── rooted collectives ─────────────────────────────────────────────────────── */

class RootedCollectiveTest : public ::testing::TestWithParam<std::string> {
protected:
    static constexpr int N      = 8;
    static constexpr int MASTER = 0;
    static constexpr int ALT    = 3;
    void SetUp() override {
        if (blink::gpu_unsupported(GetParam()))
            GTEST_SKIP() << GetParam() << " on GPU memory is unsupported by this MPI";
    }
};

TEST_P(RootedCollectiveTest, Coverage) {
    auto m = run_debug(GetParam(), N);
    check_coverage(m, N, GetParam());
}
TEST_P(RootedCollectiveTest, Nprocs) {
    auto m = run_debug(GetParam(), N);
    check_nprocs(m, N, GetParam());
}
TEST_P(RootedCollectiveTest, DefaultRoot) {
    auto m = run_debug(GetParam(), N);
    check_root(m, MASTER, GetParam() + " default root");
}
TEST_P(RootedCollectiveTest, DataIntegrity) {
    auto m = run_debug(GetParam(), N);
    check_coverage(m, N, GetParam());
    check_all_ok(m, GetParam());
}
TEST_P(RootedCollectiveTest, DataIntegrity_burst) {
    auto m = run_debug(GetParam(), N, "-blength 0.001");
    check_coverage(m, N, GetParam() + " burst");
    check_all_ok(m, GetParam() + " burst");
}
/* Granularity batching (-grty > 1): grty concurrent rooted ops per window,
 * each into a disjoint receive slot.  Guards the buffer-indexing fix.        */
TEST_P(RootedCollectiveTest, DataIntegrity_grty) {
    auto m = run_debug(GetParam(), N, "-grty 4");
    check_coverage(m, N, GetParam() + " grty=4");
    check_all_ok(m, GetParam() + " grty=4");
}
/* Randomised burst + pause: see the unrooted note. */
TEST_P(RootedCollectiveTest, DataIntegrity_burstdist) {
    auto m = run_debug(GetParam(), N, "-iter 3 -blength 0.0004 -bldist exp -bpause 0.0003 -bpdist exp");
    check_coverage(m, N, GetParam() + " burstdist");
    check_all_ok(m, GetParam() + " burstdist");
}
TEST_P(RootedCollectiveTest, NonDefaultRoot) {
    auto m = run_debug(GetParam(), N, "-mrank " + std::to_string(ALT));
    check_coverage(m, N, GetParam() + " mrank=3");
    check_nprocs(m, N, GetParam() + " mrank=3");
    check_root(m, ALT, GetParam() + " mrank=3");
    check_all_ok(m, GetParam() + " mrank=3");
}

INSTANTIATE_TEST_SUITE_P(
    Collectives, RootedCollectiveTest,
    ::testing::Values(
        "broadcast_b", "broadcast_nb",
        "gather_b",    "gather_nb",
        "scatter_b",   "scatter_nb",
        "reduce_b",    "reduce_nb"
    )
);

/* ── nb variants agree with b variants ─────────────────────────────────────── */

class CollectiveAgreementTest : public ::testing::Test {
protected:
    static constexpr int N = 8;

    static void check_agreement(const std::string& b_bin, const std::string& nb_bin) {
        if (blink::gpu_unsupported(b_bin) || blink::gpu_unsupported(nb_bin))
            GTEST_SKIP() << b_bin << "/" << nb_bin << " on GPU memory is unsupported by this MPI";
        auto b  = run_debug(b_bin,  N);
        auto nb = run_debug(nb_bin, N);
        for (const auto& [r, fields_b] : b) {
            if (!nb.count(r)) continue;
            EXPECT_EQ(get_int(nb, r, "nprocs"), get_int(b, r, "nprocs"))
                << "rank " << r << " nprocs differs: " << b_bin << " vs " << nb_bin;
            EXPECT_EQ(get_int(nb, r, "root"), get_int(b, r, "root"))
                << "rank " << r << " root differs: " << b_bin << " vs " << nb_bin;
            EXPECT_EQ(get_str(nb, r, "check"), get_str(b, r, "check"))
                << "rank " << r << " check differs: " << b_bin << " vs " << nb_bin;
        }
    }
};

TEST_F(CollectiveAgreementTest, Allgather)      { check_agreement("allgather_b",      "allgather_nb");      }
TEST_F(CollectiveAgreementTest, Allreduce)      { check_agreement("allreduce_b",      "allreduce_nb");      }
TEST_F(CollectiveAgreementTest, Alltoall)       { check_agreement("alltoall_b",       "alltoall_nb");       }
TEST_F(CollectiveAgreementTest, Barrier)        { check_agreement("barrier_b",        "barrier_nb");        }
TEST_F(CollectiveAgreementTest, Broadcast)      { check_agreement("broadcast_b",      "broadcast_nb");      }
TEST_F(CollectiveAgreementTest, Gather)         { check_agreement("gather_b",         "gather_nb");         }
TEST_F(CollectiveAgreementTest, Scatter)        { check_agreement("scatter_b",        "scatter_nb");        }
TEST_F(CollectiveAgreementTest, Reduce)         { check_agreement("reduce_b",         "reduce_nb");         }
TEST_F(CollectiveAgreementTest, ReduceScatter)  { check_agreement("reduce_scatter_b", "reduce_scatter_nb"); }

/* ── transfer-volume agreement ─────────────────────────────────────────────── */
/* For a given -msgsize, every allgather variant must move the SAME per-rank
 * payload.  Regression guard: allgather_comm_only previously treated -msgsize as
 * the total gathered size (msg_size/w_size per rank) while allgather_b/nb treat
 * it as the per-rank contribution, making their numbers non-comparable.  The
 * benchmarks now emit bytes=<per-rank send bytes> under -debug.              */
TEST_F(CollectiveAgreementTest, AllgatherTransferVolume) {
    const int M = 2048; /* multiple of sizeof(int) for the comm_only variant */
    const std::string sz = "-msgsize " + std::to_string(M);
    auto b  = run_debug("allgather_b",         N, sz);
    auto nb = run_debug("allgather_nb",        N, sz);
    auto co = run_debug("allgather_comm_only", N, sz);
    for (int r = 0; r < N; r++) {
        EXPECT_EQ(get_int(b,  r, "bytes"), M) << "allgather_b rank "         << r;
        EXPECT_EQ(get_int(nb, r, "bytes"), M) << "allgather_nb rank "        << r;
        EXPECT_EQ(get_int(co, r, "bytes"), M) << "allgather_comm_only rank " << r;
    }
}
