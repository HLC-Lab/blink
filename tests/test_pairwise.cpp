/*
 * test_pairwise.cpp — correctness tests for pairwise benchmarks.
 *
 * Strategy: spawn the *actual* benchmark binary under mpirun with -debug,
 * parse the "DEBUG rank=X nprocs=N partner=Y check=OK|FAIL" lines it emits,
 * and assert the pairing properties below.  Breaking pairwise_b.c /
 * pairwise_nb.c / pairwise_bsnbr.c will break these tests.
 *
 * Properties checked:
 *   Coverage    — every rank 0..N-1 appears exactly once in the output.
 *   Validity    — every reported partner is in [0, N-1].
 *   Symmetry    — A→B implies B→A  (holds for offpair and rpair modes).
 *   Bijection   — no partner value appears more than once.
 *   ExactPairs  — for deterministic modes we verify the specific expected mapping.
 *   DataIntegrity — every rank reports check=OK.
 */
#include <gtest/gtest.h>
#include <map>
#include <set>
#include <string>
#include "popen_helpers.h"

using blink::DebugMap;
using blink::run_debug;
using blink::get_int;
using blink::check_all_ok;
using blink::check_coverage;

/* ── shared assertion helpers ───────────────────────────────────────────────── */

static void check_validity(const DebugMap& m, int n, const std::string& ctx)
{
    for (const auto& [r, fields] : m) {
        int p = get_int(m, r, "partner");
        EXPECT_GE(p, 0)  << ctx << ": rank " << r << " has partner " << p << " < 0";
        EXPECT_LT(p, n)  << ctx << ": rank " << r << " has partner " << p << " >= " << n;
    }
}

static void check_symmetric(const DebugMap& m, const std::string& ctx)
{
    for (const auto& [r, fields] : m) {
        int p = get_int(m, r, "partner");
        auto it = m.find(p);
        if (it == m.end()) continue;          /* coverage failures reported elsewhere */
        int pp = get_int(m, p, "partner");
        EXPECT_EQ(pp, r)
            << ctx << ": rank " << r << "→" << p
            << " but rank " << p << "→" << pp << " (not symmetric)";
    }
}

static void check_bijection(const DebugMap& m, const std::string& ctx)
{
    std::set<int> seen;
    for (const auto& [r, fields] : m) {
        int p = get_int(m, r, "partner");
        EXPECT_TRUE(seen.insert(p).second)
            << ctx << ": partner " << p << " assigned to more than one rank";
    }
}

static void check_exact(const DebugMap& m,
                         const std::map<int,int>& expected,
                         const std::string& ctx)
{
    for (const auto& [r, p] : expected) {
        auto it = m.find(r);
        if (it == m.end()) continue;   /* coverage failures reported elsewhere */
        int got = get_int(m, r, "partner");
        EXPECT_EQ(got, p)
            << ctx << ": rank " << r << " expected partner " << p
            << " but got " << got;
    }
}

/* ── pairwise_b ─────────────────────────────────────────────────────────────── */

class PairwiseBTest : public ::testing::Test {
protected:
    static constexpr int N = 8;
};

TEST_F(PairwiseBTest, Offpair_Offset1_Coverage) {
    auto m = run_debug("pairwise_b", N, "-mode offpair -offset 1");
    check_coverage(m, N, "offpair offset=1");
    check_all_ok(m, "offpair offset=1");
}
TEST_F(PairwiseBTest, Offpair_Offset1_Validity) {
    auto m = run_debug("pairwise_b", N, "-mode offpair -offset 1");
    check_validity(m, N, "offpair offset=1");
}
TEST_F(PairwiseBTest, Offpair_Offset1_Symmetric) {
    auto m = run_debug("pairwise_b", N, "-mode offpair -offset 1");
    check_symmetric(m, "offpair offset=1");
}
TEST_F(PairwiseBTest, Offpair_Offset1_ExactPairs) {
    auto m = run_debug("pairwise_b", N, "-mode offpair -offset 1");
    /* offset_pairs(n=8, offset=1) → 0↔1, 2↔3, 4↔5, 6↔7 */
    check_exact(m, {{0,1},{1,0},{2,3},{3,2},{4,5},{5,4},{6,7},{7,6}}, "offpair offset=1");
}

TEST_F(PairwiseBTest, Offpair_Offset2_Symmetric) {
    auto m = run_debug("pairwise_b", N, "-mode offpair -offset 2");
    check_symmetric(m, "offpair offset=2");
}
TEST_F(PairwiseBTest, Offpair_Offset2_ExactPairs) {
    auto m = run_debug("pairwise_b", N, "-mode offpair -offset 2");
    /* offset_pairs(n=8, offset=2) → 0↔2, 1↔3, 4↔6, 5↔7 */
    check_exact(m, {{0,2},{2,0},{1,3},{3,1},{4,6},{6,4},{5,7},{7,5}}, "offpair offset=2");
}
TEST_F(PairwiseBTest, Offpair_DefaultOffset_ExactPairs) {
    auto m = run_debug("pairwise_b", N, "-mode offpair");
    /* default offset = N/2 = 4 → 0↔4, 1↔5, 2↔6, 3↔7 */
    check_exact(m, {{0,4},{4,0},{1,5},{5,1},{2,6},{6,2},{3,7},{7,3}}, "offpair default offset");
}

TEST_F(PairwiseBTest, RandomPair_Coverage) {
    auto m = run_debug("pairwise_b", N, "-mode rpair -seed 1");
    check_coverage(m, N, "rpair");
    check_all_ok(m, "rpair");
}
TEST_F(PairwiseBTest, RandomPair_Validity) {
    auto m = run_debug("pairwise_b", N, "-mode rpair -seed 1");
    check_validity(m, N, "rpair");
}
TEST_F(PairwiseBTest, RandomPair_Symmetric) {
    auto m = run_debug("pairwise_b", N, "-mode rpair -seed 1");
    check_symmetric(m, "rpair");
}
TEST_F(PairwiseBTest, RandomPair_Bijection) {
    auto m = run_debug("pairwise_b", N, "-mode rpair -seed 1");
    check_bijection(m, "rpair");
}
TEST_F(PairwiseBTest, Burst) {
    auto m = run_debug("pairwise_b", N, "-mode offpair -offset 1 -blength 0.001");
    check_coverage(m, N, "pairwise_b burst");
    check_symmetric(m, "pairwise_b burst");
    check_all_ok(m, "pairwise_b burst");
}
TEST_F(PairwiseBTest, Burstdist) {
    auto m = run_debug("pairwise_b", N, "-mode offpair -offset 1 -iter 3 -blength 0.0004 -bldist exp -bpause 0.0003 -bpdist exp");
    check_coverage(m, N, "pairwise_b burstdist");
    check_all_ok(m, "pairwise_b burstdist");
}

/* ── non-reciprocal modes (rot / perm) ──────────────────────────────────────
 * rot and perm are permutations, not symmetric pairings, so the partner graph
 * is a bijection but NOT symmetric (A→B does not imply B→A).  These exercise
 * the debug path that must avoid deadlock with a non-reciprocal partner.    */
TEST_F(PairwiseBTest, Rot_Offset1_Coverage) {
    auto m = run_debug("pairwise_b", N, "-mode rot -offset 1");
    check_coverage(m, N, "rot offset=1");
    check_all_ok(m, "rot offset=1");
}
TEST_F(PairwiseBTest, Rot_Offset1_Validity) {
    auto m = run_debug("pairwise_b", N, "-mode rot -offset 1");
    check_validity(m, N, "rot offset=1");
}
TEST_F(PairwiseBTest, Rot_Offset1_Bijection) {
    auto m = run_debug("pairwise_b", N, "-mode rot -offset 1");
    check_bijection(m, "rot offset=1");
}
TEST_F(PairwiseBTest, Rot_Offset1_ExactPairs) {
    auto m = run_debug("pairwise_b", N, "-mode rot -offset 1");
    /* rot: rank r sends to (r+1)%N */
    check_exact(m, {{0,1},{1,2},{2,3},{3,4},{4,5},{5,6},{6,7},{7,0}}, "rot offset=1");
}
TEST_F(PairwiseBTest, Perm_Coverage) {
    auto m = run_debug("pairwise_b", N, "-mode perm -seed 1");
    check_coverage(m, N, "perm");
    check_all_ok(m, "perm");
}
TEST_F(PairwiseBTest, Perm_Validity) {
    auto m = run_debug("pairwise_b", N, "-mode perm -seed 1");
    check_validity(m, N, "perm");
}
TEST_F(PairwiseBTest, Perm_Bijection) {
    auto m = run_debug("pairwise_b", N, "-mode perm -seed 1");
    check_bijection(m, "perm");
}

/* ── pairwise_nb — same pairing logic, non-blocking MPI ────────────────────── */

class PairwiseNbTest : public ::testing::Test {
protected:
    static constexpr int N = 8;
};

TEST_F(PairwiseNbTest, Offpair_Offset1_Symmetric) {
    auto m = run_debug("pairwise_nb", N, "-mode offpair -offset 1");
    check_symmetric(m, "nb offpair offset=1");
    check_all_ok(m, "nb offpair offset=1");
}
TEST_F(PairwiseNbTest, Offpair_Offset1_ExactPairs) {
    auto m = run_debug("pairwise_nb", N, "-mode offpair -offset 1");
    check_exact(m, {{0,1},{1,0},{2,3},{3,2},{4,5},{5,4},{6,7},{7,6}}, "nb offpair offset=1");
}
TEST_F(PairwiseNbTest, RandomPair_Symmetric) {
    auto m = run_debug("pairwise_nb", N, "-mode rpair -seed 1");
    check_symmetric(m, "nb rpair");
    check_all_ok(m, "nb rpair");
}
TEST_F(PairwiseNbTest, Burst) {
    auto m = run_debug("pairwise_nb", N, "-mode offpair -offset 1 -blength 0.001");
    check_coverage(m, N, "pairwise_nb burst");
    check_symmetric(m, "pairwise_nb burst");
    check_all_ok(m, "pairwise_nb burst");
}
TEST_F(PairwiseNbTest, Burstdist) {
    auto m = run_debug("pairwise_nb", N, "-mode offpair -offset 1 -iter 3 -blength 0.0004 -bldist exp -bpause 0.0003 -bpdist exp");
    check_coverage(m, N, "pairwise_nb burstdist");
    check_all_ok(m, "pairwise_nb burstdist");
}
TEST_F(PairwiseNbTest, Rot_Offset1_DataIntegrity) {
    auto m = run_debug("pairwise_nb", N, "-mode rot -offset 1");
    check_coverage(m, N, "nb rot offset=1");
    check_bijection(m, "nb rot offset=1");
    check_all_ok(m, "nb rot offset=1");
}
TEST_F(PairwiseNbTest, Perm_DataIntegrity) {
    auto m = run_debug("pairwise_nb", N, "-mode perm -seed 1");
    check_coverage(m, N, "nb perm");
    check_bijection(m, "nb perm");
    check_all_ok(m, "nb perm");
}

/* ── pairwise_bsnbr — same pairing logic, non-blocking send / blocking recv ── */

class PairwiseBsnbrTest : public ::testing::Test {
protected:
    static constexpr int N = 8;
};

TEST_F(PairwiseBsnbrTest, Offpair_Offset1_Symmetric) {
    auto m = run_debug("pairwise_bsnbr", N, "-mode offpair -offset 1");
    check_symmetric(m, "bsnbr offpair offset=1");
    check_all_ok(m, "bsnbr offpair offset=1");
}
TEST_F(PairwiseBsnbrTest, Offpair_Offset1_ExactPairs) {
    auto m = run_debug("pairwise_bsnbr", N, "-mode offpair -offset 1");
    check_exact(m, {{0,1},{1,0},{2,3},{3,2},{4,5},{5,4},{6,7},{7,6}}, "bsnbr offpair offset=1");
}
TEST_F(PairwiseBsnbrTest, RandomPair_Symmetric) {
    auto m = run_debug("pairwise_bsnbr", N, "-mode rpair -seed 1");
    check_symmetric(m, "bsnbr rpair");
    check_all_ok(m, "bsnbr rpair");
}
TEST_F(PairwiseBsnbrTest, Burst) {
    auto m = run_debug("pairwise_bsnbr", N, "-mode offpair -offset 1 -blength 0.001");
    check_coverage(m, N, "pairwise_bsnbr burst");
    check_symmetric(m, "pairwise_bsnbr burst");
    check_all_ok(m, "pairwise_bsnbr burst");
}
TEST_F(PairwiseBsnbrTest, Burstdist) {
    auto m = run_debug("pairwise_bsnbr", N, "-mode offpair -offset 1 -iter 3 -blength 0.0004 -bldist exp -bpause 0.0003 -bpdist exp");
    check_coverage(m, N, "pairwise_bsnbr burstdist");
    check_all_ok(m, "pairwise_bsnbr burstdist");
}
TEST_F(PairwiseBsnbrTest, Rot_Offset1_DataIntegrity) {
    auto m = run_debug("pairwise_bsnbr", N, "-mode rot -offset 1");
    check_coverage(m, N, "bsnbr rot offset=1");
    check_bijection(m, "bsnbr rot offset=1");
    check_all_ok(m, "bsnbr rot offset=1");
}
TEST_F(PairwiseBsnbrTest, Perm_DataIntegrity) {
    auto m = run_debug("pairwise_bsnbr", N, "-mode perm -seed 1");
    check_coverage(m, N, "bsnbr perm");
    check_bijection(m, "bsnbr perm");
    check_all_ok(m, "bsnbr perm");
}
