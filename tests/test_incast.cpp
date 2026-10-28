/*
 * test_incast.cpp — correctness tests for incast benchmarks.
 *
 * Strategy: spawn the actual benchmark binary with -debug and parse:
 *   "DEBUG rank=X nprocs=N nsenders=K check=OK|FAIL"  (receiver)
 *   "DEBUG rank=X nprocs=N target=R check=OK"         (senders)
 * No MPI in this binary.
 *
 * A rank is the receiver iff its parsed fields contain "nsenders".
 *
 * Properties checked:
 *   Coverage      — every rank 0..N-1 appears exactly once.
 *   Roles         — exactly one receiver (master_rank), N-1 senders.
 *   Targets       — every sender targets master_rank.
 *   Count         — receiver's nsenders equals N-1.
 *   NonDefault    — -mrank flag correctly changes the receiver.
 *   DataIntegrity — every rank reports check=OK.
 *   Agreement     — incast_nb / incast_bsnbr / incast_get / incast_put report
 *                   the same topology as incast_b.
 *
 * Requires 8 MPI ranks.
 */
#include <gtest/gtest.h>
#include <string>
#include "popen_helpers.h"

using blink::DebugMap;
using blink::DebugFields;
using blink::run_debug;
using blink::get_int;
using blink::get_str;
using blink::check_all_ok;
using blink::check_coverage;

/* ── role helpers ───────────────────────────────────────────────────────────── */

static bool is_receiver(const DebugMap& m, int r)
{
    auto it = m.find(r);
    if (it == m.end()) return false;
    return it->second.count("nsenders") > 0;
}

static void check_roles(const DebugMap& m, int n, int master, const std::string& ctx)
{
    int receiver_count = 0;
    for (const auto& [r, fields] : m) {
        if (fields.count("nsenders")) {
            receiver_count++;
            EXPECT_EQ(r, master)
                << ctx << ": rank " << r << " is receiver but master_rank=" << master;
            int nsenders = get_int(m, r, "nsenders");
            EXPECT_EQ(nsenders, n - 1)
                << ctx << ": receiver reports nsenders=" << nsenders
                << " expected " << (n - 1);
        } else {
            int target = get_int(m, r, "target");
            EXPECT_EQ(target, master)
                << ctx << ": rank " << r << " sender targets " << target
                << " expected master=" << master;
        }
    }
    EXPECT_EQ(receiver_count, 1) << ctx << ": expected exactly 1 receiver";
}

/* ── incast_b ───────────────────────────────────────────────────────────────── */

class IncastBTest : public ::testing::Test {
protected:
    static constexpr int N      = 8;
    static constexpr int MASTER = 0;
};

TEST_F(IncastBTest, Coverage) {
    auto m = run_debug("incast_b", N);
    check_coverage(m, N, "incast_b");
    check_all_ok(m, "incast_b");
}
TEST_F(IncastBTest, Roles) {
    auto m = run_debug("incast_b", N);
    check_roles(m, N, MASTER, "incast_b");
}
TEST_F(IncastBTest, Burst) {
    auto m = run_debug("incast_b", N, "-blength 0.001");
    check_coverage(m, N, "incast_b burst");
    check_roles(m, N, MASTER, "incast_b burst");
    check_all_ok(m, "incast_b burst");
}
TEST_F(IncastBTest, Burstdist) {
    auto m = run_debug("incast_b", N, "-iter 3 -blength 0.0004 -bldist exp -bpause 0.0003 -bpdist exp");
    check_coverage(m, N, "incast_b burstdist");
    check_roles(m, N, MASTER, "incast_b burstdist");
    check_all_ok(m, "incast_b burstdist");
}
TEST_F(IncastBTest, NonDefaultMaster) {
    const int ALT = 3;
    auto m = run_debug("incast_b", N, "-mrank " + std::to_string(ALT));
    check_coverage(m, N, "incast_b mrank=3");
    check_roles(m, N, ALT, "incast_b mrank=3");
    check_all_ok(m, "incast_b mrank=3");
}

/* ── incast_nb ──────────────────────────────────────────────────────────────── */

class IncastNbTest : public ::testing::Test {
protected:
    static constexpr int N      = 8;
    static constexpr int MASTER = 0;
};

TEST_F(IncastNbTest, Coverage) {
    auto m = run_debug("incast_nb", N);
    check_coverage(m, N, "incast_nb");
    check_all_ok(m, "incast_nb");
}
TEST_F(IncastNbTest, Roles) {
    auto m = run_debug("incast_nb", N);
    check_roles(m, N, MASTER, "incast_nb");
}
TEST_F(IncastNbTest, Burst) {
    auto m = run_debug("incast_nb", N, "-blength 0.001");
    check_coverage(m, N, "incast_nb burst");
    check_roles(m, N, MASTER, "incast_nb burst");
    check_all_ok(m, "incast_nb burst");
}
TEST_F(IncastNbTest, Burstdist) {
    auto m = run_debug("incast_nb", N, "-iter 3 -blength 0.0004 -bldist exp -bpause 0.0003 -bpdist exp");
    check_coverage(m, N, "incast_nb burstdist");
    check_roles(m, N, MASTER, "incast_nb burstdist");
    check_all_ok(m, "incast_nb burstdist");
}
/* -grty>1: the root posts grty*(N-1) concurrent Irecvs, each into a disjoint
 * slot.  A shared slot would be an MPI overlap violation; this guards it.   */
TEST_F(IncastNbTest, Grty) {
    auto m = run_debug("incast_nb", N, "-grty 4");
    check_coverage(m, N, "incast_nb grty=4");
    check_roles(m, N, MASTER, "incast_nb grty=4");
    check_all_ok(m, "incast_nb grty=4");
}
TEST_F(IncastNbTest, NonDefaultMaster) {
    const int ALT = 3;
    auto m = run_debug("incast_nb", N, "-mrank " + std::to_string(ALT));
    check_coverage(m, N, "incast_nb mrank=3");
    check_roles(m, N, ALT, "incast_nb mrank=3");
    check_all_ok(m, "incast_nb mrank=3");
}
TEST_F(IncastNbTest, AgreesWith_IncastB) {
    auto b  = run_debug("incast_b",  N);
    auto nb = run_debug("incast_nb", N);
    for (const auto& [r, fields_b] : b) {
        if (!nb.count(r)) continue;
        bool b_rx  = is_receiver(b,  r);
        bool nb_rx = is_receiver(nb, r);
        EXPECT_EQ(nb_rx, b_rx)
            << "rank " << r << " role differs between incast_b and incast_nb";
        if (!b_rx)
            EXPECT_EQ(get_int(nb, r, "target"), get_int(b, r, "target"))
                << "rank " << r << " target differs between incast_b and incast_nb";
    }
}

/* ── incast_bsnbr ───────────────────────────────────────────────────────────── */

class IncastBsnbrTest : public ::testing::Test {
protected:
    static constexpr int N      = 8;
    static constexpr int MASTER = 0;
};

TEST_F(IncastBsnbrTest, Coverage) {
    auto m = run_debug("incast_bsnbr", N);
    check_coverage(m, N, "incast_bsnbr");
    check_all_ok(m, "incast_bsnbr");
}
TEST_F(IncastBsnbrTest, Roles) {
    auto m = run_debug("incast_bsnbr", N);
    check_roles(m, N, MASTER, "incast_bsnbr");
}
TEST_F(IncastBsnbrTest, Burst) {
    auto m = run_debug("incast_bsnbr", N, "-blength 0.001");
    check_coverage(m, N, "incast_bsnbr burst");
    check_roles(m, N, MASTER, "incast_bsnbr burst");
    check_all_ok(m, "incast_bsnbr burst");
}
TEST_F(IncastBsnbrTest, Burstdist) {
    auto m = run_debug("incast_bsnbr", N, "-iter 3 -blength 0.0004 -bldist exp -bpause 0.0003 -bpdist exp");
    check_coverage(m, N, "incast_bsnbr burstdist");
    check_roles(m, N, MASTER, "incast_bsnbr burstdist");
    check_all_ok(m, "incast_bsnbr burstdist");
}
TEST_F(IncastBsnbrTest, NonDefaultMaster) {
    const int ALT = 3;
    auto m = run_debug("incast_bsnbr", N, "-mrank " + std::to_string(ALT));
    check_coverage(m, N, "incast_bsnbr mrank=3");
    check_roles(m, N, ALT, "incast_bsnbr mrank=3");
    check_all_ok(m, "incast_bsnbr mrank=3");
}
TEST_F(IncastBsnbrTest, AgreesWith_IncastB) {
    auto b     = run_debug("incast_b",     N);
    auto bsnbr = run_debug("incast_bsnbr", N);
    for (const auto& [r, fields_b] : b) {
        if (!bsnbr.count(r)) continue;
        bool b_rx  = is_receiver(b,     r);
        bool bs_rx = is_receiver(bsnbr, r);
        EXPECT_EQ(bs_rx, b_rx)
            << "rank " << r << " role differs between incast_b and incast_bsnbr";
        if (!b_rx)
            EXPECT_EQ(get_int(bsnbr, r, "target"), get_int(b, r, "target"))
                << "rank " << r << " target differs between incast_b and incast_bsnbr";
    }
}

/* ── incast_get ─────────────────────────────────────────────────────────────── */

class IncastGetTest : public ::testing::Test {
protected:
    static constexpr int N      = 8;
    static constexpr int MASTER = 0;
    /* GPU twin: one-sided on device memory is not supported by every MPI (the -gpu runtime
     * aborts with "not supported by this MPI" when its probe sees no data arrive). */
    void SetUp() override {
        if (blink::gpu_unsupported("incast_get"))
            GTEST_SKIP() << "one-sided on GPU memory unsupported by this MPI";
    }
};

TEST_F(IncastGetTest, Coverage) {
    auto m = run_debug("incast_get", N);
    check_coverage(m, N, "incast_get");
    check_all_ok(m, "incast_get");
}
TEST_F(IncastGetTest, Roles) {
    auto m = run_debug("incast_get", N);
    check_roles(m, N, MASTER, "incast_get");
}
TEST_F(IncastGetTest, Burst) {
    auto m = run_debug("incast_get", N, "-blength 0.001");
    check_coverage(m, N, "incast_get burst");
    check_roles(m, N, MASTER, "incast_get burst");
    check_all_ok(m, "incast_get burst");
}
TEST_F(IncastGetTest, Burstdist) {
    auto m = run_debug("incast_get", N, "-iter 3 -blength 0.0004 -bldist exp -bpause 0.0003 -bpdist exp");
    check_coverage(m, N, "incast_get burstdist");
    check_roles(m, N, MASTER, "incast_get burstdist");
    check_all_ok(m, "incast_get burstdist");
}
/* -grty>1: the root issues grty Gets per sender into distinct window-offset
 * slots; guards the RMA displacement math (j*M*grty + i*M).                 */
TEST_F(IncastGetTest, Grty) {
    auto m = run_debug("incast_get", N, "-grty 4");
    check_coverage(m, N, "incast_get grty=4");
    check_roles(m, N, MASTER, "incast_get grty=4");
    check_all_ok(m, "incast_get grty=4");
}
TEST_F(IncastGetTest, NonDefaultMaster) {
    const int ALT = 3;
    auto m = run_debug("incast_get", N, "-mrank " + std::to_string(ALT));
    check_coverage(m, N, "incast_get mrank=3");
    check_roles(m, N, ALT, "incast_get mrank=3");
    check_all_ok(m, "incast_get mrank=3");
}
TEST_F(IncastGetTest, AgreesWith_IncastB) {
    auto b   = run_debug("incast_b",   N);
    auto get = run_debug("incast_get", N);
    for (const auto& [r, fields_b] : b) {
        if (!get.count(r)) continue;
        bool b_rx = is_receiver(b,   r);
        bool g_rx = is_receiver(get, r);
        EXPECT_EQ(g_rx, b_rx)
            << "rank " << r << " role differs between incast_b and incast_get";
        if (!b_rx)
            EXPECT_EQ(get_int(get, r, "target"), get_int(b, r, "target"))
                << "rank " << r << " target differs between incast_b and incast_get";
    }
}

/* ── incast_put ─────────────────────────────────────────────────────────────── */

class IncastPutTest : public ::testing::Test {
protected:
    static constexpr int N      = 8;
    static constexpr int MASTER = 0;
    /* GPU twin: one-sided on device memory is not supported by every MPI (the -gpu runtime
     * aborts with "not supported by this MPI" when its probe sees no data arrive). */
    void SetUp() override {
        if (blink::gpu_unsupported("incast_put"))
            GTEST_SKIP() << "one-sided on GPU memory unsupported by this MPI";
    }
};

TEST_F(IncastPutTest, Coverage) {
    auto m = run_debug("incast_put", N);
    check_coverage(m, N, "incast_put");
    check_all_ok(m, "incast_put");
}
TEST_F(IncastPutTest, Roles) {
    auto m = run_debug("incast_put", N);
    check_roles(m, N, MASTER, "incast_put");
}
TEST_F(IncastPutTest, Burst) {
    auto m = run_debug("incast_put", N, "-blength 0.001");
    check_coverage(m, N, "incast_put burst");
    check_roles(m, N, MASTER, "incast_put burst");
    check_all_ok(m, "incast_put burst");
}
TEST_F(IncastPutTest, Burstdist) {
    auto m = run_debug("incast_put", N, "-iter 3 -blength 0.0004 -bldist exp -bpause 0.0003 -bpdist exp");
    check_coverage(m, N, "incast_put burstdist");
    check_roles(m, N, MASTER, "incast_put burstdist");
    check_all_ok(m, "incast_put burstdist");
}
/* -grty>1: each sender issues grty Puts into distinct window-offset slots;
 * guards the RMA displacement math (rank*M*grty + i*M).                     */
TEST_F(IncastPutTest, Grty) {
    auto m = run_debug("incast_put", N, "-grty 4");
    check_coverage(m, N, "incast_put grty=4");
    check_roles(m, N, MASTER, "incast_put grty=4");
    check_all_ok(m, "incast_put grty=4");
}
TEST_F(IncastPutTest, NonDefaultMaster) {
    const int ALT = 3;
    auto m = run_debug("incast_put", N, "-mrank " + std::to_string(ALT));
    check_coverage(m, N, "incast_put mrank=3");
    check_roles(m, N, ALT, "incast_put mrank=3");
    check_all_ok(m, "incast_put mrank=3");
}
TEST_F(IncastPutTest, AgreesWith_IncastB) {
    auto b   = run_debug("incast_b",   N);
    auto put = run_debug("incast_put", N);
    for (const auto& [r, fields_b] : b) {
        if (!put.count(r)) continue;
        bool b_rx = is_receiver(b,   r);
        bool p_rx = is_receiver(put, r);
        EXPECT_EQ(p_rx, b_rx)
            << "rank " << r << " role differs between incast_b and incast_put";
        if (!b_rx)
            EXPECT_EQ(get_int(put, r, "target"), get_int(b, r, "target"))
                << "rank " << r << " target differs between incast_b and incast_put";
    }
}
