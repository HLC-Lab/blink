/*
 * test_misc.cpp — miscellaneous correctness tests:
 *   - ring-buffer wrap (iter > maxsamples)
 *   - pretty-print formatter doesn't crash and produces a recognisable table
 *   - -mrand picks a deterministic master rank for a given seed
 *   - sampler/dist_test runs (validates the burst distribution implementation)
 *
 * All tests use 8 ranks except where noted.
 */
#include <gtest/gtest.h>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>   /* std::count */
#include "popen_helpers.h"

using blink::run_debug_capture;
using blink::DebugRun;

/* Helper: count lines in s containing a substring */
static int count_lines_containing(const std::string& s, const std::string& needle)
{
    int n = 0;
    size_t p = 0;
    while ((p = s.find(needle, p)) != std::string::npos) { n++; p += needle.size(); }
    return n;
}

/* ── ring-buffer wrap ───────────────────────────────────────────────────────── */

/*
 * Configure max_samples to be smaller than the iteration count, ensuring the
 * ring buffer wraps.  The benchmark should still print exactly max_samples
 * "Average,..." data lines (one per recorded sample), and the summary line
 * should report "Measured <max_samples>" iterations.
 */
TEST(RingBufferWrap, BarrierWrapsCorrectly)
{
    /* run barrier_nb (cheap) with iter=50 maxsamples=10 warmup=0 — buffer
     * should hold the last 10 samples.                                    */
    blink::DebugRun r = run_debug_capture("barrier_nb", 8,
        "-iter 50 -maxsamples 10 -warmup 0");
    /* run_debug_capture does NOT assert; do the assertions explicitly */
    ASSERT_TRUE(r.normal_exit) << "stderr:\n" << r.stderr_raw;
    ASSERT_EQ(r.exit_code, 0)  << "stderr:\n" << r.stderr_raw;

    /* the master rank's CSV body has lines of the form "%.9f,%.9f,..." */
    int data_lines = 0;
    size_t p = 0;
    while ((p = r.stdout_raw.find('\n', p)) != std::string::npos) {
        /* heuristic: a data line starts with a digit and has 4 commas */
        size_t line_start = r.stdout_raw.rfind('\n', p > 0 ? p - 1 : 0);
        size_t s = (line_start == std::string::npos) ? 0 : line_start + 1;
        std::string line = r.stdout_raw.substr(s, p - s);
        if (!line.empty() && isdigit((unsigned char)line[0]))
            if (std::count(line.begin(), line.end(), ',') == 4)
                data_lines++;
        p++;
    }
    EXPECT_EQ(data_lines, 10)
        << "ring buffer wrap: expected 10 measured samples (maxsamples), got "
        << data_lines << "\nstdout:\n" << r.stdout_raw;

    EXPECT_NE(r.stdout_raw.find("Ran 50 iterations. Measured 10 iterations."),
              std::string::npos)
        << "summary line mismatch.\nstdout:\n" << r.stdout_raw;
}

/* ── pretty-print formatter ─────────────────────────────────────────────────── */

TEST(PrettyPrint, FormatterDoesNotCrash)
{
    DebugRun r = run_debug_capture("barrier_nb", 8, "-iter 3 -pretty-print");
    ASSERT_TRUE(r.normal_exit) << "stderr:\n" << r.stderr_raw;
    ASSERT_EQ(r.exit_code, 0);
    /* pretty-print uses a column-header row and a separator line */
    EXPECT_NE(r.stdout_raw.find("avg"),    std::string::npos);
    EXPECT_NE(r.stdout_raw.find("median"), std::string::npos);
    EXPECT_NE(r.stdout_raw.find("samples"),std::string::npos);
}

/* ── -mrand determinism ─────────────────────────────────────────────────────── */

TEST(MasterRand, SameSeedSameMaster)
{
    /* with -mrand and a fixed seed, the chosen master_rank should be
     * deterministic across runs.  We grep the "receiver rank: R" line from
     * incast_b's own startup banner.                                       */
    DebugRun r1 = run_debug_capture("incast_b", 8, "-mrand -seed 42");
    DebugRun r2 = run_debug_capture("incast_b", 8, "-mrand -seed 42");
    ASSERT_TRUE(r1.normal_exit);
    ASSERT_TRUE(r2.normal_exit);
    auto p1 = r1.stdout_raw.find("receiver rank:");
    auto p2 = r2.stdout_raw.find("receiver rank:");
    ASSERT_NE(p1, std::string::npos) << r1.stdout_raw;
    ASSERT_NE(p2, std::string::npos) << r2.stdout_raw;
    int m1, m2;
    sscanf(r1.stdout_raw.c_str() + p1, "receiver rank: %d", &m1);
    sscanf(r2.stdout_raw.c_str() + p2, "receiver rank: %d", &m2);
    EXPECT_EQ(m1, m2)
        << "same -seed should give same -mrand master, got " << m1 << " vs " << m2;
}

/* ── dist_test sampler validation ───────────────────────────────────────────── */

TEST(SamplerValidation, DistTestRuns)
{
    /* dist_test is single-rank.  If it exits 0 the sampler implementations
     * are internally consistent (mean / KS test).                          */
    DebugRun r = run_debug_capture("dist_test", 1, "");
    /* dist_test may not accept -debug; treat absence of failure as success */
    ASSERT_EQ(r.exit_code, 0)
        << "dist_test failed.\nstderr:\n" << r.stderr_raw
        << "\nstdout:\n" << r.stdout_raw;
}

/* ── warm-up exclusion ──────────────────────────────────────────────────────── */

/*
 * Warm-up iterations must NOT be recorded.  With -iter 5 -warmup 3 the benchmark
 * runs 8 outer iterations but records only the last 5; the summary must report
 * "Ran 8 iterations. Measured 5 iterations." and emit exactly 5 CSV data lines.
 */
TEST(WarmupExclusion, WarmupSamplesNotRecorded)
{
    DebugRun r = run_debug_capture("barrier_nb", 8, "-iter 5 -warmup 3 -maxsamples 1000");
    ASSERT_TRUE(r.normal_exit) << "stderr:\n" << r.stderr_raw;
    ASSERT_EQ(r.exit_code, 0)  << "stderr:\n" << r.stderr_raw;

    int data_lines = 0;
    size_t p = 0;
    while ((p = r.stdout_raw.find('\n', p)) != std::string::npos) {
        size_t line_start = r.stdout_raw.rfind('\n', p > 0 ? p - 1 : 0);
        size_t s = (line_start == std::string::npos) ? 0 : line_start + 1;
        std::string line = r.stdout_raw.substr(s, p - s);
        if (!line.empty() && isdigit((unsigned char)line[0]))
            if (std::count(line.begin(), line.end(), ',') == 4)
                data_lines++;
        p++;
    }
    EXPECT_EQ(data_lines, 5)
        << "expected 5 recorded samples (3 warmup excluded), got " << data_lines
        << "\nstdout:\n" << r.stdout_raw;
    EXPECT_NE(r.stdout_raw.find("Ran 8 iterations. Measured 5 iterations."),
              std::string::npos)
        << "summary should report 8 run / 5 measured.\nstdout:\n" << r.stdout_raw;
}

/* ── SIGUSR1 clean shutdown ─────────────────────────────────────────────────── */

/*
 * Regression test for the endless-mode SIGUSR1 shutdown.  The shutdown flag is
 * set only on the rank that receives the signal, so all ranks must agree
 * collectively (check_shutdown) to leave the measurement loop on the same
 * iteration.  Sending SIGUSR1 to exactly ONE rank of an endless run must shut
 * the whole job down cleanly (no deadlock) and still print collected results.
 *
 * Orchestrated in POSIX sh via popen: launch the endless run in the background,
 * signal one rank (a process whose comm is exactly "alltoall_nb" — i.e. a rank,
 * not the launcher), then wait up to 30s for a clean exit, killing it if it
 * hangs so the test itself can never block indefinitely.
 */
TEST(Shutdown, Sigusr1ToSingleRankShutsDownCleanly)
{
    /* BLINK_MPIEXEC overrides the launcher (e.g. srun inside an allocation): the ranks then run on a compute node and
     * cannot be found with pgrep from here. */
    if (getenv("BLINK_MPIEXEC"))
        GTEST_SKIP() << "BLINK_MPIEXEC is set: rank processes may run on another node, cannot signal one from here";
    char outtmpl[] = "/tmp/blink_endl_XXXXXX";
    int ofd = mkstemp(outtmpl);
    ASSERT_GE(ofd, 0);
    close(ofd);
    std::string outfile = outtmpl;
    std::string bin = blink::bin_dir() + "/alltoall_nb";

    std::string cmd =
        "\"" + blink::mpiexec() + "\" " + blink::nproc_flag() + " 4 \"" + bin + "\""
        " -endl -msgsize 1024 -seed 999983 > \"" + outfile + "\" 2>&1 &\n"
        "MPIPID=$!\n"
        "sleep 3\n"
        "RP=\"\"\n"
        "for q in $(pgrep -f 'alltoall_nb -endl'); do\n"
        "  if [ -r /proc/$q/comm ] && [ \"$(cat /proc/$q/comm)\" = alltoall_nb ]; then RP=$q; break; fi\n"
        "done\n"
        "[ -n \"$RP\" ] && kill -USR1 \"$RP\"\n"
        "CLEAN=HANG\n"
        "for i in $(seq 1 30); do kill -0 \"$MPIPID\" 2>/dev/null || { CLEAN=CLEAN; break; }; sleep 1; done\n"
        "if [ \"$CLEAN\" = HANG ]; then kill -9 \"$MPIPID\" 2>/dev/null; pkill -9 -f 'alltoall_nb -endl' 2>/dev/null; fi\n"
        "echo SIGNALED=$RP\n"
        "echo RESULT=$CLEAN\n";

    FILE *pp = popen(cmd.c_str(), "r");
    ASSERT_NE(pp, nullptr);
    std::string out;
    char buf[512];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), pp)) > 0) out.append(buf, n);
    pclose(pp);

    std::string bench;
    FILE *of = fopen(outfile.c_str(), "r");
    if (of) { while ((n = fread(buf, 1, sizeof(buf), of)) > 0) bench.append(buf, n); fclose(of); }
    unlink(outfile.c_str());

    EXPECT_EQ(out.find("SIGNALED=\n"), std::string::npos)
        << "no rank process was found to signal.\norchestration:\n" << out;
    EXPECT_NE(out.find("RESULT=CLEAN"), std::string::npos)
        << "endless job did not shut down after SIGUSR1 to one rank (deadlock?).\n"
        << "orchestration:\n" << out << "\nbenchmark output:\n" << bench;
    EXPECT_NE(bench.find("Measured"), std::string::npos)
        << "no results footer after clean shutdown.\nbenchmark output:\n" << bench;
}

/* ── CLI validation guards ──────────────────────────────────────────────────── */

/*
 * An out-of-range -mrank must fail loudly (clean MPI_Abort with a diagnostic),
 * never crash or run with an invalid collective root.  Regression guard for the
 * master_rank range check added to parse_common_args().
 */
TEST(CliGuards, OutOfRangeMrankAbortsCleanly)
{
    DebugRun r = run_debug_capture("alltoall_b", 8, "-mrank 99");
    EXPECT_FALSE(r.normal_exit && r.exit_code == 0)
        << "out-of-range -mrank should abort, but the run exited 0.\n"
        << "stdout:\n" << r.stdout_raw;
    EXPECT_NE(r.stderr_raw.find("mrank must be in"), std::string::npos)
        << "expected an -mrank range diagnostic on stderr.\nstderr:\n" << r.stderr_raw;
}

/*
 * A value-taking flag supplied as the final token must be rejected with a
 * "Missing value" message rather than dereferencing argv[argc] (== NULL).
 * Regression guard for arg_value() and the benchmark-specific parsers that
 * now route through it.
 */
TEST(CliGuards, MissingFlagValueAbortsCleanly)
{
    DebugRun r = run_debug_capture("pairwise_b", 8, "-mode");
    EXPECT_FALSE(r.normal_exit && r.exit_code == 0)
        << "missing -mode value should abort, but the run exited 0.\n"
        << "stdout:\n" << r.stdout_raw;
    EXPECT_NE(r.stderr_raw.find("Missing value for option -mode"), std::string::npos)
        << "expected a 'Missing value' diagnostic on stderr.\nstderr:\n" << r.stderr_raw;
}

/* ── -plot distribution histogram ───────────────────────────────────────────── */

TEST(Plot, RendersHistogramAdditively)
{
    DebugRun r = run_debug_capture("alltoall_nb", 8, "-iter 100 -plot");
    ASSERT_TRUE(r.normal_exit) << "stderr:\n" << r.stderr_raw;
    ASSERT_EQ(r.exit_code, 0)  << "stderr:\n" << r.stderr_raw;
    EXPECT_NE(r.stdout_raw.find("Per-iteration latency"), std::string::npos)
        << "no plot header.\nstdout:\n" << r.stdout_raw;
    EXPECT_NE(r.stdout_raw.find("stat=max"), std::string::npos)        /* default stat */
        << "default -plotstat should be max.\nstdout:\n" << r.stdout_raw;
    EXPECT_NE(r.stdout_raw.find("Measured"), std::string::npos)        /* additive */
        << "plot should be additive (the listing is still printed).\nstdout:\n" << r.stdout_raw;
}

TEST(Plot, StatAndBinningSelectors)
{
    DebugRun avg = run_debug_capture("alltoall_nb", 8, "-iter 100 -plot -plotstat avg");
    EXPECT_EQ(avg.exit_code, 0) << avg.stderr_raw;
    EXPECT_NE(avg.stdout_raw.find("stat=avg"), std::string::npos) << avg.stdout_raw;

    DebugRun lg = run_debug_capture("alltoall_nb", 8, "-iter 100 -plot -plotlog");
    EXPECT_EQ(lg.exit_code, 0) << lg.stderr_raw;
    EXPECT_NE(lg.stdout_raw.find("log bins"), std::string::npos) << lg.stdout_raw;

    DebugRun fx = run_debug_capture("alltoall_nb", 8, "-iter 100 -plot -plotbinsize 1us");
    EXPECT_EQ(fx.exit_code, 0) << fx.stderr_raw;
    EXPECT_NE(fx.stdout_raw.find("fixed bins"), std::string::npos) << fx.stdout_raw;
}

TEST(Plot, InvalidStatAbortsCleanly)
{
    DebugRun r = run_debug_capture("alltoall_nb", 8, "-plot -plotstat bogus");
    EXPECT_FALSE(r.normal_exit && r.exit_code == 0)
        << "invalid -plotstat should abort.\nstdout:\n" << r.stdout_raw;
    EXPECT_NE(r.stderr_raw.find("plotstat must be one of"), std::string::npos)
        << "expected a -plotstat diagnostic.\nstderr:\n" << r.stderr_raw;
}

TEST(Plot, BinsizeWithLogConflictAborts)
{
    DebugRun r = run_debug_capture("alltoall_nb", 8, "-plot -plotbinsize 1us -plotlog");
    EXPECT_FALSE(r.normal_exit && r.exit_code == 0)
        << "combining -plotbinsize with -plotlog should abort.\nstdout:\n" << r.stdout_raw;
    EXPECT_NE(r.stderr_raw.find("cannot be combined with -plotlog"), std::string::npos)
        << "expected a conflict diagnostic.\nstderr:\n" << r.stderr_raw;
}

/* ── burst / pause distribution sampling ─────────────────────────────────────
 * Exercises the randomised inter-arrival path from the benchmark side:
 * sample_burst_length/sample_pause_length -> rand_duration -> rand_{exp,pareto,
 * lognormal}, plus dsleep().  barrier_nb is the cheapest benchmark to drive it. */
TEST(BurstSampling, RandomisedBurstAndPause)
{
    for (const char *d : {"exp", "pareto", "lognormal"}) {
        std::string extra = std::string("-iter 5 -blength 0.0004 -bldist ") + d;
        if (std::string(d) != "exp") extra += " -blshape 1.6";   /* shape ignored for exp */
        DebugRun r = run_debug_capture("barrier_nb", 8, extra);
        EXPECT_TRUE(r.normal_exit && r.exit_code == 0)
            << "barrier_nb -bldist " << d << " failed.\nstderr:\n" << r.stderr_raw;
    }
    /* randomised pause drives dsleep() + sample_pause_length() */
    DebugRun p = run_debug_capture("barrier_nb", 8, "-iter 4 -bpause 0.0004 -bpdist exp");
    EXPECT_TRUE(p.normal_exit && p.exit_code == 0)
        << "barrier_nb -bpdist failed.\nstderr:\n" << p.stderr_raw;
}

TEST(BurstSampling, InvalidShapeAbortsCleanly)
{
    DebugRun r = run_debug_capture("barrier_nb", 8, "-blength 0.001 -bldist pareto -blshape 0.5");
    EXPECT_FALSE(r.normal_exit && r.exit_code == 0)
        << "a Pareto shape <= 1 should abort.\nstdout:\n" << r.stdout_raw;
    EXPECT_NE(r.stderr_raw.find("pareto shape"), std::string::npos)
        << "expected a pareto-shape diagnostic.\nstderr:\n" << r.stderr_raw;
}

/*
 * Duration unit suffixes (s, ms, us, ns) — previously dropped silently by atof
 * (so "-blength 1ms" became 1 SECOND, not 1 ms).  parse_duration_arg() now
 * recognises them; a bare number is still seconds for backwards compatibility.
 */
TEST(BurstSampling, DurationSuffixesAccepted)
{
    DebugRun r = run_debug_capture("barrier_nb", 4,
        "-iter 3 -blength 400us -bpause 300us -bldist exp -bpdist exp");
    EXPECT_TRUE(r.normal_exit && r.exit_code == 0)
        << "barrier_nb with unit-suffix durations failed.\nstderr:\n" << r.stderr_raw;
}

TEST(BurstSampling, InvalidDurationAbortsCleanly)
{
    /* old atof would have silently returned 0.0; we now abort instead. */
    DebugRun r = run_debug_capture("barrier_nb", 2, "-blength garbage");
    EXPECT_FALSE(r.normal_exit && r.exit_code == 0)
        << "non-numeric -blength should abort.\nstdout:\n" << r.stdout_raw;
    EXPECT_NE(r.stderr_raw.find("must be a non-negative duration"), std::string::npos)
        << "expected a duration diagnostic.\nstderr:\n" << r.stderr_raw;
}

/* ── -h / --help / -help ────────────────────────────────────────────────────
 *
 * The help system: weak `benchmark_help` symbol in common.c overridden by
 * benchmarks with custom flags (pairwise, kpartners, ring, stencil).
 * parse_common_args handles -h/-help/--help by printing on rank 0 and calling
 * MPI_Finalize + exit(0) on every rank.
 */

TEST(Help, CommonHelpPrintsAndExitsZero)
{
    /* barrier_nb has NO custom args — only the common block should appear */
    DebugRun r = run_debug_capture("barrier_nb", 2, "-h");
    ASSERT_TRUE(r.normal_exit) << "stderr:\n" << r.stderr_raw;
    EXPECT_EQ(r.exit_code, 0) << "stderr:\n" << r.stderr_raw;
    EXPECT_NE(r.stdout_raw.find("Usage: barrier_nb"), std::string::npos)
        << "expected 'Usage: barrier_nb'.\nstdout:\n" << r.stdout_raw;
    EXPECT_NE(r.stdout_raw.find("-mrank"),  std::string::npos)
        << "common help should list -mrank.\nstdout:\n" << r.stdout_raw;
    EXPECT_NE(r.stdout_raw.find("-blength"), std::string::npos)
        << "common help should list burst flags.\nstdout:\n" << r.stdout_raw;
    EXPECT_NE(r.stdout_raw.find("-plot"),   std::string::npos)
        << "common help should list plot flags.\nstdout:\n" << r.stdout_raw;
    /* benchmarks without custom args must NOT show the benchmark-specific section */
    EXPECT_EQ(r.stdout_raw.find("Benchmark-specific options"), std::string::npos)
        << "barrier_nb has no custom args — the section should be absent.\nstdout:\n"
        << r.stdout_raw;
}

TEST(Help, BenchmarkSpecificSectionAppears)
{
    /* pairwise_b defines benchmark_help with -mode and -offset */
    DebugRun r = run_debug_capture("pairwise_b", 8, "--help");
    ASSERT_TRUE(r.normal_exit) << "stderr:\n" << r.stderr_raw;
    EXPECT_EQ(r.exit_code, 0) << "stderr:\n" << r.stderr_raw;
    EXPECT_NE(r.stdout_raw.find("Benchmark-specific options"), std::string::npos)
        << "pairwise_b --help should include the benchmark-specific section.\nstdout:\n"
        << r.stdout_raw;
    EXPECT_NE(r.stdout_raw.find("-mode"),   std::string::npos)
        << "pairwise_b --help should mention -mode.\nstdout:\n" << r.stdout_raw;
    EXPECT_NE(r.stdout_raw.find("-offset"), std::string::npos)
        << "pairwise_b --help should mention -offset.\nstdout:\n" << r.stdout_raw;
}

TEST(Help, AllThreeFlagFormsAccepted)
{
    for (const char *flag : {"-h", "-help", "--help"}) {
        DebugRun r = run_debug_capture("barrier_nb", 2, flag);
        EXPECT_TRUE(r.normal_exit && r.exit_code == 0)
            << "flag '" << flag << "' should print help and exit 0.\nstderr:\n"
            << r.stderr_raw;
        EXPECT_NE(r.stdout_raw.find("Usage:"), std::string::npos)
            << "flag '" << flag << "' — expected 'Usage:' in stdout.\nstdout:\n"
            << r.stdout_raw;
    }
}

TEST(Help, OnlyMasterRankPrints)
{
    /* With -np 4, help must appear exactly once, not 4×.  Catches a regression
     * where print_help would skip its rank-0 guard.                            */
    DebugRun r = run_debug_capture("barrier_nb", 4, "-h");
    EXPECT_EQ(r.exit_code, 0) << "stderr:\n" << r.stderr_raw;
    int usage_count = 0;
    size_t p = 0;
    while ((p = r.stdout_raw.find("Usage:", p)) != std::string::npos) {
        usage_count++;
        p += 6;
    }
    EXPECT_EQ(usage_count, 1)
        << "help should print once (rank 0 only), got " << usage_count
        << " copies.\nstdout:\n" << r.stdout_raw;
}

/* ── auxiliary tools (checker, null_dummy) ──────────────────────────────────── */

TEST(Tools, CheckerRuns)
{
    /* checker = all-to-all plus a per-iteration timestamp log; just exercise it. */
    DebugRun r = run_debug_capture("checker", 8, "-iter 10");
    EXPECT_TRUE(r.normal_exit) << "stderr:\n" << r.stderr_raw;
    EXPECT_EQ(r.exit_code, 0)  << "stderr:\n" << r.stderr_raw;
    EXPECT_NE(r.stdout_raw.find("Measured"), std::string::npos)
        << "checker produced no results footer.\nstdout:\n" << r.stdout_raw;
}

TEST(Tools, NullDummyRuns)
{
    DebugRun r = run_debug_capture("null_dummy", 1, "");
    EXPECT_TRUE(r.normal_exit) << "stderr:\n" << r.stderr_raw;
    EXPECT_EQ(r.exit_code, 0)  << "stderr:\n" << r.stderr_raw;
}
