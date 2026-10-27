#pragma once
/*
 * popen_helpers.h — shared helpers for spawning a benchmark under mpirun
 * and parsing structured DEBUG output.
 *
 * Used by every test_*.cpp.  Provides:
 *   - run_debug_capture(): launches the benchmark, returns parsed DEBUG
 *     fields keyed by rank, plus exit-status decoded via WIFEXITED/WEXITSTATUS.
 *     stderr is redirected into a temp file and surfaced on failure so the
 *     diagnostic is not lost.
 *
 * Configure-time defaults are baked in by CMake via -DBLINK_MPIEXEC=...
 * etc.  Each can be overridden at run time via environment variables
 * (BLINK_MPIEXEC, BLINK_NPROC_FLAG, BLINK_BIN_DIR).
 */
#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace blink {

inline std::string env_or(const char *key, const char *def)
{
    const char *v = std::getenv(key);
    return v ? v : def;
}

inline std::string mpiexec()    { return env_or("BLINK_MPIEXEC",    BLINK_MPIEXEC);    }
inline std::string nproc_flag() { return env_or("BLINK_NPROC_FLAG", BLINK_NPROC_FLAG); }
inline std::string bin_dir()    { return env_or("BLINK_BIN_DIR",    BLINK_BIN_DIR);    }

/* true when the suite should run every benchmark in GPU mode (-gpu appended). */
inline bool gpu_mode() { return env_or("BLINK_TEST_GPU", "0") == "1"; }

/* A single parsed DEBUG line: rank => key/value map. */
using DebugFields = std::map<std::string, std::string>;
using DebugMap    = std::map<int, DebugFields>;

/* Result of a debug run. */
struct DebugRun {
    DebugMap    ranks;          /* parsed DEBUG lines, keyed by rank        */
    int         exit_code;      /* WEXITSTATUS, or -signum if killed        */
    bool        normal_exit;    /* true iff WIFEXITED                       */
    std::string stdout_raw;     /* full stdout                              */
    std::string stderr_raw;     /* full stderr (drained from temp file)     */
    std::string command;        /* the command line that was run            */
};

/* Parse a single DEBUG line into key/value fields.  Returns rank, or -1
 * if the line does not begin with "DEBUG rank=<int>". */
inline int parse_debug_line(const std::string& line, DebugFields& out)
{
    auto p = line.find("DEBUG rank=");
    if (p == std::string::npos) return -1;
    p += strlen("DEBUG rank=");
    int rank = 0;
    size_t consumed = 0;
    try { rank = std::stoi(line.substr(p), &consumed); }
    catch (...) { return -1; }
    out["rank"] = std::to_string(rank);
    p += consumed;
    /* parse key=val tokens separated by whitespace */
    while (p < line.size()) {
        while (p < line.size() && isspace((unsigned char)line[p])) p++;
        size_t eq = line.find('=', p);
        if (eq == std::string::npos) break;
        std::string k = line.substr(p, eq - p);
        size_t v_start = eq + 1;
        size_t v_end   = v_start;
        while (v_end < line.size() && !isspace((unsigned char)line[v_end])) v_end++;
        std::string v = line.substr(v_start, v_end - v_start);
        out[k] = v;
        p = v_end;
    }
    return rank;
}

/* Spawn the benchmark, capture stdout + stderr, parse DEBUG lines, decode
 * the wait status, and return a DebugRun.  This does NOT assert anything —
 * callers should check `run.normal_exit && run.exit_code == 0` themselves
 * and use gtest macros (so context strings can be tailored per call).    */
inline DebugRun run_debug_capture(const std::string& binary,
                                  int nprocs,
                                  const std::string& extra = "")
{
    DebugRun r;
    char errfile_template[] = "/tmp/blink_test_err_XXXXXX";
    int fd = mkstemp(errfile_template);
    std::string errfile = (fd >= 0) ? errfile_template : "/dev/null";
    if (fd >= 0) close(fd);

    r.command = mpiexec() + " " + nproc_flag() + " "
              + std::to_string(nprocs) + " "
              + bin_dir() + "/" + binary
              + " -debug -iter 1 -warmup 0 " + (gpu_mode() ? "-gpu " : "") + extra
              + " 2>" + errfile;

    FILE *pipe = popen(r.command.c_str(), "r");
    if (!pipe) {
        r.exit_code   = -1;
        r.normal_exit = false;
        return r;
    }

    char line[1024];
    while (fgets(line, sizeof(line), pipe)) {
        r.stdout_raw += line;
        DebugFields fields;
        int rank = parse_debug_line(line, fields);
        if (rank >= 0) {
            /* merge: a rank may print several DEBUG lines (e.g. the -gpu runtime's own) */
            auto &dst = r.ranks[rank];
            for (auto &kv : fields) dst[kv.first] = kv.second;
        }
    }
    int status = pclose(pipe);
    if (status == -1) {
        r.exit_code   = -1;
        r.normal_exit = false;
    } else if (WIFEXITED(status)) {
        r.exit_code   = WEXITSTATUS(status);
        r.normal_exit = true;
    } else if (WIFSIGNALED(status)) {
        r.exit_code   = -WTERMSIG(status);
        r.normal_exit = false;
    } else {
        r.exit_code   = status;
        r.normal_exit = false;
    }

    /* drain stderr from the temp file */
    if (errfile != "/dev/null") {
        FILE *ef = fopen(errfile.c_str(), "r");
        if (ef) {
            char buf[1024];
            size_t n;
            while ((n = fread(buf, 1, sizeof(buf), ef)) > 0)
                r.stderr_raw.append(buf, n);
            fclose(ef);
        }
        unlink(errfile.c_str());
    }
    return r;
}

/* Convenience: assert the benchmark ran successfully and return its parsed
 * DEBUG lines.  On failure, dump captured stderr / stdout to gtest output. */
inline DebugMap run_debug(const std::string& binary,
                         int nprocs,
                         const std::string& extra = "")
{
    DebugRun r = run_debug_capture(binary, nprocs, extra);
    EXPECT_TRUE(r.normal_exit)
        << "command: " << r.command << "\n"
        << "exit_code: " << r.exit_code << "\n"
        << "stderr:\n" << r.stderr_raw << "\n"
        << "stdout:\n" << r.stdout_raw;
    EXPECT_EQ(r.exit_code, 0)
        << "command: " << r.command << "\n"
        << "stderr:\n" << r.stderr_raw << "\n"
        << "stdout:\n" << r.stdout_raw;
    return r.ranks;
}

/* Look up an integer field for the given rank.  Returns -1 if absent. */
inline int get_int(const DebugMap& m, int rank, const std::string& key, int def = -1)
{
    auto rit = m.find(rank);
    if (rit == m.end()) return def;
    auto fit = rit->second.find(key);
    if (fit == rit->second.end()) return def;
    try { return std::stoi(fit->second); }
    catch (...) { return def; }
}

/* Look up a string field for the given rank.  Returns "" if absent. */
inline std::string get_str(const DebugMap& m, int rank, const std::string& key)
{
    auto rit = m.find(rank);
    if (rit == m.end()) return "";
    auto fit = rit->second.find(key);
    if (fit == rit->second.end()) return "";
    return fit->second;
}

/* Assert that every rank emitted a check= field AND reported check=OK.
 * Requiring the field to be present (rather than silently skipping ranks that
 * omit it) makes data-integrity tests fail loudly if a benchmark ever stops
 * emitting its self-check, instead of passing vacuously.                    */
inline void check_all_ok(const DebugMap& m, const std::string& ctx)
{
    for (const auto& [r, fields] : m) {
        auto it = fields.find("check");
        EXPECT_NE(it, fields.end())
            << ctx << ": rank " << r << " emitted no check= field";
        if (it == fields.end()) continue;
        EXPECT_EQ(it->second, "OK")
            << ctx << ": rank " << r << " check=" << it->second;
    }
}

/* Assert that ranks 0..n-1 are all present in the map. */
inline void check_coverage(const DebugMap& m, int n, const std::string& ctx)
{
    ASSERT_EQ((int)m.size(), n)
        << ctx << ": expected " << n << " DEBUG lines, got " << m.size();
    for (int r = 0; r < n; r++)
        EXPECT_TRUE(m.count(r)) << ctx << ": rank " << r << " missing";
}

/* Assert that every rank that emits nprocs= reports the expected value. */
inline void check_nprocs(const DebugMap& m, int expected, const std::string& ctx)
{
    for (const auto& [r, fields] : m) {
        auto it = fields.find("nprocs");
        if (it == fields.end()) continue;
        EXPECT_EQ(std::stoi(it->second), expected)
            << ctx << ": rank " << r << " nprocs=" << it->second
            << " expected " << expected;
    }
}

} // namespace blink
