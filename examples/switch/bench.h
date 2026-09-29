#pragma once

// In-app benchmark of the loaded model, llama-bench style. Portable (no libnx): runs on the host too.
//
//   pp<N>  prompt processing: N synthetic tokens decoded from an empty context -> tokens/s
//   tg<N>  generation: N tokens decoded one by one (greedy) after a short fixed prompt -> tokens/s;
//          "GB/s" = model size x tokens/s, the weight bytes read per second while generating
//   mem    memory read bandwidth: a large buffer summed with NEON by 1..n threads, best of a few runs.
//          If tg GB/s is close to this, generation is bandwidth bound; far below it, compute bound.
//
// Every test runs once untimed (warm-up), then `reps` timed times with the KV cache cleared before each.
// The thread configurations are applied with engine_bench_set_threads() and the model's own pool is put
// back afterwards. The chat's cached tokens are dropped (generate_reset_cache), the chat history is kept.

#include "app.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

enum bench_test {
    BENCH_TEST_PP,
    BENCH_TEST_TG,
    BENCH_TEST_MEM,
};

struct bench_config {
    int  n_threads = 1;
    bool pinned    = false;
};

struct bench_params {
    int    n_pp      = 128;
    int    n_tg      = 32;
    int    reps      = 2;
    std::vector<bench_config> configs; // run in this order; pp and tg for each
    int    mem_threads = 0;            // mem test with 1..mem_threads threads (0 = no mem test)
    size_t mem_bytes   = 128u * 1024 * 1024;
};

struct bench_result {
    bench_test test      = BENCH_TEST_PP;
    int        n_threads = 1;
    bool       pinned    = false;
    int        n         = 0;       // tokens (pp/tg) or buffer MiB (mem)
    std::vector<double> samples;    // tokens/s (pp/tg) or GB/s (mem), one per timed run
    double     gbps      = 0.0;     // tg: model size x mean tokens/s; mem: best run
    uint32_t   cpu_hz    = 0;       // clocks read right after the last run (0 = unknown)
    uint32_t   gpu_hz    = 0;
    uint32_t   emc_hz    = 0;
    bool       clocks_changed = false; // CPU or EMC clock differed between the runs of this test
    std::string err;                // non-empty: the test failed
};

struct bench_progress {
    std::string step;       // e.g. "tg32, 3 threads, pinned, run 2/2"
    int         step_index = 0;
    int         n_steps    = 0;
    int         done       = 0; // tokens of the current step (tg), 0..total
    int         total      = 0;
    bool        timing     = false; // a timed run is in progress: keep the callback cheap
};

struct bench_hooks {
    // called before every step and after every generated token; return false to cancel
    std::function<bool(const bench_progress & p)> progress;
    // optional: current CPU/GPU/EMC clocks in Hz
    std::function<void(uint32_t & cpu_hz, uint32_t & gpu_hz, uint32_t & emc_hz)> read_clocks;
};

struct bench_report {
    // filled by the caller
    std::string timestamp;
    std::string app;        // app version / commit
    std::string device;     // operating mode, clocks setup, ...

    // filled by bench_run
    std::string model_name;
    std::string model_path;
    std::string model_desc; // llama_model_desc, e.g. "lfm2 1.2B Q4_0"
    uint64_t    model_size     = 0;
    uint64_t    model_n_params = 0;
    uint32_t    n_ctx     = 0;
    uint32_t    n_batch   = 0;
    uint32_t    n_ubatch  = 0;
    int         n_threads = 0; // the model's own configuration
    bool        pinned    = false;
    std::string flash_attn;
    std::string system_info;
    int         n_pp = 0;
    int         n_tg = 0;
    int         reps = 0;
    size_t      mem_bytes = 0;

    std::vector<bench_result> results;
    bool        cancelled  = false;
    std::string error;         // set when the benchmark could not run at all
    double      duration_s = 0.0;
};

// 1..n_threads unpinned, then n_threads pinned (when n_threads > 1)
std::vector<bench_config> bench_default_configs(int n_threads);

// Runs the benchmark on the loaded model. Returns false if it could not run (report.error) or was
// cancelled (report.cancelled); the results measured so far are kept in both cases.
bool bench_run(app_state & state, const bench_params & params, const bench_hooks & hooks, bench_report & report);

// mean and sample standard deviation (0 for fewer than two values)
void bench_stats(const std::vector<double> & v, double & mean, double & stdev);

// Memory read bandwidth in GB/s (1e9 bytes/s) of n_threads new threads summing `bytes` of buf, best of
// `runs`. The calling thread only waits. Returns 0 on bad arguments.
double bench_mem_read_gbps(const uint8_t * buf, size_t bytes, int n_threads, int runs);

// results table, at most 78 columns per line (screen and file)
std::vector<std::string> bench_format_table(const bench_report & r);

// the full plain-text block appended to bench.txt
std::string bench_format_report(const bench_report & r);

// appends text to path (creating its directory); false + err on failure
bool bench_append_file(const std::string & path, const std::string & text, std::string & err);

// local time as "YYYY-MM-DD HH:MM:SS"
std::string bench_timestamp();

// name of the loaded model's entry in state.models, else the file name of state.loaded_model_path
std::string bench_loaded_model_name(const app_state & state);

// snprintf into a std::string (output longer than 511 characters is cut)
std::string bench_printf(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
