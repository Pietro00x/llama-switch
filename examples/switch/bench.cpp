#include "bench.h"
#include "engine.h"
#include "generate.h"

#include "llama.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <new>
#include <thread>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

// short fixed prompt of the tg test; the generated tokens are greedy (argmax), so runs are repeatable
static const char * k_bench_tg_prompt = "The history of the Nintendo Switch began in";

static constexpr int k_bench_warmup_pp = 32; // tokens of the untimed prompt decoded per thread configuration
static constexpr int k_bench_mem_runs  = 3;  // mem test: best of this many runs, each reading the buffer once

std::vector<bench_config> bench_default_configs(int n_threads) {
    std::vector<bench_config> configs;
    for (int n = 1; n <= n_threads; ++n) {
        bench_config c;
        c.n_threads = n;
        c.pinned    = false;
        configs.push_back(c);
    }
    if (n_threads > 1) {
        bench_config c;
        c.n_threads = n_threads;
        c.pinned    = true;
        configs.push_back(c);
    }
    return configs;
}

void bench_stats(const std::vector<double> & v, double & mean, double & stdev) {
    mean  = 0.0;
    stdev = 0.0;
    if (v.empty()) {
        return;
    }
    for (double x : v) {
        mean += x;
    }
    mean /= (double) v.size();
    if (v.size() < 2) {
        return;
    }
    double sq = 0.0;
    for (double x : v) {
        sq += (x - mean) * (x - mean);
    }
    stdev = std::sqrt(sq / (double) (v.size() - 1));
}

//
// memory bandwidth
//

static std::atomic<uint64_t> g_bench_sink(0); // keeps the sums alive

static uint64_t bench_sum(const uint8_t * p, size_t n) {
    size_t   i   = 0;
    uint64_t sum = 0;
#if defined(__ARM_NEON)
    uint32x4_t a0 = vdupq_n_u32(0);
    uint32x4_t a1 = vdupq_n_u32(0);
    uint32x4_t a2 = vdupq_n_u32(0);
    uint32x4_t a3 = vdupq_n_u32(0);
    for (; i + 64 <= n; i += 64) {
        a0 = vaddq_u32(a0, vld1q_u32((const uint32_t *) (p + i)));
        a1 = vaddq_u32(a1, vld1q_u32((const uint32_t *) (p + i + 16)));
        a2 = vaddq_u32(a2, vld1q_u32((const uint32_t *) (p + i + 32)));
        a3 = vaddq_u32(a3, vld1q_u32((const uint32_t *) (p + i + 48)));
    }
    sum = vaddlvq_u32(vaddq_u32(vaddq_u32(a0, a1), vaddq_u32(a2, a3)));
#else
    uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (; i + 32 <= n; i += 32) {
        uint64_t w[4];
        std::memcpy(w, p + i, sizeof(w));
        s0 += w[0];
        s1 += w[1];
        s2 += w[2];
        s3 += w[3];
    }
    sum = s0 + s1 + s2 + s3;
#endif
    for (; i < n; ++i) {
        sum += p[i];
    }
    return sum;
}

double bench_mem_read_gbps(const uint8_t * buf, size_t bytes, int n_threads, int runs) {
    if (!buf || n_threads < 1 || runs < 1 || bytes < (size_t) n_threads * 64) {
        return 0.0;
    }
    // slices start on 64-byte boundaries, the last thread also takes the remainder
    const size_t slice = (bytes / (size_t) n_threads) & ~(size_t) 63;

    double best = 0.0;
    for (int run = 0; run < runs; ++run) {
        std::atomic<int>     ready(0);
        std::atomic<bool>    go(false);
        std::vector<int64_t> t_end((size_t) n_threads, 0);

        auto work = [&](int ith) {
            const size_t begin = (size_t) ith * slice;
            const size_t len   = ith == n_threads - 1 ? bytes - begin : slice;
            const uint64_t s   = bench_sum(buf + begin, len);
            t_end[(size_t) ith] = ggml_time_us();
            g_bench_sink.fetch_add(s, std::memory_order_relaxed);
        };

        // Only equal-priority worker threads read; the calling thread just waits. On the Switch the main
        // thread (priority 44) would starve a worker (0x3B) that the kernel placed on its core.
        std::vector<std::thread> workers;
        for (int ith = 0; ith < n_threads; ++ith) {
            workers.emplace_back([&, ith]() {
                ready.fetch_add(1);
                while (!go.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
                work(ith);
            });
        }
        // sleep, not spin, for the same reason
        while (ready.load() < n_threads) {
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }

        const int64_t t0 = ggml_time_us();
        go.store(true, std::memory_order_release);
        for (std::thread & w : workers) {
            w.join();
        }
        int64_t t1 = t0;
        for (int64_t t : t_end) {
            t1 = std::max(t1, t);
        }
        if (t1 > t0) {
            best = std::max(best, (double) bytes / ((double) (t1 - t0) * 1e-6) / 1e9);
        }
    }
    return best;
}

//
// llama tests
//

static bool bench_decode(llama_context * ctx, llama_token * tokens, int n, std::string & err) {
    const int32_t rc = llama_decode(ctx, llama_batch_get_one(tokens, n));
    if (rc != 0) {
        err = "llama_decode failed (" + std::to_string(rc) + ")";
        return false;
    }
    return true;
}

// greedy pick from the logits of the last decoded token (waits for the decode to finish)
static llama_token bench_argmax(llama_context * ctx, int32_t n_vocab) {
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (!logits) {
        return 0;
    }
    int32_t best = 0;
    for (int32_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return best;
}

// deterministic pseudo-random tokens (like llama-bench), BOS first if the model wants one
static std::vector<llama_token> bench_synthetic_tokens(const llama_vocab * vocab, int n) {
    const int32_t n_vocab = std::max(1, llama_vocab_n_tokens(vocab));
    std::vector<llama_token> tokens((size_t) n);
    uint32_t x = 12345;
    for (int i = 0; i < n; ++i) {
        x = x * 1664525u + 1013904223u;
        tokens[(size_t) i] = (llama_token) ((x >> 8) % (uint32_t) n_vocab);
    }
    if (n > 0 && llama_vocab_get_add_bos(vocab) && llama_vocab_bos(vocab) != LLAMA_TOKEN_NULL) {
        tokens[0] = llama_vocab_bos(vocab);
    }
    return tokens;
}

static std::vector<llama_token> bench_prompt_tokens(const llama_vocab * vocab) {
    const int32_t len = (int32_t) std::strlen(k_bench_tg_prompt);
    const int32_t n   = -llama_tokenize(vocab, k_bench_tg_prompt, len, nullptr, 0, true, false);
    std::vector<llama_token> tokens;
    if (n > 0) {
        tokens.resize((size_t) n);
        if (llama_tokenize(vocab, k_bench_tg_prompt, len, tokens.data(), n, true, false) != n) {
            tokens.clear();
        }
    }
    if (tokens.empty()) {
        tokens = bench_synthetic_tokens(vocab, 8);
    }
    return tokens;
}

static void bench_clear(llama_context * ctx) {
    llama_memory_t mem = llama_get_memory(ctx);
    if (mem) {
        llama_memory_clear(mem, true);
    }
}

// decodes `tokens` from an empty context in chunks of n_batch; tps = tokens/s of the whole prompt
static bool bench_pp_once(llama_context * ctx, std::vector<llama_token> & tokens, double & tps, std::string & err) {
    const int n       = (int) tokens.size();
    const int n_batch = std::max(1, (int) llama_n_batch(ctx));
    bench_clear(ctx);

    const int64_t t0 = ggml_time_us();
    for (int i = 0; i < n; i += n_batch) {
        if (!bench_decode(ctx, tokens.data() + i, std::min(n_batch, n - i), err)) {
            return false;
        }
    }
    llama_synchronize(ctx);
    const int64_t t1 = ggml_time_us();
    tps = t1 > t0 ? (double) n / ((double) (t1 - t0) * 1e-6) : 0.0;
    return true;
}

// the fixed prompt and one token untimed, then n_tg greedy tokens one by one; only decode + argmax
// are timed (the progress callback is not). Returns false on errors and when cancelled.
static bool bench_tg_once(
    llama_context * ctx, const llama_vocab * vocab, int n_tg, const bench_hooks & hooks, bench_progress & p,
    double & tps, bool & cancelled, std::string & err) {
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> prompt = bench_prompt_tokens(vocab);
    bench_clear(ctx);

    if (!bench_decode(ctx, prompt.data(), (int) prompt.size(), err)) {
        return false;
    }
    llama_token tok = bench_argmax(ctx, n_vocab);
    // untimed: the first single-token graph after the prompt
    if (!bench_decode(ctx, &tok, 1, err)) {
        return false;
    }
    tok = bench_argmax(ctx, n_vocab);

    int64_t t_total = 0;
    p.timing = true;
    for (int i = 0; i < n_tg; ++i) {
        const int64_t t0 = ggml_time_us();
        if (!bench_decode(ctx, &tok, 1, err)) {
            p.timing = false;
            return false;
        }
        tok = bench_argmax(ctx, n_vocab);
        t_total += ggml_time_us() - t0;

        p.done = i + 1;
        if (hooks.progress && !hooks.progress(p)) {
            cancelled = true;
            p.timing  = false;
            return false;
        }
    }
    p.timing = false;
    tps = t_total > 0 ? (double) n_tg / ((double) t_total * 1e-6) : 0.0;
    return true;
}

static std::string bench_config_name(int n_threads, bool pinned) {
    return std::to_string(n_threads) + (n_threads == 1 ? " thread" : " threads") + (pinned ? ", pinned" : "");
}

std::string bench_loaded_model_name(const app_state & state) {
    for (const model_entry & m : state.models) {
        if (m.path == state.loaded_model_path) {
            return m.name;
        }
    }
    const size_t slash = state.loaded_model_path.find_last_of('/');
    return slash == std::string::npos ? state.loaded_model_path : state.loaded_model_path.substr(slash + 1);
}

static void bench_fill_info(const app_state & state, const bench_params & params, bench_report & r) {
    r.model_path = state.loaded_model_path;
    r.model_name = bench_loaded_model_name(state);

    char desc[256] = { 0 };
    llama_model_desc(state.model, desc, sizeof(desc));
    r.model_desc     = desc;
    r.model_size     = llama_model_size(state.model);
    r.model_n_params = llama_model_n_params(state.model);
    r.n_ctx          = llama_n_ctx(state.ctx);
    r.n_batch        = llama_n_batch(state.ctx);
    r.n_ubatch       = llama_n_ubatch(state.ctx);
    r.n_threads      = state.n_threads;
    r.pinned         = state.threads_pinned;

    // engine.cpp keeps the llama.cpp default; "auto" enables it when every layer is on one device (CPU)
    const llama_flash_attn_type fa = llama_context_default_params().flash_attn_type;
    r.flash_attn = llama_flash_attn_type_name(fa);
    if (fa == LLAMA_FLASH_ATTN_TYPE_AUTO) {
        r.flash_attn += " (CPU only: on)";
    }

    const char * sys = llama_print_system_info();
    r.system_info = trim(sys ? sys : "");
    r.n_pp      = params.n_pp;
    r.n_tg      = params.n_tg;
    r.reps      = params.reps;
    r.mem_bytes = 0;
}

static void bench_sample_clocks(const bench_hooks & hooks, bench_result & r, bool first) {
    if (!hooks.read_clocks) {
        return;
    }
    uint32_t cpu = 0, gpu = 0, emc = 0;
    hooks.read_clocks(cpu, gpu, emc);
    const auto differs = [](uint32_t a, uint32_t b) { return (a > b ? a - b : b - a) > 1000000; };
    if (!first && (differs(cpu, r.cpu_hz) || differs(emc, r.emc_hz))) {
        r.clocks_changed = true;
    }
    r.cpu_hz = cpu;
    r.gpu_hz = gpu;
    r.emc_hz = emc;
}

bool bench_run(app_state & state, const bench_params & params, const bench_hooks & hooks, bench_report & report) {
    const int64_t t_start = ggml_time_us();
    report.results.clear();
    report.cancelled  = false;
    report.error.clear();
    report.duration_s = 0.0;

    if (!state.model || !state.ctx || !state.threadpool) {
        report.error = "no model loaded";
        return false;
    }
    bench_fill_info(state, params, report);
    if (llama_model_has_encoder(state.model)) {
        report.error = "encoder-decoder models are not supported";
        return false;
    }

    const int reps  = std::max(1, params.reps);
    const int n_pp  = std::max(1, params.n_pp);
    const int n_tg  = std::max(1, params.n_tg);
    const int n_mem = std::max(0, std::min(params.mem_threads, k_pool_threads_max));

    bench_progress p;
    p.n_steps = (int) params.configs.size() * 2 * reps + n_mem;
    int step = 0;
    const auto begin_step = [&](const std::string & name, int total) {
        p.step       = name;
        p.step_index = step++;
        p.done       = 0;
        p.total      = total;
        p.timing     = false;
        return !hooks.progress || hooks.progress(p);
    };

    // 1. memory bandwidth, first: the thread pools are idle now (asleep), later they would still be
    //    spinning for a few ms after the last decode and compete with the test threads
    if (n_mem > 0) {
        size_t    bytes = params.mem_bytes;
        uint8_t * buf   = nullptr;
        while (!buf && bytes >= 16u * 1024 * 1024) {
            buf = new (std::nothrow) uint8_t[bytes];
            if (!buf) {
                bytes /= 2;
            }
        }
        if (!buf) {
            bench_result r;
            r.test = BENCH_TEST_MEM;
            r.err  = "cannot allocate a 16 MiB buffer";
            report.results.push_back(r);
            step += n_mem;
        } else {
            report.mem_bytes = bytes;
            std::memset(buf, 1, bytes); // commits the pages before timing (host)
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            for (int n = 1; n <= n_mem; ++n) {
                if (!begin_step("mem read " + std::to_string(bytes >> 20) + " MiB, " + bench_config_name(n, false), 0)) {
                    report.cancelled = true;
                    break;
                }
                bench_result r;
                r.test      = BENCH_TEST_MEM;
                r.n_threads = n;
                r.n         = (int) (bytes >> 20);
                r.gbps      = bench_mem_read_gbps(buf, bytes, n, k_bench_mem_runs);
                r.samples.push_back(r.gbps);
                bench_sample_clocks(hooks, r, true);
                report.results.push_back(r);
            }
            delete[] buf;
        }
    }

    // 2. pp and tg for every thread configuration
    llama_context *     ctx   = state.ctx;
    const llama_vocab * vocab = llama_model_get_vocab(state.model);
    const int           n_ctx = (int) llama_n_ctx(ctx);
    const int           n_prompt_tg = (int) bench_prompt_tokens(vocab).size();

    // the chat's cached tokens are overwritten: forget them (the chat history itself is kept)
    generate_reset_cache(state);

    for (size_t ci = 0; ci < params.configs.size() && !report.cancelled; ++ci) {
        const bench_config & cfg = params.configs[ci];
        const std::string cfg_name = bench_config_name(cfg.n_threads, cfg.pinned);

        bench_result pp;
        pp.test      = BENCH_TEST_PP;
        pp.n_threads = cfg.n_threads;
        pp.pinned    = cfg.pinned;
        pp.n         = n_pp;
        bench_result tg = pp;
        tg.test = BENCH_TEST_TG;
        tg.n    = n_tg;

        std::string err;
        if (!engine_bench_set_threads(state, cfg.n_threads, cfg.pinned, err)) {
            pp.err = err;
            tg.err = err;
            report.results.push_back(pp);
            report.results.push_back(tg);
            step += 2 * reps;
            continue;
        }

        // prompt processing (step: where the progress continues even if a run fails or does not fit)
        int step_next = step + reps;
        if (n_pp > n_ctx) {
            pp.err = "pp" + std::to_string(n_pp) + " does not fit the context";
        } else {
            std::vector<llama_token> tokens = bench_synthetic_tokens(vocab, n_pp);
            for (int rep = 0; rep < reps; ++rep) {
                if (!begin_step("pp" + std::to_string(n_pp) + ", " + cfg_name + ", run " +
                        std::to_string(rep + 1) + "/" + std::to_string(reps), 0)) {
                    report.cancelled = true;
                    break;
                }
                double tps = 0.0;
                if (rep == 0) {
                    // untimed warm-up: wakes the pool and fills the caches
                    std::vector<llama_token> warm = bench_synthetic_tokens(vocab, std::min(n_pp, k_bench_warmup_pp));
                    if (!bench_pp_once(ctx, warm, tps, err)) {
                        pp.err = err;
                        break;
                    }
                }
                if (!bench_pp_once(ctx, tokens, tps, err)) {
                    pp.err = err;
                    break;
                }
                pp.samples.push_back(tps);
                bench_sample_clocks(hooks, pp, rep == 0);
            }
        }
        step = step_next;
        report.results.push_back(pp);
        if (report.cancelled) {
            break;
        }

        // generation
        step_next = step + reps;
        if (n_prompt_tg + 1 + n_tg > n_ctx) {
            tg.err = "tg" + std::to_string(n_tg) + " does not fit the context";
        } else {
            for (int rep = 0; rep < reps; ++rep) {
                if (!begin_step("tg" + std::to_string(n_tg) + ", " + cfg_name + ", run " +
                        std::to_string(rep + 1) + "/" + std::to_string(reps), n_tg)) {
                    report.cancelled = true;
                    break;
                }
                double tps = 0.0;
                bool cancelled = false;
                if (!bench_tg_once(ctx, vocab, n_tg, hooks, p, tps, cancelled, err)) {
                    if (cancelled) {
                        report.cancelled = true;
                    } else {
                        tg.err = err;
                    }
                    break;
                }
                tg.samples.push_back(tps);
                bench_sample_clocks(hooks, tg, rep == 0);
            }
        }
        step = step_next;
        if (!tg.samples.empty()) {
            double mean = 0.0, stdev = 0.0;
            bench_stats(tg.samples, mean, stdev);
            tg.gbps = (double) report.model_size * mean / 1e9;
        }
        report.results.push_back(tg);
    }

    engine_bench_restore_threads(state);
    generate_reset_cache(state);

    report.duration_s = (double) (ggml_time_us() - t_start) * 1e-6;
    return !report.cancelled;
}

//
// report
//

std::string bench_printf(const char * fmt, ...) {
    char buf[512];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    return std::string(buf);
}

static std::string bench_mhz(uint32_t hz) {
    return hz == 0 ? std::string("-") : std::to_string((hz + 500000) / 1000000);
}

static std::string bench_test_name(const bench_result & r) {
    switch (r.test) {
        case BENCH_TEST_PP:  return "pp" + std::to_string(r.n);
        case BENCH_TEST_TG:  return "tg" + std::to_string(r.n);
        case BENCH_TEST_MEM: return "mem-read";
    }
    return "?";
}

static std::string bench_format_row(const bench_result & r) {
    const std::string name = bench_test_name(r);
    const char *      pin  = r.test == BENCH_TEST_MEM ? "-" : (r.pinned ? "yes" : "no");
    if (!r.err.empty()) {
        return clip_line(bench_printf("%-8s %3d %3s  failed: %s", name.c_str(), r.n_threads, pin, r.err.c_str()), 78);
    }
    const std::string clocks = bench_printf("%5s %5s %5s%s",
        bench_mhz(r.cpu_hz).c_str(), bench_mhz(r.gpu_hz).c_str(), bench_mhz(r.emc_hz).c_str(), r.clocks_changed ? "*" : "");
    if (r.test == BENCH_TEST_MEM) {
        return bench_printf("%-8s %3d %3s %7s %6s %7s %6.2f %s",
            name.c_str(), r.n_threads, pin, "-", "-", "-", r.gbps, clocks.c_str());
    }
    double mean = 0.0, stdev = 0.0;
    bench_stats(r.samples, mean, stdev);
    const double ms = mean > 0.0 ? 1000.0 / mean : 0.0;
    const std::string gbps = r.test == BENCH_TEST_TG ? bench_printf("%6.2f", r.gbps) : std::string("     -");
    return bench_printf("%-8s %3d %3s %7.2f %6.2f %7.1f %s %s",
        name.c_str(), r.n_threads, pin, mean, stdev, ms, gbps.c_str(), clocks.c_str());
}

std::vector<std::string> bench_format_table(const bench_report & r) {
    std::vector<std::string> lines;
    lines.push_back(bench_printf("%-8s %3s %3s %7s %6s %7s %6s %5s %5s %5s",
        "test", "thr", "pin", "t/s", "+/-", "ms/t", "GB/s", "CPU", "GPU", "EMC"));
    // llama tests first, then the memory test (which runs first)
    for (int pass = 0; pass < 2; ++pass) {
        for (const bench_result & res : r.results) {
            if ((res.test == BENCH_TEST_MEM) != (pass == 1)) {
                continue;
            }
            if (res.samples.empty() && res.err.empty()) {
                continue; // cancelled before its first run
            }
            lines.push_back(bench_format_row(res));
        }
    }
    return lines;
}

std::string bench_format_report(const bench_report & r) {
    std::string out;
    const auto line = [&](const std::string & s) { out += s + "\n"; };

    line("=== llama.cpp Switch benchmark | " + r.timestamp + " ===");
    line("app     : " + r.app);
    line("device  : " + r.device);
    line("model   : " + r.model_name);
    line(bench_printf("          %s | %.1f MiB | %.2f B params",
        r.model_desc.c_str(), (double) r.model_size / (1024.0 * 1024.0), (double) r.model_n_params / 1e9));
    line("file    : " + r.model_path);
    line(bench_printf("context : n_ctx %u | n_batch %u | n_ubatch %u | flash_attn %s",
        (unsigned) r.n_ctx, (unsigned) r.n_batch, (unsigned) r.n_ubatch, r.flash_attn.c_str()));
    line(bench_printf("threads : model pool %d, %s", r.n_threads, r.pinned ? "pinned" : "unpinned"));
    std::string tests = bench_printf("pp%d, tg%d (greedy): %d timed run%s each after a warm-up",
        r.n_pp, r.n_tg, r.reps, r.reps == 1 ? "" : "s");
    if (r.mem_bytes > 0) {
        tests += bench_printf("; mem: %u MiB read, best of %d", (unsigned) (r.mem_bytes >> 20), k_bench_mem_runs);
    }
    line("tests   : " + tests);
    // system info can be long: wrap at '|' separators
    {
        std::string cur;
        bool first = true;
        size_t pos = 0;
        while (pos < r.system_info.size()) {
            size_t bar = r.system_info.find('|', pos);
            const size_t end = bar == std::string::npos ? r.system_info.size() : bar + 1;
            const std::string part = r.system_info.substr(pos, end - pos);
            if (!cur.empty() && cur.size() + part.size() > 68) {
                line(std::string(first ? "system  : " : "          ") + trim(cur));
                first = false;
                cur.clear();
            }
            cur += part;
            pos = end;
        }
        if (!cur.empty()) {
            line(std::string(first ? "system  : " : "          ") + trim(cur));
        }
    }
    for (const std::string & l : bench_format_table(r)) {
        line(l);
    }
    line("note    : tg GB/s = model size x t/s; clocks in MHz after the runs (* = changed during them)");
    if (!r.error.empty()) {
        line("result  : ERROR: " + r.error);
    } else if (r.cancelled) {
        line(bench_printf("result  : CANCELLED after %.1f s", r.duration_s));
    } else {
        line(bench_printf("result  : completed in %.1f s", r.duration_s));
    }
    return out;
}

bool bench_append_file(const std::string & path, const std::string & text, std::string & err) {
    ensure_parent_dir(path);
    FILE * f = std::fopen(path.c_str(), "a");
    if (!f) {
        err = "cannot open " + path;
        return false;
    }
    const bool ok = std::fwrite(text.data(), 1, text.size(), f) == text.size() && std::fputs("\n", f) >= 0;
    const bool closed = std::fclose(f) == 0;
    if (!ok || !closed) {
        err = "cannot write " + path;
        return false;
    }
    return true;
}

std::string bench_timestamp() {
    const std::time_t t = std::time(nullptr);
    const std::tm * tm = std::localtime(&t);
    char buf[32] = { 0 };
    if (!tm || std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", tm) == 0) {
        return "unknown time";
    }
    return std::string(buf);
}
