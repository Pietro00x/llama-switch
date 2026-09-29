#include "engine.h"
#include "generate.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <thread>

#ifdef __SWITCH__
#include <switch.h>

#include <mutex>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

// Engine tuning for the Tegra X1 (4x Cortex-A57, ARMv8.0-A + NEON, 32 KB L1D per core, 2 MB shared L2).
//
// Threads: homebrew may only run on cores 0-2 (core 3 belongs to the OS), so the ggml default of
// 4 threads puts two workers on one core and every graph barrier waits for the descheduled one.
//
// Thread placement: by default no cpumask is set. libnx's pthread_create gives every worker the whole
// process core mask with no fixed ideal core, which lets the kernel spread the workers over the idle
// cores; the main thread (ggml worker 0, also the UI thread) stays on core 0. app_state::pin_threads
// (off by default, an experiment measured by the benchmark screen) pins every thread to one core of
// the process mask instead, through the Horizon branch of ggml_thread_apply_affinity()
// (svcSetThreadCoreMask). ggml's strict placement puts workers 1..n-1 on the first cores and the main
// thread on the last one; engine.cpp moves the main thread itself as well (see engine_attach_pool) and
// puts it back on its original core when an unpinned pool takes over, which ggml never does.
// Priorities stay at the libnx defaults (see ggml_thread_apply_priority).
//
// Threadpool: without one, ggml creates and joins a disposable pool (n_threads - 1 pthreads, each with
// a freshly allocated stack) on every llama_decode call, i.e. per generated token. A persistent pool
// avoids that and keeps the workers warm between tokens.
//
// n_ubatch: physical batch for prompt processing. The CPU compute buffer is dominated by the worst-case
// logits tensor, n_ubatch * n_vocab floats: ~516 MiB for a 262k vocab (Gemma 3) at the llama.cpp default
// of 512. Prefill speed is nearly flat in n_ubatch (host, 3 threads, 512-token prompt: 64 -> 512 changed
// prefill by < 5% for Qwen3-0.6B and Gemma 3 1B), so n_ubatch is the largest power of two in
// [k_n_ubatch_min, k_n_ubatch_max] that keeps the logits within k_logits_budget.
static constexpr int      k_switch_max_threads = 3;
static constexpr uint32_t k_n_ubatch_min       = 64;
static constexpr uint32_t k_n_ubatch_max       = 256;
static constexpr uint64_t k_logits_budget      = 64ull * 1024 * 1024;
// prompt tokens per llama_decode call during prefill (generate.cpp checks for B between calls)
static constexpr uint32_t k_n_batch_prefill    = 256;

//
// logging
//

#ifdef __SWITCH__
static const char *    k_log_path      = "sdmc:/switch/llama_cpp_switch/llama.log";
static const char *    k_log_path_old  = "sdmc:/switch/llama_cpp_switch/llama.log.1";
static constexpr long  k_log_max_bytes = 1024 * 1024;

static std::mutex g_log_mutex;
static FILE *     g_log_file = nullptr;
static long       g_log_size = 0;
static ggml_log_level g_log_last_level = GGML_LOG_LEVEL_INFO;

// caller holds g_log_mutex
static void engine_log_open() {
    ensure_parent_dir(k_log_path);
    g_log_file = std::fopen(k_log_path, "a");
    if (!g_log_file) {
        return;
    }
    std::fseek(g_log_file, 0, SEEK_END);
    g_log_size = std::max(0L, std::ftell(g_log_file));
}

// caller holds g_log_mutex; keeps at most two files of ~k_log_max_bytes (llama.log and llama.log.1)
static void engine_log_rotate() {
    if (g_log_file) {
        std::fclose(g_log_file);
        g_log_file = nullptr;
    }
    std::remove(k_log_path_old);
    if (std::rename(k_log_path, k_log_path_old) != 0) {
        std::remove(k_log_path);
    }
    engine_log_open();
}

// caller holds g_log_mutex
static void engine_log_write(const char * text, bool flush) {
    if (!g_log_file) {
        return;
    }
    const size_t len = std::strlen(text);
    if (g_log_size > 0 && g_log_size + (long) len > k_log_max_bytes) {
        engine_log_rotate();
        if (!g_log_file) {
            return;
        }
    }
    std::fwrite(text, 1, len, g_log_file);
    g_log_size += (long) len;
    if (flush) {
        std::fflush(g_log_file);
    }
}

static void engine_log_callback(ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    if (text == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_log_mutex);
    // continuation fragments (e.g. the load progress dots) inherit the level of the message they continue
    if (level == GGML_LOG_LEVEL_CONT) {
        level = g_log_last_level;
    } else {
        g_log_last_level = level;
    }
    if (level == GGML_LOG_LEVEL_DEBUG) {
        return;
    }
    engine_log_write(text, level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR);
}

// GGML_ABORT/GGML_ASSERT: keep the message in the log file before the process dies
static void engine_abort_callback(const char * message) {
    // try_lock: never deadlock if the abort fires while another thread is inside the logger
    if (g_log_mutex.try_lock()) {
        engine_log_write("GGML_ABORT: ", false);
        engine_log_write(message, false);
        engine_log_write("\n", true);
        g_log_mutex.unlock();
    }
    std::fprintf(stderr, "GGML_ABORT: %s\n", message);
}
#endif

static void engine_log_printf(const char * fmt, ...) __attribute__((format(printf, 1, 2)));

static void engine_log_printf(const char * fmt, ...) {
#ifdef __SWITCH__
    char buf[512];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    std::lock_guard<std::mutex> lock(g_log_mutex);
    engine_log_write(buf, true);
#else
    // host: stay quiet, the caller owns stderr
    (void) fmt;
#endif
}

static void engine_log_flush() {
#ifdef __SWITCH__
    std::lock_guard<std::mutex> lock(g_log_mutex);
    if (g_log_file) {
        std::fflush(g_log_file);
    }
#endif
}

void engine_log_init() {
#ifdef __SWITCH__
    static bool initialized = false;
    if (initialized) {
        return;
    }
    initialized = true;
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        engine_log_open();
        engine_log_write("\n=== llama.cpp Switch: new session ===\n", true);
    }
    llama_log_set(engine_log_callback, nullptr);
    ggml_set_abort_callback(engine_abort_callback);
#endif
}

//
// platform helpers
//

static int engine_pick_n_threads() {
#ifdef __SWITCH__
    // allowed cores of this process, without core 3 (OS core: it would be the straggler at every barrier);
    // engine_pin_cores() picks the pinned cores from the same mask
    const int n = __builtin_popcountll(engine_core_mask() & 0x7);
    if (n > 0) {
        return std::min(n, k_switch_max_threads);
    }
    return k_switch_max_threads;
#else
    // host: roughly the number of physical cores
    const int hw = (int) std::thread::hardware_concurrency();
    if (hw <= 0) {
        return GGML_DEFAULT_N_THREADS;
    }
    return std::max(1, std::min(hw / 2, 8));
#endif
}

// FastLoad boosts the CPU to 1785 MHz (and drops the GPU to its minimum, unused by the console UI).
// Available to all applet types on 7.0.0+; failures are logged and otherwise ignored.
static bool engine_set_cpu_boost(bool enable) {
#ifdef __SWITCH__
    const Result rc = appletSetCpuBoostMode(enable ? ApmCpuBoostMode_FastLoad : ApmCpuBoostMode_Normal);
    if (R_FAILED(rc)) {
        engine_log_printf("appletSetCpuBoostMode(%d) failed: 0x%x\n", enable ? 1 : 0, (unsigned) rc);
        return false;
    }
    return enable;
#else
    (void) enable;
    return false;
#endif
}

static uint32_t engine_pick_n_ubatch(const llama_model * model, uint32_t n_ctx) {
    // llama_encode needs the whole encoder input in one ubatch (asserts n_ubatch >= n_tokens)
    if (llama_model_has_encoder(model)) {
        return n_ctx;
    }
    const int32_t  n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint64_t row_size = (uint64_t) std::max(1, n_vocab) * sizeof(float);

    uint32_t n_ubatch = k_n_ubatch_max;
    while (n_ubatch > k_n_ubatch_min && n_ubatch * row_size > k_logits_budget) {
        n_ubatch /= 2;
    }
    return std::min(n_ubatch, n_ctx);
}

//
// thread pools
//

uint64_t engine_core_mask() {
#ifdef __SWITCH__
    u64 mask = 0;
    if (R_SUCCEEDED(svcGetInfo(&mask, InfoType_CoreMask, CUR_PROCESS_HANDLE, 0))) {
        return mask;
    }
#endif
    return 0;
}

// Cores used by pinned pools, lowest first; returns how many. Switch: the process core mask without
// core 3 (the OS core, see engine_pick_n_threads); host: the first online CPUs.
static int engine_pin_cores(int * cores) {
#ifdef __SWITCH__
    uint64_t mask = engine_core_mask() & 0x7;
    if (mask == 0) {
        mask = 0x7;
    }
#else
    const int hw = std::max(1, (int) std::thread::hardware_concurrency());
    const uint64_t mask = hw >= 64 ? ~0ull : (1ull << hw) - 1;
#endif
    int n = 0;
    for (int core = 0; core < 64 && n < k_pool_threads_max; ++core) {
        if ((mask >> core) & 1) {
            cores[n++] = core;
        }
    }
    return n;
}

// Core mask of the main thread before the first pinned pool moved it (ggml never moves it back).
struct engine_main_affinity {
    bool saved = false;
#ifdef __SWITCH__
    s32 core = 0;
    u64 mask = 0;
#elif defined(__linux__)
    cpu_set_t set;
#endif
};
static engine_main_affinity g_main_affinity;

// Moves the main thread (ggml worker 0) to one core. On the Switch it must never share a core with a
// pinned worker: at priority 44 it would spin in the graph barrier forever while the worker (0x3B) waits.
static void engine_main_affinity_pin(int core) {
#ifdef __SWITCH__
    if (!g_main_affinity.saved) {
        const Result rc = svcGetThreadCoreMask(&g_main_affinity.core, &g_main_affinity.mask, CUR_THREAD_HANDLE);
        if (R_FAILED(rc)) {
            // the main thread of a homebrew process starts on core 0 only
            engine_log_printf("svcGetThreadCoreMask(main) failed: 0x%x, will restore core 0\n", (unsigned) rc);
            g_main_affinity.core = 0;
            g_main_affinity.mask = 0x1;
        }
        g_main_affinity.saved = true;
    }
    const Result rc = svcSetThreadCoreMask(CUR_THREAD_HANDLE, core, 1u << core);
    if (R_FAILED(rc)) {
        engine_log_printf("svcSetThreadCoreMask(main, core %d) failed: 0x%x\n", core, (unsigned) rc);
    }
#elif defined(__linux__)
    if (!g_main_affinity.saved) {
        if (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &g_main_affinity.set) != 0) {
            CPU_ZERO(&g_main_affinity.set);
            const int hw = std::max(1, (int) std::thread::hardware_concurrency());
            for (int i = 0; i < hw && i < CPU_SETSIZE; ++i) {
                CPU_SET(i, &g_main_affinity.set);
            }
        }
        g_main_affinity.saved = true;
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &set);
#else
    (void) core;
#endif
}

static void engine_main_affinity_restore() {
    if (!g_main_affinity.saved) {
        return;
    }
#ifdef __SWITCH__
    const Result rc = svcSetThreadCoreMask(CUR_THREAD_HANDLE, g_main_affinity.core, (u32) g_main_affinity.mask);
    if (R_FAILED(rc)) {
        engine_log_printf("svcSetThreadCoreMask(main, restore core %d mask 0x%llx) failed: 0x%x\n",
            (int) g_main_affinity.core, (unsigned long long) g_main_affinity.mask, (unsigned) rc);
        return;
    }
#elif defined(__linux__)
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &g_main_affinity.set);
#endif
    g_main_affinity.saved = false;
}

static void engine_log_placement(const char * what, int n_threads, bool pinned) {
#ifdef __SWITCH__
    s32 core = 0;
    u64 mask = 0;
    if (R_FAILED(svcGetThreadCoreMask(&core, &mask, CUR_THREAD_HANDLE))) {
        core = -9;
    }
    engine_log_printf("%s: pool %d threads %s | process core mask 0x%llx | main thread on core %u (ideal %d, mask 0x%llx)\n",
        what, n_threads, pinned ? "pinned" : "unpinned", (unsigned long long) engine_core_mask(),
        (unsigned) svcGetCurrentProcessorNumber(), (int) core, (unsigned long long) mask);
#else
    engine_log_printf("%s: pool %d threads %s\n", what, n_threads, pinned ? "pinned" : "unpinned");
#endif
}

// Returns the context's pool for this configuration, creating it on first use (kept until engine_free).
static ggml_threadpool * engine_get_pool(app_state & state, int n_threads, bool pinned, std::string & err) {
    if (n_threads < 1 || n_threads > k_pool_threads_max) {
        err = "invalid thread count " + std::to_string(n_threads);
        return nullptr;
    }
    ggml_threadpool * & pool = state.threadpools[pinned ? 1 : 0][n_threads - 1];
    if (pool) {
        return pool;
    }

    ggml_threadpool_params tp_params = ggml_threadpool_params_default(n_threads);
    // workers sleep until the first graph; ggml also applies the main thread's placement at that resume
    tp_params.paused = true;
    if (pinned) {
        int cores[k_pool_threads_max];
        if (engine_pin_cores(cores) < n_threads) {
            err = "not enough cores to pin " + std::to_string(n_threads) + " threads";
            return nullptr;
        }
        // strict: one core each, workers 1..n-1 on cores[0..n-2], the main thread on cores[n-1] (the order of
        // ggml_thread_cpumask_next in ggml-cpu.c; engine_attach_pool relies on it)
        for (int i = 0; i < n_threads; ++i) {
            tp_params.cpumask[cores[i]] = true;
        }
        tp_params.strict_cpu = true;
    }
    pool = ggml_threadpool_new(&tp_params);
    if (!pool) {
        err = "failed to create the thread pool";
    }
    return pool;
}

// The CPU backend switches to the new pool (and pauses the old one) at the next decode.
static void engine_attach_pool(app_state & state, ggml_threadpool * pool, int n_threads, bool pinned) {
    llama_attach_threadpool(state.ctx, pool, nullptr);
    llama_set_n_threads(state.ctx, n_threads, n_threads);
    if (pinned) {
        // same core ggml gives worker 0, applied right away: ggml only does it when it resumes a paused
        // pool, and that pool may still be the backend's current one (not paused) after a swap
        int cores[k_pool_threads_max];
        engine_pin_cores(cores); // at least n_threads, checked by engine_get_pool
        engine_main_affinity_pin(cores[n_threads - 1]);
    } else {
        engine_main_affinity_restore();
    }
}

//
// model lifecycle
//

bool engine_rebuild_sampler(app_state & state, std::string & err) {
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;

    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    if (!smpl) {
        err = "failed to create sampler chain";
        return false;
    }

    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(clamp_top_k(state.top_k)));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(clamp_top_p(state.top_p), 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(clamp_temp(state.temperature)));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    if (state.sampler) {
        llama_sampler_free(state.sampler);
    }
    state.sampler = smpl;
    return true;
}

// frees everything but keeps the CPU boost state (load_model reuses it when switching models)
static void engine_free(app_state & state) {
    // drops the KV bookkeeping and state snapshots (up to 128 MB) that refer to the context
    generate_reset_cache(state);
    if (state.sampler) {
        llama_sampler_free(state.sampler);
        state.sampler = nullptr;
    }
    if (state.ctx) {
        llama_free(state.ctx);
        state.ctx = nullptr;
    }
    // after the context: it references the pools
    for (auto & by_n : state.threadpools) {
        for (ggml_threadpool * & pool : by_n) {
            if (pool) {
                ggml_threadpool_free(pool);
                pool = nullptr;
            }
        }
    }
    state.threadpool     = nullptr;
    state.threads_pinned = false;
    engine_main_affinity_restore();
    if (state.model) {
        llama_model_free(state.model);
        state.model = nullptr;
    }
    state.loaded_model_path.clear();
    state.load_ms   = 0.0;
    state.n_threads = 0;
    state.n_ubatch  = 0;
}

void unload_model(app_state & state) {
    engine_free(state);
    if (state.cpu_boost) {
        engine_set_cpu_boost(false);
        state.cpu_boost = false;
    }
    engine_log_flush();
}

bool load_model(app_state & state, std::string & err) {
    engine_log_init();

    if (state.models.empty()) {
        err = "no models available";
        return false;
    }

    const auto & selected = state.models[state.selected_model];
    if (!file_exists(selected.path)) {
        err = "model file not found: " + selected.path;
        return false;
    }

    engine_free(state);

    engine_log_printf("load_model: %s\n", selected.path.c_str());
    const int64_t t_start_us = ggml_time_us();

    // boost while a model is loaded: loading, prefill and generation are all CPU bound
    if (!state.cpu_boost) {
        state.cpu_boost = engine_set_cpu_boost(true);
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    // newlib has no mmap/mlock (llama_mmap::SUPPORTED is false on Switch): read the weights straight into
    // the CPU buffer instead of letting the loader fall back with a warning. Host keeps mmap.
    model_params.use_mmap      = llama_supports_mmap();
    model_params.use_mlock     = false;
    model_params.use_direct_io = false;

    state.model = llama_model_load_from_file(selected.path.c_str(), model_params);
    if (!state.model) {
        unload_model(state);
        err = "failed to load model";
        return false;
    }

    const int      n_ctx     = clamp_ctx(state.n_ctx);
    const int      n_threads = engine_pick_n_threads();
    const uint32_t n_ubatch  = engine_pick_n_ubatch(state.model, (uint32_t) n_ctx);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = (uint32_t) n_ctx;
    // generate.cpp decodes the prompt in chunks of n_batch tokens and calls the UI between chunks, so
    // n_batch sets how quickly B can stop a long prefill; the compute buffer size only depends on n_ubatch.
    // Encoder models need the whole input in one batch (n_ubatch == n_ctx for them, see above).
    ctx_params.n_batch         = std::min((uint32_t) n_ctx, std::max(n_ubatch, k_n_batch_prefill));
    ctx_params.n_ubatch        = n_ubatch;
    ctx_params.n_threads       = n_threads;
    ctx_params.n_threads_batch = n_threads;
    ctx_params.no_perf         = false;

    state.ctx = llama_init_from_model(state.model, ctx_params);
    if (!state.ctx) {
        unload_model(state);
        err = "failed to create context";
        return false;
    }

    // default placement unless the pinning experiment is on, see the note at the top of this file
    bool pinned = state.pin_threads;
    state.threadpool = engine_get_pool(state, n_threads, pinned, err);
    if (!state.threadpool && pinned) {
        engine_log_printf("load_model: %s, using unpinned threads\n", err.c_str());
        pinned = false;
        state.threadpool = engine_get_pool(state, n_threads, pinned, err);
    }
    if (!state.threadpool) {
        unload_model(state);
        return false;
    }
    state.threads_pinned = pinned;
    engine_attach_pool(state, state.threadpool, n_threads, pinned);
    engine_log_placement("load_model", n_threads, pinned);

    if (!engine_rebuild_sampler(state, err)) {
        unload_model(state);
        return false;
    }

    state.loaded_model_path = selected.path;
    state.load_ms   = (double) (ggml_time_us() - t_start_us) / 1000.0;
    state.n_threads = n_threads;
    state.n_ubatch  = n_ubatch;

    engine_log_printf("load_model: done in %.0f ms (n_ctx=%d n_threads=%d n_ubatch=%u cpu_boost=%d)\n",
        state.load_ms, n_ctx, n_threads, (unsigned) n_ubatch, state.cpu_boost ? 1 : 0);
    return true;
}

//
// thread configuration (pinning setting, benchmark)
//

bool engine_set_pin_threads(app_state & state, bool pinned, std::string & err) {
    if (!state.ctx || !state.threadpool) {
        state.pin_threads = pinned; // applied by the next load_model
        return true;
    }
    ggml_threadpool * pool = engine_get_pool(state, state.n_threads, pinned, err);
    if (!pool) {
        return false;
    }
    state.pin_threads    = pinned;
    state.threadpool     = pool;
    state.threads_pinned = pinned;
    engine_attach_pool(state, pool, state.n_threads, pinned);
    engine_log_placement("engine_set_pin_threads", state.n_threads, pinned);
    return true;
}

bool engine_bench_set_threads(app_state & state, int n_threads, bool pinned, std::string & err) {
    if (!state.ctx || !state.threadpool) {
        err = "no model loaded";
        return false;
    }
    // more threads than the model's pool (= usable cores) would put two threads on one core
    if (n_threads < 1 || n_threads > state.n_threads) {
        err = "thread count must be 1.." + std::to_string(state.n_threads);
        return false;
    }
    ggml_threadpool * pool = engine_get_pool(state, n_threads, pinned, err);
    if (!pool) {
        return false;
    }
    engine_attach_pool(state, pool, n_threads, pinned);
    engine_log_placement("engine_bench_set_threads", n_threads, pinned);
    return true;
}

void engine_bench_restore_threads(app_state & state) {
    if (!state.ctx || !state.threadpool) {
        return;
    }
    engine_attach_pool(state, state.threadpool, state.n_threads, state.threads_pinned);
    engine_log_placement("engine_bench_restore_threads", state.n_threads, state.threads_pinned);
    engine_log_flush();
}
