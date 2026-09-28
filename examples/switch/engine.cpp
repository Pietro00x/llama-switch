#include "engine.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <thread>

#ifdef __SWITCH__
#include <switch.h>

#include <mutex>
#endif

// Engine tuning for the Tegra X1 (4x Cortex-A57, ARMv8.0-A + NEON, 32 KB L1D per core, 2 MB shared L2).
//
// Threads: homebrew may only run on cores 0-2 (core 3 belongs to the OS), so the ggml default of
// 4 threads puts two workers on one core and every graph barrier waits for the descheduled one.
//
// Thread placement: ggml-cpu has no affinity implementation for Horizon (ggml_thread_apply_affinity()
// takes the "unsupported platform" no-op path and newlib has no pthread_setaffinity_np), so a
// threadpool cpumask would be silently ignored and none is set. libnx's pthread_create already gives
// every worker the whole process core mask with no fixed ideal core, which lets the kernel spread the
// workers over the idle cores. If that mask ever included core 3, workers could still land there;
// pinning them would need a Horizon branch in ggml_thread_apply_affinity() using svcSetThreadCoreMask.
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
    // allowed cores of this process, without core 3 (OS core: it would be the straggler at every barrier)
    u64 core_mask = 0;
    if (R_SUCCEEDED(svcGetInfo(&core_mask, InfoType_CoreMask, CUR_PROCESS_HANDLE, 0))) {
        const int n = __builtin_popcountll(core_mask & 0x7);
        if (n > 0) {
            return std::min(n, k_switch_max_threads);
        }
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
// model lifecycle
//

// frees everything but keeps the CPU boost state (load_model reuses it when switching models)
static void engine_free(app_state & state) {
    if (state.sampler) {
        llama_sampler_free(state.sampler);
        state.sampler = nullptr;
    }
    if (state.ctx) {
        llama_free(state.ctx);
        state.ctx = nullptr;
    }
    // after the context: it references the pool
    if (state.threadpool) {
        ggml_threadpool_free(state.threadpool);
        state.threadpool = nullptr;
    }
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
    // the prompt is decoded with a single llama_decode call and may use up to the whole context,
    // which must not exceed n_batch; the compute buffer size only depends on n_ubatch
    ctx_params.n_batch         = (uint32_t) n_ctx;
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

    // cpumask stays all-zero (default placement), see the note at the top of this file
    ggml_threadpool_params tp_params = ggml_threadpool_params_default(n_threads);
    state.threadpool = ggml_threadpool_new(&tp_params);
    llama_attach_threadpool(state.ctx, state.threadpool, nullptr);

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;

    state.sampler = llama_sampler_chain_init(sparams);
    if (!state.sampler) {
        unload_model(state);
        err = "failed to create sampler chain";
        return false;
    }

    llama_sampler_chain_add(state.sampler, llama_sampler_init_top_k(clamp_top_k(state.top_k)));
    llama_sampler_chain_add(state.sampler, llama_sampler_init_top_p(clamp_top_p(state.top_p), 1));
    llama_sampler_chain_add(state.sampler, llama_sampler_init_temp(clamp_temp(state.temperature)));
    llama_sampler_chain_add(state.sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    state.loaded_model_path = selected.path;
    state.load_ms   = (double) (ggml_time_us() - t_start_us) / 1000.0;
    state.n_threads = n_threads;
    state.n_ubatch  = n_ubatch;

    engine_log_printf("load_model: done in %.0f ms (n_ctx=%d n_threads=%d n_ubatch=%u cpu_boost=%d)\n",
        state.load_ms, n_ctx, n_threads, (unsigned) n_ubatch, state.cpu_boost ? 1 : 0);
    return true;
}
