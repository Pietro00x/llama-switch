#pragma once

// Shared state for the llama.cpp Switch app.
// Keep this header free of libnx includes so portable parts (util.cpp, generate.cpp)
// can also be compiled on the host.

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

constexpr int k_ctx_min = 256;
constexpr int k_ctx_max = 8192;
constexpr int k_pool_threads_max = 8; // largest thread pool engine.cpp creates (host; the Switch uses 3)

// CPU/memory clock policy while a model is loaded (clocks.h), persisted in settings.json (settings.h)
enum clock_mode {
    CLOCK_MODE_STOCK = 0, // leave the system clocks alone
    CLOCK_MODE_BOOST,     // official maximum: CPU 1785 MHz, EMC (memory) 1600 MHz, also in handheld
    CLOCK_MODE_OVERCLOCK, // highest rates the system reports above that (needs sys-clk-OC)
    CLOCK_MODE_COUNT,
};

struct model_entry {
    std::string name;
    std::string url;
    std::string path;
    int   default_ctx  = 4096;
    float default_temp = 0.8f;
    float default_top_p = 0.95f;
    int   default_top_k = 40;
    std::string system_prompt; // optional; empty = none
    std::string chat_template; // optional override; empty = use the template stored in the GGUF

    // Runtime-only fields (never written to models.json). File fields are refreshed by
    // refresh_model_files() / load_model_list() in models.cpp, not every frame.
    uint64_t    file_size    = 0;     // bytes; 0 when the file is not on the SD card
    bool        on_sd        = false; // a regular file exists at `path`
    std::string arch         = "";    // GGUF general.architecture; empty if unknown
    bool        from_scan    = false; // found in the models dir but not listed in models.json
    bool        from_catalog = false; // built-in catalog entry that models.json does not list
};

struct chat_turn {
    std::string user;
    std::string assistant;
    bool stopped = false; // display only: the reply was cancelled by the user
};

// Snapshot of the part of the memory that cannot be rolled back (recurrent state of recurrent/hybrid models).
struct chat_kv_checkpoint {
    size_t n_tokens = 0;       // number of leading cached tokens it covers
    std::vector<uint8_t> data;
};

// Bookkeeping of what generate.cpp left in the KV cache (sequence 0) of a llama_context.
// Validated on every reply against the live context, so it is safe to leave stale after unload/reload.
struct chat_kv_cache {
    const llama_context * ctx = nullptr;         // context the tokens below belong to
    std::vector<llama_token> tokens;             // tokens at positions 0..n-1
    size_t first_turn = 0;                       // oldest chat_turns index still in the prompt window
    std::vector<chat_kv_checkpoint> checkpoints; // oldest first
};

struct app_state {
    std::vector<model_entry> models;
    size_t selected_model = 0;

    int n_ctx = 4096;
    float temperature = 0.8f;
    float top_p = 0.95f;
    int top_k = 40;

    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    llama_sampler * sampler = nullptr;
    ggml_threadpool * threadpool = nullptr; // owned by engine.cpp, attached to ctx
    std::string loaded_model_path;
    std::vector<chat_turn> chat_turns;
    chat_kv_cache kv_cache; // owned by generate.cpp

    // engine settings of the loaded model, filled by load_model (reset by unload_model)
    double   load_ms   = 0.0;   // wall time of the last successful load_model
    int      n_threads = 0;     // CPU threads used for prompt and generation
    uint32_t n_ubatch  = 0;     // physical batch size used for prompt processing
    bool     cpu_boost = false; // clocks are raised above the system configuration (Switch only)

    // clocks (clocks.h): the chosen mode and what is applied right now
    clock_mode clock_setting   = CLOCK_MODE_BOOST; // chosen in the menu, applied while a model is loaded
    uint32_t   clock_oc_cpu_hz = 0;                // CPU step in overclock mode, 0 = highest available
    clock_mode clock_applied   = CLOCK_MODE_STOCK; // effective mode now (STOCK = untouched; OC falls back to BOOST)
    uint32_t   clock_cpu_target_hz = 0;            // CPU rate we asked for, 0 = untouched
    uint32_t   clock_emc_target_hz = 0;            // EMC rate we asked for, 0 = untouched
    bool       clock_oc_available  = false;        // the system reports CPU or EMC rates above the official max
    bool       clock_sysclk        = false;        // sys-clk is running and may override the clocks

    // thread pinning experiment (default off): one core per ggml thread instead of the kernel's placement.
    // pin_threads is the setting (applied by load_model / engine_set_pin_threads), threads_pinned what
    // `threadpool` uses.
    bool pin_threads    = false;
    bool threads_pinned = false;
    // every pool created for the loaded context, by [pinned][n_threads - 1] (engine.cpp; `threadpool` is one
    // of them), reused by later benchmark runs and freed after the context. The CPU backend keeps a pointer
    // to the pool of its last decode and pauses it when the next decode uses another one, so a swapped-out
    // pool may still be referenced; engine.cpp does not track which, and the table is small (Switch: at
    // most 6 pools with 6 worker threads in total, sleeping while unused).
    ggml_threadpool * threadpools[2][k_pool_threads_max] = {};

    uint64_t process_total = 0;
    uint64_t process_used = 0;
    uint32_t cpu_hz = 0;
    uint32_t gpu_hz = 0;
    uint32_t emc_hz = 0;
    uint64_t last_stats_us = 0;

    std::string status;
    bool quit = false;
};

// util.cpp
std::string to_lower_ascii(std::string s);
std::string trim(const std::string & s);
std::string fmt_mb(uint64_t bytes);
std::string fmt_hz(uint32_t hz);
int   clamp_ctx(int n_ctx);
float clamp_temp(float t);
float clamp_top_p(float p);
int   clamp_top_k(int k);
void  ensure_parent_dir(const std::string & file_path); // mkdir -p of the file's directory
bool  file_exists(const std::string & path);
std::string clip_line(const std::string & input, size_t max_len = 78);
