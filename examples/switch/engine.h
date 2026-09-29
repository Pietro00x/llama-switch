#pragma once

#include "app.h"

#include <string>

// Loads the selected model from state.models into state.model/ctx/sampler (CPU only).
// On success also fills state.load_ms/n_threads/n_ubatch/cpu_boost.
bool load_model(app_state & state, std::string & err);

// (Re)creates state.sampler from state.top_k/top_p/temperature, e.g. after the menu changed them.
// The old sampler is kept if creating the new one fails.
bool engine_rebuild_sampler(app_state & state, std::string & err);

// Frees sampler, context, thread pool and model. Safe to call when nothing is loaded.
void unload_model(app_state & state);

// Replaces the loaded model's thread pool according to pinned (updates state.pin_threads and
// state.threads_pinned). Keeps the old pool if the new one cannot be created.
bool engine_set_pin_threads(app_state & state, bool pinned, std::string & err);

// Benchmark support: attaches a temporary pool of n_threads (pinned: one core each) to the loaded
// context and makes it use n_threads for prompt and generation; the model's own pool is kept (paused).
// engine_bench_restore_threads() puts the model's pool and thread count back; unload_model() also does.
bool engine_bench_set_threads(app_state & state, int n_threads, bool pinned, std::string & err);
void engine_bench_restore_threads(app_state & state);

// Bitmask of the cores this process may use (Switch: svcGetInfo(InfoType_CoreMask); host: 0 = unknown).
uint64_t engine_core_mask();

// Redirects libllama/ggml logging away from the console (Switch: sdmc:/switch/llama_cpp_switch/llama.log,
// host: unchanged stderr). Called by load_model; call it earlier (before llama_backend_init) to also
// capture backend init logs. Safe to call more than once.
void engine_log_init();
