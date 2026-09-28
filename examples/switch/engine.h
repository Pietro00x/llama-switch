#pragma once

#include "app.h"

#include <string>

// Loads the selected model from state.models into state.model/ctx/sampler (CPU only).
// On success also fills state.load_ms/n_threads/n_ubatch/cpu_boost.
bool load_model(app_state & state, std::string & err);

// Frees sampler, context, thread pool and model. Safe to call when nothing is loaded.
void unload_model(app_state & state);

// Redirects libllama/ggml logging away from the console (Switch: sdmc:/switch/llama_cpp_switch/llama.log,
// host: unchanged stderr). Called by load_model; call it earlier (before llama_backend_init) to also
// capture backend init logs. Safe to call more than once.
void engine_log_init();
