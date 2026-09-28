#include "engine.h"

#include <algorithm>

void unload_model(app_state & state) {
    if (state.sampler) {
        llama_sampler_free(state.sampler);
        state.sampler = nullptr;
    }
    if (state.ctx) {
        llama_free(state.ctx);
        state.ctx = nullptr;
    }
    if (state.model) {
        llama_model_free(state.model);
        state.model = nullptr;
    }
    state.loaded_model_path.clear();
}

bool load_model(app_state & state, std::string & err) {
    if (state.models.empty()) {
        err = "no models available";
        return false;
    }

    const auto & selected = state.models[state.selected_model];
    if (!file_exists(selected.path)) {
        err = "model file not found: " + selected.path;
        return false;
    }

    unload_model(state);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;

    state.model = llama_model_load_from_file(selected.path.c_str(), model_params);
    if (!state.model) {
        err = "failed to load model";
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = (uint32_t) clamp_ctx(state.n_ctx);
    ctx_params.n_batch = (uint32_t) std::min(512, clamp_ctx(state.n_ctx));
    ctx_params.no_perf = false;

    state.ctx = llama_init_from_model(state.model, ctx_params);
    if (!state.ctx) {
        unload_model(state);
        err = "failed to create context";
        return false;
    }

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
    return true;
}
