#include "generate.h"

#include <algorithm>
#include <chrono>
#include <vector>

using clock_type = std::chrono::steady_clock;

static double elapsed_ms(clock_type::time_point t0) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
}

static std::string build_prompt(const std::vector<chat_turn> & turns, size_t start, const std::string & user_input) {
    std::string prompt;
    for (size_t i = start; i < turns.size(); ++i) {
        prompt += "User: " + turns[i].user + "\n";
        prompt += "Assistant: " + turns[i].assistant + "\n";
    }
    prompt += "User: " + user_input + "\nAssistant:";
    return prompt;
}

static bool build_prompt_fit_context(
    const app_state & state,
    const std::string & user_input,
    std::string & prompt_out,
    std::string & err) {
    if (!state.model || !state.ctx) {
        err = "model is not loaded";
        return false;
    }

    const llama_vocab * vocab = llama_model_get_vocab(state.model);
    size_t start = state.chat_turns.size() > 16 ? state.chat_turns.size() - 16 : 0;

    while (start <= state.chat_turns.size()) {
        std::string prompt = build_prompt(state.chat_turns, start, user_input);
        int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
        if (n_prompt > 0 && n_prompt < clamp_ctx(state.n_ctx) - 32) {
            prompt_out = std::move(prompt);
            return true;
        }
        if (start == state.chat_turns.size()) {
            break;
        }
        ++start;
    }

    err = "prompt too long for current context";
    return false;
}


bool generate_assistant_reply(
    app_state & state,
    const std::string & user_input,
    std::string & assistant_out,
    const gen_partial_cb & on_partial,
    gen_stats & stats,
    std::string & err) {
    stats = gen_stats();
    if (!state.model || !state.ctx || !state.sampler) {
        err = "model is not loaded";
        return false;
    }

    std::string prompt;
    if (!build_prompt_fit_context(state, user_input, prompt, err)) {
        return false;
    }

    const llama_vocab * vocab = llama_model_get_vocab(state.model);
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    if (n_prompt <= 0) {
        err = "failed to tokenize prompt";
        return false;
    }

    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        err = "failed to tokenize prompt";
        return false;
    }

    llama_memory_t mem = llama_get_memory(state.ctx);
    if (mem) {
        llama_memory_clear(mem, false);
    }
    llama_sampler_reset(state.sampler);

    const auto t_prefill = clock_type::now();

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    int n_pos = batch.n_tokens;

    if (llama_model_has_encoder(state.model)) {
        if (llama_encode(state.ctx, batch)) {
            err = "failed to encode prompt";
            return false;
        }
        llama_token decoder_start_token_id = llama_model_decoder_start_token(state.model);
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
            decoder_start_token_id = llama_vocab_bos(vocab);
        }
        batch = llama_batch_get_one(&decoder_start_token_id, 1);
        n_pos = 1;
    }

    if (llama_decode(state.ctx, batch)) {
        err = "failed to evaluate prompt";
        return false;
    }

    stats.n_prompt   = n_prompt;
    stats.prefill_ms = elapsed_ms(t_prefill);

    const auto t_gen = clock_type::now();

    assistant_out.clear();
    const int max_predict = std::min(256, clamp_ctx(state.n_ctx) / 2);
    for (int i = 0; i < max_predict && n_pos < clamp_ctx(state.n_ctx) - 1; ++i) {
        llama_token tok = llama_sampler_sample(state.sampler, state.ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) {
            break;
        }

        char piece[256];
        int n_piece = llama_token_to_piece(vocab, tok, piece, sizeof(piece), 0, true);
        if (n_piece < 0) {
            err = "failed converting token to text";
            return false;
        }
        assistant_out.append(piece, (size_t) n_piece);

        batch = llama_batch_get_one(&tok, 1);
        if (llama_decode(state.ctx, batch)) {
            err = "failed during decoding";
            return false;
        }
        ++n_pos;

        stats.n_gen  = i + 1;
        stats.gen_ms = elapsed_ms(t_gen);
        if (on_partial && !on_partial(assistant_out, stats)) {
            stats.cancelled = true;
            break;
        }

        if (assistant_out.size() > 12 && assistant_out.find("\nUser:") != std::string::npos) {
            size_t p = assistant_out.find("\nUser:");
            assistant_out.resize(p);
            break;
        }
    }

    return true;
}
