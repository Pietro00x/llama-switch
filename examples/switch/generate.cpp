#include "generate.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <vector>

using clock_type = std::chrono::steady_clock;

static const char * k_think_open    = "<think>";
static const char * k_think_close   = "</think>";
static const char * k_think_force   = "\n</think>\n\n"; // injected when the thinking budget runs out
static const char * k_think_empty   = "<think>\n\n</think>\n\n"; // Qwen3-style "thinking disabled" prefix
static const char * k_no_think      = "/no_think";                // system prompt switch (Qwen3, SmolLM3)
static const char * k_fallback_stop = "\nUser:";

constexpr int k_max_answer_tokens = 1024; // generated tokens outside a <think> block
constexpr int k_max_think_tokens  = 1024; // then the <think> block is closed by force
constexpr size_t k_max_turns      = 48;   // history turns considered at most (bounds the prompt fitting loop)

static double elapsed_ms(clock_type::time_point t0) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
}

//
// reasoning blocks
//

bool reply_split_thinking(const std::string & reply, std::string & thought, std::string & answer) {
    const size_t open  = reply.find(k_think_open);
    const size_t close = reply.find(k_think_close, open == std::string::npos ? 0 : open);
    if (open == std::string::npos && close == std::string::npos) {
        thought.clear();
        answer = reply;
        return false;
    }

    // "</think>" without "<think>": the template opened the block, everything before it is reasoning
    const size_t head   = open == std::string::npos ? 0 : open;
    const size_t t_from = open == std::string::npos ? 0 : open + std::strlen(k_think_open);
    if (close == std::string::npos) {
        // still thinking (or stopped while thinking)
        thought = trim(reply.substr(t_from));
        answer  = trim(reply.substr(0, head));
        return true;
    }
    thought = trim(reply.substr(t_from, close - t_from));
    answer  = trim(reply.substr(0, head) + reply.substr(close + std::strlen(k_think_close)));
    return true;
}

std::string reply_strip_thinking(const std::string & reply) {
    std::string thought;
    std::string answer;
    reply_split_thinking(reply, thought, answer);
    return answer;
}

//
// text helpers
//

static bool is_space(char c) {
    return c == ' ' || c == '\n' || c == '\r' || c == '\t';
}

// length of s without a trailing incomplete UTF-8 sequence
static size_t utf8_complete_len(const std::string & s) {
    const size_t n = s.size();
    for (size_t k = 1; k <= 4 && k <= n; ++k) {
        const unsigned char c = (unsigned char) s[n - k];
        if ((c & 0xC0) == 0x80) {
            continue; // continuation byte
        }
        const size_t need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        return need > k ? n - k : n;
    }
    return n;
}

// length of the longest suffix of s[0, n) that is a proper prefix of stop
static size_t stop_prefix_len(const std::string & s, size_t n, const std::string & stop) {
    for (size_t len = std::min(n, stop.size() - 1); len > 0; --len) {
        if (s.compare(n - len, len, stop, 0, len) == 0) {
            return len;
        }
    }
    return 0;
}

// the part of the raw reply that can be shown: complete UTF-8 only; while streaming, a trailing
// partial match of the stop string is held back
static std::string reply_visible(const std::string & raw, const std::string & stop, bool streaming, bool trim_left) {
    size_t n = utf8_complete_len(raw);
    if (streaming && !stop.empty()) {
        n -= stop_prefix_len(raw, n, stop);
    }
    std::string out = raw.substr(0, n);

    // an empty reasoning block ("thinking disabled") carries nothing worth showing
    const size_t open = out.find(k_think_open);
    if (open != std::string::npos) {
        size_t e = open + std::strlen(k_think_open);
        while (e < out.size() && is_space(out[e])) {
            ++e;
        }
        if (streaming && e == out.size()) {
            out.erase(open); // may still become an empty block
        } else if (out.compare(e, std::strlen(k_think_close), k_think_close) == 0) {
            e += std::strlen(k_think_close);
            while (e < out.size() && is_space(out[e])) {
                ++e;
            }
            out.erase(open, e - open);
        }
    }

    size_t b = 0;
    if (trim_left) {
        while (b < out.size() && is_space(out[b])) {
            ++b;
        }
    }
    return out.substr(b);
}

//
// prompt formatting
//

struct chat_format {
    std::string tmpl;          // for llama_chat_apply_template; empty = plain "User:/Assistant:" transcript
    std::string system_prompt;
    bool        no_think = false; // prefill an empty <think> block after the assistant header
};

static const model_entry * find_loaded_entry(const app_state & state) {
    if (state.loaded_model_path.empty()) {
        return nullptr;
    }
    if (state.selected_model < state.models.size() && state.models[state.selected_model].path == state.loaded_model_path) {
        return &state.models[state.selected_model];
    }
    for (const auto & m : state.models) {
        if (m.path == state.loaded_model_path) {
            return &m;
        }
    }
    return nullptr;
}

// llama_chat_apply_template only knows built-in templates (detected heuristically from the Jinja source)
static bool chat_template_supported(const std::string & tmpl) {
    if (tmpl.empty()) {
        return false;
    }
    llama_chat_message msg = { "user", "hi" };
    return llama_chat_apply_template(tmpl.c_str(), &msg, 1, true, nullptr, 0) >= 0;
}

static chat_format chat_format_init(const app_state & state) {
    chat_format fmt;
    const model_entry * entry = find_loaded_entry(state);
    if (entry) {
        fmt.system_prompt = trim(entry->system_prompt);
        if (chat_template_supported(entry->chat_template)) {
            fmt.tmpl = entry->chat_template;
            return fmt;
        }
    }
    const char * tmpl = llama_model_chat_template(state.model, nullptr);
    if (tmpl && chat_template_supported(tmpl)) {
        fmt.tmpl = tmpl;
    }
    return fmt;
}

static bool tokenize_text(const llama_vocab * vocab, const std::string & text, bool add_special, std::vector<llama_token> & out) {
    out.resize(text.size() + 2);
    int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), out.data(), (int32_t) out.size(), add_special, true);
    if (n < 0 && n != INT32_MIN) {
        out.resize((size_t) -n);
        n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), out.data(), (int32_t) out.size(), add_special, true);
    }
    if (n < 0) {
        out.clear();
        return false;
    }
    out.resize((size_t) n);

    // some templates emit the BOS text themselves
    const llama_token bos = llama_vocab_bos(vocab);
    if (add_special && out.size() >= 2 && bos != LLAMA_TOKEN_NULL && out[0] == bos && out[1] == bos) {
        out.erase(out.begin());
    }
    return true;
}

// the single token that spells text exactly, or LLAMA_TOKEN_NULL
static llama_token single_token(const llama_vocab * vocab, const char * text) {
    std::vector<llama_token> toks;
    if (!tokenize_text(vocab, text, false, toks) || toks.size() != 1) {
        return LLAMA_TOKEN_NULL;
    }
    return toks[0];
}

// "/no_think" in the system prompt disables reasoning the way the official Qwen3/SmolLM3 templates do:
// the assistant turn starts with an empty <think> block (needed for Qwen3.5, which has no switch)
static void chat_format_init_no_think(chat_format & fmt, const llama_vocab * vocab) {
    fmt.no_think = !fmt.tmpl.empty() && fmt.system_prompt.find(k_no_think) != std::string::npos &&
        single_token(vocab, k_think_open) != LLAMA_TOKEN_NULL && single_token(vocab, k_think_close) != LLAMA_TOKEN_NULL;
}

static bool chat_format_apply(
    const chat_format & fmt,
    const std::vector<chat_turn> & turns,
    size_t start,
    const std::string & user_input,
    std::string & out) {
    // old reasoning blocks are never sent back to the model
    std::vector<std::string> answers;
    answers.reserve(turns.size() - start);
    for (size_t i = start; i < turns.size(); ++i) {
        // with thinking disabled the empty block stays in front of old answers, exactly as it was
        // in the prompt, so the cached reply tokens still match
        answers.push_back((fmt.no_think ? k_think_empty : "") + reply_strip_thinking(turns[i].assistant));
    }

    if (fmt.tmpl.empty()) {
        out.clear();
        if (!fmt.system_prompt.empty()) {
            out += fmt.system_prompt + "\n\n";
        }
        for (size_t i = start; i < turns.size(); ++i) {
            out += "User: " + turns[i].user + "\n";
            out += "Assistant: " + answers[i - start] + "\n";
        }
        out += "User: " + user_input + "\nAssistant:";
        return true;
    }

    std::vector<llama_chat_message> msgs;
    msgs.reserve(2 * answers.size() + 2);
    size_t n_chars = user_input.size() + fmt.system_prompt.size();
    if (!fmt.system_prompt.empty()) {
        msgs.push_back({ "system", fmt.system_prompt.c_str() });
    }
    for (size_t i = start; i < turns.size(); ++i) {
        msgs.push_back({ "user",      turns[i].user.c_str() });
        msgs.push_back({ "assistant", answers[i - start].c_str() });
        n_chars += turns[i].user.size() + answers[i - start].size();
    }
    msgs.push_back({ "user", user_input.c_str() });

    std::vector<char> buf(2 * n_chars + 256);
    int32_t n = llama_chat_apply_template(fmt.tmpl.c_str(), msgs.data(), msgs.size(), true, buf.data(), (int32_t) buf.size());
    if (n > (int32_t) buf.size()) {
        buf.resize(n);
        n = llama_chat_apply_template(fmt.tmpl.c_str(), msgs.data(), msgs.size(), true, buf.data(), (int32_t) buf.size());
    }
    if (n < 0) {
        return false;
    }
    out.assign(buf.data(), (size_t) n);
    if (fmt.no_think) {
        out += k_think_empty;
    }
    return true;
}

static bool token_to_text(const llama_vocab * vocab, llama_token tok, std::string & out) {
    char buf[128];
    int32_t n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, false); // control tokens render as ""
    if (n >= 0) {
        out.assign(buf, (size_t) n);
        return true;
    }
    out.resize((size_t) -n);
    n = llama_token_to_piece(vocab, tok, &out[0], (int32_t) out.size(), 0, false);
    if (n < 0) {
        return false;
    }
    out.resize((size_t) n);
    return true;
}

// Formats the conversation window [first_turn, end) + user_input and tokenizes it. When it does not
// fit in n_limit tokens, the oldest turns are dropped until it fits in 3/4 of n_limit, so that
// the next few replies can reuse the cache again instead of dropping one turn every time.
static bool build_prompt_tokens(
    app_state & state,
    const chat_format & fmt,
    const std::string & user_input,
    int n_limit,
    int n_limit_hard,
    std::vector<llama_token> & tokens,
    std::string & err) {
    const llama_vocab * vocab = llama_model_get_vocab(state.model);
    const std::vector<chat_turn> & turns = state.chat_turns;

    size_t start = state.kv_cache.first_turn <= turns.size() ? state.kv_cache.first_turn : 0;
    if (turns.size() - start > k_max_turns) {
        start = turns.size() - k_max_turns * 3 / 4; // jump, so the window stays stable for a while
    }
    const int n_limit_low = n_limit - n_limit / 4;
    bool dropping = false;
    std::string text;
    for (;;) {
        if (!chat_format_apply(fmt, turns, start, user_input, text)) {
            err = "failed to apply the chat template";
            return false;
        }
        if (!tokenize_text(vocab, text, true, tokens) || tokens.empty()) {
            err = "failed to tokenize prompt";
            return false;
        }
        const int n = (int) tokens.size();
        if (n <= (dropping ? n_limit_low : n_limit)) {
            break;
        }
        if (start >= turns.size()) {
            // no history left: accept a shorter reply budget
            if (n <= n_limit_hard) {
                break;
            }
            err = "message too long for the context";
            return false;
        }
        dropping = true;
        ++start;
    }

    state.kv_cache.first_turn = start;
    return true;
}

//
// KV cache bookkeeping
//

// checkpoints of the recurrent state (recurrent/hybrid models): a few, within a memory budget
constexpr size_t k_ckpt_max      = 3;
constexpr size_t k_ckpt_budget   = 128u * 1024 * 1024; // total bytes kept when there is more than one
constexpr size_t k_ckpt_size_max = 64u * 1024 * 1024; // larger states are never checkpointed

// first_turn is kept: it only chooses the history window and keeps the drop-loop short after a reload
static void kv_cache_forget(chat_kv_cache & cache) {
    cache.ctx = nullptr;
    cache.tokens.clear();
    cache.checkpoints.clear();
    cache.checkpoints.shrink_to_fit();
}

void generate_reset_cache(app_state & state) {
    kv_cache_forget(state.kv_cache);
    if (state.ctx) {
        llama_memory_t mem = llama_get_memory(state.ctx);
        if (mem) {
            llama_memory_clear(mem, true);
        }
    }
}

// the record is only trusted if it belongs to this context and matches what the memory holds
// (a new context, possibly at the same address, starts empty)
static bool kv_cache_matches(const chat_kv_cache & cache, const llama_context * ctx, llama_memory_t mem) {
    if (cache.ctx != ctx) {
        return false;
    }
    return (llama_pos) cache.tokens.size() == llama_memory_seq_pos_max(mem, 0) + 1;
}

// Snapshots the recurrent state at the current end of the cache (checkpoints stay sorted by n_tokens).
// When the budget cannot hold k_ckpt_max snapshots of this size, the newest snapshot is replaced if it
// covers exactly replace_pos tokens (the prompt snapshot of this reply), instead of evicting older ones.
static void kv_cache_save_checkpoint(llama_context * ctx, chat_kv_cache & cache, size_t replace_pos = 0) {
    const size_t n_tokens = cache.tokens.size();
    if (!cache.checkpoints.empty() && cache.checkpoints.back().n_tokens >= n_tokens) {
        return;
    }
    const size_t size = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (size == 0 || size > k_ckpt_size_max) {
        return;
    }

    chat_kv_checkpoint ckpt;
    const bool few_fit = size * k_ckpt_max > k_ckpt_budget;
    if (few_fit && replace_pos != 0 && !cache.checkpoints.empty() && cache.checkpoints.back().n_tokens == replace_pos) {
        ckpt.data = std::move(cache.checkpoints.back().data); // reuse the allocation
        cache.checkpoints.pop_back();
    }
    size_t total = size;
    for (const auto & c : cache.checkpoints) {
        total += c.data.size();
    }
    while (!cache.checkpoints.empty() && (cache.checkpoints.size() >= k_ckpt_max || total > k_ckpt_budget)) {
        total -= cache.checkpoints.front().data.size();
        ckpt.data = std::move(cache.checkpoints.front().data);
        cache.checkpoints.erase(cache.checkpoints.begin());
    }

    ckpt.data.resize(size);
    if (llama_state_seq_get_data_ext(ctx, ckpt.data.data(), size, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != size) {
        return;
    }
    ckpt.n_tokens = n_tokens;
    cache.checkpoints.push_back(std::move(ckpt));
}

// Removes everything after the first n_keep cached tokens. Returns the number of tokens actually
// kept, which is smaller when the memory cannot be rolled back that far (recurrent state, pruned SWA).
static size_t kv_cache_truncate(llama_context * ctx, llama_memory_t mem, const llama_model * model, chat_kv_cache & cache, size_t n_keep) {
    if (n_keep >= cache.tokens.size()) {
        return cache.tokens.size();
    }

    bool ok = true;
    const int n_swa = llama_model_n_swa(model);
    if (n_swa > 0 && n_keep > 0) {
        // a non-full SWA cache only holds the last n_swa positions
        const llama_pos pos_min = llama_memory_seq_pos_min(mem, 0);
        ok = pos_min <= std::max(0, (int) n_keep - n_swa);
    }
    if (ok) {
        ok = llama_memory_seq_rm(mem, 0, (llama_pos) n_keep, -1);
    }
    if (!ok && n_swa == 0) {
        // recurrent/hybrid: roll the state back to the latest usable checkpoint, then drop the attention tail
        for (size_t k = cache.checkpoints.size(); k-- > 0; ) {
            const chat_kv_checkpoint & c = cache.checkpoints[k];
            if (c.n_tokens == 0 || c.n_tokens > n_keep) {
                continue;
            }
            const size_t size = c.data.size();
            ok = llama_state_seq_set_data_ext(ctx, c.data.data(), size, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == size &&
                 llama_memory_seq_rm(mem, 0, (llama_pos) c.n_tokens, -1);
            n_keep = c.n_tokens;
            break;
        }
    }
    if (!ok) {
        llama_memory_clear(mem, true);
        n_keep = 0;
    }

    cache.tokens.resize(n_keep);
    while (!cache.checkpoints.empty() && cache.checkpoints.back().n_tokens > n_keep) {
        cache.checkpoints.pop_back();
    }
    return n_keep;
}

static bool generate_fail(app_state & state, std::string & err, const char * msg) {
    generate_reset_cache(state);
    err = msg;
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
    assistant_out.clear();
    if (!state.model || !state.ctx || !state.sampler) {
        err = "model is not loaded";
        return false;
    }

    llama_context * ctx   = state.ctx;
    const llama_model * model = state.model;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_memory_t mem = llama_get_memory(ctx);
    if (!mem) {
        err = "context has no memory";
        return false;
    }
    chat_kv_cache & cache = state.kv_cache;

    const int  n_ctx       = (int) llama_n_ctx(ctx);
    const int  n_batch     = std::max(1, (int) llama_n_batch(ctx));
    const bool has_encoder = llama_model_has_encoder(model);
    const bool use_ckpt    = llama_model_is_recurrent(model) || llama_model_is_hybrid(model);

    // time spent inside on_partial (UI redraws) is not counted in prefill_ms / gen_ms
    double cb_ms = 0.0;
    auto call_partial = [&](const std::string & text) {
        if (!on_partial) {
            return true;
        }
        const auto t_cb = clock_type::now();
        const bool keep_going = on_partial(text, stats);
        cb_ms += elapsed_ms(t_cb);
        return keep_going;
    };

    if (has_encoder || !kv_cache_matches(cache, ctx, mem)) {
        // unknown memory contents (new context, reloaded model, earlier error): start from scratch
        kv_cache_forget(cache);
        llama_memory_clear(mem, true);
    }

    // keep a reply budget free; the prompt may use more only when there is no history left to drop
    const int n_reserve = std::min(1024, std::max(64, n_ctx / 4));
    int n_limit      = n_ctx - n_reserve;
    int n_limit_hard = n_ctx - 32;
    if (has_encoder) {
        // the encoder needs the whole prompt in one ubatch
        n_limit      = std::min(n_limit,      (int) llama_n_ubatch(ctx));
        n_limit_hard = std::min(n_limit_hard, (int) llama_n_ubatch(ctx));
    }
    chat_format fmt = chat_format_init(state);
    chat_format_init_no_think(fmt, vocab);
    std::vector<llama_token> tokens;
    if (!build_prompt_tokens(state, fmt, user_input, n_limit, n_limit_hard, tokens, err)) {
        return false;
    }
    stats.n_turns_skipped = (int) cache.first_turn;
    stats.n_prompt_total  = (int) tokens.size();

    llama_sampler_reset(state.sampler);

    const auto t_prefill = clock_type::now();
    int n_pos = 0; // positions used in the (decoder) memory

    if (has_encoder) {
        llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size());
        if (llama_encode(ctx, batch)) {
            return generate_fail(state, err, "failed to encode prompt");
        }
        llama_token decoder_start = llama_model_decoder_start_token(model);
        if (decoder_start == LLAMA_TOKEN_NULL) {
            decoder_start = llama_vocab_bos(vocab);
        }
        if (llama_decode(ctx, llama_batch_get_one(&decoder_start, 1))) {
            return generate_fail(state, err, "failed to evaluate prompt");
        }
        n_pos = 1;
        stats.n_prompt   = (int) tokens.size();
        stats.prefill_ms = elapsed_ms(t_prefill);
    } else {
        cache.ctx = ctx;

        // reuse the common prefix; at least the last prompt token is decoded again to get its logits
        size_t n_common = 0;
        const size_t n_max = std::min(cache.tokens.size(), tokens.size() - 1);
        while (n_common < n_max && cache.tokens[n_common] == tokens[n_common]) {
            ++n_common;
        }
        n_common = kv_cache_truncate(ctx, mem, model, cache, n_common);

        stats.n_cached   = (int) n_common;
        stats.prefilling = true;

        // recurrent state cannot be rolled back: snapshot it before the last prompt token, which
        // makes the prompt reusable when the reply diverges (stripped <think>, regenerate)
        const size_t ckpt_at = use_ckpt ? tokens.size() - 1 : tokens.size() + 1;
        size_t i = n_common;
        while (i < tokens.size()) {
            size_t n = std::min((size_t) n_batch, tokens.size() - i);
            if (i < ckpt_at && i + n > ckpt_at) {
                n = ckpt_at - i;
            }
            if (llama_decode(ctx, llama_batch_get_one(&tokens[i], (int32_t) n))) {
                return generate_fail(state, err, "failed to evaluate prompt");
            }
            cache.tokens.insert(cache.tokens.end(), tokens.begin() + i, tokens.begin() + i + n);
            i += n;

            if (i == ckpt_at) {
                kv_cache_save_checkpoint(ctx, cache);
            }

            stats.n_prompt   = (int) (i - n_common);
            stats.prefill_ms = elapsed_ms(t_prefill) - cb_ms;
            if (i < tokens.size() && !call_partial(std::string())) {
                stats.cancelled  = true;
                stats.prefilling = false;
                return true;
            }
        }
        stats.prefilling = false;
        stats.prefill_ms = elapsed_ms(t_prefill) - cb_ms;
        n_pos = (int) cache.tokens.size();
    }

    // generation
    const bool        plain       = fmt.tmpl.empty();
    const std::string stop        = plain ? k_fallback_stop : "";
    const llama_token tok_open    = single_token(vocab, k_think_open);
    const llama_token tok_close   = single_token(vocab, k_think_close);
    const int         answer_room = std::min(256, n_ctx / 8); // context kept free for the answer after thinking

    std::string raw;
    std::string piece;
    bool think_done = false;
    int  n_answer   = 0;

    const auto t_gen = clock_type::now();
    cb_ms = 0.0;
    for (;;) {
        if (n_pos >= n_ctx || (!stats.thinking && n_answer >= k_max_answer_tokens)) {
            stats.truncated = true;
            break;
        }

        llama_token tok = llama_sampler_sample(state.sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) {
            break;
        }
        if (llama_vocab_is_control(vocab, tok) && tok != tok_open && tok != tok_close) {
            break; // a turn marker that is not flagged EOG (e.g. mismatching template override)
        }

        if (tok == tok_open) {
            piece = k_think_open; // may be a control token that would render as ""
        } else if (tok == tok_close) {
            piece = k_think_close;
        } else if (!token_to_text(vocab, tok, piece)) {
            return generate_fail(state, err, "failed converting token to text");
        }

        if (llama_decode(ctx, llama_batch_get_one(&tok, 1))) {
            return generate_fail(state, err, "failed during decoding");
        }
        ++n_pos;
        if (!has_encoder) {
            cache.tokens.push_back(tok);
        }

        ++stats.n_gen;
        if (stats.thinking) {
            ++stats.n_think;
        } else {
            ++n_answer;
        }

        const size_t scan_from = raw.size() > 8 ? raw.size() - 8 : 0;
        raw += piece;
        if (!think_done) {
            // a reasoning block only counts at the start of the reply, not when the answer mentions the tag
            const size_t open = stats.thinking ? std::string::npos : raw.find(k_think_open, scan_from);
            if (open != std::string::npos) {
                size_t b = 0;
                while (b < open && is_space(raw[b])) {
                    ++b;
                }
                stats.thinking = b == open;
                think_done     = b != open;
            }
            if (raw.find(k_think_close, scan_from) != std::string::npos) {
                stats.thinking = false;
                think_done     = true;
            }
        }

        bool stop_hit = false;
        if (!stop.empty()) {
            const size_t p = raw.find(stop, scan_from);
            if (p != std::string::npos) {
                raw.resize(p);
                stop_hit = true;
            }
        }

        // thinking budget exhausted or context nearly full: close the block so an answer follows
        if (stats.thinking && !stop_hit && (stats.n_think >= k_max_think_tokens || n_ctx - n_pos <= answer_room)) {
            std::vector<llama_token> forced;
            if (!tokenize_text(vocab, k_think_force, false, forced) || forced.empty()) {
                return generate_fail(state, err, "failed to tokenize prompt");
            }
            if (n_pos + (int) forced.size() < n_ctx) {
                if (llama_decode(ctx, llama_batch_get_one(forced.data(), (int32_t) forced.size()))) {
                    return generate_fail(state, err, "failed during decoding");
                }
                n_pos += (int) forced.size();
                if (!has_encoder) {
                    cache.tokens.insert(cache.tokens.end(), forced.begin(), forced.end());
                }
                raw += k_think_force;
                stats.thinking = false;
                think_done     = true;
            }
        }

        stats.gen_ms = elapsed_ms(t_gen) - cb_ms;
        if (!call_partial(reply_visible(raw, stop, true, plain))) {
            stats.cancelled = true;
            break;
        }
        if (stop_hit) {
            break;
        }
    }

    stats.gen_ms   = elapsed_ms(t_gen) - cb_ms;
    stats.thinking = false;
    assistant_out  = reply_visible(raw, stop, false, plain);

    // Snapshot the end of the reply too: it serves a popped/edited next message (and, with a few more
    // tokens to decode, a regenerate) when only one or two snapshots fit in the budget. Not when the
    // reply has a reasoning block: that is stripped from the history, so only the prompt one is useful.
    if (use_ckpt && !has_encoder && stats.n_gen > 0 && raw.find(k_think_close) == std::string::npos && raw.find(k_think_open) == std::string::npos) {
        kv_cache_save_checkpoint(ctx, cache, tokens.size() - 1);
    }

    if (has_encoder) {
        kv_cache_forget(cache);
    }
    return true;
}
