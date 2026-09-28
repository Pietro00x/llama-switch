#pragma once

// Prompt building and token generation. Portable (no libnx): can be compiled on the host.
//
// The conversation is formatted with the model's chat template (or model_entry::chat_template,
// falling back to a plain "User:/Assistant:" transcript for base models / unsupported templates).
// Tokens left in the KV cache by the previous reply are reused: only the part of the conversation
// that differs from the cache (normally just the new user message plus template glue) is decoded.

#include "app.h"

#include <functional>
#include <string>

struct gen_stats {
    int    n_prompt        = 0;     // prompt tokens decoded for this reply (excludes the reused ones)
    int    n_gen           = 0;     // tokens generated so far
    double prefill_ms      = 0.0;   // time spent evaluating the prompt (excluding the callback)
    double gen_ms          = 0.0;   // time spent generating: sampling + decoding (excluding the callback)
    bool   cancelled       = false; // generation was stopped by the callback

    int    n_cached        = 0;     // prompt tokens reused from the KV cache
    int    n_prompt_total  = 0;     // n_cached + tokens that have to be decoded for the prompt
    int    n_turns_skipped = 0;     // oldest chat turns left out because the context is full
    int    n_think         = 0;     // generated tokens that belong to a <think> block
    bool   prefilling      = false; // the prompt is still being evaluated (n_gen == 0)
    bool   thinking        = false; // the model is currently inside a <think> block
    bool   truncated       = false; // stopped by the token budget or a full context, not by the model
};

// Called while the prompt is evaluated (partial is empty, stats.prefilling is true) and after
// every generated token with the reply text so far (complete UTF-8 only, including any
// <think> block; use reply_split_thinking() to separate it).
// Return false to stop early (the partial reply is kept, stats.cancelled is set).
using gen_partial_cb = std::function<bool(const std::string & partial, const gen_stats & stats)>;

// Generates the assistant reply to user_input given state.chat_turns as history.
// Returns false only on errors (err is set); a cancelled generation returns true (assistant_out may
// be empty if it was cancelled during the prompt evaluation).
// Budgets: at most 1024 answer tokens; a <think> block is closed by force after 1024 tokens or when
// the context is nearly full, so an answer always follows. "/no_think" in the model's system prompt
// starts the assistant turn with an empty <think> block (Qwen3 / Qwen3.5 / SmolLM3 style).
bool generate_assistant_reply(
    app_state & state,
    const std::string & user_input,
    std::string & assistant_out,
    const gen_partial_cb & on_partial,
    gen_stats & stats,
    std::string & err);

// Splits a reply into its reasoning ("<think>...</think>", possibly still open) and the answer.
// Returns true if the reply contains a reasoning block.
bool reply_split_thinking(const std::string & reply, std::string & thought, std::string & answer);

// Returns the reply without its reasoning block (what is sent back to the model as history).
std::string reply_strip_thinking(const std::string & reply);

// Forgets the KV-cache bookkeeping (and frees its recurrent-state checkpoints, up to ~128 MB) and
// clears the context memory if a context is loaded. Not required for correctness after clearing
// chat_turns or reloading the model (both are detected automatically), but calling it from
// unload_model() releases the checkpoint memory right away.
void generate_reset_cache(app_state & state);
