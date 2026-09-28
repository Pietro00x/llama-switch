#pragma once

// Prompt building and token generation. Portable (no libnx): can be compiled on the host.

#include "app.h"

#include <functional>
#include <string>

struct gen_stats {
    int    n_prompt   = 0;     // prompt tokens evaluated for this reply
    int    n_gen      = 0;     // tokens generated so far
    double prefill_ms = 0.0;   // time spent evaluating the prompt
    double gen_ms     = 0.0;   // time spent generating (sampling + decoding)
    bool   cancelled  = false; // generation was stopped by the callback
};

// Called after every generated token with the reply text so far.
// Return false to stop generation early (the partial reply is kept, stats.cancelled is set).
using gen_partial_cb = std::function<bool(const std::string & partial, const gen_stats & stats)>;

// Generates the assistant reply to user_input given state.chat_turns as history.
// Returns false only on errors (err is set); a cancelled generation returns true.
bool generate_assistant_reply(
    app_state & state,
    const std::string & user_input,
    std::string & assistant_out,
    const gen_partial_cb & on_partial,
    gen_stats & stats,
    std::string & err);
