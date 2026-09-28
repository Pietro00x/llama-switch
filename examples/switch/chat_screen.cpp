#include "generate.h"
#include "screens.h"
#include "sysinfo.h"

#include <cstdio>
#include <string>
#include <vector>

static bool show_prompt_keyboard(const std::string & current, std::string & output, std::string & err) {
    SwkbdConfig kbd {};
    Result rc = swkbdCreate(&kbd, 0);
    if (R_FAILED(rc)) {
        err = "swkbdCreate failed: " + rc_to_hex(rc);
        return false;
    }

    swkbdConfigMakePresetDefault(&kbd);
    swkbdConfigSetHeaderText(&kbd, "Chat Prompt");
    swkbdConfigSetGuideText(&kbd, "Enter prompt for the model");
    swkbdConfigSetSubText(&kbd, "Press + to submit");
    swkbdConfigSetOkButtonText(&kbd, "Send");
    swkbdConfigSetStringLenMax(&kbd, 512);
    swkbdConfigSetInitialText(&kbd, current.c_str());

    char buffer[513] = {};
    rc = swkbdShow(&kbd, buffer, sizeof(buffer));
    swkbdClose(&kbd);

    if (R_FAILED(rc)) {
        err = "keyboard canceled or failed: " + rc_to_hex(rc);
        return false;
    }

    output = buffer;
    return true;
}

static void render_chat_ui(
    const app_state & state,
    const std::string & draft,
    const std::string & partial_reply,
    bool generating) {
    consoleClear();

    std::printf("llama.cpp Switch Chat\n");
    std::printf("Model: %s\n", state.model ? state.loaded_model_path.c_str() : "(not loaded)");
    std::printf("Ctx=%d Temp=%.2f Top-p=%.2f Top-k=%d\n\n",
        clamp_ctx(state.n_ctx), clamp_temp(state.temperature), clamp_top_p(state.top_p), clamp_top_k(state.top_k));

    std::printf("Controls: X edit prompt, A send, Y clear chat, B back\n");
    std::printf("Prompt: %s\n\n", clip_line(draft).c_str());

    std::printf("Conversation:\n");
    std::vector<std::string> lines;
    lines.reserve(state.chat_turns.size() * 2 + 2);
    for (const auto & turn : state.chat_turns) {
        lines.push_back("U: " + clip_line(turn.user));
        lines.push_back("A: " + clip_line(turn.assistant));
    }
    if (generating || !partial_reply.empty()) {
        lines.push_back("A*: " + clip_line(partial_reply));
    }

    const size_t max_lines = 14;
    const size_t start = lines.size() > max_lines ? lines.size() - max_lines : 0;
    for (size_t i = start; i < lines.size(); ++i) {
        std::printf("%s\n", lines[i].c_str());
    }

    std::printf("\nSwitch stats (always visible)\n");
    std::printf("RAM: %s / %s\n", fmt_mb(state.process_used).c_str(), fmt_mb(state.process_total).c_str());
    std::printf("CPU: %s  GPU: %s  EMC: %s\n", fmt_hz(state.cpu_hz).c_str(), fmt_hz(state.gpu_hz).c_str(), fmt_hz(state.emc_hz).c_str());
    std::printf("Status: %s\n", state.status.c_str());
}

void run_chat_screen(app_state & state, PadState & pad) {
    if (!state.model || !state.ctx || !state.sampler) {
        state.status = "Load a model first";
        return;
    }

    std::string draft_prompt;
    std::string partial_reply;
    bool in_chat = true;

    while (appletMainLoop() && in_chat) {
        update_stats(state);
        render_chat_ui(state, draft_prompt, partial_reply, false);
        consoleUpdate(nullptr);

        padUpdate(&pad);
        const u64 k_down = padGetButtonsDown(&pad);

        if (k_down & HidNpadButton_B) {
            in_chat = false;
            break;
        }

        if (k_down & HidNpadButton_X) {
            std::string entered;
            std::string kbd_err;
            if (show_prompt_keyboard(draft_prompt, entered, kbd_err)) {
                draft_prompt = entered;
                state.status = "Prompt updated";
            } else {
                state.status = kbd_err;
            }
        }

        if (k_down & HidNpadButton_Y) {
            state.chat_turns.clear();
            partial_reply.clear();
            state.status = "Chat history cleared";
        }

        if ((k_down & HidNpadButton_A) && !draft_prompt.empty()) {
            state.status = "Generating...";
            partial_reply.clear();

            render_chat_ui(state, draft_prompt, partial_reply, true);
            consoleUpdate(nullptr);

            std::string assistant;
            std::string gen_err;
            gen_stats stats;
            uint64_t last_draw = 0;
            bool ok = generate_assistant_reply(
                state,
                draft_prompt,
                assistant,
                [&](const std::string & partial, const gen_stats &) {
                    const uint64_t t = now_us();
                    if (t - last_draw < 100000) {
                        return true;
                    }
                    last_draw = t;
                    partial_reply = partial;
                    update_stats(state);
                    render_chat_ui(state, draft_prompt, partial_reply, true);
                    consoleUpdate(nullptr);
                    return true;
                },
                stats,
                gen_err);

            if (ok) {
                state.chat_turns.push_back({ draft_prompt, assistant });
                draft_prompt.clear();
                partial_reply.clear();
                char buf[96];
                const double tps = stats.gen_ms > 0.0 ? 1000.0 * stats.n_gen / stats.gen_ms : 0.0;
                std::snprintf(buf, sizeof(buf), "Reply complete (%d tokens, %.2f tok/s)", stats.n_gen, tps);
                state.status = buf;
            } else {
                state.status = "Generation failed: " + gen_err;
            }
        }
    }
}
