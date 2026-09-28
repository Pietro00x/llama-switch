#include "download.h"
#include "engine.h"
#include "models.h"
#include "screens.h"
#include "sysinfo.h"

#include <algorithm>
#include <cstdio>
#include <string>

enum menu_item : size_t {
    MENU_MODEL = 0,
    MENU_DOWNLOAD,
    MENU_CONTEXT,
    MENU_TEMP,
    MENU_TOP_P,
    MENU_TOP_K,
    MENU_LOAD,
    MENU_CHAT,
    MENU_UNLOAD,
    MENU_RELOAD_JSON,
    MENU_EXIT,
    MENU_COUNT
};

static void render_menu(const app_state & state, size_t cursor) {
    consoleClear();

    const model_entry * sel = state.models.empty() ? nullptr : &state.models[state.selected_model];
    const bool model_present = sel && file_exists(sel->path);

    std::printf("llama.cpp Switch Menu\n");
    std::printf("JSON list: %s\n\n", k_model_list_path);

    std::printf("%c Model        : %s\n", cursor == MENU_MODEL ? '>' : ' ', sel ? sel->name.c_str() : "(none)");
    std::printf("%c Download     : %s\n", cursor == MENU_DOWNLOAD ? '>' : ' ', sel ? "A to fetch selected model" : "no model selected");
    std::printf("%c Context      : %d (hard cap %d)\n", cursor == MENU_CONTEXT ? '>' : ' ', clamp_ctx(state.n_ctx), k_ctx_max);
    std::printf("%c Temperature  : %.2f\n", cursor == MENU_TEMP ? '>' : ' ', clamp_temp(state.temperature));
    std::printf("%c Top-p        : %.2f\n", cursor == MENU_TOP_P ? '>' : ' ', clamp_top_p(state.top_p));
    std::printf("%c Top-k        : %d\n", cursor == MENU_TOP_K ? '>' : ' ', clamp_top_k(state.top_k));
    std::printf("%c Load model   : %s\n", cursor == MENU_LOAD ? '>' : ' ', model_present ? "A to load" : "file missing");
    std::printf("%c Chat         : %s\n", cursor == MENU_CHAT ? '>' : ' ', state.model ? "A to open chat UI" : "load model first");
    std::printf("%c Unload model : %s\n", cursor == MENU_UNLOAD ? '>' : ' ', state.model ? "A to unload" : "(not loaded)");
    std::printf("%c Reload JSON  : A to reload model list\n", cursor == MENU_RELOAD_JSON ? '>' : ' ');
    std::printf("%c Exit         : A or +\n", cursor == MENU_EXIT ? '>' : ' ');

    std::printf("\nModel path : %s\n", sel ? sel->path.c_str() : "(none)");
    std::printf("Model URL  : %s\n", sel ? normalize_hf_url(sel->url).c_str() : "(none)");
    std::printf("Loaded     : %s\n", state.model ? state.loaded_model_path.c_str() : "(none)");

    std::printf("\nSwitch stats (always visible)\n");
    std::printf("RAM  : %s / %s\n", fmt_mb(state.process_used).c_str(), fmt_mb(state.process_total).c_str());
    std::printf("CPU  : %s\n", fmt_hz(state.cpu_hz).c_str());
    std::printf("GPU  : %s\n", fmt_hz(state.gpu_hz).c_str());
    std::printf("EMC  : %s\n", fmt_hz(state.emc_hz).c_str());

    std::printf("\nStatus: %s\n", state.status.c_str());
    std::printf("Controls: Up/Down navigate, Left/Right adjust, A action, + quit\n");
}

static void adjust_selected_item(app_state & state, size_t cursor, int dir) {
    switch (cursor) {
        case MENU_MODEL: {
            if (state.models.empty()) {
                break;
            }
            const size_t n = state.models.size();
            state.selected_model = (state.selected_model + n + (dir > 0 ? 1 : n - 1)) % n;
            apply_selected_model_defaults(state);
            state.status = "Selected model: " + state.models[state.selected_model].name;
        } break;
        case MENU_CONTEXT:
            state.n_ctx = clamp_ctx(state.n_ctx + dir * 256);
            state.status = "Context updated";
            break;
        case MENU_TEMP:
            state.temperature = clamp_temp(state.temperature + dir * 0.05f);
            state.status = "Temperature updated";
            break;
        case MENU_TOP_P:
            state.top_p = clamp_top_p(state.top_p + dir * 0.01f);
            state.status = "Top-p updated";
            break;
        case MENU_TOP_K:
            state.top_k = clamp_top_k(state.top_k + dir);
            state.status = "Top-k updated";
            break;
        default:
            break;
    }
}

static void run_action(app_state & state, size_t cursor, PadState & pad) {
    std::string err;

    switch (cursor) {
        case MENU_DOWNLOAD:
            run_download_screen(state, pad);
            break;

        case MENU_LOAD:
            state.status = "Loading model...";
            if (load_model(state, err)) {
                state.status = "Model loaded";
            } else {
                state.status = "Load failed: " + err;
            }
            break;

        case MENU_CHAT:
            run_chat_screen(state, pad);
            break;

        case MENU_UNLOAD:
            unload_model(state);
            state.status = "Model unloaded";
            break;

        case MENU_RELOAD_JSON:
            if (load_model_list(k_model_list_path, state.models, err)) {
                state.selected_model = std::min(state.selected_model, state.models.size() - 1);
                apply_selected_model_defaults(state);
                state.status = "Reloaded model list from JSON";
            } else {
                state.status = "Reload JSON failed: " + err;
            }
            break;

        case MENU_EXIT:
            state.quit = true;
            break;

        default:
            break;
    }
}

void run_main_menu(app_state & state, PadState & pad) {
    size_t cursor = 0;

    while (appletMainLoop() && !state.quit) {
        update_stats(state);
        render_menu(state, cursor);
        consoleUpdate(NULL);

        padUpdate(&pad);
        const u64 k_down = padGetButtonsDown(&pad);

        if (k_down & HidNpadButton_Up) {
            cursor = (cursor + MENU_COUNT - 1) % MENU_COUNT;
        }
        if (k_down & HidNpadButton_Down) {
            cursor = (cursor + 1) % MENU_COUNT;
        }
        if (k_down & HidNpadButton_Left) {
            adjust_selected_item(state, cursor, -1);
        }
        if (k_down & HidNpadButton_Right) {
            adjust_selected_item(state, cursor, +1);
        }
        if (k_down & HidNpadButton_A) {
            run_action(state, cursor, pad);
        }
        if (k_down & HidNpadButton_Plus) {
            state.quit = true;
        }
    }
}
