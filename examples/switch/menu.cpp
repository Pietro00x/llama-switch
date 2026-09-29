#include "clocks.h"
#include "download.h"
#include "engine.h"
#include "models.h"
#include "screens.h"
#include "sysinfo.h"
#include "ui.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <sys/stat.h>

// Screen layout (80x45 console, rows/cols are 0-based, nothing is drawn in column 79):
//
//   row  0       title bar
//   rows 1-2     applet mode warning banner (blank in application mode)
//   rows 3-19    menu (cols 0-47)            | System box rows 3-11, Help box rows 13-19 (cols 50-78)
//                MODEL 3-5, SETTINGS 7-12, ACTIONS 13-19 (no blank row before ACTIONS: 13 items + 3 headers)
//   rows 20-28   selected model info card (cols 0-78)
//   rows 30-39   model list box (cols 0-78, 8 entries)
//   row  41      separator
//   rows 42-43   status line
//   row  44      button hints footer

enum menu_item : size_t {
    MENU_MODEL = 0,
    MENU_DOWNLOAD,
    MENU_CONTEXT,
    MENU_TEMP,
    MENU_TOP_P,
    MENU_TOP_K,
    MENU_CLOCK,
    MENU_LOAD,
    MENU_CHAT,
    MENU_BENCH,
    MENU_UNLOAD,
    MENU_RELOAD_JSON,
    MENU_EXIT,
    MENU_COUNT
};

struct menu_section {
    const char * name;
    int          row;
    size_t       first;
    size_t       last;
};

static const menu_section k_sections[] = {
    { "MODEL",    3,  MENU_MODEL,   MENU_DOWNLOAD },
    { "SETTINGS", 7,  MENU_CONTEXT, MENU_CLOCK    },
    { "ACTIONS",  13, MENU_LOAD,    MENU_EXIT     },
};
static const size_t k_section_count = sizeof(k_sections) / sizeof(k_sections[0]);

static const int k_item_rows[MENU_COUNT] = { 4, 5, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19 };

constexpr int k_menu_col   = 1;  // menu rows span cols 1-47, the selection marker is in col 0
constexpr int k_menu_width = 47;
constexpr int k_label_width = 12;
constexpr int k_side_col   = 50; // System/Help boxes span cols 50-78
constexpr int k_side_width = 29;
constexpr int k_card_row   = 20;
constexpr int k_list_row   = 30;
constexpr int k_list_rows  = 8;  // model entries visible in the list box
constexpr int k_status_row = 42;
constexpr int k_full_width = 79; // widest line that does not wrap the cursor

constexpr uint64_t k_stats_interval_us  = 500000;
constexpr uint64_t k_repeat_delay_us    = 400000;
constexpr uint64_t k_repeat_interval_us = 70000;
constexpr uint64_t k_frame_sleep_ns     = 16 * 1000 * 1000ULL;

constexpr uint64_t k_repeat_buttons =
    HidNpadButton_AnyUp | HidNpadButton_AnyDown | HidNpadButton_AnyLeft | HidNpadButton_AnyRight;

struct menu_view {
    size_t cursor = 0;
    bool   dirty  = true;
    bool   applet_mode = false;

    std::vector<long long> file_sizes; // per model entry, -1 = not on the SD card

    // what the last frame showed, to detect changes
    std::string drawn_status;
    uint64_t    drawn_used  = 0;
    uint64_t    drawn_total = 0;
    uint32_t    drawn_cpu   = 0;
    uint32_t    drawn_gpu   = 0;
    uint32_t    drawn_emc   = 0;
    uint32_t    drawn_cpu_target = 0;
    uint32_t    drawn_emc_target = 0;
    uint64_t    last_stats_check_us = 0;

    // hold-to-repeat for the direction buttons
    uint64_t repeat_buttons  = 0;
    uint64_t repeat_start_us = 0;
    uint64_t repeat_last_us  = 0;
};

static bool menu_is_applet_mode() {
    const AppletType type = appletGetAppletType();
    return type != AppletType_Application && type != AppletType_SystemApplication;
}

static long long menu_file_size(const std::string & path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
        return -1;
    }
    return (long long) st.st_size;
}

static void menu_refresh_files(const app_state & state, menu_view & view) {
    view.file_sizes.assign(state.models.size(), -1);
    for (size_t i = 0; i < state.models.size(); ++i) {
        view.file_sizes[i] = menu_file_size(state.models[i].path);
    }
    view.dirty = true;
}

static const model_entry * menu_selected(const app_state & state) {
    if (state.models.empty() || state.selected_model >= state.models.size()) {
        return nullptr;
    }
    return &state.models[state.selected_model];
}

static long long menu_selected_size(const app_state & state, const menu_view & view) {
    if (!menu_selected(state) || state.selected_model >= view.file_sizes.size()) {
        return -1;
    }
    return view.file_sizes[state.selected_model];
}

static bool menu_selected_is_loaded(const app_state & state) {
    const model_entry * sel = menu_selected(state);
    return sel && state.model && state.loaded_model_path == sel->path;
}

static int menu_section_of(size_t item) {
    for (size_t s = 0; s < k_section_count; ++s) {
        if (item >= k_sections[s].first && item <= k_sections[s].last) {
            return (int) s;
        }
    }
    return 0;
}

// error messages red, warnings yellow, success green, everything else white
static const char * menu_status_color(const std::string & status) {
    const std::string s = to_lower_ascii(status);
    static const char * const k_error[] = { "fail", "error", "not found", "invalid", "cannot", "missing" };
    static const char * const k_warn[]  = { "cancel", "fallback", "first", "warning", "no model", "not available", "sys-clk" };
    static const char * const k_ok[]    = { "loaded", "complete", "selected", "updated", "reloaded", "ready" };
    for (const char * w : k_error) {
        if (s.find(w) != std::string::npos) {
            return UI_FG_RED;
        }
    }
    for (const char * w : k_warn) {
        if (s.find(w) != std::string::npos) {
            return UI_FG_YELLOW;
        }
    }
    for (const char * w : k_ok) {
        if (s.find(w) != std::string::npos) {
            return UI_FG_GREEN;
        }
    }
    return UI_FG_WHITE;
}

static const char * menu_usage_color(double frac) {
    if (frac >= 0.9) {
        return UI_FG_RED;
    }
    if (frac >= 0.7) {
        return UI_FG_YELLOW;
    }
    return UI_FG_GREEN;
}

static std::string menu_fmt(double v, int decimals) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", decimals, v);
    return std::string(buf);
}

static std::string menu_fmt_int(int v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%d", v);
    return std::string(buf);
}

static std::string menu_align_right(const std::string & s, int width) {
    if ((int) s.size() >= width) {
        return s.substr(0, (size_t) width);
    }
    return std::string((size_t) width - s.size(), ' ') + s;
}

// "used / total MB" without decimals, short enough for the title bar and the System box
static std::string menu_ram_text(const app_state & state) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.0f / %.0f MB",
        (double) state.process_used / (1024.0 * 1024.0), (double) state.process_total / (1024.0 * 1024.0));
    return std::string(buf);
}

static double menu_ram_frac(const app_state & state) {
    return state.process_total > 0 ? (double) state.process_used / (double) state.process_total : 0.0;
}

//
// drawing
//

static void draw_title(const app_state & state, const menu_view & view) {
    const std::string left  = " llama.cpp for Switch";
    const std::string right = std::string(view.applet_mode ? "APPLET  " : "") + "RAM " + menu_ram_text(state) + " ";
    std::string line = left;
    if ((int) (left.size() + right.size()) < k_full_width) {
        line += std::string((size_t) k_full_width - left.size() - right.size(), ' ') + right;
    }
    ui_move(0, 0);
    std::printf(UI_BG_BLUE UI_FG_WHITE "%s" UI_RESET, ui_fit(line, k_full_width).c_str());
}

static void draw_applet_banner(const app_state & state, const menu_view & view) {
    if (!view.applet_mode) {
        return;
    }
    const std::string total = state.process_total > 0 ? fmt_mb(state.process_total) : std::string("~400 MB");
    const std::string line1 = " ! APPLET MODE: only " + total + " of RAM - most models will fail to load.";
    const std::string line2 = " ! For full RAM (~3.2 GB) hold R while starting any game (title override).";
    ui_move(1, 0);
    std::printf(UI_BG_RED UI_FG_WHITE "%s" UI_RESET, ui_fit(line1, k_full_width).c_str());
    ui_move(2, 0);
    std::printf(UI_BG_RED UI_FG_WHITE "%s" UI_RESET, ui_fit(line2, k_full_width).c_str());
}

static void draw_section_header(const menu_section & sec, bool active) {
    const std::string name = std::string(" ") + sec.name + " ";
    ui_move(sec.row, k_menu_col);
    std::printf("%s%s" UI_RESET UI_FAINT "%s" UI_RESET,
        active ? UI_FG_YELLOW : UI_FG_CYAN, name.c_str(),
        std::string((size_t) (k_menu_width - (int) name.size()), '-').c_str());
}

// one menu row: " label        value extra", value/extra colored unless the row is selected
static void draw_item(
    int row, bool selected, bool enabled,
    const std::string & label,
    const std::string & value, const char * value_color,
    const std::string & extra, const char * extra_color) {
    const std::string head = " " + ui_fit(label, k_label_width) + " ";
    const int room = k_menu_width - (int) head.size();
    const std::string v = ui_fit(value, std::min(room, (int) value.size()));
    const std::string e = ui_fit(extra, std::max(0, room - (int) v.size()));

    ui_move(row, 0);
    if (selected) {
        std::printf(UI_FG_YELLOW ">" UI_RESET UI_REVERSE "%s%s%s" UI_RESET, head.c_str(), v.c_str(), e.c_str());
        return;
    }
    std::printf(" %s%s" UI_RESET "%s%s" UI_RESET "%s%s" UI_RESET,
        enabled ? UI_FG_WHITE : UI_FAINT, head.c_str(),
        enabled ? value_color : UI_FAINT, v.c_str(),
        enabled ? extra_color : UI_FAINT, e.c_str());
}

// editable value: "<  4096 >  [#####.......]"
static void draw_value_item(
    int row, bool selected, const std::string & label,
    const std::string & value, double frac, const std::string & note) {
    const std::string v = "< " + menu_align_right(value, 5) + " >";
    const std::string extra = "  " + ui_progress_bar(frac, 14) + note;
    draw_item(row, selected, true, label, v, UI_FG_YELLOW, extra, note.empty() ? UI_FG_CYAN : UI_FG_YELLOW);
}

static void draw_menu(const app_state & state, const menu_view & view) {
    const model_entry * sel = menu_selected(state);
    const long long size    = menu_selected_size(state, view);
    const bool on_sd        = size >= 0;
    const bool loaded_sel   = menu_selected_is_loaded(state);
    const size_t c          = view.cursor;

    const int active = menu_section_of(c);
    for (size_t s = 0; s < k_section_count; ++s) {
        draw_section_header(k_sections[s], (int) s == active);
    }

    // model
    {
        const int name_w = k_menu_width - (k_label_width + 2) - 4;
        const std::string name = sel ? clip_line(ui_ascii(sel->name), (size_t) name_w) : std::string("(none)");
        draw_item(k_item_rows[MENU_MODEL], c == MENU_MODEL, sel != nullptr, "Model",
            "< " + ui_fit(name, name_w) + " >", UI_FG_YELLOW, "", UI_RESET);
    }
    if (!sel) {
        draw_item(k_item_rows[MENU_DOWNLOAD], c == MENU_DOWNLOAD, false, "Download", "no model selected", UI_FAINT, "", UI_RESET);
    } else if (sel->url.empty()) {
        draw_item(k_item_rows[MENU_DOWNLOAD], c == MENU_DOWNLOAD, true, "Download", "no URL in models.json", UI_FG_RED, "", UI_RESET);
    } else if (on_sd) {
        draw_item(k_item_rows[MENU_DOWNLOAD], c == MENU_DOWNLOAD, true, "Download",
            "on SD, " + fmt_mb((uint64_t) size), UI_FG_GREEN, " (A: fetch again)", UI_FAINT);
    } else {
        draw_item(k_item_rows[MENU_DOWNLOAD], c == MENU_DOWNLOAD, true, "Download",
            "not on SD - A to download", UI_FG_YELLOW, "", UI_RESET);
    }

    // sampling
    const int ctx = clamp_ctx(state.n_ctx);
    // llama.cpp pads n_ctx up to a multiple of 256, compare against the padded request
    const uint32_t ctx_padded = (uint32_t) ((ctx + 255) / 256 * 256);
    const bool ctx_pending = state.ctx && ctx_padded != llama_n_ctx(state.ctx);
    draw_value_item(k_item_rows[MENU_CONTEXT], c == MENU_CONTEXT, "Context", menu_fmt_int(ctx),
        (double) (ctx - k_ctx_min) / (double) (k_ctx_max - k_ctx_min), ctx_pending ? " reload" : "");
    draw_value_item(k_item_rows[MENU_TEMP], c == MENU_TEMP, "Temperature",
        menu_fmt(clamp_temp(state.temperature), 2), clamp_temp(state.temperature) / 2.0, "");
    draw_value_item(k_item_rows[MENU_TOP_P], c == MENU_TOP_P, "Top-p",
        menu_fmt(clamp_top_p(state.top_p), 2), clamp_top_p(state.top_p), "");
    draw_value_item(k_item_rows[MENU_TOP_K], c == MENU_TOP_K, "Top-k",
        menu_fmt_int(clamp_top_k(state.top_k)), (double) (clamp_top_k(state.top_k) - 1) / 199.0, "");
    {
        // "< Boost     >  1785/1600 MHz": the CPU/EMC rates the chosen mode asks for
        const clock_plan plan = clocks_plan_for(state);
        const std::string value = "< " + ui_fit(clock_mode_name(state.clock_setting), 9) + " >";
        const std::string rates = plan.mode == CLOCK_MODE_STOCK ? std::string("  system clocks")
            : "  " + clock_fmt_mhz(plan.cpu_hz) + "/" + clock_fmt_mhz(plan.emc_hz) + " MHz";
        draw_item(k_item_rows[MENU_CLOCK], c == MENU_CLOCK, true, "CPU clock", value,
            state.clock_setting == CLOCK_MODE_OVERCLOCK ? UI_FG_MAGENTA : UI_FG_YELLOW, rates, UI_FG_CYAN);
    }

    // actions
    if (!sel) {
        draw_item(k_item_rows[MENU_LOAD], c == MENU_LOAD, false, "Load model", "no model selected", UI_FAINT, "", UI_RESET);
    } else if (!on_sd) {
        draw_item(k_item_rows[MENU_LOAD], c == MENU_LOAD, true, "Load model", "file missing", UI_FG_RED, " (A: download)", UI_FAINT);
    } else if (loaded_sel) {
        draw_item(k_item_rows[MENU_LOAD], c == MENU_LOAD, true, "Load model", "loaded", UI_FG_GREEN, " (A: reload)", UI_FAINT);
    } else {
        draw_item(k_item_rows[MENU_LOAD], c == MENU_LOAD, true, "Load model", "A to load into RAM", UI_FG_CYAN, "", UI_RESET);
    }
    if (state.model) {
        draw_item(k_item_rows[MENU_CHAT], c == MENU_CHAT, true, "Chat", "A to open chat",
            UI_FG_CYAN, " (" + menu_fmt_int((int) state.chat_turns.size()) + " turns)", UI_FAINT);
        draw_item(k_item_rows[MENU_BENCH], c == MENU_BENCH, true, "Benchmark", "A to measure speed", UI_FG_CYAN, "", UI_RESET);
        draw_item(k_item_rows[MENU_UNLOAD], c == MENU_UNLOAD, true, "Unload model", "A to free memory", UI_FG_CYAN, "", UI_RESET);
    } else {
        draw_item(k_item_rows[MENU_CHAT], c == MENU_CHAT, false, "Chat", "load a model first", UI_FAINT, "", UI_RESET);
        draw_item(k_item_rows[MENU_BENCH], c == MENU_BENCH, false, "Benchmark", "load a model first", UI_FAINT, "", UI_RESET);
        draw_item(k_item_rows[MENU_UNLOAD], c == MENU_UNLOAD, false, "Unload model", "nothing loaded", UI_FAINT, "", UI_RESET);
    }
    draw_item(k_item_rows[MENU_RELOAD_JSON], c == MENU_RELOAD_JSON, true, "Reload JSON", "re-read models.json", UI_FG_CYAN, "", UI_RESET);
    draw_item(k_item_rows[MENU_EXIT], c == MENU_EXIT, true, "Exit", "quit the app", UI_FG_CYAN, "", UI_RESET);
}

static void draw_side_line(int row, const char * color, const std::string & text) {
    ui_move(row, k_side_col + 2);
    std::printf("%s%s" UI_RESET, color, ui_fit(text, k_side_width - 4).c_str());
}

// raised clock: green while it holds, yellow when something else (the system, sys-clk) changed it
static const char * menu_clock_color(uint32_t live_hz, uint32_t target_hz) {
    if (target_hz == 0) {
        return UI_FG_GRAY;
    }
    const uint32_t diff = live_hz > target_hz ? live_hz - target_hz : target_hz - live_hz;
    return diff <= 2000000 ? UI_FG_GREEN : UI_FG_YELLOW;
}

static void draw_stats_box(const app_state & state, const menu_view & view) {
    const int top = 3;
    std::printf(UI_FG_CYAN);
    ui_box(top, k_side_col, 9, k_side_width, "System");
    std::printf(UI_RESET);

    const double frac = menu_ram_frac(state);
    const uint64_t free_bytes = state.process_total > state.process_used ? state.process_total - state.process_used : 0;

    draw_side_line(top + 1, UI_FG_WHITE, "RAM  " + menu_ram_text(state));
    ui_move(top + 2, k_side_col + 2);
    std::printf("%s%s" UI_RESET " %s", menu_usage_color(frac), ui_progress_bar(frac, 18).c_str(),
        menu_align_right(menu_fmt_int((int) (frac * 100.0 + 0.5)) + "%", 4).c_str());
    draw_side_line(top + 3, UI_FG_GRAY, "Free " + fmt_mb(free_bytes));
    const char * tag = state.clock_applied == CLOCK_MODE_OVERCLOCK ? "  OC" :
        state.clock_applied == CLOCK_MODE_BOOST ? "  Boost" : "";
    draw_side_line(top + 4, menu_clock_color(state.cpu_hz, state.clock_cpu_target_hz), "CPU  " + fmt_hz(state.cpu_hz) + tag);
    draw_side_line(top + 5, UI_FG_GRAY, "GPU  " + fmt_hz(state.gpu_hz));
    draw_side_line(top + 6, menu_clock_color(state.emc_hz, state.clock_emc_target_hz), "EMC  " + fmt_hz(state.emc_hz));
    if (view.applet_mode) {
        draw_side_line(top + 7, UI_FG_RED, "Mode applet (low RAM)");
    } else {
        draw_side_line(top + 7, UI_FG_GREEN, "Mode application");
    }
}

static const char * menu_clock_help(const app_state & state) {
    switch (state.clock_setting) {
        case CLOCK_MODE_STOCK:
            return "Clocks while a model is loaded. Stock: system clocks. Right: Boost.";
        case CLOCK_MODE_OVERCLOCK:
            return "Overclock: highest rates allowed. Hot, drains battery. Left: lower CPU.";
        default:
            return state.clock_oc_available
                ? "Boost: CPU 1785, RAM 1600 MHz while a model is loaded. Right: Overclock."
                : "Boost: CPU 1785, RAM 1600 MHz. Overclock n/a: needs sys-clk-OC.";
    }
}

static const char * menu_help_text(const app_state & state, size_t item) {
    switch (item) {
        case MENU_MODEL:       return "Left/Right picks a model from models.json and applies its preset.";
        case MENU_DOWNLOAD:    return "Downloads the GGUF file from its URL to the SD card. B cancels.";
        case MENU_CONTEXT:     return "Context size in tokens. Bigger uses more RAM. Applied on the next Load.";
        case MENU_TEMP:        return "Randomness of the reply. 0 = always the most likely token.";
        case MENU_TOP_P:       return "Keeps the smallest set of tokens whose probability adds up to top-p.";
        case MENU_TOP_K:       return "Samples only among the K most likely tokens.";
        case MENU_CLOCK:       return menu_clock_help(state);
        case MENU_LOAD:        return "Loads the selected model into RAM (CPU inference).";
        case MENU_CHAT:        return "Opens the chat screen for the loaded model.";
        case MENU_BENCH:       return "Measures prompt/generation speed and memory bandwidth; saves bench.txt.";
        case MENU_UNLOAD:      return "Frees the model and its context memory.";
        case MENU_RELOAD_JSON: return "Re-reads models.json from the SD card after editing it.";
        case MENU_EXIT:        return "Frees everything and returns to the homebrew menu.";
        default:               return "";
    }
}

static void draw_help_box(const app_state & state, const menu_view & view) {
    const int top = 13;
    const int height = 7;
    std::printf(UI_FG_CYAN);
    ui_box(top, k_side_col, height, k_side_width, "Help");
    std::printf(UI_RESET);

    // the sys-clk note goes to the bottom lines, the help text keeps the rest
    std::vector<std::string> note;
    if (view.cursor == MENU_CLOCK && state.clock_sysclk) {
        note = ui_wrap("sys-clk detected: it may override these clocks.", k_side_width - 4);
    }
    const int text_rows = height - 2 - (int) note.size();
    const std::vector<std::string> lines = ui_wrap(menu_help_text(state, view.cursor), k_side_width - 4);
    for (size_t i = 0; i < lines.size() && (int) i < text_rows; ++i) {
        draw_side_line(top + 1 + (int) i, UI_FG_GRAY, lines[i]);
    }
    for (size_t i = 0; i < note.size(); ++i) {
        draw_side_line(top + 1 + text_rows + (int) i, UI_FG_YELLOW, note[i]);
    }
}

static void draw_card_line(int row, const std::string & label, const char * color, const std::string & value) {
    const int value_w = k_full_width - 4 - 9;
    ui_move(row, 2);
    std::printf(UI_FAINT "%s" UI_RESET "%s%s" UI_RESET, ui_fit(label, 9).c_str(), color, ui_fit(value, value_w).c_str());
}

static void draw_model_card(const app_state & state, const menu_view & view) {
    const model_entry * sel = menu_selected(state);
    const int top = k_card_row;
    const int value_w = k_full_width - 4 - 9;

    std::string title = "Selected model";
    if (sel) {
        title += " (" + menu_fmt_int((int) state.selected_model + 1) + " of " + menu_fmt_int((int) state.models.size()) + ")";
    }
    std::printf(UI_FG_CYAN);
    ui_box(top, 0, 9, k_full_width, title);
    std::printf(UI_RESET);

    if (!sel) {
        draw_card_line(top + 1, "", UI_FG_YELLOW, "No model in the list. Edit models.json and choose Reload JSON.");
        return;
    }

    const long long size = menu_selected_size(state, view);
    const std::string url = normalize_hf_url(sel->url);

    draw_card_line(top + 1, "Name", UI_FG_WHITE, clip_line(ui_ascii(sel->name), (size_t) value_w));
    draw_card_line(top + 2, "File", UI_FG_GRAY, clip_line(ui_ascii(sel->path), (size_t) value_w));
    if (size >= 0) {
        draw_card_line(top + 3, "On SD", UI_FG_GREEN, "yes, " + fmt_mb((uint64_t) size));
    } else {
        draw_card_line(top + 3, "On SD", UI_FG_RED, "no - use Download to fetch it");
    }
    if (menu_selected_is_loaded(state)) {
        const std::string info = "yes, ctx " + menu_fmt_int((int) llama_n_ctx(state.ctx)) + ", " +
            menu_fmt_int((int) state.chat_turns.size()) + " chat turns";
        draw_card_line(top + 4, "Loaded", UI_FG_GREEN, info);
    } else if (state.model) {
        draw_card_line(top + 4, "Loaded", UI_FG_YELLOW, clip_line("no (other: " + ui_ascii(state.loaded_model_path), (size_t) value_w - 1) + ")");
    } else {
        draw_card_line(top + 4, "Loaded", UI_FG_GRAY, "no");
    }

    const std::string u = ui_ascii(url.empty() ? std::string("(none)") : url);
    draw_card_line(top + 5, "URL", UI_FG_CYAN, u.substr(0, (size_t) value_w));
    if (u.size() > (size_t) value_w) {
        draw_card_line(top + 6, "", UI_FG_CYAN, clip_line(u.substr((size_t) value_w), (size_t) value_w));
    }

    std::string preset = "ctx " + menu_fmt_int(clamp_ctx(sel->default_ctx)) +
        "  temp " + menu_fmt(clamp_temp(sel->default_temp), 2) +
        "  top-p " + menu_fmt(clamp_top_p(sel->default_top_p), 2) +
        "  top-k " + menu_fmt_int(clamp_top_k(sel->default_top_k));
    if (!sel->system_prompt.empty()) {
        preset += "  +system prompt";
    }
    if (!sel->chat_template.empty()) {
        preset += "  +template";
    }
    draw_card_line(top + 7, "Preset", UI_FG_GRAY, preset);
}

static void draw_model_list(const app_state & state, const menu_view & view) {
    const int top = k_list_row;
    const size_t n = state.models.size();

    std::printf(UI_FG_CYAN);
    ui_box(top, 0, k_list_rows + 2, k_full_width, menu_fmt_int((int) n) + " models - " + k_model_list_path);
    std::printf(UI_RESET);

    size_t first = 0;
    if (n > (size_t) k_list_rows) {
        first = state.selected_model > 3 ? state.selected_model - 3 : 0;
        first = std::min(first, n - (size_t) k_list_rows);
    }

    const int name_w = 47;
    for (int r = 0; r < k_list_rows && first + (size_t) r < n; ++r) {
        const size_t i = first + (size_t) r;
        const model_entry & m = state.models[i];
        const long long size = i < view.file_sizes.size() ? view.file_sizes[i] : -1;
        const bool selected = i == state.selected_model;
        const bool loaded = state.model && state.loaded_model_path == m.path;

        const std::string name = ui_fit(clip_line(ui_ascii(m.name), (size_t) name_w), name_w);
        const std::string size_text = menu_align_right(size >= 0 ? fmt_mb((uint64_t) size) : std::string("-"), 10);

        std::string marker = selected ? "> " : "  ";
        if (r == 0 && first > 0) {
            marker = "^ ";
        } else if (r == k_list_rows - 1 && first + (size_t) k_list_rows < n) {
            marker = "v ";
        }
        if (selected) {
            marker = "> ";
        }

        ui_move(top + 1 + r, 2);
        std::printf(UI_FG_YELLOW "%s" UI_RESET "%s%s" UI_RESET " %s%s " UI_RESET "%s%s" UI_RESET " " UI_FG_GREEN "%s" UI_RESET,
            marker.c_str(),
            size >= 0 ? UI_FG_GREEN : UI_FAINT, size >= 0 ? "[SD]" : "[  ]",
            selected ? UI_FG_YELLOW : UI_FG_GRAY, name.c_str(),
            UI_FG_GRAY, size_text.c_str(),
            loaded ? "LOADED" : "      ");
    }
}

static void draw_status(const app_state & state) {
    const int text_w = k_full_width - 1 - 9;
    const std::vector<std::string> lines = ui_wrap(state.status, text_w);
    const char * color = menu_status_color(state.status);

    ui_hline(k_status_row - 1, '-');
    ui_move(k_status_row, 0);
    std::printf(UI_FAINT " Status: " UI_RESET "%s%s" UI_RESET, color, lines.empty() ? "" : lines[0].c_str());
    if (lines.size() > 1) {
        std::string second = lines[1];
        if (lines.size() > 2) {
            second = clip_line(second + " " + lines[2], (size_t) text_w);
        }
        ui_move(k_status_row + 1, 9);
        std::printf("%s%s" UI_RESET, color, second.c_str());
    }
}

static void render_menu(const app_state & state, const menu_view & view) {
    ui_frame_begin();
    draw_title(state, view);
    draw_applet_banner(state, view);
    draw_menu(state, view);
    draw_stats_box(state, view);
    draw_help_box(state, view);
    draw_model_card(state, view);
    draw_model_list(state, view);
    draw_status(state);
    ui_footer("Up/Down move  L/R section  Left/Right change  A select  + quit");
    ui_frame_end();
}

// draws the current state right away, e.g. before a blocking call
static void menu_redraw_now(const app_state & state, menu_view & view) {
    render_menu(state, view);
    view.drawn_status = state.status;
    view.dirty = false;
}

//
// input and actions
//

static float menu_round_2(float v) {
    return std::round(v * 100.0f) / 100.0f;
}

// Left/Right on "CPU clock": Stock, Boost, Overclock (asks first), then the Overclock CPU steps
static void menu_adjust_clock(app_state & state, menu_view & view, PadState & pad, int dir) {
    // one step per press, no hold-to-repeat: every step is applied and saved, and holding Right from Stock
    // must not run into the overclock warning
    if (!(padGetButtonsDown(&pad) & (HidNpadButton_AnyLeft | HidNpadButton_AnyRight))) {
        return;
    }
    const clock_step step = clock_step_setting(state.clock_setting, state.clock_oc_cpu_hz, clocks_oc_cpu_steps(),
        state.clock_oc_available, dir);
    if (step.unavailable) {
        state.status = "Overclock not available (needs sys-clk-OC): staying on Boost";
        return;
    }
    if (step.mode == state.clock_setting && step.oc_cpu_hz == state.clock_oc_cpu_hz) {
        return;
    }
    if (step.confirm) {
        const bool ok = ui_confirm(pad, "Overclock goes above Nintendo's limits: more heat, faster battery drain, "
            "possible freezes or crashes. At your own risk - enable it?");
        // a direction still held after the dialog must not open it again
        view.repeat_buttons = 0;
        if (!ok) {
            state.status = "Overclock cancelled";
            return;
        }
    }

    std::string err;
    const bool saved = clocks_set_setting(state, step.mode, step.oc_cpu_hz, err);

    const clock_plan plan = clocks_plan_for(state);
    std::string msg = std::string("CPU clock updated: ") + clock_mode_name(state.clock_setting);
    if (plan.mode == CLOCK_MODE_STOCK) {
        msg += " (system clocks)";
    } else {
        msg += " (CPU " + clock_fmt_mhz(plan.cpu_hz) + " / EMC " + clock_fmt_mhz(plan.emc_hz) + " MHz)";
    }
    if (!state.model) {
        msg += ", used from the next Load";
    }
    if (state.clock_sysclk && plan.mode != CLOCK_MODE_STOCK) {
        msg += " - sys-clk detected: it may override it";
    }
    if (!saved) {
        msg += " - not saved: " + err;
    }
    state.status = msg;
    state.last_stats_us = 0; // show the new clocks right away
}

static void adjust_selected_item(app_state & state, menu_view & view, PadState & pad, int dir) {
    switch (view.cursor) {
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
            state.n_ctx = clamp_ctx(clamp_ctx(state.n_ctx) + dir * 256);
            state.status = "Context updated";
            break;
        case MENU_TEMP:
            state.temperature = menu_round_2(clamp_temp(state.temperature + dir * 0.05f));
            state.status = "Temperature updated";
            break;
        case MENU_TOP_P:
            state.top_p = menu_round_2(clamp_top_p(state.top_p + dir * 0.01f));
            state.status = "Top-p updated";
            break;
        case MENU_TOP_K:
            state.top_k = clamp_top_k(state.top_k + dir);
            state.status = "Top-k updated";
            break;
        case MENU_CLOCK:
            menu_adjust_clock(state, view, pad, dir);
            break;
        default:
            break;
    }

    // sampling changes apply to the loaded model right away (the context needs a reload, the sampler does not)
    if (state.sampler && (view.cursor == MENU_TEMP || view.cursor == MENU_TOP_P || view.cursor == MENU_TOP_K)) {
        std::string err;
        if (!engine_rebuild_sampler(state, err)) {
            state.status = "Sampler update failed: " + err;
        }
    }
}

static void menu_do_download(app_state & state, menu_view & view, PadState & pad) {
    const model_entry * sel = menu_selected(state);
    if (sel && sel->url.empty()) {
        state.status = "Local file only: this model has no download URL";
        return;
    }
    if (menu_selected_size(state, view) >= 0 &&
        !ui_confirm(pad, "Download again? The file on SD is only replaced once the new copy is complete.")) {
        state.status = "Download cancelled";
        return;
    }
    run_download_screen(state, pad);
    menu_refresh_files(state, view);
}

static void menu_do_load(app_state & state, menu_view & view, PadState & pad) {
    const model_entry * sel = menu_selected(state);
    if (!sel) {
        state.status = "No model selected";
        return;
    }
    if (menu_selected_size(state, view) < 0) {
        if (sel->url.empty()) {
            state.status = "Model file missing on SD (local entry, no download URL): " + sel->path;
            return;
        }
        if (ui_confirm(pad, "The model file is not on the SD card. Download it now?")) {
            run_download_screen(state, pad);
            menu_refresh_files(state, view);
        } else {
            state.status = "Model file missing on SD: " + sel->path;
        }
        return;
    }
    if (state.model) {
        const std::string q = menu_selected_is_loaded(state)
            ? "Reload " + sel->name + " with the current settings?"
            : "Unload the current model and load " + sel->name + "?";
        if (!ui_confirm(pad, q)) {
            view.dirty = true;
            return;
        }
    }

    const std::string name = sel->name;
    state.status = "Loading " + name + ", please wait...";
    menu_redraw_now(state, view);

    std::string err;
    const uint64_t t0 = now_us();
    if (load_model(state, err)) {
        const double secs = (double) (now_us() - t0) / 1e6;
        state.status = "Model loaded: " + name + " (ctx " + menu_fmt_int((int) llama_n_ctx(state.ctx)) +
            ", " + menu_fmt(secs, 1) + " s)";
    } else {
        state.status = "Load failed: " + err;
    }
    state.last_stats_us = 0; // memory usage changed a lot: refresh the stats right away
    update_stats(state);
}

static void run_action(app_state & state, menu_view & view, PadState & pad) {
    std::string err;

    switch (view.cursor) {
        case MENU_DOWNLOAD:
            if (!menu_selected(state)) {
                state.status = "No model selected";
                break;
            }
            menu_do_download(state, view, pad);
            break;

        case MENU_LOAD:
            menu_do_load(state, view, pad);
            break;

        case MENU_CHAT:
            run_chat_screen(state, pad);
            break;

        case MENU_BENCH:
            run_bench_screen(state, pad);
            break;

        case MENU_UNLOAD:
            if (!state.model) {
                state.status = "No model loaded";
                break;
            }
            if (ui_confirm(pad, "Unload the model and free its memory?")) {
                unload_model(state);
                state.status = "Model unloaded";
                state.last_stats_us = 0;
                update_stats(state);
            }
            break;

        case MENU_RELOAD_JSON:
            // keeps the selection on the same model (by path): the list is re-sorted on every load
            if (reload_model_list(state, err)) {
                state.status = "Reloaded model list (" + menu_fmt_int((int) state.models.size()) + " models)";
                if (!err.empty()) {
                    state.status += " - warning: " + err;
                }
            } else {
                state.status = "Reload JSON failed: " + err;
            }
            menu_refresh_files(state, view);
            break;

        case MENU_EXIT:
            if (ui_confirm(pad, "Exit llama.cpp for Switch?")) {
                state.quit = true;
            }
            break;

        default:
            break;
    }
    view.dirty = true;
}

// returns the direction buttons that fired this frame: new presses plus hold-to-repeat
static uint64_t menu_repeat(menu_view & view, uint64_t k_down, uint64_t k_held, uint64_t t) {
    const uint64_t pressed = k_down & k_repeat_buttons;
    if (pressed) {
        view.repeat_buttons  = pressed;
        view.repeat_start_us = t;
        view.repeat_last_us  = t;
        return pressed;
    }
    const uint64_t held = k_held & view.repeat_buttons;
    if (!held) {
        view.repeat_buttons = 0;
        return 0;
    }
    if (t - view.repeat_start_us >= k_repeat_delay_us && t - view.repeat_last_us >= k_repeat_interval_us) {
        view.repeat_last_us = t;
        return held;
    }
    return 0;
}

static void menu_jump_section(menu_view & view, int dir) {
    const int n = (int) k_section_count;
    const int s = (menu_section_of(view.cursor) + n + dir) % n;
    view.cursor = k_sections[s].first;
}

static bool menu_stats_changed(const app_state & state, const menu_view & view) {
    // ignore RAM changes below 1 MB to avoid redrawing for allocator noise
    const uint64_t mb = 1024 * 1024;
    return state.process_used / mb != view.drawn_used / mb || state.process_total != view.drawn_total ||
        state.cpu_hz != view.drawn_cpu || state.gpu_hz != view.drawn_gpu || state.emc_hz != view.drawn_emc ||
        state.clock_cpu_target_hz != view.drawn_cpu_target || state.clock_emc_target_hz != view.drawn_emc_target;
}

void run_main_menu(app_state & state, PadState & pad) {
    menu_view view;
    view.applet_mode = menu_is_applet_mode();
    menu_refresh_files(state, view);
    update_stats(state);

    while (ui_app_running() && !state.quit) {
        const uint64_t t = now_us();
        if (t - view.last_stats_check_us >= k_stats_interval_us) {
            view.last_stats_check_us = t;
            update_stats(state);
            if (menu_stats_changed(state, view)) {
                view.dirty = true;
            }
        }
        if (state.status != view.drawn_status) {
            view.dirty = true;
        }

        if (view.dirty) {
            render_menu(state, view);
            view.drawn_status = state.status;
            view.drawn_used   = state.process_used;
            view.drawn_total  = state.process_total;
            view.drawn_cpu    = state.cpu_hz;
            view.drawn_gpu    = state.gpu_hz;
            view.drawn_emc    = state.emc_hz;
            view.drawn_cpu_target = state.clock_cpu_target_hz;
            view.drawn_emc_target = state.clock_emc_target_hz;
            view.dirty = false;
        }

        padUpdate(&pad);
        const uint64_t k_down = padGetButtonsDown(&pad);
        const uint64_t k_move = menu_repeat(view, k_down, padGetButtons(&pad), t);

        if (k_move & HidNpadButton_AnyUp) {
            view.cursor = (view.cursor + MENU_COUNT - 1) % MENU_COUNT;
            view.dirty = true;
        } else if (k_move & HidNpadButton_AnyDown) {
            view.cursor = (view.cursor + 1) % MENU_COUNT;
            view.dirty = true;
        }
        if (k_move & HidNpadButton_AnyLeft) {
            adjust_selected_item(state, view, pad, -1);
            view.dirty = true;
        } else if (k_move & HidNpadButton_AnyRight) {
            adjust_selected_item(state, view, pad, +1);
            view.dirty = true;
        }
        if (k_down & (HidNpadButton_L | HidNpadButton_ZL)) {
            menu_jump_section(view, -1);
            view.dirty = true;
        }
        if (k_down & (HidNpadButton_R | HidNpadButton_ZR)) {
            menu_jump_section(view, +1);
            view.dirty = true;
        }
        if (k_down & HidNpadButton_A) {
            run_action(state, view, pad);
        } else if (k_down & HidNpadButton_Plus) {
            if (ui_confirm(pad, "Exit llama.cpp for Switch?")) {
                state.quit = true;
            }
            view.dirty = true;
        }

        svcSleepThread(k_frame_sleep_ns);
    }
}
