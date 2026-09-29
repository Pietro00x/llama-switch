#include "bench.h"
#include "engine.h"
#include "screens.h"
#include "sysinfo.h"
#include "ui.h"

#include <cstdio>
#include <string>
#include <vector>

// set for this file by CMakeLists.txt
#ifndef LLAMA_SWITCH_VERSION
#define LLAMA_SWITCH_VERSION "unknown"
#endif
#ifndef LLAMA_SWITCH_COMMIT
#define LLAMA_SWITCH_COMMIT "unknown"
#endif
#ifndef LLAMA_SWITCH_BUILD
#define LLAMA_SWITCH_BUILD 0
#endif

// Screen layout (80x45 console, rows/cols are 0-based, nothing is drawn in column 79):
//
//   row  0       title bar
//   rows 2-10    Setup box
//   rows 12-27   Results box (table header + up to 13 rows)
//   rows 29-39   help text, or the progress of the running benchmark
//   row  41      separator
//   rows 42-43   status line
//   row  44      button hints footer

static const char * k_bench_path = "sdmc:/switch/llama_cpp_switch/bench.txt";

constexpr int k_bench_width       = 79;
constexpr int k_bench_setup_row   = 2;
constexpr int k_bench_results_row = 12;
constexpr int k_bench_results_h   = 16;
constexpr int k_bench_info_row    = 29;
constexpr int k_bench_status_row  = 42;

constexpr uint64_t k_bench_redraw_timed_us = 1000000; // at most one redraw per second inside a timed run
constexpr uint64_t k_frame_sleep_ns        = 16 * 1000 * 1000ULL;

struct bench_view {
    bench_report   report;            // last finished run
    bool           has_report = false;
    bool           running    = false;
    bench_progress progress;          // of the running benchmark
    std::string    status;
    const char *   status_color = UI_FG_WHITE;

    // what the last idle frame showed, to redraw only on changes
    uint32_t drawn_cpu  = 0;
    uint32_t drawn_gpu  = 0;
    uint32_t drawn_emc  = 0;
    uint64_t drawn_used = 0;
};

static std::string bench_app_text() {
    return bench_printf("llama.cpp Switch %s (commit %s, build %d)", LLAMA_SWITCH_VERSION, LLAMA_SWITCH_COMMIT, (int) LLAMA_SWITCH_BUILD);
}

static std::string bench_device_text() {
    const bool docked = appletGetOperationMode() == AppletOperationMode_Console;
    const AppletType type = appletGetAppletType();
    const bool applet = type != AppletType_Application && type != AppletType_SystemApplication;
    const ApmPerformanceMode perf = appletGetPerformanceMode();
    const char * perf_name = perf == ApmPerformanceMode_Boost ? "boost" : perf == ApmPerformanceMode_Normal ? "normal" : "?";
    const u32 fw = hosversionGet();
    return bench_printf("%s | %s mode | perf mode %s | fw %u.%u.%u%s | core mask 0x%llx",
        docked ? "docked" : "handheld", applet ? "applet" : "application", perf_name,
        (unsigned) HOSVER_MAJOR(fw), (unsigned) HOSVER_MINOR(fw), (unsigned) HOSVER_MICRO(fw),
        hosversionIsAtmosphere() ? " AMS" : "", (unsigned long long) engine_core_mask());
}

static void bench_draw_line(int row, int col, const char * color, const std::string & text, int width) {
    ui_move(row, col);
    std::printf("%s%s" UI_RESET, color, ui_fit(ui_ascii(text), width).c_str());
}

static void bench_draw_field(int row, const std::string & label, const char * color, const std::string & value) {
    ui_move(row, 2);
    std::printf(UI_FAINT "%s" UI_RESET "%s%s" UI_RESET, ui_fit(label, 9).c_str(), color,
        ui_fit(clip_line(ui_ascii(value), 66), 66).c_str());
}

static void bench_draw_setup(const app_state & state) {
    const int top = k_bench_setup_row;
    std::printf(UI_FG_CYAN);
    ui_box(top, 0, 9, k_bench_width, "Setup");
    std::printf(UI_RESET);

    if (!state.model || !state.ctx) {
        bench_draw_field(top + 1, "Model", UI_FG_YELLOW, "no model loaded - load one in the main menu first");
    } else {
        char desc[128] = { 0 };
        llama_model_desc(state.model, desc, sizeof(desc));
        bench_draw_field(top + 1, "Model", UI_FG_WHITE, bench_loaded_model_name(state) + "  (" + desc + ", " +
            fmt_mb(llama_model_size(state.model)) + ")");
        bench_draw_field(top + 2, "Context", UI_FG_GRAY, bench_printf("n_ctx %u  n_batch %u  n_ubatch %u  flash-attn %s",
            (unsigned) llama_n_ctx(state.ctx), (unsigned) llama_n_batch(state.ctx), (unsigned) llama_n_ubatch(state.ctx),
            llama_flash_attn_type_name(llama_context_default_params().flash_attn_type)));
        bench_draw_field(top + 3, "Threads", UI_FG_GRAY, bench_printf("%d (model pool), %s", state.n_threads,
            state.threads_pinned ? "pinned to cores" : "unpinned (kernel placement)"));
    }
    bench_draw_field(top + 4, "Device", UI_FG_GRAY, bench_device_text());
    bench_draw_field(top + 5, "Clocks", UI_FG_GRAY, "CPU " + fmt_hz(state.cpu_hz) + "  GPU " + fmt_hz(state.gpu_hz) +
        "  EMC " + fmt_hz(state.emc_hz));
    bench_draw_field(top + 6, "Build", UI_FG_GRAY, bench_app_text());
    bench_draw_field(top + 7, "Output", UI_FG_GRAY, std::string(k_bench_path) + " (appended)");
}

static void bench_draw_results(const bench_report * report) {
    const int top = k_bench_results_row;
    std::printf(UI_FG_CYAN);
    ui_box(top, 0, k_bench_results_h, k_bench_width, "Results");
    std::printf(UI_RESET);

    if (!report) {
        bench_draw_line(top + 2, 2, UI_FAINT, "No results yet.", 74);
        return;
    }
    const std::vector<std::string> lines = bench_format_table(*report);
    for (size_t i = 0; i < lines.size() && (int) i < k_bench_results_h - 2; ++i) {
        const char * color = i == 0 ? UI_FG_CYAN : (lines[i].find("failed:") != std::string::npos ? UI_FG_RED : UI_FG_WHITE);
        bench_draw_line(top + 1 + (int) i, 2, color, lines[i], 75);
    }
    if (lines.size() <= 1) {
        bench_draw_line(top + 2, 2, UI_FAINT, report->error.empty() ? "Waiting for the first result..." : report->error, 74);
    }
}

static void bench_draw_help(const app_state & state) {
    const int row = k_bench_info_row;
    if (!state.ctx) {
        bench_draw_line(row, 1, UI_FG_YELLOW, "Load a model in the main menu first (Load model), then come back here.", 77);
        return;
    }
    // the thread setups of the full run, as bench_default_configs() makes them
    const int n = state.n_threads;
    std::string setups;
    for (int i = 1; i <= n; ++i) {
        setups += (i == 1 ? "" : (i == n ? " and " : ", ")) + std::to_string(i);
    }
    setups += n == 1 ? " thread" : " threads";
    if (n > 1) {
        setups += " plus " + std::to_string(n) + " pinned";
    }
    const std::string lines[] = {
        "A  Full run: pp128 + tg32 with " + setups + ", and the",
        "   memory bandwidth test (2 timed runs each; about 2-4 minutes).",
        "X  Quick run: the current thread setup only (3 timed runs, under a minute).",
        std::string("Y  Pin threads for chat: ") + (state.threads_pinned ? "ON -> OFF" : "OFF -> ON") +
            " (experiment, reset when the app restarts).",
        "",
        "Results are appended to bench.txt on the SD card: send us that file (or the",
        "last block of it). Leave the console alone while it runs.",
        "The chat history is kept; the next chat reply re-reads it once (slower).",
    };
    for (size_t i = 0; i < sizeof(lines) / sizeof(lines[0]); ++i) {
        bench_draw_line(row + (int) i, 1, i < 4 ? UI_FG_WHITE : UI_FG_GRAY, lines[i], 77);
    }
}

static void bench_draw_progress(const bench_view & view) {
    const int row = k_bench_info_row;
    const bench_progress & p = view.progress;
    bench_draw_line(row, 1, UI_FG_YELLOW, "Running: " + p.step, 77);

    const double frac = p.n_steps > 0 ? (double) p.step_index / (double) p.n_steps : 0.0;
    ui_move(row + 2, 1);
    std::printf("%s " UI_FG_WHITE "step %d of %d" UI_RESET, ui_progress_bar(frac, 50).c_str(), p.step_index + 1, p.n_steps);
    if (p.total > 0) {
        ui_move(row + 3, 1);
        std::printf("%s " UI_FG_WHITE "%d / %d tokens" UI_RESET,
            ui_progress_bar((double) p.done / (double) p.total, 50).c_str(), p.done, p.total);
    }
    bench_draw_line(row + 5, 1, UI_FG_GRAY, "Hold B to cancel. It is checked between steps and generated tokens, so a", 77);
    bench_draw_line(row + 6, 1, UI_FG_GRAY, "prompt step (up to ~20 s with 1 thread) finishes first.", 77);
    bench_draw_line(row + 7, 1, UI_FG_GRAY, "The screen updates at most once per second while a test is timed.", 77);
}

static void bench_render(const app_state & state, const bench_view & view, const bench_report * live) {
    ui_frame_begin();
    ui_title_bar("Benchmark", bench_printf("RAM %.0f / %.0f MB", (double) state.process_used / (1024.0 * 1024.0),
        (double) state.process_total / (1024.0 * 1024.0)));
    bench_draw_setup(state);
    bench_draw_results(live ? live : (view.has_report ? &view.report : nullptr));
    if (view.running) {
        bench_draw_progress(view);
    } else {
        bench_draw_help(state);
    }

    ui_hline(k_bench_status_row - 1, '-');
    const std::vector<std::string> status = ui_wrap(view.status, k_bench_width - 2);
    for (size_t i = 0; i < status.size() && i < 2; ++i) {
        bench_draw_line(k_bench_status_row + (int) i, 1, view.status_color, status[i], k_bench_width - 2);
    }
    if (view.running) {
        ui_footer("Hold B Cancel");
    } else if (state.ctx) {
        ui_footer(std::string("A Full run   X Quick run   Y Pin threads: ") + (state.threads_pinned ? "ON " : "OFF") + "   B Back");
    } else {
        ui_footer("B Back");
    }
    ui_frame_end();
}

// one line for the main menu status, e.g. "tg32 6.12 t/s, pp128 25.3 t/s (3 threads)"
static std::string bench_summary(const bench_report & r) {
    const bench_result * pp = nullptr;
    const bench_result * tg = nullptr;
    for (const bench_result & res : r.results) {
        if (res.samples.empty() || res.n_threads != r.n_threads || res.pinned != r.pinned) {
            continue;
        }
        if (res.test == BENCH_TEST_PP) {
            pp = &res;
        } else if (res.test == BENCH_TEST_TG) {
            tg = &res;
        }
    }
    std::string s;
    double mean = 0.0, stdev = 0.0;
    if (tg) {
        bench_stats(tg->samples, mean, stdev);
        s += bench_printf("tg%d %.2f t/s", tg->n, mean);
    }
    if (pp) {
        bench_stats(pp->samples, mean, stdev);
        s += bench_printf("%spp%d %.1f t/s", s.empty() ? "" : ", ", pp->n, mean);
    }
    if (!s.empty()) {
        s += bench_printf(" (%d threads%s)", r.n_threads, r.pinned ? ", pinned" : "");
    }
    return s;
}

static bool bench_has_samples(const bench_report & r) {
    for (const bench_result & res : r.results) {
        if (!res.samples.empty()) {
            return true;
        }
    }
    return false;
}

static void bench_start(app_state & state, PadState & pad, bench_view & view, bool full) {
    bench_params params;
    if (full) {
        params.configs     = bench_default_configs(state.n_threads);
        params.reps        = 2;
        params.mem_threads = state.n_threads;
    } else {
        bench_config c;
        c.n_threads = state.n_threads;
        c.pinned    = state.threads_pinned;
        params.configs.push_back(c);
        params.reps        = 3;
        params.mem_threads = 0;
    }

    bench_report report;
    report.timestamp = bench_timestamp();
    report.app       = bench_app_text();
    report.device    = bench_device_text();

    view.running  = true;
    view.progress = bench_progress();
    view.status       = full ? "Full benchmark running..." : "Quick benchmark running...";
    view.status_color = UI_FG_YELLOW;

    uint64_t last_draw = 0;
    bench_hooks hooks;
    hooks.progress = [&](const bench_progress & p) {
        view.progress = p;
        // held, not only pressed: a short press during a long decode is not seen by padUpdate
        padUpdate(&pad);
        const uint64_t buttons = padGetButtonsDown(&pad) | padGetButtons(&pad);
        if (!ui_app_running() || (buttons & HidNpadButton_B)) {
            return false;
        }
        const uint64_t t = now_us();
        if (!p.timing || t - last_draw >= k_bench_redraw_timed_us) {
            last_draw = t;
            bench_render(state, view, &report);
        }
        return true;
    };
    hooks.read_clocks = [&](uint32_t & cpu_hz, uint32_t & gpu_hz, uint32_t & emc_hz) {
        state.last_stats_us = 0; // bypass the 250 ms rate limit
        update_stats(state);
        cpu_hz = state.cpu_hz;
        gpu_hz = state.gpu_hz;
        emc_hz = state.emc_hz;
    };

    bench_render(state, view, &report);
    const bool completed = bench_run(state, params, hooks, report);
    view.running = false;

    view.report     = report;
    view.has_report = true;
    state.last_stats_us = 0;
    update_stats(state);

    if (!report.error.empty() && !bench_has_samples(report)) {
        view.status       = "Benchmark failed: " + report.error;
        view.status_color = UI_FG_RED;
        return;
    }
    if (!bench_has_samples(report)) {
        view.status       = "Benchmark cancelled before the first result (nothing saved)";
        view.status_color = UI_FG_YELLOW;
        return;
    }

    std::string err;
    const bool saved = bench_append_file(k_bench_path, bench_format_report(report), err);
    const std::string summary = bench_summary(report);
    if (!saved) {
        view.status       = "Could not save the results: " + err;
        view.status_color = UI_FG_RED;
    } else if (!completed) {
        view.status       = "Benchmark cancelled; partial results appended to bench.txt";
        view.status_color = UI_FG_YELLOW;
    } else {
        view.status       = "Done in " + bench_printf("%.0f s", report.duration_s) + (summary.empty() ? "" : ": " + summary) +
            ". Appended to bench.txt";
        view.status_color = UI_FG_GREEN;
    }
    if (saved && !summary.empty()) {
        state.status = "Benchmark: " + summary + " - saved to bench.txt";
    }
}

void run_bench_screen(app_state & state, PadState & pad) {
    bench_view view;
    view.status = state.ctx ? "Ready. A starts the full benchmark, X a quick one." : "No model loaded";
    view.status_color = state.ctx ? UI_FG_WHITE : UI_FG_YELLOW;

    bool dirty = true;
    update_stats(state);
    while (ui_app_running()) {
        update_stats(state); // rate limited to 4 per second
        const uint64_t mb = 1024 * 1024;
        if (state.cpu_hz != view.drawn_cpu || state.gpu_hz != view.drawn_gpu || state.emc_hz != view.drawn_emc ||
            state.process_used / mb != view.drawn_used / mb) {
            dirty = true;
        }
        if (dirty) {
            bench_render(state, view, nullptr);
            view.drawn_cpu  = state.cpu_hz;
            view.drawn_gpu  = state.gpu_hz;
            view.drawn_emc  = state.emc_hz;
            view.drawn_used = state.process_used;
            dirty = false;
        }

        padUpdate(&pad);
        const uint64_t k_down = padGetButtonsDown(&pad);
        if (k_down & HidNpadButton_B) {
            break;
        }
        if (state.ctx && (k_down & (HidNpadButton_A | HidNpadButton_X))) {
            bench_start(state, pad, view, (k_down & HidNpadButton_A) != 0);
            dirty = true;
        } else if (state.ctx && (k_down & HidNpadButton_Y)) {
            std::string err;
            const bool pin = !state.threads_pinned;
            if (engine_set_pin_threads(state, pin, err)) {
                view.status       = pin ? "Threads pinned to one core each (chat and quick run)" : "Threads unpinned (default placement)";
                view.status_color = UI_FG_GREEN;
            } else {
                view.status       = "Pinning failed: " + err;
                view.status_color = UI_FG_RED;
            }
            dirty = true;
        }
        svcSleepThread(k_frame_sleep_ns);
    }
}
