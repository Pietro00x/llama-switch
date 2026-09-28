#include "download.h"
#include "screens.h"
#include "sysinfo.h"
#include "ui.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

struct download_view {
    const model_entry * model       = nullptr;
    std::string         url;
    uint64_t            part_before = 0;     // .part size found before starting
    bool                has_free    = false;
    uint64_t            sd_free     = 0;
};

void render_header(const app_state & state, const download_view & v, const char * right) {
    ui_title_bar("Download model", right);

    ui_move(2, 1);
    std::printf(UI_FG_WHITE "Model " UI_RESET "%s", ui_fit(ui_ascii(v.model->name), 72).c_str());
    ui_move(3, 1);
    std::printf(UI_FG_WHITE "URL   " UI_RESET "%s", ui_fit(clip_line(ui_ascii(v.url), 72), 72).c_str());
    ui_move(4, 1);
    std::printf(UI_FG_WHITE "Path  " UI_RESET "%s", ui_fit(clip_line(ui_ascii(v.model->path), 72), 72).c_str());

    ui_move(5, 1);
    std::string sd = v.has_free ? dl_fmt_size(v.sd_free) + " free" : std::string("unknown");
    std::printf(UI_FG_WHITE "SD    " UI_RESET "%s", ui_fit(sd, 72).c_str());

    ui_move(20, 1);
    std::printf(UI_FG_GRAY "RAM %s / %s" UI_RESET,
        fmt_mb(state.process_used).c_str(), fmt_mb(state.process_total).c_str());
}

void render_progress(const app_state & state, const download_view & v, const download_progress & p, bool stalled) {
    ui_frame_begin();
    render_header(state, v, p.resumed_from > 0 ? "resuming" : "");

    ui_move(7, 1);
    if (p.connecting) {
        std::printf(UI_FG_YELLOW "Connecting..." UI_RESET);
        if (v.part_before > 0) {
            std::printf("  (resuming, %s already downloaded)", dl_fmt_size(v.part_before).c_str());
        }
    } else if (stalled) {
        std::printf(UI_FG_YELLOW "Waiting for data..." UI_RESET);
    } else {
        std::printf(UI_FG_GREEN "Downloading" UI_RESET);
    }

    if (!p.connecting) {
        ui_move(9, 1);
        if (p.total > 0) {
            const double frac = (double) p.received / (double) p.total;
            std::printf("%s " UI_FG_WHITE "%5.1f%%" UI_RESET, ui_progress_bar(frac, 70).c_str(), 100.0 * frac);
        } else {
            std::printf("%s", ui_fit("(size unknown)", 70).c_str());
        }

        ui_move(11, 1);
        const std::string done  = dl_fmt_size(p.received);
        const std::string total = p.total > 0 ? dl_fmt_size(p.total) : std::string("?");
        std::printf("Received  " UI_FG_WHITE "%s" UI_RESET " / %s", done.c_str(), total.c_str());
        ui_move(12, 1);
        std::printf("Speed     " UI_FG_WHITE "%s" UI_RESET, dl_fmt_speed(p.speed_bps).c_str());
        ui_move(13, 1);
        std::printf("Time left " UI_FG_WHITE "%s" UI_RESET, dl_fmt_duration(p.eta_s).c_str());
        if (p.resumed_from > 0) {
            ui_move(14, 1);
            std::printf("Resumed   %s were already on the SD card", dl_fmt_size(p.resumed_from).c_str());
        }
    }

    ui_move(17, 1);
    std::printf(UI_FG_GRAY "Cancelling keeps the partial file (.part);" UI_RESET);
    ui_move(18, 1);
    std::printf(UI_FG_GRAY "start the download again later to resume it." UI_RESET);

    ui_footer("B cancel");
    ui_frame_end();
}

// Shows the final message until A or B is pressed.
void show_result(app_state & state, PadState & pad, const download_view & v, bool ok, const std::string & msg) {
    ui_frame_begin();
    render_header(state, v, ok ? "done" : "failed");

    ui_move(7, 1);
    if (ok) {
        std::printf(UI_FG_GREEN "Download complete." UI_RESET);
    } else {
        std::printf(UI_FG_RED "Download failed:" UI_RESET);
    }
    const std::vector<std::string> lines = ui_wrap(msg, 76);
    for (size_t i = 0; i < lines.size() && i < 8; ++i) {
        ui_move(9 + (int) i, 3);
        std::printf("%s", lines[i].c_str());
    }
    if (!ok) {
        const uint64_t part = download_part_size(v.model->path);
        if (part > 0) {
            ui_move(18, 1);
            std::printf(UI_FG_GRAY "%s kept in the .part file; download again to resume." UI_RESET,
                dl_fmt_size(part).c_str());
        }
    }

    ui_footer("A/B back");
    ui_frame_end();

    while (ui_app_running()) {
        padUpdate(&pad);
        if (padGetButtonsDown(&pad) & (HidNpadButton_A | HidNpadButton_B)) {
            return;
        }
        svcSleepThread(16 * 1000 * 1000ULL);
    }
}

} // namespace

void run_download_screen(app_state & state, PadState & pad) {
    if (state.models.empty()) {
        state.status = "No model selected";
        return;
    }
    const model_entry & selected = state.models[state.selected_model];

    download_view view;
    view.model       = &selected;
    view.url         = normalize_hf_url(selected.url);
    view.part_before = download_part_size(selected.path);
    view.has_free    = download_free_space(selected.path, view.sd_free);

    uint64_t          last_draw     = 0;
    uint64_t          last_applet   = 0;
    uint64_t          last_received = 0;
    uint64_t          last_data_us  = now_us();
    download_progress last;
    last.connecting   = true;
    last.received     = view.part_before;
    last.resumed_from = view.part_before;

    auto progress_cb = [&](const download_progress & p) {
        const uint64_t t = now_us();
        if (p.received != last_received || p.connecting) {
            last_received = p.received;
            last_data_us  = t;
        }
        last = p;

        // honor HOME -> close / system exit requests during long transfers
        if (t - last_applet >= 100000) {
            last_applet = t;
            if (!ui_app_running()) {
                state.quit = true;
                return false;
            }
        }

        padUpdate(&pad);
        if (padGetButtonsDown(&pad) & HidNpadButton_B) {
            render_progress(state, view, p, false);
            if (ui_confirm(pad, "Cancel the download? The partial file is kept for resuming.")) {
                return false;
            }
            last_draw = 0; // redraw over the dialog right away
        }

        const bool finished = p.total > 0 && p.received >= p.total;
        if (!finished && t - last_draw < 200000) {
            return true;
        }
        last_draw = t;
        update_stats(state);
        render_progress(state, view, p, t - last_data_us > 2000000);
        return true;
    };

    update_stats(state);
    render_progress(state, view, last, false);

    std::string err;
    const download_result res = download_file(selected.url, selected.path, err, progress_cb);
    if (res == DOWNLOAD_RESULT_CANCELLED) {
        const uint64_t part = download_part_size(selected.path);
        state.status = part > 0 ? "Download cancelled; " + dl_fmt_size(part) + " kept for resume" : "Download cancelled";
        return;
    }
    if (state.quit) {
        return;
    }
    if (res == DOWNLOAD_RESULT_OK) {
        state.status = "Download complete: " + selected.path;
        show_result(state, pad, view, true, "Saved " + dl_fmt_size(last.total > 0 ? last.total : last.received) +
            " to " + selected.path);
    } else {
        state.status = "Download failed: " + err;
        show_result(state, pad, view, false, err);
    }
}
